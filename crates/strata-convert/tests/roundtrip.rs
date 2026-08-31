//! build a safetensors file, convert it, read it back, compare every byte.
//!
//! the converter's only job is to move weights without changing them, so the
//! test that matters is byte equality against the source rather than anything
//! about sizes or counts. a converter that loses a row produces a file of
//! exactly the right length full of plausible numbers, and a model that is
//! subtly wrong is worse than one that fails to load.
//!
//! the fixture is synthesised here rather than checked in, because a real
//! checkpoint is gigabytes and the thing under test is the byte arithmetic,
//! which does not care how large the tensors are.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use strata_convert::{SafeTensors, convert, plan};
use strata_format::{ExpertKey, LayoutReader};

/// a directory that removes itself, so a failing test does not leave gigabytes
/// of fixtures behind.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "strata-convert-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        fs::create_dir_all(&path).expect("create temp dir");
        Self(path)
    }

    fn join(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// one tensor to write into the fixture.
struct Tensor {
    name: String,
    shape: Vec<u64>,
    bytes: Vec<u8>,
}

/// write a minimal safetensors file: 8 byte length, json header, then data.
fn write_safetensors(path: &Path, tensors: &[Tensor]) {
    let mut header = String::from("{");
    let mut offset = 0u64;
    for (i, t) in tensors.iter().enumerate() {
        if i > 0 {
            header.push(',');
        }
        let shape: Vec<String> = t.shape.iter().map(ToString::to_string).collect();
        let end = offset + t.bytes.len() as u64;
        write!(
            header,
            r#""{}":{{"dtype":"BF16","shape":[{}],"data_offsets":[{},{}]}}"#,
            t.name,
            shape.join(","),
            offset,
            end
        )
        .expect("writing to a string cannot fail");
        offset = end;
    }
    header.push('}');

    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for t in tensors {
        out.extend_from_slice(&t.bytes);
    }
    fs::write(path, out).expect("write fixture");
}

/// distinct, position dependent bytes, so a slice copied from the wrong offset
/// cannot accidentally match.
fn fill(seed: u64, len: usize) -> Vec<u8> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 24) as u8
        })
        .collect()
}

/// a two layer, three expert model in the granite naming convention.
fn build_fixture(dir: &TempDir) -> (PathBuf, BTreeMap<(u32, u32), Vec<u8>>) {
    const LAYERS: u32 = 2;
    const EXPERTS: u32 = 3;
    // bf16, so an even number of bytes per row
    const IN_BYTES: usize = 64;
    const OUT_BYTES: usize = 32;

    let mut tensors = Vec::new();
    let mut expected: BTreeMap<(u32, u32), Vec<u8>> = BTreeMap::new();

    for layer in 0..LAYERS {
        let input = fill(u64::from(layer) + 1, IN_BYTES * EXPERTS as usize);
        let output = fill(u64::from(layer) + 100, OUT_BYTES * EXPERTS as usize);

        for expert in 0..EXPERTS {
            // the payload order is the sorted projection name:
            // input_linear.weight then output_linear.weight
            let mut payload = Vec::new();
            let i0 = expert as usize * IN_BYTES;
            let o0 = expert as usize * OUT_BYTES;
            payload.extend_from_slice(&input[i0..i0 + IN_BYTES]);
            payload.extend_from_slice(&output[o0..o0 + OUT_BYTES]);
            expected.insert((layer, expert), payload);
        }

        tensors.push(Tensor {
            name: format!("model.layers.{layer}.block_sparse_moe.input_linear.weight"),
            shape: vec![u64::from(EXPERTS), 4, (IN_BYTES / 8) as u64],
            bytes: input,
        });
        tensors.push(Tensor {
            name: format!("model.layers.{layer}.block_sparse_moe.output_linear.weight"),
            shape: vec![u64::from(EXPERTS), 4, (OUT_BYTES / 8) as u64],
            bytes: output,
        });
        // the router must be skipped, and it is deliberately shaped like an
        // expert tensor so that only the naming rule can exclude it
        tensors.push(Tensor {
            name: format!("model.layers.{layer}.block_sparse_moe.router.layer.weight"),
            shape: vec![u64::from(EXPERTS), 8],
            bytes: fill(u64::from(layer) + 900, 2 * EXPERTS as usize * 8),
        });
    }

    // attention and embeddings must not be pulled in
    tensors.push(Tensor {
        name: "model.embed_tokens.weight".to_string(),
        shape: vec![16, 8],
        bytes: fill(7, 2 * 16 * 8),
    });
    tensors.push(Tensor {
        name: "model.layers.0.self_attn.k_proj.weight".to_string(),
        shape: vec![8, 8],
        bytes: fill(8, 2 * 8 * 8),
    });

    let path = dir.join("model.safetensors");
    write_safetensors(&path, &tensors);
    (path, expected)
}

#[test]
fn every_expert_byte_survives_the_round_trip() {
    let dir = TempDir::new("roundtrip");
    let (source_path, expected) = build_fixture(&dir);
    let out = dir.join("model.strata");

    let mut source = SafeTensors::open(&source_path).expect("open fixture");
    let plan = plan::plan(&source).expect("plan fixture");

    assert_eq!(plan.layers, 2);
    assert_eq!(plan.experts_per_layer, 3);
    assert_eq!(plan.experts.len(), 6);
    assert_eq!(
        plan.projections,
        vec![
            "input_linear.weight".to_string(),
            "output_linear.weight".to_string()
        ],
        "the router and attention must not be projections"
    );

    let report = convert(&mut source, &plan, &out, "fixture").expect("convert");
    assert_eq!(report.experts, 6);

    let reader = LayoutReader::open(&out).expect("open layout");
    for ((layer, expert), want) in &expected {
        let key = ExpertKey::new(*layer, *expert);
        let got = reader.read_expert(key).expect("read expert back");
        assert_eq!(&got, want, "{key} came back different from what went in");
    }
}

#[test]
fn a_router_shaped_like_an_expert_is_still_not_an_expert() {
    let dir = TempDir::new("router");
    let (source_path, _) = build_fixture(&dir);

    let source = SafeTensors::open(&source_path).expect("open fixture");
    let plan = plan::plan(&source).expect("plan fixture");

    // three experts per layer, not six: the router tensor also has the expert
    // count as its outermost dimension and must be excluded by name
    assert_eq!(plan.experts_per_layer, 3);
    for expert in &plan.experts {
        for part in &expert.parts {
            assert!(
                !part.tensor.contains("router"),
                "{} pulled the router into an expert payload",
                part.tensor
            );
        }
    }
}

#[test]
fn a_checkpoint_with_no_experts_is_refused_rather_than_written_empty() {
    let dir = TempDir::new("dense");
    let path = dir.join("dense.safetensors");
    write_safetensors(
        &path,
        &[Tensor {
            name: "model.layers.0.self_attn.q_proj.weight".to_string(),
            shape: vec![8, 8],
            bytes: fill(3, 2 * 8 * 8),
        }],
    );

    let source = SafeTensors::open(&path).expect("open");
    assert!(
        plan::plan(&source).is_err(),
        "a dense model has no experts and converting it would produce an empty \
         layout file that looks valid"
    );
}

#[test]
fn a_header_that_disagrees_with_itself_is_rejected() {
    let dir = TempDir::new("bad");
    let path = dir.join("bad.safetensors");

    // shape says 4 elements of bf16, so 8 bytes, but the offsets span 16
    let header = r#"{"a":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,16]}}"#;
    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&[0u8; 16]);
    fs::write(&path, out).expect("write");

    let err = SafeTensors::open(&path).expect_err("must reject");
    let text = err.to_string();
    assert!(
        text.contains('8') && text.contains("16"),
        "the error should name both sizes, got: {text}"
    );
}
