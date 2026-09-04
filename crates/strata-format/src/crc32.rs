//! crc32 (ieee 802.3 polynomial, reflected) over byte slices.
//!
//! the format checksums the header, the index region, and every expert payload.
//! the point is not security, it is catching a truncated or torn write before
//! those bytes are fed to a gemm and turn into silently wrong logits.

const POLY: u32 = 0xEDB8_8320;

const fn build_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ POLY
            } else {
                crc >> 1
            };
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

static TABLE: [u32; 256] = build_table();

/// running crc32 state, so a payload can be checksummed as it is streamed to disk
/// without ever holding the whole expert in memory twice.
#[derive(Debug, Clone, Copy)]
pub struct Crc32(u32);

impl Default for Crc32 {
    fn default() -> Self {
        Self::new()
    }
}

impl Crc32 {
    /// a fresh checksum.
    #[must_use]
    pub const fn new() -> Self {
        Self(0xFFFF_FFFF)
    }

    /// fold more bytes into the checksum.
    pub fn update(&mut self, bytes: &[u8]) {
        let mut crc = self.0;
        for &b in bytes {
            let idx = ((crc ^ u32::from(b)) & 0xFF) as usize;
            crc = (crc >> 8) ^ TABLE[idx];
        }
        self.0 = crc;
    }

    /// the finished checksum.
    #[must_use]
    pub const fn finish(self) -> u32 {
        self.0 ^ 0xFFFF_FFFF
    }
}

/// one shot checksum of a slice.
#[must_use]
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut c = Crc32::new();
    c.update(bytes);
    c.finish()
}

#[cfg(test)]
mod tests {
    use super::{Crc32, crc32};

    #[test]
    fn known_vectors() {
        assert_eq!(crc32(b""), 0x0000_0000);
        assert_eq!(crc32(b"a"), 0xE8B7_BE43);
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(
            crc32(b"The quick brown fox jumps over the lazy dog"),
            0x414F_A339
        );
    }

    #[test]
    fn streaming_matches_one_shot() {
        let data: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
        let mut c = Crc32::new();
        for chunk in data.chunks(97) {
            c.update(chunk);
        }
        assert_eq!(c.finish(), crc32(&data));
    }
}
