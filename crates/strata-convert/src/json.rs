//! just enough json to read a safetensors header.
//!
//! # why this is here rather than a dependency
//!
//! the workspace has no external dependencies on purpose: the layout file is an
//! on-disk contract other processes read, so its byte handling should be
//! visible in the source. pulling in a general json crate to read one header
//! would trade that for convenience on the one file in the pipeline that is not
//! performance sensitive and is at most a few tens of kilobytes.
//!
//! so this is a complete parser for the json subset safetensors headers use,
//! and it is strict: anything it does not understand is an error rather than a
//! best guess. a header that parses here parsed exactly, and a header that does
//! not says where it stopped.

use std::fmt;

/// a parsed json value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// key-value pairs, in the order they appeared.
    ///
    /// a vector rather than a map because a safetensors header holds a few
    /// hundred keys, lookup is not hot, and preserving order keeps error
    /// messages in the order a human reads the file.
    Object(Vec<(String, Value)>),
    /// a json array.
    Array(Vec<Value>),
    /// a json string, with escapes already resolved.
    Str(String),
    /// a json number. safetensors only uses integers, but a number is a number.
    Num(f64),
    /// `true` or `false`.
    Bool(bool),
    /// `null`.
    Null,
}

impl Value {
    /// look up a key, if this is an object.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// the pairs of an object.
    #[must_use]
    pub fn entries(&self) -> Option<&[(String, Value)]> {
        match self {
            Value::Object(pairs) => Some(pairs),
            _ => None,
        }
    }

    /// this value as a string slice.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    /// this value as a non-negative integer.
    ///
    /// returns `None` for a negative or fractional number, because every number
    /// in a safetensors header is a length, a shape or a byte offset, and none
    /// of those can be either.
    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Num(n) if *n >= 0.0 && n.fract() == 0.0 => Some(*n as u64),
            _ => None,
        }
    }

    /// the elements of an array.
    #[must_use]
    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(items) => Some(items),
            _ => None,
        }
    }
}

/// where and why parsing stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    /// byte offset into the header at which parsing failed.
    pub at: usize,
    /// what was expected there.
    pub message: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "json byte {}: {}", self.at, self.message)
    }
}

impl std::error::Error for ParseError {}

/// parse a whole json document, rejecting trailing content.
///
/// # Errors
/// returns the byte offset and what was expected there.
pub fn parse(input: &[u8]) -> Result<Value, ParseError> {
    let mut p = Parser { input, at: 0 };
    p.skip_ws();
    let value = p.value()?;
    p.skip_ws();
    if p.at != input.len() {
        return Err(p.err("trailing content after the top level value"));
    }
    Ok(value)
}

struct Parser<'a> {
    input: &'a [u8],
    at: usize,
}

impl Parser<'_> {
    fn err(&self, message: &str) -> ParseError {
        ParseError {
            at: self.at,
            message: message.to_string(),
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.at).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.at += 1;
        }
    }

    fn expect(&mut self, byte: u8) -> Result<(), ParseError> {
        if self.peek() == Some(byte) {
            self.at += 1;
            Ok(())
        } else {
            Err(self.err(&format!("expected {:?}", byte as char)))
        }
    }

    fn literal(&mut self, word: &str, value: Value) -> Result<Value, ParseError> {
        if self.input[self.at..].starts_with(word.as_bytes()) {
            self.at += word.len();
            Ok(value)
        } else {
            Err(self.err(&format!("expected {word}")))
        }
    }

    fn value(&mut self) -> Result<Value, ParseError> {
        match self.peek() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => self.string().map(Value::Str),
            Some(b't') => self.literal("true", Value::Bool(true)),
            Some(b'f') => self.literal("false", Value::Bool(false)),
            Some(b'n') => self.literal("null", Value::Null),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number(),
            Some(_) => Err(self.err("expected a value")),
            None => Err(self.err("unexpected end of input")),
        }
    }

    fn object(&mut self) -> Result<Value, ParseError> {
        self.expect(b'{')?;
        let mut pairs = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.at += 1;
            return Ok(Value::Object(pairs));
        }
        loop {
            self.skip_ws();
            let key = self.string()?;
            self.skip_ws();
            self.expect(b':')?;
            self.skip_ws();
            let value = self.value()?;
            pairs.push((key, value));
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b'}') => {
                    self.at += 1;
                    return Ok(Value::Object(pairs));
                }
                _ => return Err(self.err("expected ',' or '}'")),
            }
        }
    }

    fn array(&mut self) -> Result<Value, ParseError> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.at += 1;
            return Ok(Value::Array(items));
        }
        loop {
            self.skip_ws();
            items.push(self.value()?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b']') => {
                    self.at += 1;
                    return Ok(Value::Array(items));
                }
                _ => return Err(self.err("expected ',' or ']'")),
            }
        }
    }

    fn string(&mut self) -> Result<String, ParseError> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            let Some(c) = self.peek() else {
                return Err(self.err("unterminated string"));
            };
            self.at += 1;
            match c {
                b'"' => return Ok(out),
                b'\\' => {
                    let Some(esc) = self.peek() else {
                        return Err(self.err("unterminated escape"));
                    };
                    self.at += 1;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => out.push(self.unicode_escape()?),
                        _ => return Err(self.err("unknown escape")),
                    }
                }
                // a raw byte. tensor names are ascii in practice, but metadata
                // values are arbitrary, so decode utf-8 properly rather than
                // assuming.
                _ => {
                    let start = self.at - 1;
                    let len = utf8_len(c).ok_or_else(|| self.err("invalid utf-8 lead byte"))?;
                    if start + len > self.input.len() {
                        return Err(self.err("truncated utf-8 sequence"));
                    }
                    let s = std::str::from_utf8(&self.input[start..start + len])
                        .map_err(|_| self.err("invalid utf-8"))?;
                    out.push_str(s);
                    self.at = start + len;
                }
            }
        }
    }

    /// a `\uXXXX` escape, joining a surrogate pair if one follows.
    fn unicode_escape(&mut self) -> Result<char, ParseError> {
        let high = self.hex4()?;
        if (0xD800..0xDC00).contains(&high) {
            if !self.input[self.at..].starts_with(b"\\u") {
                return Err(self.err("high surrogate without a low surrogate"));
            }
            self.at += 2;
            let low = self.hex4()?;
            if !(0xDC00..0xE000).contains(&low) {
                return Err(self.err("expected a low surrogate"));
            }
            let combined = 0x1_0000 + ((high - 0xD800) << 10) + (low - 0xDC00);
            return char::from_u32(combined).ok_or_else(|| self.err("invalid code point"));
        }
        char::from_u32(high).ok_or_else(|| self.err("invalid code point"))
    }

    fn hex4(&mut self) -> Result<u32, ParseError> {
        if self.at + 4 > self.input.len() {
            return Err(self.err("truncated \\u escape"));
        }
        let mut value = 0u32;
        for i in 0..4 {
            let d = self.input[self.at + i];
            let digit = (d as char)
                .to_digit(16)
                .ok_or_else(|| self.err("bad hex digit in \\u escape"))?;
            value = value * 16 + digit;
        }
        self.at += 4;
        Ok(value)
    }

    fn number(&mut self) -> Result<Value, ParseError> {
        let start = self.at;
        if self.peek() == Some(b'-') {
            self.at += 1;
        }
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.at += 1;
        }
        if self.peek() == Some(b'.') {
            self.at += 1;
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.at += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.at += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.at += 1;
            }
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.at += 1;
            }
        }
        let text = std::str::from_utf8(&self.input[start..self.at])
            .map_err(|_| self.err("number is not utf-8"))?;
        text.parse::<f64>().map(Value::Num).map_err(|_| ParseError {
            at: start,
            message: "not a number".to_string(),
        })
    }
}

/// length in bytes of a utf-8 sequence with this lead byte.
const fn utf8_len(lead: u8) -> Option<usize> {
    match lead {
        0x00..=0x7F => Some(1),
        0xC2..=0xDF => Some(2),
        0xE0..=0xEF => Some(3),
        0xF0..=0xF4 => Some(4),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{Value, parse};

    #[test]
    fn parses_the_shape_a_safetensors_header_has() {
        let src = br#"{"__metadata__":{"format":"pt"},
            "a.weight":{"dtype":"BF16","shape":[32,1024,512],"data_offsets":[0,33554432]}}"#;
        let v = parse(src).expect("parses");

        let t = v.get("a.weight").expect("tensor present");
        assert_eq!(t.get("dtype").and_then(Value::as_str), Some("BF16"));
        let shape: Vec<u64> = t
            .get("shape")
            .and_then(Value::as_array)
            .expect("shape")
            .iter()
            .map(|d| d.as_u64().expect("dim"))
            .collect();
        assert_eq!(shape, vec![32, 1024, 512]);
        let offsets = t
            .get("data_offsets")
            .and_then(Value::as_array)
            .expect("off");
        assert_eq!(offsets[1].as_u64(), Some(33_554_432));
    }

    #[test]
    fn resolves_escapes_including_surrogate_pairs() {
        // u+1f600 arrives as a surrogate pair, which is the case a naive
        // \uXXXX reader turns into two replacement characters
        let v = parse(br#"{"k":"a\"b\\c\nA\ud83d\ude00"}"#).expect("parses");
        assert_eq!(
            v.get("k").and_then(Value::as_str),
            Some("a\"b\\c\nA\u{1F600}")
        );
    }

    #[test]
    fn passes_through_raw_multibyte_utf8() {
        // metadata values are arbitrary text, so those bytes arrive
        // undecorated rather than as escapes
        let mut src = br#"{"k":""#.to_vec();
        src.extend_from_slice("h\u{e9}llo\u{2192}".as_bytes());
        src.extend_from_slice(br#""}"#);

        let v = parse(&src).expect("parses");
        assert_eq!(
            v.get("k").and_then(Value::as_str),
            Some("h\u{e9}llo\u{2192}")
        );
    }

    #[test]
    fn rejects_trailing_content_rather_than_ignoring_it() {
        let err = parse(br#"{"a":1} junk"#).expect_err("must reject");
        assert!(err.message.contains("trailing"), "{err}");
    }

    #[test]
    fn reports_where_it_stopped() {
        let err = parse(br#"{"a":}"#).expect_err("must reject");
        assert_eq!(err.at, 5, "{err}");
    }

    #[test]
    fn a_negative_or_fractional_number_is_not_a_length() {
        let v = parse(br#"{"a":-1,"b":1.5,"c":7}"#).expect("parses");
        assert_eq!(v.get("a").and_then(Value::as_u64), None);
        assert_eq!(v.get("b").and_then(Value::as_u64), None);
        assert_eq!(v.get("c").and_then(Value::as_u64), Some(7));
    }
}
