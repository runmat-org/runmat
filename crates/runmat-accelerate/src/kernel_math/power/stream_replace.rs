//! Bounded-state replacement used by the established power WGSL spelling.
//! Each stage streams into the next; neither count nor emission materializes
//! an intermediate copy of the complete helper source.

use std::fmt::{self, Write};

pub(super) struct ReplaceWriter<'a, W: Write> {
    target: &'a mut W,
    needle: &'static str,
    replacement: [&'a str; 2],
    pending: [char; 32],
    used: usize,
}

impl<'a, W: Write> ReplaceWriter<'a, W> {
    pub(super) fn new(target: &'a mut W, needle: &'static str, replacement: [&'a str; 2]) -> Self {
        assert!(needle.is_ascii() && !needle.is_empty() && needle.len() <= 32);
        Self {
            target,
            needle,
            replacement,
            pending: ['\0'; 32],
            used: 0,
        }
    }

    fn pending_is_prefix(&self) -> bool {
        self.used <= self.needle.len()
            && self
                .needle
                .chars()
                .zip(self.pending[..self.used].iter().copied())
                .take(self.used)
                .all(|(expected, actual)| expected == actual)
    }

    fn flush_first(&mut self) -> fmt::Result {
        self.target.write_char(self.pending[0])?;
        self.pending.copy_within(1..self.used, 0);
        self.used -= 1;
        Ok(())
    }

    pub(super) fn finish(mut self) -> fmt::Result {
        while self.used != 0 {
            self.flush_first()?;
        }
        Ok(())
    }
}

impl<W: Write> Write for ReplaceWriter<'_, W> {
    fn write_str(&mut self, fragment: &str) -> fmt::Result {
        for character in fragment.chars() {
            self.pending[self.used] = character;
            self.used += 1;
            while !self.pending_is_prefix() {
                self.flush_first()?;
            }
            if self.used == self.needle.len() {
                self.target.write_str(self.replacement[0])?;
                self.target.write_str(self.replacement[1])?;
                self.used = 0;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replacement_spans_fragments_and_preserves_other_unicode() {
        let mut output = String::new();
        let mut writer = ReplaceWriter::new(&mut output, "finite(", ["rm_", "finite("]);
        writer.write_str("λfini").unwrap();
        writer.write_str("te(a) and fin").unwrap();
        writer.write_str("ite(b)").unwrap();
        writer.finish().unwrap();
        assert_eq!(output, "λrm_finite(a) and rm_finite(b)");
    }

    #[test]
    fn nested_streams_match_ordered_string_replacement() {
        let original = "fn finite(a: T) { return nan(T(0.0)); }";
        let expected = original
            .replace('T', "f64")
            .replace("finite(", "rm_finite(")
            .replace("nan(", "rm_nan(");
        let mut output = String::new();
        let mut nan = ReplaceWriter::new(&mut output, "nan(", ["rm_", "nan("]);
        let mut finite = ReplaceWriter::new(&mut nan, "finite(", ["rm_", "finite("]);
        let mut ty = ReplaceWriter::new(&mut finite, "T", ["f64", ""]);
        ty.write_str("fn fini").unwrap();
        ty.write_str("te(a: T) { return nan(T(0.0)); }").unwrap();
        ty.finish().unwrap();
        finite.finish().unwrap();
        nan.finish().unwrap();
        assert_eq!(output, expected);
    }
}
