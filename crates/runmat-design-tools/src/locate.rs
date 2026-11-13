//! Source positions: byte offsets (what the compilers report) to the UTF-16 offsets editors
//! count and one-based lines and columns people read.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A range of one file: UTF-16 offsets `start..end` and the one-based line and column (in
/// characters) of `start`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Range {
    pub path: String,
    pub start: usize,
    pub end: usize,
    pub line: usize,
    pub column: usize,
}

/// Converts positions in one text.
pub struct Locator<'a> {
    text: &'a str,
    /// Byte offset of each line's start.
    lines: Vec<usize>,
}

impl<'a> Locator<'a> {
    pub fn new(text: &'a str) -> Self {
        let mut lines = vec![0];
        lines.extend(text.match_indices('\n').map(|(i, _)| i + 1));
        Locator { text, lines }
    }

    fn floor(&self, mut at: usize) -> usize {
        at = at.min(self.text.len());
        while at > 0 && !self.text.is_char_boundary(at) {
            at -= 1;
        }
        at
    }

    /// The UTF-16 offset of byte offset `at`.
    pub fn utf16(&self, at: usize) -> usize {
        let head = &self.text[..self.floor(at)];
        if head.is_ascii() {
            head.len()
        } else {
            head.encode_utf16().count()
        }
    }

    /// The byte offset of UTF-16 offset `units`.
    pub fn byte(&self, units: usize) -> usize {
        if self.text.is_ascii() {
            return units.min(self.text.len());
        }
        let mut n = 0;
        for (b, ch) in self.text.char_indices() {
            if n >= units {
                return b;
            }
            n += ch.len_utf16();
        }
        self.text.len()
    }

    /// One-based line and column of byte offset `at`.
    pub fn line_column(&self, at: usize) -> (usize, usize) {
        let at = self.floor(at);
        let line = match self.lines.binary_search(&at) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        let column = self.text[self.lines[line]..at].chars().count();
        (line + 1, column + 1)
    }

    /// The range of bytes `start..end` in `path`.
    pub fn range(&self, path: &str, start: usize, end: usize) -> Range {
        let (line, column) = self.line_column(start);
        Range {
            path: path.to_owned(),
            start: self.utf16(start),
            end: self.utf16(end.max(start)),
            line,
            column,
        }
    }

    /// The range of UTF-16 offsets `start..end` in `path`.
    pub fn range_utf16(&self, path: &str, start: usize, end: usize) -> Range {
        let b = self.byte(start);
        let (line, column) = self.line_column(b);
        Range {
            path: path.to_owned(),
            start,
            end: end.max(start),
            line,
            column,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_and_columns_are_one_based_and_utf16_counts_units() {
        let text = "ab\nc±d\n";
        let l = Locator::new(text);
        assert_eq!(l.line_column(0), (1, 1));
        assert_eq!(l.line_column(3), (2, 1));
        let d = text.find('d').unwrap();
        assert_eq!(l.line_column(d), (2, 3));
        assert_eq!(l.utf16(d), 5);
        assert_eq!(l.byte(5), d);
    }
}
