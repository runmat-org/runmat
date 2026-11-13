//! Bounded, fallible copying of user-visible grouping text.

use super::super::super::{allocation_failure, BOXPLOT_ERROR_INVALID_ARGUMENT, NAME};
use runmat_value::{CharArray, Utf16DecodeError};

const COPY_QUANTUM_BYTES: usize = 256;

#[derive(Clone, Copy)]
enum TextSource<'a> {
    Utf8(&'a str),
    CharacterRow { array: &'a CharArray, row: usize },
}

pub(super) enum TextCopyProgress {
    Work(u64),
    Complete(Option<String>),
}

pub(super) struct TextCopy<'a> {
    source: TextSource<'a>,
    cursor: usize,
    output: String,
    last_non_space_end: usize,
}

impl<'a> TextCopy<'a> {
    pub(super) fn utf8(text: &'a str) -> Self {
        Self::new(TextSource::Utf8(text))
    }

    pub(super) fn character_row(array: &'a CharArray, row: usize) -> Self {
        Self::new(TextSource::CharacterRow { array, row })
    }

    fn new(source: TextSource<'a>) -> Self {
        Self {
            source,
            cursor: 0,
            output: String::new(),
            last_non_space_end: 0,
        }
    }

    pub(super) fn step(&mut self) -> crate::BuiltinResult<TextCopyProgress> {
        let mut copied = 0usize;
        while copied < COPY_QUANTUM_BYTES {
            let Some(character) = self.next_character()? else {
                if matches!(&self.source, TextSource::CharacterRow { array, .. } if array.rows > 1)
                {
                    self.output.truncate(self.last_non_space_end);
                }
                let text = std::mem::take(&mut self.output);
                // Typed absence is handled before text copying. Empty strings
                // and blank character rows are omitted by boxplot's grouping
                // contract; present spellings never encode missingness.
                let present = match self.source {
                    TextSource::Utf8(_) => !text.is_empty(),
                    TextSource::CharacterRow { .. } => self.last_non_space_end != 0,
                };
                return Ok(TextCopyProgress::Complete(present.then_some(text)));
            };
            let bytes = character.len_utf8();
            if self.output.capacity() - self.output.len() < bytes {
                self.output.try_reserve(bytes).map_err(|error| {
                    allocation_failure("boxplot: grouping label exceeds available memory", error)
                })?;
            }
            self.output.push(character);
            if !character.is_whitespace() {
                self.last_non_space_end = self.output.len();
            }
            copied += bytes;
        }
        Ok(TextCopyProgress::Work(copied as u64))
    }

    fn next_character(&mut self) -> crate::BuiltinResult<Option<char>> {
        match self.source {
            TextSource::Utf8(text) => {
                let Some(character) = text.get(self.cursor..).and_then(|tail| tail.chars().next())
                else {
                    return Ok(None);
                };
                self.cursor += character.len_utf8();
                Ok(Some(character))
            }
            TextSource::CharacterRow { array, row } => {
                let Some(first) = row_unit(array, row, self.cursor) else {
                    return Ok(None);
                };
                let index = self.cursor;
                let raw = first.raw();
                self.cursor += 1;
                let scalar = match raw {
                    0xD800..=0xDBFF => {
                        let Some(second) = row_unit(array, row, self.cursor) else {
                            return Err(invalid_utf16(index, raw));
                        };
                        let low = second.raw();
                        if !(0xDC00..=0xDFFF).contains(&low) {
                            return Err(invalid_utf16(index, raw));
                        }
                        self.cursor += 1;
                        0x1_0000 + (((raw - 0xD800) as u32) << 10) + (low - 0xDC00) as u32
                    }
                    0xDC00..=0xDFFF => return Err(invalid_utf16(index, raw)),
                    _ => raw as u32,
                };
                Ok(char::from_u32(scalar))
            }
        }
    }
}

fn row_unit(array: &CharArray, row: usize, column: usize) -> Option<runmat_value::Utf16CodeUnit> {
    if array.rows <= 1 {
        array.data().get(column).copied()
    } else {
        let offset = row.checked_mul(array.cols)?.checked_add(column)?;
        (column < array.cols)
            .then(|| array.data().get(offset).copied())
            .flatten()
    }
}

fn invalid_utf16(index: usize, unpaired_surrogate: u16) -> crate::RuntimeError {
    let source = Utf16DecodeError {
        index,
        unpaired_surrogate,
    };
    let mut builder =
        crate::build_runtime_error(format!("boxplot: invalid grouping label text: {source}"))
            .with_builtin(NAME)
            .with_source(source);
    if let Some(identifier) = BOXPLOT_ERROR_INVALID_ARGUMENT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incremental_missing_classification_matches_public_text_policy() {
        for (text, present) in [
            ("", false),
            ("  ", true),
            ("<missing>", true),
            ("<undefined>", true),
            (" MiSsInG  ", true),
            ("missing text", true),
            ("å missing", true),
            ("A", true),
        ] {
            let mut copy = TextCopy::utf8(text);
            let copied = loop {
                match copy.step().unwrap() {
                    TextCopyProgress::Work(_) => {}
                    TextCopyProgress::Complete(value) => break value,
                }
            };
            assert_eq!(copied.is_some(), present, "{text:?}");
        }
    }

    #[test]
    fn long_unicode_and_character_rows_copy_in_bounded_steps_without_changing_text() {
        let original = "💠é".repeat(400);
        let mut copy = TextCopy::utf8(&original);
        let mut turns = 0;
        let result = loop {
            match copy.step().unwrap() {
                TextCopyProgress::Work(units) => {
                    assert!(units <= (COPY_QUANTUM_BYTES + 3) as u64);
                    turns += 1;
                }
                TextCopyProgress::Complete(value) => break value.unwrap(),
            }
        };
        assert_eq!(result, original);
        assert!(turns > 1);

        let characters =
            CharArray::from_unicode_scalars(vec!['A', '1', ' ', 'B', '2', ' '], 2, 3).unwrap();
        for (row, expected) in [(0, "A1"), (1, "B2")] {
            let mut copy = TextCopy::character_row(&characters, row);
            let result = loop {
                match copy.step().unwrap() {
                    TextCopyProgress::Work(_) => {}
                    TextCopyProgress::Complete(value) => break value.unwrap(),
                }
            };
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn character_rows_decode_pairs_and_preserve_malformed_utf16_causes() {
        let characters = CharArray::from_code_units_with_shape(
            vec![
                runmat_value::Utf16CodeUnit::from_raw(0xD83D),
                runmat_value::Utf16CodeUnit::from_raw(0xDCA0),
                runmat_value::Utf16CodeUnit::from_raw(0xD800),
            ],
            vec![1, 3],
        )
        .unwrap();
        let mut copy = TextCopy::character_row(&characters, 0);
        let error = loop {
            match copy.step() {
                Ok(TextCopyProgress::Work(_)) => {}
                Ok(TextCopyProgress::Complete(_)) => panic!("malformed row unexpectedly decoded"),
                Err(error) => break error,
            }
        };
        let source = error
            .source
            .as_deref()
            .and_then(|source| source.downcast_ref::<Utf16DecodeError>())
            .expect("typed UTF-16 source");
        assert_eq!(source.index, 2);
        assert_eq!(source.unpaired_surrogate, 0xD800);
    }
}
