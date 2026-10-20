//! Bounded, fallible copying of user-visible grouping text.

use super::super::super::allocation_failure;
use runmat_value::CharArray;

const COPY_QUANTUM_BYTES: usize = 256;

#[derive(Clone, Copy)]
enum TextSource<'a> {
    Utf8(&'a str),
    CharacterRow { array: &'a CharArray, row: usize },
}

#[derive(Default)]
struct TextPresence {
    significant: usize,
    seen_non_space: bool,
    gap: bool,
    angle_missing: bool,
    angle_undefined: bool,
    bare_missing: bool,
}

impl TextPresence {
    fn new() -> Self {
        Self {
            angle_missing: true,
            angle_undefined: true,
            bare_missing: true,
            ..Self::default()
        }
    }

    fn observe(&mut self, character: char) {
        if character.is_whitespace() {
            self.gap |= self.seen_non_space;
            return;
        }
        if self.gap {
            self.angle_missing = false;
            self.angle_undefined = false;
            self.bare_missing = false;
        }
        self.seen_non_space = true;
        self.angle_missing &= matches_ascii_at("<missing>", self.significant, character, false);
        self.angle_undefined &= matches_ascii_at("<undefined>", self.significant, character, false);
        self.bare_missing &= matches_ascii_at("missing", self.significant, character, true);
        self.significant = self.significant.saturating_add(1);
        self.gap = false;
    }

    fn present(&self) -> bool {
        self.seen_non_space
            && !(self.angle_missing && self.significant == "<missing>".len())
            && !(self.angle_undefined && self.significant == "<undefined>".len())
            && !(self.bare_missing && self.significant == "missing".len())
    }
}

fn matches_ascii_at(token: &str, position: usize, character: char, ignore_case: bool) -> bool {
    token.as_bytes().get(position).is_some_and(|expected| {
        character.is_ascii()
            && (if ignore_case {
                character.to_ascii_lowercase() as u8
            } else {
                character as u8
            }) == *expected
    })
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
    presence: TextPresence,
    honor_missing: bool,
}

impl<'a> TextCopy<'a> {
    pub(super) fn utf8(text: &'a str, honor_missing: bool) -> Self {
        Self::new(TextSource::Utf8(text), honor_missing)
    }

    pub(super) fn character_row(array: &'a CharArray, row: usize) -> Self {
        Self::new(TextSource::CharacterRow { array, row }, true)
    }

    fn new(source: TextSource<'a>, honor_missing: bool) -> Self {
        Self {
            source,
            cursor: 0,
            output: String::new(),
            last_non_space_end: 0,
            presence: TextPresence::new(),
            honor_missing,
        }
    }

    pub(super) fn step(&mut self) -> crate::BuiltinResult<TextCopyProgress> {
        let mut copied = 0usize;
        while copied < COPY_QUANTUM_BYTES {
            let Some(character) = self.next_character() else {
                if matches!(&self.source, TextSource::CharacterRow { array, .. } if array.rows > 1)
                {
                    self.output.truncate(self.last_non_space_end);
                }
                let text = std::mem::take(&mut self.output);
                return Ok(TextCopyProgress::Complete(
                    (!self.honor_missing || self.presence.present()).then_some(text),
                ));
            };
            let bytes = character.len_utf8();
            if self.output.capacity() - self.output.len() < bytes {
                self.output.try_reserve(bytes).map_err(|error| {
                    allocation_failure("boxplot: grouping label exceeds available memory", error)
                })?;
            }
            self.output.push(character);
            self.presence.observe(character);
            if !character.is_whitespace() {
                self.last_non_space_end = self.output.len();
            }
            copied += bytes;
        }
        Ok(TextCopyProgress::Work(copied as u64))
    }

    fn next_character(&mut self) -> Option<char> {
        match self.source {
            TextSource::Utf8(text) => {
                let character = text.get(self.cursor..)?.chars().next()?;
                self.cursor += character.len_utf8();
                Some(character)
            }
            TextSource::CharacterRow { array, row } => {
                let character = if array.rows <= 1 {
                    array.data().get(self.cursor).copied()
                } else {
                    let offset = row.checked_mul(array.cols)?.checked_add(self.cursor)?;
                    (self.cursor < array.cols)
                        .then(|| array.data().get(offset).copied())
                        .flatten()
                }?;
                self.cursor += 1;
                Some(character)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incremental_missing_classification_matches_public_text_policy() {
        for (text, present) in [
            ("", false),
            ("  ", false),
            ("<missing>", false),
            ("<undefined>", false),
            (" MiSsInG  ", false),
            ("missing text", true),
            ("å missing", true),
            ("A", true),
        ] {
            let mut copy = TextCopy::utf8(text, true);
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
        let mut copy = TextCopy::utf8(&original, true);
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

        let characters = CharArray::new(vec!['A', '1', ' ', 'B', '2', ' '], 2, 3).unwrap();
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
}
