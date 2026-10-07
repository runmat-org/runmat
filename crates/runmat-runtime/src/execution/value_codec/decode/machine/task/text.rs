use super::super::*;

/// Incremental exact UTF-8 copy used by every payload field that materializes
/// owned runtime text. Copying Unicode scalars directly into the final String
/// avoids a terminal full-buffer UTF-8 validation scan.
pub(in crate::execution::value_codec::decode::machine) struct TextCopy<'a> {
    source: &'a str,
    source_offset: usize,
    output: String,
}

impl<'a> TextCopy<'a> {
    pub(super) fn new(source: &'a str) -> Self {
        Self {
            source,
            source_offset: 0,
            output: String::with_capacity(source.len()),
        }
    }

    pub(super) fn advance(&mut self, maximum_work: usize) -> usize {
        let mut completed = 0;
        for character in self.source[self.source_offset..].chars().take(maximum_work) {
            self.output.push(character);
            self.source_offset += character.len_utf8();
            completed += 1;
        }
        completed.max(1)
    }

    pub(super) fn is_complete(&self) -> bool {
        self.source_offset == self.source.len()
    }

    pub(super) fn finish(self) -> String {
        self.output
    }
}

pub(in crate::execution::value_codec::decode::machine) struct ShapeCopy<'a> {
    source: &'a [u64],
    output: Vec<usize>,
    path: String,
}

impl<'a> ShapeCopy<'a> {
    pub(super) fn new(source: &'a [u64], path: String) -> Self {
        Self {
            source,
            output: Vec::with_capacity(source.len()),
            path,
        }
    }

    pub(super) fn advance(&mut self, maximum_work: usize) -> Result<usize, ValueCodecError> {
        let count = (self.source.len() - self.output.len()).min(maximum_work);
        for value in &self.source[self.output.len()..self.output.len() + count] {
            self.output.push(usize::try_from(*value).map_err(|_| {
                ValueCodecError::invalid(&self.path, "value exceeds platform limits")
            })?);
        }
        Ok(count.max(1))
    }

    pub(super) fn is_complete(&self) -> bool {
        self.output.len() == self.source.len()
    }

    pub(super) fn finish(self) -> Vec<usize> {
        self.output
    }
}
