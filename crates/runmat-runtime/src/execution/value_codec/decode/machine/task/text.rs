use super::super::*;

pub(in crate::execution::value_codec) use crate::execution::value_codec::text_copy::TextCopy;

pub(in crate::execution::value_codec::decode::machine) struct ShapeCopy<'a> {
    source: &'a [u64],
    output: Vec<usize>,
    path: ControlPath<'a>,
}

impl<'a> ShapeCopy<'a> {
    pub(super) fn new(source: &'a [u64], path: ControlPath<'a>) -> Self {
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

    pub(super) fn take_finished(&mut self) -> Vec<usize> {
        assert!(self.is_complete(), "shape conversion is complete");
        std::mem::take(&mut self.output)
    }
}
