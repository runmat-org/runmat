//! Cooperative borrowed-UTF-8 copy into one final exact byte owner.

use std::num::NonZeroUsize;

use crate::{
    ExactArrayStorage, ExactArrayStorageBuilder, ExactHostAllocationPermit,
    ExactHostNumericAllocationError,
};

use super::Utf8Text;

/// The source borrow keeps its bytes unchanged while a partially initialized
/// prefix may end inside a multibyte scalar. Only completion publishes text;
/// no temporary String or terminal validation scan is needed.
#[derive(Debug)]
pub struct Utf8TextExactCopy<'a> {
    source: &'a str,
    copied: usize,
    bytes: ExactArrayStorageBuilder<u8>,
}

impl<'a> Utf8TextExactCopy<'a> {
    pub(super) fn try_new(
        source: &'a str,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        Ok(Self {
            source,
            copied: 0,
            bytes: ExactArrayStorage::<u8>::try_begin(source.len(), permit)?,
        })
    }

    pub fn copied_bytes(&self) -> usize {
        self.copied
    }

    pub fn is_complete(&self) -> bool {
        self.copied == self.source.len()
    }

    /// Copy at most the requested number of source bytes into the final
    /// backing. This method performs no allocation or UTF-8 rescan.
    pub fn advance(&mut self, maximum_work: NonZeroUsize) -> usize {
        let end = self
            .copied
            .saturating_add(maximum_work.get())
            .min(self.source.len());
        for &byte in &self.source.as_bytes()[self.copied..end] {
            self.bytes
                .try_push(byte)
                .expect("quoted UTF-8 owner has one slot per source byte");
        }
        let completed = end - self.copied;
        self.copied = end;
        completed
    }

    /// Refuse premature publication and return the same claim-bearing owner.
    pub fn try_finish(self) -> Result<Utf8Text, Self> {
        if !self.is_complete() {
            return Err(self);
        }
        let bytes = self
            .bytes
            .try_finish()
            .unwrap_or_else(|_| unreachable!("complete UTF-8 copy fills exact owner"));
        Ok(Utf8Text(bytes))
    }
}
