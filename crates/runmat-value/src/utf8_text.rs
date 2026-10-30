//! Immutable UTF-8 text with an optional exact retained backing.
//!
//! Ordinary construction preserves the caller's `String` allocation. Exact
//! promotion moves those same bytes into the canonical exact array owner;
//! all aliases share those original bytes in O(1), and promotion attaches one
//! physical claim to that same owner.

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Deref;

use crate::{
    ExactArrayClaimPromotionError, ExactArrayStorage, ExactHostAllocationPermit,
    ExactHostNumericAllocationError,
};

mod staged_copy;
mod utf16_copy;
pub use staged_copy::Utf8TextExactCopy;
pub use utf16_copy::Utf16TextCopyError;

#[derive(Debug, Clone)]
pub struct Utf8Text(ExactArrayStorage<u8>);

impl Default for Utf8Text {
    fn default() -> Self {
        Self::new(String::new())
    }
}

#[derive(Debug)]
pub struct Utf8TextAdmissionError {
    text: Utf8Text,
    source: ExactHostNumericAllocationError,
}
impl fmt::Display for Utf8TextAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.source.fmt(formatter)
    }
}
impl std::error::Error for Utf8TextAdmissionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}
impl Utf8TextAdmissionError {
    pub fn source(&self) -> ExactHostNumericAllocationError {
        self.source
    }
    /// Recovery cannot detach an already-attached physical claim, including
    /// one attached through another alias after this reservation failed.
    pub fn try_into_unclaimed_text(self) -> Result<Utf8Text, Self> {
        if self.text.has_retained_claim() {
            Err(self)
        } else {
            Ok(self.text)
        }
    }
}

impl Utf8Text {
    pub fn shared_storage_identity(
        &self,
    ) -> Option<crate::exact_array_storage::SharedStorageIdentity> {
        self.0.shared_storage_identity()
    }
    /// Move an existing String's bytes into the canonical exact text owner.
    /// Both the original byte capacity and new shared header are reserved
    /// before the header allocation; no intermediate text copy is made.
    pub fn try_adopt_string_exact(
        text: String,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, crate::ExactArrayAdoptionError<u8>> {
        ExactArrayStorage::try_adopt_exact(text.into_bytes(), permit).map(Self)
    }

    pub fn new(text: String) -> Self {
        let bytes = text.into_bytes();
        Self(if bytes.capacity() == 0 {
            ExactArrayStorage::empty()
        } else {
            ExactArrayStorage::shared_unbounded(bytes)
        })
    }
    pub fn as_str(&self) -> &str {
        // SAFETY: constructors move validated String bytes or complete an
        // unchanged str/char encoding copy. Published backing is immutable.
        unsafe { std::str::from_utf8_unchecked(self.0.as_slice()) }
    }
    pub fn capacity(&self) -> usize {
        self.0.capacity()
    }
    pub fn has_retained_claim(&self) -> bool {
        self.0.is_exact()
    }
    pub(crate) fn unclaimed_bytes(&self) -> Result<Option<usize>, ExactHostNumericAllocationError> {
        self.0
            .unclaimed_shared_bytes()
            .map_err(|source| match source {
                ExactArrayClaimPromotionError::Allocation(source) => source,
                ExactArrayClaimPromotionError::UnsharedBacking => {
                    unreachable!("text is shared from original construction")
                }
            })
    }

    pub fn requested_bytes_for_capacity(
        capacity: usize,
    ) -> Result<usize, ExactHostNumericAllocationError> {
        ExactArrayStorage::<u8>::requested_bytes_for_capacity(capacity)
    }

    /// Promote the original String allocation without copying its bytes.
    /// Failure returns the original text owner intact.
    pub fn try_admit_exact(
        self,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, Utf8TextAdmissionError> {
        match self.0.try_promote_shared_claim(permit) {
            Ok(()) => Ok(self),
            Err(source) => {
                let source = match source {
                    ExactArrayClaimPromotionError::Allocation(source) => source,
                    ExactArrayClaimPromotionError::UnsharedBacking => {
                        unreachable!("text is shared from original construction")
                    }
                };
                Err(Utf8TextAdmissionError { text: self, source })
            }
        }
    }

    /// Copy borrowed UTF-8 into a separately admitted immutable generation.
    /// Both the exact byte vector and shared control are allocated before the
    /// first byte is copied; this never uses `String::clone` as finite work.
    pub fn try_copy_exact(
        &self,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        self.try_copy_exact_with(permit, |_| Ok::<(), std::convert::Infallible>(()))
            .map_err(|error| match error {
                crate::ExactArrayInitializationError::Allocation(error) => error,
                crate::ExactArrayInitializationError::Element(never) => match never {},
            })
    }

    /// Copy borrowed text into an exact generation while allowing the caller
    /// to fence long copies without exposing a mutable or unclaimed backing.
    pub fn try_copy_exact_with<E>(
        &self,
        permit: &dyn ExactHostAllocationPermit,
        checkpoint: impl FnMut(usize) -> Result<(), E>,
    ) -> Result<Self, crate::ExactArrayInitializationError<E>> {
        Self::try_copy_str_exact_with(self.as_str(), permit, checkpoint)
    }

    /// Copy borrowed UTF-8 directly into an exact retained byte owner. No
    /// temporary String or unclaimed byte vector is materialized.
    pub fn try_begin_copy_exact<'a>(
        source: &'a str,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Utf8TextExactCopy<'a>, ExactHostNumericAllocationError> {
        Utf8TextExactCopy::try_new(source, permit)
    }

    /// Synchronously drain the same staged exact owner, checkpointing before
    /// each bounded byte copy rather than constructing a second representation.
    pub fn try_copy_str_exact_with<E>(
        source: &str,
        permit: &dyn ExactHostAllocationPermit,
        mut checkpoint: impl FnMut(usize) -> Result<(), E>,
    ) -> Result<Self, crate::ExactArrayInitializationError<E>> {
        use crate::ExactArrayInitializationError::{Allocation, Element};

        let mut copy = Self::try_begin_copy_exact(source, permit).map_err(Allocation)?;
        while !copy.is_complete() {
            checkpoint(copy.copied_bytes()).map_err(Element)?;
            copy.advance(std::num::NonZeroUsize::new(4096).expect("nonzero copy quantum"));
        }
        Ok(copy.try_finish().expect("complete staged UTF-8 copy"))
    }

    /// Quote the exact UTF-8 owner for a borrowed character sequence while
    /// fencing long Unicode scans before allocation.
    pub fn requested_bytes_for_chars_with<E>(
        chars: &[char],
        mut checkpoint: impl FnMut(usize) -> Result<(), E>,
    ) -> Result<(usize, usize), crate::ExactArrayInitializationError<E>> {
        use crate::ExactArrayInitializationError::{Allocation, Element};
        let mut byte_len = 0usize;
        for (index, character) in chars.iter().enumerate() {
            if index % 4096 == 0 {
                checkpoint(index).map_err(Element)?;
            }
            byte_len = byte_len
                .checked_add(character.len_utf8())
                .ok_or(Allocation(ExactHostNumericAllocationError::LayoutOverflow))?;
        }
        let requested = Self::requested_bytes_for_capacity(byte_len).map_err(Allocation)?;
        Ok((byte_len, requested))
    }

    /// Concatenate borrowed UTF-8 parts directly in one exact immutable owner.
    /// No temporary String or byte vector is created. The immutable part slice
    /// guarantees the same UTF-8 boundaries during length and fill passes.
    pub fn try_copy_parts_exact_with<E>(
        parts: &[&str],
        permit: &dyn ExactHostAllocationPermit,
        mut checkpoint: impl FnMut(usize) -> Result<(), E>,
    ) -> Result<Self, crate::ExactArrayInitializationError<E>> {
        use crate::ExactArrayInitializationError::{Allocation, Element};
        let mut length = 0usize;
        for part in parts {
            checkpoint(length).map_err(Element)?;
            length = length
                .checked_add(part.len())
                .ok_or(Allocation(ExactHostNumericAllocationError::LayoutOverflow))?;
        }
        let mut source = parts.iter().flat_map(|part| part.bytes());
        let bytes = ExactArrayStorage::try_initialize(length, permit, |index| {
            if index % 4096 == 0 {
                checkpoint(index)?;
            }
            Ok(source.next().expect("stable UTF-8 parts byte length"))
        })?;
        Ok(Self(bytes))
    }

    /// Encode borrowed characters directly into an exact UTF-8 owner. The
    /// caller quotes this same input first; both scans remain cancellable.
    pub fn try_copy_chars_exact_with<E>(
        chars: &[char],
        permit: &dyn ExactHostAllocationPermit,
        mut checkpoint: impl FnMut(usize) -> Result<(), E>,
    ) -> Result<Self, crate::ExactArrayInitializationError<E>> {
        let (byte_len, _) = Self::requested_bytes_for_chars_with(chars, &mut checkpoint)?;
        let mut source = chars.iter();
        let mut encoded = [0u8; 4];
        let mut encoded_len = 0usize;
        let mut encoded_index = 0usize;
        let bytes = ExactArrayStorage::try_initialize(byte_len, permit, |index| {
            if index % 4096 == 0 {
                checkpoint(index)?;
            }
            if encoded_index == encoded_len {
                let character = source.next().expect("quoted character byte count");
                encoded_len = character.encode_utf8(&mut encoded).len();
                encoded_index = 0;
            }
            let byte = encoded[encoded_index];
            encoded_index += 1;
            Ok(byte)
        })?;
        Ok(Self(bytes))
    }

    /// Materialize ordinary String ownership. Exact callers should normally
    /// borrow `as_str`; this explicit boundary may copy shared bytes.
    pub fn into_string_unbounded(self) -> String {
        let bytes = match self.0.try_into_owned() {
            Ok(values) => match values.try_into_unbounded_vec() {
                Ok(bytes) => bytes,
                Err(values) => values.as_slice().to_vec(),
            },
            Err(shared) => shared.as_slice().to_vec(),
        };
        // SAFETY: the original immutable UTF-8 sequence is unchanged.
        unsafe { String::from_utf8_unchecked(bytes) }
    }
}

impl PartialEq for Utf8Text {
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}

impl Eq for Utf8Text {}

impl Hash for Utf8Text {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}

impl Deref for Utf8Text {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

impl AsRef<str> for Utf8Text {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl Borrow<str> for Utf8Text {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl PartialEq<str> for Utf8Text {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<&str> for Utf8Text {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl PartialEq<String> for Utf8Text {
    fn eq(&self, other: &String) -> bool {
        self.as_str() == other.as_str()
    }
}

impl PartialEq<Utf8Text> for String {
    fn eq(&self, other: &Utf8Text) -> bool {
        self.as_str() == other.as_str()
    }
}

impl PartialOrd for Utf8Text {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Utf8Text {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_str().cmp(other.as_str())
    }
}

impl fmt::Display for Utf8Text {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl From<String> for Utf8Text {
    fn from(text: String) -> Self {
        Self::new(text)
    }
}

impl From<&str> for Utf8Text {
    fn from(text: &str) -> Self {
        Self::new(text.to_owned())
    }
}
impl From<&Utf8Text> for Utf8Text {
    fn from(text: &Utf8Text) -> Self {
        text.clone()
    }
}
impl From<&String> for Utf8Text {
    fn from(text: &String) -> Self {
        Self::from(text.as_str())
    }
}

#[cfg(test)]
mod tests;
