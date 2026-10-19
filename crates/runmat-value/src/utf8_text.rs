//! Immutable UTF-8 text with an optional exact retained backing.
//!
//! Ordinary construction preserves the caller's `String` allocation. Exact
//! promotion moves those same bytes into the canonical exact array owner;
//! aliases of an exact owner are O(1) and retain its physical claim.

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Deref;

use crate::{
    ExactArrayAdoptionError, ExactArrayStorage, ExactHostAllocationPermit,
    ExactHostNumericAllocationError,
};

#[derive(Debug)]
enum Utf8TextInner {
    Owned(String),
    Exact(ExactArrayStorage<u8>),
}

#[derive(Debug)]
pub struct Utf8Text(Utf8TextInner);

impl Default for Utf8Text {
    fn default() -> Self {
        Self::new(String::new())
    }
}

#[derive(Debug)]
enum Utf8TextAdmissionFailure {
    Unclaimed(Utf8Text),
    Claimed(ExactArrayAdoptionError<u8>),
}

#[derive(Debug)]
pub struct Utf8TextAdmissionError {
    failure: Utf8TextAdmissionFailure,
    source: ExactHostNumericAllocationError,
}

impl Utf8TextAdmissionError {
    pub fn source(&self) -> ExactHostNumericAllocationError {
        self.source
    }

    /// Recover the original text only when admission failed before a claim
    /// was issued. A claim-bearing failure must retire intact.
    pub fn try_into_unclaimed_text(self) -> Result<Utf8Text, Self> {
        let Self { failure, source } = self;
        match failure {
            Utf8TextAdmissionFailure::Unclaimed(text) => Ok(text),
            Utf8TextAdmissionFailure::Claimed(error) => Err(Self {
                failure: Utf8TextAdmissionFailure::Claimed(error),
                source,
            }),
        }
    }
}

impl Utf8Text {
    pub fn new(text: String) -> Self {
        Self(Utf8TextInner::Owned(text))
    }

    pub fn as_str(&self) -> &str {
        match &self.0 {
            Utf8TextInner::Owned(text) => text,
            Utf8TextInner::Exact(bytes) => {
                // SAFETY: the only exact constructor moves bytes out of a
                // validated String, and the immutable owner cannot mutate them.
                unsafe { std::str::from_utf8_unchecked(bytes.as_slice()) }
            }
        }
    }

    pub fn capacity(&self) -> usize {
        match &self.0 {
            Utf8TextInner::Owned(text) => text.capacity(),
            Utf8TextInner::Exact(bytes) => bytes.capacity(),
        }
    }

    pub fn has_retained_claim(&self) -> bool {
        matches!(&self.0, Utf8TextInner::Exact(bytes) if bytes.is_exact())
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
        let text = match self.0 {
            Utf8TextInner::Owned(text) => text,
            Utf8TextInner::Exact(bytes) => return Ok(Self(Utf8TextInner::Exact(bytes))),
        };
        let bytes = text.into_bytes();
        match ExactArrayStorage::try_adopt_exact(bytes, permit) {
            Ok(bytes) => Ok(Self(Utf8TextInner::Exact(bytes))),
            Err(error) => {
                let source = *error.source();
                match error.try_into_unclaimed_values() {
                    Ok(bytes) => {
                        // SAFETY: the adopter returned the unchanged bytes
                        // produced by `String::into_bytes`.
                        let text = unsafe { String::from_utf8_unchecked(bytes) };
                        Err(Utf8TextAdmissionError {
                            failure: Utf8TextAdmissionFailure::Unclaimed(Self(
                                Utf8TextInner::Owned(text),
                            )),
                            source,
                        })
                    }
                    Err(error) => Err(Utf8TextAdmissionError {
                        failure: Utf8TextAdmissionFailure::Claimed(error),
                        source,
                    }),
                }
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
    pub fn try_copy_str_exact_with<E>(
        source: &str,
        permit: &dyn ExactHostAllocationPermit,
        mut checkpoint: impl FnMut(usize) -> Result<(), E>,
    ) -> Result<Self, crate::ExactArrayInitializationError<E>> {
        let source = source.as_bytes();
        let bytes = ExactArrayStorage::try_initialize(source.len(), permit, |index| {
            if index % 4096 == 0 {
                checkpoint(index)?;
            }
            Ok(source[index])
        })?;
        Ok(Self(Utf8TextInner::Exact(bytes)))
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
        Ok(Self(Utf8TextInner::Exact(bytes)))
    }

    /// Materialize ordinary String ownership. Exact callers should normally
    /// borrow `as_str`; this explicit boundary may copy shared bytes.
    pub fn into_string_unbounded(self) -> String {
        match self.0 {
            Utf8TextInner::Owned(text) => text,
            Utf8TextInner::Exact(bytes) => {
                // SAFETY: exact text bytes preserve the constructor invariant.
                unsafe { String::from_utf8_unchecked(bytes.as_slice().to_vec()) }
            }
        }
    }
}

impl Clone for Utf8Text {
    fn clone(&self) -> Self {
        match &self.0 {
            Utf8TextInner::Owned(text) => Self::new(text.clone()),
            Utf8TextInner::Exact(bytes) => Self(Utf8TextInner::Exact(bytes.clone())),
        }
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

#[cfg(test)]
mod tests;
