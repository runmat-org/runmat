//! Strict two-pass UTF-16 decoding directly into one exact UTF-8 owner.

use crate::{
    ExactArrayInitializationError, ExactArrayStorage, ExactHostAllocationPermit,
    ExactHostNumericAllocationError, Utf16CodeUnit, Utf16DecodeError,
};

use super::Utf8Text;

#[derive(Debug)]
pub enum Utf16TextCopyError<E> {
    Decode(Utf16DecodeError),
    Checkpoint(E),
}

impl Utf8Text {
    pub fn requested_bytes_for_utf16_with<E>(
        units: &[Utf16CodeUnit],
        mut checkpoint: impl FnMut(usize) -> Result<(), E>,
    ) -> Result<(usize, usize), ExactArrayInitializationError<Utf16TextCopyError<E>>> {
        use ExactArrayInitializationError::{Allocation, Element};

        let mut length = 0usize;
        let mut unit_index = 0usize;
        let mut next_checkpoint = 0usize;
        for decoded in char::decode_utf16(units.iter().map(|unit| unit.raw())) {
            if unit_index >= next_checkpoint {
                checkpoint(unit_index)
                    .map_err(|error| Element(Utf16TextCopyError::Checkpoint(error)))?;
                next_checkpoint = unit_index.saturating_add(4096);
            }
            let character = decoded.map_err(|error| {
                Element(Utf16TextCopyError::Decode(Utf16DecodeError {
                    index: unit_index,
                    unpaired_surrogate: error.unpaired_surrogate(),
                }))
            })?;
            length = length
                .checked_add(character.len_utf8())
                .ok_or(Allocation(ExactHostNumericAllocationError::LayoutOverflow))?;
            unit_index += character.len_utf16();
        }
        checkpoint(unit_index).map_err(|error| Element(Utf16TextCopyError::Checkpoint(error)))?;
        let bytes = Self::requested_bytes_for_capacity(length).map_err(Allocation)?;
        Ok((length, bytes))
    }

    pub fn try_copy_utf16_exact_with<E>(
        units: &[Utf16CodeUnit],
        permit: &dyn ExactHostAllocationPermit,
        mut checkpoint: impl FnMut(usize) -> Result<(), E>,
    ) -> Result<Self, ExactArrayInitializationError<Utf16TextCopyError<E>>> {
        let (length, _) = Self::requested_bytes_for_utf16_with(units, &mut checkpoint)?;
        let mut characters = char::decode_utf16(units.iter().map(|unit| unit.raw()));
        let mut encoded = [0u8; 4];
        let mut encoded_len = 0usize;
        let mut encoded_index = 0usize;
        let bytes = ExactArrayStorage::try_initialize(length, permit, |index| {
            if index % 4096 == 0 {
                checkpoint(index).map_err(Utf16TextCopyError::Checkpoint)?;
            }
            if encoded_index == encoded_len {
                let character = characters
                    .next()
                    .expect("validated UTF-16 byte length")
                    .expect("immutable UTF-16 input was validated in the quote pass");
                encoded_len = character.encode_utf8(&mut encoded).len();
                encoded_index = 0;
            }
            let byte = encoded[encoded_index];
            encoded_index += 1;
            Ok(byte)
        })?;
        Ok(Self(bytes))
    }
}
