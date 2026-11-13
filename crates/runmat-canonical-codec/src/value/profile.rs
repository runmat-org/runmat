//! One structural scanner for contiguous and chunked canonical payloads.
//! Scalar scratch is at most nine bytes; strings are validated without gathering.
use super::MaterializationProfile;
use crate::{checkpoint, CanonicalCodecControl, CanonicalCodecError, NeverCancelled};

#[cfg(test)]
mod tests;

fn invalid(message: &str) -> CanonicalCodecError {
    CanonicalCodecError::invalid("canonical materialization profile", message)
}

pub(crate) fn materialization_profile(
    encoded: &[u8],
) -> Result<MaterializationProfile, CanonicalCodecError> {
    chunked_profile(std::iter::once(encoded), &NeverCancelled).map(|(profile, _)| profile)
}

pub(crate) fn chunked_profile<'a>(
    chunks: impl IntoIterator<Item = &'a [u8]>,
    control: &dyn CanonicalCodecControl,
) -> Result<(MaterializationProfile, u64), CanonicalCodecError> {
    scan(chunks, control, Representation::Envelope)
}
pub(crate) fn value_profile(
    encoded: &[u8],
    control: &dyn CanonicalCodecControl,
) -> Result<(MaterializationProfile, u64), CanonicalCodecError> {
    scan(std::iter::once(encoded), control, Representation::Value)
}
enum Representation {
    Envelope,
    Value,
}
fn scan<'a>(
    chunks: impl IntoIterator<Item = &'a [u8]>,
    control: &dyn CanonicalCodecControl,
    representation: Representation,
) -> Result<(MaterializationProfile, u64), CanonicalCodecError> {
    let mut reader = Reader {
        chunks: chunks.into_iter(),
        current: &[],
        position: 0,
        control,
    };
    if matches!(representation, Representation::Envelope) {
        while reader.byte()? != Some(0) {
            if reader.current.is_empty() && reader.peek_end()? {
                return Err(invalid("domain prefix terminator is missing"));
            }
        }
        let (major, length, _) = reader.header()?;
        if major != 4 || length != Some(2) {
            return Err(invalid(
                "canonical envelope must contain a domain and value",
            ));
        }
        reader.string()?;
    }
    let mut profile = MaterializationProfile::default();
    reader.value(0, &mut profile)?;
    if reader.byte()?.is_some() {
        return Err(invalid("trailing data is forbidden"));
    }
    Ok((profile, reader.position))
}

struct Reader<'a, 'c, I> {
    chunks: I,
    current: &'a [u8],
    position: u64,
    control: &'c dyn CanonicalCodecControl,
}
impl<'a, I: Iterator<Item = &'a [u8]>> Reader<'a, '_, I> {
    fn peek_end(&mut self) -> Result<bool, CanonicalCodecError> {
        while self.current.is_empty() {
            checkpoint(self.control)?;
            match self.chunks.next() {
                Some(chunk) => self.current = chunk,
                None => return Ok(true),
            }
        }
        Ok(false)
    }
    fn byte(&mut self) -> Result<Option<u8>, CanonicalCodecError> {
        if self.position.is_multiple_of(1024) {
            checkpoint(self.control)?;
        }
        if self.peek_end()? {
            return Ok(None);
        }
        let byte = self.current[0];
        self.current = &self.current[1..];
        self.position = self
            .position
            .checked_add(1)
            .ok_or_else(|| invalid("payload length overflows"))?;
        Ok(Some(byte))
    }
    fn required(&mut self) -> Result<u8, CanonicalCodecError> {
        self.byte()?
            .ok_or_else(|| invalid("canonical payload is truncated"))
    }
    fn header(&mut self) -> Result<(u8, Option<u64>, u8), CanonicalCodecError> {
        let head = self.required()?;
        let additional = head & 31;
        let argument = match additional {
            0..=23 => Some(u64::from(additional)),
            24..=27 => {
                let mut value = 0u64;
                for _ in 0..(1usize << (additional - 24)) {
                    value = (value << 8) | u64::from(self.required()?);
                }
                Some(value)
            }
            31 => None,
            _ => return Err(invalid("invalid CBOR argument")),
        };
        Ok((head >> 5, argument, additional))
    }
    fn string(&mut self) -> Result<u64, CanonicalCodecError> {
        let (major, length, _) = self.header()?;
        if major != 3 {
            return Err(invalid("canonical string is required"));
        }
        self.utf8(length.ok_or_else(|| invalid("indefinite strings are forbidden"))?)
    }
    fn utf8(&mut self, length: u64) -> Result<u64, CanonicalCodecError> {
        let mut remaining = length;
        while remaining > 0 {
            let mut scalar = [0u8; 4];
            scalar[0] = self.required()?;
            let width = match scalar[0] {
                0..=0x7f => 1,
                0xc2..=0xdf => 2,
                0xe0..=0xef => 3,
                0xf0..=0xf4 => 4,
                _ => return Err(invalid("invalid UTF-8 string")),
            };
            if remaining < width as u64 {
                return Err(invalid("truncated UTF-8 string"));
            }
            for byte in &mut scalar[1..width] {
                *byte = self.required()?;
            }
            std::str::from_utf8(&scalar[..width]).map_err(|_| invalid("invalid UTF-8 string"))?;
            remaining -= width as u64;
        }
        Ok(length)
    }
    fn value(
        &mut self,
        depth: usize,
        profile: &mut MaterializationProfile,
    ) -> Result<(), CanonicalCodecError> {
        checkpoint(self.control)?;
        if depth > 128 {
            return Err(invalid("nesting depth exceeds the profiling limit"));
        }
        profile.value_nodes = profile
            .value_nodes
            .checked_add(1)
            .ok_or_else(|| invalid("value count overflows"))?;
        let (major, argument, additional) = self.header()?;
        match (major, argument) {
            (0, Some(_)) => Ok(()),
            (1, Some(value)) if value <= i64::MAX as u64 => Ok(()),
            (3, Some(length)) => {
                self.utf8(length)?;
                profile.string_bytes = profile
                    .string_bytes
                    .checked_add(length)
                    .ok_or_else(|| invalid("string bytes overflow"))?;
                profile.value_string_bytes = profile
                    .value_string_bytes
                    .checked_add(length)
                    .ok_or_else(|| invalid("string bytes overflow"))?;
                profile.heap_containers += 1;
                Ok(())
            }
            (4, Some(length)) => {
                profile.heap_containers += 1;
                for _ in 0..length {
                    self.value(depth + 1, profile)?;
                }
                Ok(())
            }
            (5, Some(length)) => {
                profile.maps += 1;
                profile.map_entries = profile
                    .map_entries
                    .checked_add(length)
                    .ok_or_else(|| invalid("map entry count overflows"))?;
                for _ in 0..length {
                    let bytes = self.string()?;
                    profile.string_bytes = profile
                        .string_bytes
                        .checked_add(bytes)
                        .ok_or_else(|| invalid("string bytes overflow"))?;
                    self.value(depth + 1, profile)?;
                }
                Ok(())
            }
            (7, Some(_)) if matches!(additional, 20..=22 | 25..=27) => Ok(()),
            _ => Err(invalid("unsupported CBOR type or indefinite collection")),
        }
    }
}
