//! Validate the complete hostile representation before typed allocation, then
//! deserialize from the original bytes. No JSON tree or second wire buffer.
use crate::{checkpoint, CanonicalCodecControl, CanonicalCodecError, CanonicalLimits};
use serde::de::DeserializeOwned;
mod access;
mod key;
mod representation;
#[cfg(test)]
mod tests;
mod typed;

fn invalid(reason: impl std::fmt::Display) -> CanonicalCodecError {
    CanonicalCodecError::invalid("canonical decoding", reason.to_string())
}
impl serde::de::Error for CanonicalCodecError {
    fn custom<T: std::fmt::Display>(message: T) -> Self {
        invalid(message)
    }
}

pub(super) fn validated_value<'a>(
    domain: &str,
    payload: &'a [u8],
    limits: CanonicalLimits,
    control: &dyn CanonicalCodecControl,
) -> Result<&'a [u8], CanonicalCodecError> {
    let start = representation::validate(domain, payload, limits, control)?;
    Ok(&payload[start..])
}

pub(super) fn decode_value<T: DeserializeOwned>(
    value: &[u8],
    control: &dyn CanonicalCodecControl,
) -> Result<T, CanonicalCodecError> {
    let mut decoder = typed::Decoder {
        input: minicbor::Decoder::new(value),
        control,
    };
    let decoded = T::deserialize(&mut decoder)?;
    if decoder.input.position() != value.len() {
        return Err(invalid("typed visitor did not consume the complete value"));
    }
    checkpoint(control)?;
    Ok(decoded)
}
