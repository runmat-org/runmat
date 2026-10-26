//! Canonical serde projection without retaining a complete JSON value tree.
//! Sequences retain encoded bytes; maps retain only sorted keys and encoded values.
use crate::{
    checkpoint, encoding_error, CanonicalCodecControl, CanonicalCodecError, CanonicalLimits,
};
use serde::Serialize;
mod compound;
mod probe;
mod serializer;
#[cfg(test)]
mod tests;
use serializer::Serializer;

struct State<'a> {
    domain: &'a str,
    limits: CanonicalLimits,
    control: &'a dyn CanonicalCodecControl,
    wire_bytes: usize,
}
impl State<'_> {
    fn check(&self) -> Result<(), CanonicalCodecError> {
        checkpoint(self.control)
    }
    fn charge(&mut self, bytes: usize) -> Result<(), CanonicalCodecError> {
        let next = self.wire_bytes.checked_add(bytes).ok_or_else(|| {
            CanonicalCodecError::resource_limit(
                "canonical encoding",
                format!("contract `{}` encoded byte count overflow", self.domain),
            )
        })?;
        if next > self.limits.maximum_encoded_bytes {
            return Err(CanonicalCodecError::resource_limit(
                "canonical encoding",
                format!(
                    "contract `{}` requires at least {next} bytes, exceeding its {}-byte limit",
                    self.domain, self.limits.maximum_encoded_bytes,
                ),
            ));
        }
        self.wire_bytes = next;
        Ok(())
    }
}
fn invalid(reason: &'static str) -> CanonicalCodecError {
    CanonicalCodecError::invalid("canonical encoding", reason)
}

pub(crate) fn encode<T: Serialize>(
    prefix: &[u8],
    domain: &str,
    value: &T,
    limits: CanonicalLimits,
    control: &dyn CanonicalCodecControl,
) -> Result<Vec<u8>, CanonicalCodecError> {
    // Reserve the profiled output once so it never regrows; the encoding workspace profile
    // assumes this. Shape limits are the encoder's own; a profile it cannot produce leaves
    // the output to grow as before.
    let capacity = crate::profile_canonical_encoding_with_control(
        value,
        CanonicalLimits::new(
            limits.maximum_encoded_bytes,
            usize::MAX,
            usize::MAX,
            usize::MAX,
        ),
        control,
    )
    .ok()
    .and_then(|profile| usize::try_from(profile.encoded_bytes_upper_bound).ok())
    .filter(|capacity| *capacity <= limits.maximum_encoded_bytes);
    encode_with_capacity(prefix, domain, value, limits, control, capacity)
}

pub(crate) fn encode_with_capacity<T: Serialize>(
    prefix: &[u8],
    domain: &str,
    value: &T,
    limits: CanonicalLimits,
    control: &dyn CanonicalCodecControl,
    capacity: Option<usize>,
) -> Result<Vec<u8>, CanonicalCodecError> {
    checkpoint(control)?;
    if capacity.is_some_and(|capacity| capacity > limits.maximum_encoded_bytes) {
        return Err(CanonicalCodecError::resource_limit(
            "canonical encoding",
            "requested output capacity exceeds the codec bound",
        ));
    }
    let mut output = Vec::new();
    if let Some(capacity) = capacity {
        output.try_reserve_exact(capacity).map_err(|error| {
            CanonicalCodecError::resource_limit("canonical encoding", error.to_string())
        })?;
    }
    output.extend_from_slice(prefix);
    minicbor::Encoder::new(&mut output)
        .array(2)
        .and_then(|encoder| encoder.str(domain))
        .map_err(encoding_error)?;
    let mut state = State {
        domain,
        limits,
        control,
        wire_bytes: 0,
    };
    state.charge(output.len())?;
    value.serialize(Serializer::child(&mut output, &mut state, value))?;
    checkpoint(control)?;
    debug_assert_eq!(state.wire_bytes, output.len());
    Ok(output)
}
