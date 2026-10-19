//! Shared canonical codec controls for constraint identity, publication and cold import.
use super::*;
use runmat_canonical_codec::{
    decode_contract_with_control, encode_contract_with_control, CanonicalCodecError,
};
use serde::de::DeserializeOwned;

fn error(
    error: CanonicalCodecError,
    control: &crate::progress::FeaCancellationToken,
) -> ConstraintPlanError {
    if control.is_cancelled() {
        ConstraintPlanError::Cancelled
    } else {
        ConstraintPlanError::Encoding(error.to_string())
    }
}
pub(super) fn encode<T: Serialize>(
    domain: &'static str,
    value: &T,
) -> Result<Vec<u8>, ConstraintPlanError> {
    let control = crate::progress::cancellation_token();
    encode_contract_with_control(PLAN_PREFIX, domain, value, PLAN_LIMITS, &control)
        .map_err(|failure| error(failure, &control))
}
pub(super) fn decode<T: DeserializeOwned>(
    domain: &'static str,
    bytes: &[u8],
) -> Result<T, ConstraintPlanError> {
    let control = crate::progress::cancellation_token();
    decode_contract_with_control(PLAN_PREFIX, domain, bytes, PLAN_LIMITS, &control)
        .map_err(|failure| error(failure, &control))
}

#[cfg(test)]
mod tests;
