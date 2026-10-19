//! One governed authority controls typed materialization and validation scratch.
use super::*;
use crate::GeometryEvaluationControl;
use runmat_canonical_codec::{CanonicalCodecControl, CanonicalCodecErrorKind};

pub trait GeometryCanonicalControl: GeometryEvaluationControl + CanonicalCodecControl {}
impl<T: GeometryEvaluationControl + CanonicalCodecControl + ?Sized> GeometryCanonicalControl for T {}

pub(crate) fn decode_with_control<T: DeserializeOwned>(
    domain: &str,
    bytes: &[u8],
    limits: CanonicalLimits,
    control: &dyn GeometryCanonicalControl,
) -> Result<T, GeometryContractError> {
    // The admitted structural envelope is retained by the input's scope. Typed
    // semantic validators reserve and release their own exact scratch buffers.
    runmat_canonical_codec::decode_contract_with_admission(
        GEOMETRY_CODEC_PREFIX,
        domain,
        bytes,
        limits,
        control,
        |value| {
            let requested =
                runmat_canonical_codec::canonical_value_materialization_resident_bytes_estimate(
                    value, control,
                )?;
            control
                .consume_allocation_bytes(requested)
                .map_err(|error| CanonicalCodecError {
                    kind: CanonicalCodecErrorKind::ResourceLimit,
                    field: "geometry typed materialization admission",
                    reason: error.to_string(),
                })
        },
    )
    .map_err(map_error)
}
