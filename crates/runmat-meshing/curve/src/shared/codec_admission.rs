//! Shared byte production for admitted curve/batch encoders. Semantic admission
//! remains with each domain entry point; successful buffers retain their charge.
use super::{SharedCurveError, SharedCurveErrorKind};
use runmat_canonical_codec::{CanonicalCodecError, CanonicalCodecErrorKind, CanonicalLimits};
use runmat_geometry_core::{
    GeometryAllocationReservation, GeometryEvaluationControl, GeometryEvaluationErrorKind,
};
use runmat_meshing_core::{MeshingCancellationSignal, MeshingCanonicalCodecControl};
use serde::Serialize;

pub(super) fn encode<T: Serialize>(
    value: &T,
    prefix: &[u8],
    domain: &str,
    limits: CanonicalLimits,
    signal: &dyn MeshingCancellationSignal,
    allocations: &dyn GeometryEvaluationControl,
) -> Result<Vec<u8>, SharedCurveError> {
    let control = MeshingCanonicalCodecControl::new(signal);
    let codec_error = |error: CanonicalCodecError| {
        let kind = if control.observed_cancellation() {
            Some(GeometryEvaluationErrorKind::Cancelled)
        } else if error.kind == CanonicalCodecErrorKind::ResourceLimit {
            Some(GeometryEvaluationErrorKind::AllocationBudgetExceeded)
        } else {
            None
        };
        match kind {
            Some(kind) => SharedCurveError::new(
                SharedCurveErrorKind::GeometryEvaluation(kind),
                error.field,
                error.reason,
            ),
            None => SharedCurveError::invalid_encoding(error.field, error.reason),
        }
    };
    let profile =
        runmat_canonical_codec::profile_canonical_encoding_with_control(value, limits, &control)
            .map_err(codec_error)?;
    let mut storage =
        GeometryAllocationReservation::new(allocations, profile.resident_workspace_bytes_estimate)
            .map_err(|error| {
                SharedCurveError::new(
                    SharedCurveErrorKind::GeometryEvaluation(error.kind),
                    "curve encoding admission",
                    error.reason,
                )
            })?;
    let capacity = usize::try_from(profile.encoded_bytes_upper_bound).map_err(|_| {
        SharedCurveError::new(
            SharedCurveErrorKind::GeometryEvaluation(
                GeometryEvaluationErrorKind::AllocationBudgetExceeded,
            ),
            "curve encoding admission",
            "curve output capacity exceeds addressable storage",
        )
    })?;
    let bytes = runmat_canonical_codec::encode_contract_with_capacity(
        prefix, domain, value, limits, &control, capacity,
    )
    .map_err(codec_error)?;
    if bytes.len() as u64 > profile.encoded_bytes_upper_bound
        || bytes.capacity() as u64 > profile.resident_workspace_bytes_estimate
    {
        return Err(SharedCurveError::new(
            SharedCurveErrorKind::InvalidContract,
            "curve encoding admission",
            "curve output exceeds its admitted profile",
        ));
    }
    storage.retain_bytes(bytes.capacity() as u64);
    Ok(bytes)
}
