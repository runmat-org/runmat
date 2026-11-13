use runmat_canonical_codec::{CanonicalCodecError, CanonicalLimits};
use runmat_geometry_core::ExactBRepTopology;
use runmat_meshing_core::{MeshingCancellationSignal, MeshingCanonicalCodecControl};

use super::{SharedCurveError, SharedCurveMesh};

const CODEC_PREFIX: &[u8] = b"runmat-meshing-curve-canonical-cbor/v4\0";
const CONTRACT_DOMAIN: &str = "shared-curve-mesh/v4";
const CONTRACT_LIMITS: CanonicalLimits = CanonicalLimits::platform_content(1024 * 1024, 64);

/// Encodes an admitted shared-curve artifact in its sole canonical wire form.
pub fn encode_shared_curve_mesh(
    mesh: &SharedCurveMesh,
    topology: &ExactBRepTopology,
) -> Result<Vec<u8>, SharedCurveError> {
    mesh.validate_against(topology)?;
    runmat_canonical_codec::encode_contract(CODEC_PREFIX, CONTRACT_DOMAIN, mesh, CONTRACT_LIMITS)
        .map_err(map_codec_error)
}

/// Cooperatively validate and admit codec workspace before producing bytes.
/// The caller retains the returned buffer's charge until its value dies.
pub fn encode_shared_curve_mesh_admitted(
    mesh: &SharedCurveMesh,
    topology: &ExactBRepTopology,
    signal: &dyn MeshingCancellationSignal,
    allocations: &dyn runmat_geometry_core::GeometryEvaluationControl,
) -> Result<Vec<u8>, SharedCurveError> {
    super::validation::validate_shared_curve_mesh_with_cancellation(mesh, topology, Some(signal))?;
    super::codec_admission::encode(
        mesh,
        CODEC_PREFIX,
        CONTRACT_DOMAIN,
        CONTRACT_LIMITS,
        signal,
        allocations,
    )
}

/// Decodes bounded hostile input and independently re-admits it against exact topology.
pub fn decode_shared_curve_mesh(
    bytes: &[u8],
    topology: &ExactBRepTopology,
) -> Result<SharedCurveMesh, SharedCurveError> {
    decode_with_limits(bytes, topology, CONTRACT_LIMITS)
}

/// Re-admits a cold artifact while probing cancellation during decoding and validation.
pub fn decode_shared_curve_mesh_with_cancellation(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<SharedCurveMesh, SharedCurveError> {
    let codec_control = MeshingCanonicalCodecControl::new(cancellation);
    let mesh = runmat_canonical_codec::decode_contract_with_control(
        CODEC_PREFIX,
        CONTRACT_DOMAIN,
        bytes,
        CONTRACT_LIMITS,
        &codec_control,
    )
    .map_err(|error| {
        if codec_control.observed_cancellation() {
            SharedCurveError::new(
                super::SharedCurveErrorKind::GeometryEvaluation(
                    runmat_geometry_core::GeometryEvaluationErrorKind::Cancelled,
                ),
                error.field,
                error.reason,
            )
        } else {
            map_codec_error(error)
        }
    })?;
    super::validation::validate_shared_curve_mesh_with_cancellation(
        &mesh,
        topology,
        Some(cancellation),
    )?;
    Ok(mesh)
}

fn decode_with_limits(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    limits: CanonicalLimits,
) -> Result<SharedCurveMesh, SharedCurveError> {
    let mesh: SharedCurveMesh =
        runmat_canonical_codec::decode_contract(CODEC_PREFIX, CONTRACT_DOMAIN, bytes, limits)
            .map_err(map_codec_error)?;
    mesh.validate_against(topology)?;
    Ok(mesh)
}

#[cfg(test)]
pub(super) fn decode_with_byte_limit(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    maximum_encoded_bytes: usize,
) -> Result<SharedCurveMesh, SharedCurveError> {
    decode_with_limits(
        bytes,
        topology,
        CanonicalLimits::content_bounded(maximum_encoded_bytes, 1024 * 1024, 64),
    )
}

fn map_codec_error(error: CanonicalCodecError) -> SharedCurveError {
    SharedCurveError::invalid_encoding(error.field, error.reason)
}
