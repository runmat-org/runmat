use runmat_canonical_codec::CanonicalLimits;
use runmat_geometry_core::ExactBRepTopology;
use runmat_meshing_core::MeshingCancellationSignal;
use runmat_meshing_curve::SharedCurveMesh;

use super::{
    ExactFacePartitionResult, ExactSurfaceJoinOptions, ExactSurfaceMeshError,
    ExactSurfaceMeshErrorKind, ExactSurfacePassResult, EXACT_SURFACE_PASS_RESULT_SCHEMA_VERSION,
};

const CODEC_PREFIX: &[u8] = b"runmat-meshing-exact-surface-pass-canonical-cbor/v1\0";
const CONTRACT_DOMAIN: &str = "exact-surface-pass-result/v1";
const CONTRACT_LIMITS: CanonicalLimits = CanonicalLimits::platform_content(1024 * 1024, 64);

pub fn validate_exact_surface_pass_result(
    result: &ExactSurfacePassResult,
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
    partitions: &[ExactFacePartitionResult],
    options: ExactSurfaceJoinOptions,
) -> Result<(), ExactSurfaceMeshError> {
    validate_with_control(result, topology, curves, partitions, options, None)
}

pub(super) fn validate_with_control(
    result: &ExactSurfacePassResult,
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
    partitions: &[ExactFacePartitionResult],
    options: ExactSurfaceJoinOptions,
    control: Option<&dyn MeshingCancellationSignal>,
) -> Result<(), ExactSurfaceMeshError> {
    super::control::checkpoint(control)?;
    if result.schema_version != EXACT_SURFACE_PASS_RESULT_SCHEMA_VERSION {
        return Err(invalid("surface pass result schema is unsupported"));
    }
    super::convergence::validation::validate_decision(
        result, curves, partitions, topology, options, control,
    )
    .map_err(|error| {
        ExactSurfaceMeshError::new(
            if matches!(
                error.kind,
                super::ExactSurfaceConvergenceErrorKind::Surface(
                    ExactSurfaceMeshErrorKind::Cancelled
                ) | super::ExactSurfaceConvergenceErrorKind::Curve(
                    runmat_meshing_curve::SharedCurveErrorKind::GeometryEvaluation(
                        runmat_geometry_core::GeometryEvaluationErrorKind::Cancelled
                    )
                )
            ) {
                ExactSurfaceMeshErrorKind::Cancelled
            } else {
                ExactSurfaceMeshErrorKind::InvalidInput
            },
            error.to_string(),
        )
    })?;
    Ok(())
}

pub fn encode_exact_surface_pass_result(
    result: &ExactSurfacePassResult,
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
    partitions: &[ExactFacePartitionResult],
    options: ExactSurfaceJoinOptions,
) -> Result<Vec<u8>, ExactSurfaceMeshError> {
    validate_exact_surface_pass_result(result, topology, curves, partitions, options)?;
    super::codec::encode(CODEC_PREFIX, CONTRACT_DOMAIN, result, CONTRACT_LIMITS, None)
}

/// Encodes a pass result returned directly by [`decide_exact_surface_pass`].
///
/// Artifact consumers still reconstruct the decision during decoding. Callers with any other
/// source must use [`encode_exact_surface_pass_result`] so prerequisite validation is not skipped.
pub fn encode_decided_exact_surface_pass_result(
    result: &ExactSurfacePassResult,
) -> Result<Vec<u8>, ExactSurfaceMeshError> {
    super::codec::encode(CODEC_PREFIX, CONTRACT_DOMAIN, result, CONTRACT_LIMITS, None)
}

pub fn decode_exact_surface_pass_result(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
    partitions: &[ExactFacePartitionResult],
    options: ExactSurfaceJoinOptions,
) -> Result<ExactSurfacePassResult, ExactSurfaceMeshError> {
    decode_with_limits(
        bytes,
        topology,
        curves,
        partitions,
        options,
        CONTRACT_LIMITS,
    )
}

fn decode_with_limits(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
    partitions: &[ExactFacePartitionResult],
    options: ExactSurfaceJoinOptions,
    limits: CanonicalLimits,
) -> Result<ExactSurfacePassResult, ExactSurfaceMeshError> {
    let result = super::codec::decode(CODEC_PREFIX, CONTRACT_DOMAIN, bytes, limits, None)?;
    validate_exact_surface_pass_result(&result, topology, curves, partitions, options)?;
    Ok(result)
}

#[cfg(test)]
pub(crate) fn decode_exact_surface_pass_result_with_byte_limit(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
    partitions: &[ExactFacePartitionResult],
    options: ExactSurfaceJoinOptions,
    maximum_encoded_bytes: usize,
) -> Result<ExactSurfacePassResult, ExactSurfaceMeshError> {
    decode_with_limits(
        bytes,
        topology,
        curves,
        partitions,
        options,
        CanonicalLimits::content_bounded(maximum_encoded_bytes, 1024 * 1024, 64),
    )
}

fn invalid(reason: impl Into<String>) -> ExactSurfaceMeshError {
    ExactSurfaceMeshError::new(ExactSurfaceMeshErrorKind::InvalidInput, reason)
}

/// Cooperatively encodes a decision made by the canonical pass coordinator.
pub fn encode_decided_exact_surface_pass_result_with_cancellation(
    result: &ExactSurfacePassResult,
    control: &dyn MeshingCancellationSignal,
) -> Result<Vec<u8>, ExactSurfaceMeshError> {
    super::codec::encode(
        CODEC_PREFIX,
        CONTRACT_DOMAIN,
        result,
        CONTRACT_LIMITS,
        Some(control),
    )
}

/// Reconstructs and validates the cold decision under the caller's execution control.
pub fn decode_exact_surface_pass_result_with_cancellation(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
    partitions: &[ExactFacePartitionResult],
    options: ExactSurfaceJoinOptions,
    control: &dyn MeshingCancellationSignal,
) -> Result<ExactSurfacePassResult, ExactSurfaceMeshError> {
    let result = super::codec::decode(
        CODEC_PREFIX,
        CONTRACT_DOMAIN,
        bytes,
        CONTRACT_LIMITS,
        Some(control),
    )?;
    validate_with_control(
        &result,
        topology,
        curves,
        partitions,
        options,
        Some(control),
    )?;
    Ok(result)
}

pub(in crate::surface_mesh) fn profile_decided_pass(
    result: &ExactSurfacePassResult,
    signal: &dyn MeshingCancellationSignal,
) -> Result<runmat_canonical_codec::CanonicalEncodingProfile, ExactSurfaceMeshError> {
    let control = runmat_meshing_core::MeshingCanonicalCodecControl::new(signal);
    runmat_canonical_codec::profile_canonical_encoding_with_control(
        result,
        CONTRACT_LIMITS,
        &control,
    )
    .map_err(|error| {
        ExactSurfaceMeshError::new(
            if control.observed_cancellation() {
                ExactSurfaceMeshErrorKind::Cancelled
            } else {
                ExactSurfaceMeshErrorKind::InvalidEncoding
            },
            error.to_string(),
        )
    })
}
