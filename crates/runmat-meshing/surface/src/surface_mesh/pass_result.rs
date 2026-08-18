use runmat_canonical_codec::{CanonicalCodecError, CanonicalLimits};
use runmat_geometry_core::ExactBRepTopology;
use runmat_meshing_curve::SharedCurveMesh;

use super::{
    decide_exact_surface_pass, ExactFacePartitionResult, ExactSurfaceJoinOptions,
    ExactSurfaceMeshError, ExactSurfaceMeshErrorKind, ExactSurfacePassResult,
    EXACT_SURFACE_PASS_RESULT_SCHEMA_VERSION,
};

const CODEC_PREFIX: &[u8] = b"runmat-meshing-exact-surface-pass-canonical-cbor/v1\0";
const CONTRACT_DOMAIN: &str = "exact-surface-pass-result/v1";
const CONTRACT_LIMITS: CanonicalLimits =
    CanonicalLimits::new(512 * 1024 * 1024, 20_000_000, 1024 * 1024, 64);

pub fn validate_exact_surface_pass_result(
    result: &ExactSurfacePassResult,
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
    partitions: &[ExactFacePartitionResult],
    options: ExactSurfaceJoinOptions,
) -> Result<(), ExactSurfaceMeshError> {
    if result.schema_version != EXACT_SURFACE_PASS_RESULT_SCHEMA_VERSION {
        return Err(invalid("surface pass result schema is unsupported"));
    }
    let expected = decide_exact_surface_pass(curves, partitions, topology, options)
        .map_err(|error| invalid(error.to_string()))?;
    if result != &expected {
        return Err(invalid(
            "surface pass result does not reproduce from its exact prerequisites",
        ));
    }
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
    runmat_canonical_codec::encode_contract(CODEC_PREFIX, CONTRACT_DOMAIN, result, CONTRACT_LIMITS)
        .map_err(map_codec_error)
}

/// Encodes a pass result returned directly by [`decide_exact_surface_pass`].
///
/// Artifact consumers still reconstruct the decision during decoding. Callers with any other
/// source must use [`encode_exact_surface_pass_result`] so prerequisite validation is not skipped.
pub fn encode_decided_exact_surface_pass_result(
    result: &ExactSurfacePassResult,
) -> Result<Vec<u8>, ExactSurfaceMeshError> {
    runmat_canonical_codec::encode_contract(CODEC_PREFIX, CONTRACT_DOMAIN, result, CONTRACT_LIMITS)
        .map_err(map_codec_error)
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
    let result =
        runmat_canonical_codec::decode_contract(CODEC_PREFIX, CONTRACT_DOMAIN, bytes, limits)
            .map_err(map_codec_error)?;
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
        CanonicalLimits::new(maximum_encoded_bytes, 20_000_000, 1024 * 1024, 64),
    )
}

fn map_codec_error(error: CanonicalCodecError) -> ExactSurfaceMeshError {
    ExactSurfaceMeshError::new(
        ExactSurfaceMeshErrorKind::InvalidEncoding,
        format!("{}: {}", error.field, error.reason),
    )
}

fn invalid(reason: impl Into<String>) -> ExactSurfaceMeshError {
    ExactSurfaceMeshError::new(ExactSurfaceMeshErrorKind::InvalidInput, reason)
}
