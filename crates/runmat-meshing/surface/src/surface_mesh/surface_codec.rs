use runmat_canonical_codec::{CanonicalCodecError, CanonicalLimits};
use runmat_geometry_core::ExactBRepTopology;
use runmat_meshing_curve::SharedCurveMesh;

use super::{
    validate_exact_surface_mesh, validate_exact_surface_pass_result, ExactFaceMeshBatch,
    ExactFacePartitionOutcome, ExactFacePartitionResult, ExactSurfaceJoinOptions, ExactSurfaceMesh,
    ExactSurfaceMeshError, ExactSurfaceMeshErrorKind, ExactSurfacePassOutcome,
    ExactSurfacePassResult, EXACT_FACE_MESH_BATCH_SCHEMA_VERSION,
};

const CODEC_PREFIX: &[u8] = b"runmat-meshing-exact-surface-canonical-cbor/v1\0";
const CONTRACT_DOMAIN: &str = "exact-surface-mesh/v1";
const CONTRACT_LIMITS: CanonicalLimits =
    CanonicalLimits::new(512 * 1024 * 1024, 20_000_000, 1024 * 1024, 64);

pub fn encode_exact_surface_mesh(
    mesh: &ExactSurfaceMesh,
    topology: &ExactBRepTopology,
    batches: &[ExactFaceMeshBatch],
    options: ExactSurfaceJoinOptions,
) -> Result<Vec<u8>, ExactSurfaceMeshError> {
    validate_exact_surface_mesh(mesh, topology, batches, options)?;
    runmat_canonical_codec::encode_contract(CODEC_PREFIX, CONTRACT_DOMAIN, mesh, CONTRACT_LIMITS)
        .map_err(map_codec_error)
}

pub fn decode_exact_surface_mesh(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    batches: &[ExactFaceMeshBatch],
    options: ExactSurfaceJoinOptions,
) -> Result<ExactSurfaceMesh, ExactSurfaceMeshError> {
    decode_with_limits(bytes, topology, batches, options, CONTRACT_LIMITS)
}

pub fn encode_exact_surface_mesh_from_pass(
    pass: &ExactSurfacePassResult,
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
    partitions: &[ExactFacePartitionResult],
    options: ExactSurfaceJoinOptions,
) -> Result<Vec<u8>, ExactSurfaceMeshError> {
    validate_exact_surface_pass_result(pass, topology, curves, partitions, options)?;
    let ExactSurfacePassOutcome::Converged { surface } = &pass.outcome else {
        return Err(ExactSurfaceMeshError::new(
            ExactSurfaceMeshErrorKind::InvalidInput,
            "a curve-restart pass cannot publish a final exact surface",
        ));
    };
    encode_exact_surface_mesh(surface, topology, &converged_batches(partitions)?, options)
}

pub fn decode_exact_surface_mesh_from_pass(
    bytes: &[u8],
    pass: &ExactSurfacePassResult,
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
    partitions: &[ExactFacePartitionResult],
    options: ExactSurfaceJoinOptions,
) -> Result<ExactSurfaceMesh, ExactSurfaceMeshError> {
    validate_exact_surface_pass_result(pass, topology, curves, partitions, options)?;
    if !matches!(pass.outcome, ExactSurfacePassOutcome::Converged { .. }) {
        return Err(ExactSurfaceMeshError::new(
            ExactSurfaceMeshErrorKind::InvalidInput,
            "a curve-restart pass cannot admit a final exact surface",
        ));
    }
    decode_exact_surface_mesh(bytes, topology, &converged_batches(partitions)?, options)
}

fn converged_batches(
    partitions: &[ExactFacePartitionResult],
) -> Result<Vec<ExactFaceMeshBatch>, ExactSurfaceMeshError> {
    partitions
        .iter()
        .map(|partition| {
            let ExactFacePartitionOutcome::Converged { faces } = &partition.outcome else {
                return Err(ExactSurfaceMeshError::new(
                    ExactSurfaceMeshErrorKind::InvalidInput,
                    "final exact surface prerequisites contain a curve-restart partition",
                ));
            };
            Ok(ExactFaceMeshBatch {
                schema_version: EXACT_FACE_MESH_BATCH_SCHEMA_VERSION,
                partition: partition.partition.clone(),
                faces: faces.clone(),
            })
        })
        .collect()
}

fn decode_with_limits(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    batches: &[ExactFaceMeshBatch],
    options: ExactSurfaceJoinOptions,
    limits: CanonicalLimits,
) -> Result<ExactSurfaceMesh, ExactSurfaceMeshError> {
    let mesh =
        runmat_canonical_codec::decode_contract(CODEC_PREFIX, CONTRACT_DOMAIN, bytes, limits)
            .map_err(map_codec_error)?;
    validate_exact_surface_mesh(&mesh, topology, batches, options)?;
    Ok(mesh)
}

#[cfg(test)]
pub(crate) fn decode_exact_surface_mesh_with_byte_limit(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    batches: &[ExactFaceMeshBatch],
    options: ExactSurfaceJoinOptions,
    maximum_encoded_bytes: usize,
) -> Result<ExactSurfaceMesh, ExactSurfaceMeshError> {
    decode_with_limits(
        bytes,
        topology,
        batches,
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
