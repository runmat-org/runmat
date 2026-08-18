use runmat_canonical_codec::{CanonicalCodecError, CanonicalLimits};
use runmat_geometry_core::ExactBRepTopology;
use runmat_meshing_core::MeshingPartitionDescriptor;
use runmat_meshing_curve::{validate_shared_curve_split_set, SharedCurveMesh};

use super::{
    validate_exact_face_mesh_batch_parts, validate_face_partition_descriptor,
    ExactFacePartitionOutcome, ExactFacePartitionResult, ExactSurfaceMeshError,
    ExactSurfaceMeshErrorKind, EXACT_FACE_MESH_BATCH_SCHEMA_VERSION,
    EXACT_FACE_PARTITION_RESULT_SCHEMA_VERSION,
};

const CODEC_PREFIX: &[u8] = b"runmat-meshing-face-partition-result-canonical-cbor/v1\0";
const CONTRACT_DOMAIN: &str = "exact-face-partition-result/v1";
const CONTRACT_LIMITS: CanonicalLimits =
    CanonicalLimits::new(256 * 1024 * 1024, 10_000_000, 1024 * 1024, 64);

pub fn build_exact_face_partition_result(
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
    partition: MeshingPartitionDescriptor,
    outcome: ExactFacePartitionOutcome,
) -> Result<ExactFacePartitionResult, ExactSurfaceMeshError> {
    let result = ExactFacePartitionResult {
        schema_version: EXACT_FACE_PARTITION_RESULT_SCHEMA_VERSION,
        partition,
        outcome,
    };
    validate_exact_face_partition_result(&result, topology, curves)?;
    Ok(result)
}

pub fn validate_exact_face_partition_result(
    result: &ExactFacePartitionResult,
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
) -> Result<(), ExactSurfaceMeshError> {
    if result.schema_version != EXACT_FACE_PARTITION_RESULT_SCHEMA_VERSION {
        return Err(invalid("face partition result schema is unsupported"));
    }
    validate_face_partition_descriptor(&result.partition)?;
    match &result.outcome {
        ExactFacePartitionOutcome::Converged { faces } => validate_exact_face_mesh_batch_parts(
            EXACT_FACE_MESH_BATCH_SCHEMA_VERSION,
            &result.partition,
            faces,
            topology,
        ),
        ExactFacePartitionOutcome::RequiresCurveSplits { splits } => {
            validate_split_ownership(&result.partition, topology, splits)?;
            validate_shared_curve_split_set(curves, topology, splits)
                .map_err(|error| invalid(error.to_string()))
        }
    }
}

pub fn encode_exact_face_partition_result(
    result: &ExactFacePartitionResult,
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
) -> Result<Vec<u8>, ExactSurfaceMeshError> {
    validate_exact_face_partition_result(result, topology, curves)?;
    runmat_canonical_codec::encode_contract(CODEC_PREFIX, CONTRACT_DOMAIN, result, CONTRACT_LIMITS)
        .map_err(map_codec_error)
}

pub fn decode_exact_face_partition_result(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
) -> Result<ExactFacePartitionResult, ExactSurfaceMeshError> {
    decode_with_limits(bytes, topology, curves, CONTRACT_LIMITS)
}

fn decode_with_limits(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
    limits: CanonicalLimits,
) -> Result<ExactFacePartitionResult, ExactSurfaceMeshError> {
    let result =
        runmat_canonical_codec::decode_contract(CODEC_PREFIX, CONTRACT_DOMAIN, bytes, limits)
            .map_err(map_codec_error)?;
    validate_exact_face_partition_result(&result, topology, curves)?;
    Ok(result)
}

#[cfg(test)]
pub(crate) fn decode_exact_face_partition_result_with_byte_limit(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
    maximum_encoded_bytes: usize,
) -> Result<ExactFacePartitionResult, ExactSurfaceMeshError> {
    decode_with_limits(
        bytes,
        topology,
        curves,
        CanonicalLimits::new(maximum_encoded_bytes, 10_000_000, 1024 * 1024, 64),
    )
}

fn validate_split_ownership(
    partition: &MeshingPartitionDescriptor,
    topology: &ExactBRepTopology,
    splits: &[runmat_meshing_curve::SharedCurveSegmentSplit],
) -> Result<(), ExactSurfaceMeshError> {
    let range = partition
        .entity_range
        .as_ref()
        .expect("partition validated");
    let faces = topology
        .faces
        .iter()
        .filter(|face| face.id >= range.first && face.id <= range.last)
        .collect::<Vec<_>>();
    if faces.len() as u64 != range.entity_count
        || faces.first().is_none_or(|face| face.id != range.first)
        || faces.last().is_none_or(|face| face.id != range.last)
    {
        return Err(invalid(
            "curve-restart result does not own its declared canonical face range",
        ));
    }
    if splits.iter().any(|split| {
        !topology.coedges.iter().any(|coedge| {
            coedge.edge_id == split.source_edge_id
                && coedge.face_id >= range.first
                && coedge.face_id <= range.last
        })
    }) {
        return Err(invalid(
            "curve-restart result references an edge outside its face partition",
        ));
    }
    Ok(())
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
