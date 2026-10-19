mod record_set;
mod storage;
pub use record_set::{
    decode_exact_face_partition_records, decode_exact_face_partition_records_admitted,
    decode_exact_face_partition_records_with_cancellation, encode_exact_face_partition_completion,
    profile_exact_face_partition_completion, EXACT_FACE_PARTITION_RECORD_SET_SCHEMA_VERSION,
};
use runmat_geometry_core::ExactBRepTopology;
use runmat_meshing_core::{MeshingCancellationSignal, MeshingPartitionDescriptor};
use runmat_meshing_curve::{
    validate_shared_curve_split_set, validate_shared_curve_split_set_with_cancellation,
    SharedCurveMesh,
};

use super::{
    validate_face_partition_descriptor, ExactFacePartitionOutcome, ExactFacePartitionResult,
    ExactSurfaceMeshError, ExactSurfaceMeshErrorKind, EXACT_FACE_MESH_BATCH_SCHEMA_VERSION,
    EXACT_FACE_PARTITION_RESULT_SCHEMA_VERSION,
};

pub(crate) mod boundary;
use boundary::validate_curve_boundary;
mod codec;
#[cfg(test)]
pub(crate) use codec::decode_exact_face_partition_result_with_byte_limit;
#[cfg(test)]
pub(crate) use codec::encode_unadmitted_for_test;
pub use codec::{
    decode_exact_face_partition_result, decode_exact_face_partition_result_admitted,
    decode_exact_face_partition_result_with_cancellation,
    decode_exact_face_reuse_record_with_cancellation, encode_exact_face_partition_result,
    encode_exact_face_partition_result_with_cancellation,
    profile_exact_face_partition_result_encoding,
};
pub(crate) use codec::{encode_accepted_face, profile_accepted_face};

use super::control::checkpoint;

pub fn build_exact_face_partition_result(
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
    partition: MeshingPartitionDescriptor,
    outcome: ExactFacePartitionOutcome,
) -> Result<ExactFacePartitionResult, ExactSurfaceMeshError> {
    let boundary = crate::build_exact_surface_boundary(topology, curves)
        .map_err(|error| invalid(error.to_string()))?;
    build_exact_face_partition_result_with_boundary(
        topology, curves, &boundary, partition, outcome, None,
    )
}

pub(crate) fn build_exact_face_partition_result_with_boundary(
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
    boundary: &crate::ExactSurfaceBoundary,
    partition: MeshingPartitionDescriptor,
    outcome: ExactFacePartitionOutcome,
    cancellation: Option<&dyn MeshingCancellationSignal>,
) -> Result<ExactFacePartitionResult, ExactSurfaceMeshError> {
    let result = ExactFacePartitionResult {
        schema_version: EXACT_FACE_PARTITION_RESULT_SCHEMA_VERSION,
        partition,
        outcome,
    };
    validate_with_boundary(&result, topology, curves, boundary, cancellation)?;
    Ok(result)
}

pub fn validate_exact_face_partition_result(
    result: &ExactFacePartitionResult,
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
) -> Result<(), ExactSurfaceMeshError> {
    validate_with_cancellation(result, topology, curves, None)
}

pub fn validate_exact_face_partition_result_with_cancellation(
    result: &ExactFacePartitionResult,
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), ExactSurfaceMeshError> {
    validate_with_cancellation(result, topology, curves, Some(cancellation))
}

fn validate_with_cancellation(
    result: &ExactFacePartitionResult,
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
    cancellation: Option<&dyn MeshingCancellationSignal>,
) -> Result<(), ExactSurfaceMeshError> {
    checkpoint(cancellation)?;
    let boundary = match cancellation {
        Some(control) => {
            crate::build_exact_surface_boundary_with_cancellation(topology, curves, control)
        }
        None => crate::build_exact_surface_boundary(topology, curves),
    }
    .map_err(|error| {
        if error.kind == crate::ExactSurfaceBoundaryErrorKind::Cancelled {
            ExactSurfaceMeshError::new(ExactSurfaceMeshErrorKind::Cancelled, error.to_string())
        } else {
            invalid(error.to_string())
        }
    })?;
    validate_with_boundary(result, topology, curves, &boundary, cancellation)
}

pub(super) fn validate_with_boundary(
    result: &ExactFacePartitionResult,
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
    boundary: &crate::ExactSurfaceBoundary,
    cancellation: Option<&dyn MeshingCancellationSignal>,
) -> Result<(), ExactSurfaceMeshError> {
    checkpoint(cancellation)?;
    if result.schema_version != EXACT_FACE_PARTITION_RESULT_SCHEMA_VERSION {
        return Err(invalid("face partition result schema is unsupported"));
    }
    validate_face_partition_descriptor(&result.partition)?;
    match &result.outcome {
        ExactFacePartitionOutcome::Converged { faces } => {
            super::batch::validate_exact_face_mesh_batch_parts_with_cancellation(
                EXACT_FACE_MESH_BATCH_SCHEMA_VERSION,
                &result.partition,
                faces,
                topology,
                cancellation,
            )?;
            validate_curve_boundary(faces, boundary, topology, curves, cancellation)
        }
        ExactFacePartitionOutcome::RequiresCurveSplits { splits } => {
            validate_split_ownership(&result.partition, topology, splits, cancellation)?;
            let validation = match cancellation {
                Some(control) => validate_shared_curve_split_set_with_cancellation(
                    curves, topology, splits, control,
                ),
                None => validate_shared_curve_split_set(curves, topology, splits),
            };
            validation.map_err(|error| {
                if error.kind
                    == runmat_meshing_curve::SharedCurveErrorKind::GeometryEvaluation(
                        runmat_geometry_core::GeometryEvaluationErrorKind::Cancelled,
                    )
                {
                    ExactSurfaceMeshError::new(
                        ExactSurfaceMeshErrorKind::Cancelled,
                        error.to_string(),
                    )
                } else {
                    invalid(error.to_string())
                }
            })
        }
    }
}

fn validate_split_ownership(
    partition: &MeshingPartitionDescriptor,
    topology: &ExactBRepTopology,
    splits: &[runmat_meshing_curve::SharedCurveSegmentSplit],
    cancellation: Option<&dyn MeshingCancellationSignal>,
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
    for split in splits {
        checkpoint(cancellation)?;
        if topology.coedges.iter().any(|coedge| {
            coedge.edge_id == split.source_edge_id
                && coedge.face_id >= range.first
                && coedge.face_id <= range.last
        }) {
            continue;
        }
        return Err(invalid(
            "curve-restart result references an edge outside its face partition",
        ));
    }
    Ok(())
}

fn invalid(reason: impl Into<String>) -> ExactSurfaceMeshError {
    ExactSurfaceMeshError::new(ExactSurfaceMeshErrorKind::InvalidInput, reason)
}
