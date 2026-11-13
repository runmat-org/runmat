//! Cold records are admitted before reconstruction. Wire/codec scratch ends at
//! decode completion; the exact physical projection retains its own reservation.
use super::*;
use crate::{codec_control::CodecControl, MeshingGeometryEvaluationControl};
use runmat_geometry_core::{
    GeometryAllocationReservation, GeometryAllocationScope, GeometryEvaluationControl,
};
use runmat_meshing_core::MeshingContractError;

pub(super) fn decode_projection<'a>(
    publication: &PreparedMeshingResultPublication,
    host: &MeshingHostWorkload,
    control: &'a MeshingGeometryEvaluationControl<'_>,
) -> Result<(GeometryAllocationReservation<'a>, SolverMeshProjection), Box<MeshingFailure>> {
    let scope = GeometryAllocationScope::new(control);
    let codec = CodecControl::new(&scope);
    let bytes = one_record(
        publication,
        MeshingStageKind::OrderElevation,
        MeshingChunkMediaType::SolverMeshProjection,
        SOLVER_MESH_PROJECTION_SCHEMA_VERSION,
        host.workload.stage,
        &codec,
    )?;
    let projection = SolverMeshProjection::canonical_decode_with_resource_admission(
        &bytes,
        &codec,
        &codec,
        |bytes| {
            codec.consume_allocation_bytes(bytes).map_err(|error| {
                MeshingContractError::invalid("terminal projection admission", error.to_string())
            })
        },
    )
    .map_err(|error| map_error(host.workload.stage, &codec, error))?;
    if projection.geometry != host.stage_identity.geometry
        || projection.resolved_request != host.resolved_request
    {
        return Err(invalid_inputs(host.workload.stage));
    }
    let retained = projection
        .retained_storage_bytes(u64::MAX, &mut || {
            codec.checkpoint().map_err(|error| {
                MeshingContractError::invalid("terminal projection retention", error.to_string())
            })
        })
        .map_err(|error| map_error(host.workload.stage, &codec, error))?;
    let storage = scope.transfer_reservation(retained).map_err(|error| {
        crate::budget::failure(
            host.workload.stage,
            crate::geometry_failure::category(error.kind),
            &error.reason,
            None,
        )
    })?;
    drop(bytes);
    Ok((storage, projection))
}

pub(super) fn decode_validation(
    publication: &PreparedMeshingResultPublication,
    control: &MeshingGeometryEvaluationControl<'_>,
) -> Result<SolverMeshValidation, Box<MeshingFailure>> {
    // Small receipt ownership remains charged to this kernel phase. Its wire and
    // decoder allowances can conservatively live until the kernel ends.
    let codec = CodecControl::new(control);
    let bytes = one_record(
        publication,
        MeshingStageKind::Validation,
        MeshingChunkMediaType::ValidationEvidence,
        SOLVER_MESH_VALIDATION_SCHEMA_VERSION,
        MeshingStageKind::Serialization,
        &codec,
    )?;
    SolverMeshValidation::canonical_decode_with_admission(
        &bytes,
        &codec,
        |bytes| {
            codec.consume_allocation_bytes(bytes).map_err(|error| {
                MeshingContractError::invalid("terminal validation admission", error.to_string())
            })
        },
        |_| Ok(0),
    )
    .map_err(|error| map_error(MeshingStageKind::Serialization, &codec, error))
}

pub(super) fn serialized_artifact_bytes(
    publication: &PreparedMeshingResultPublication,
    control: &MeshingGeometryEvaluationControl<'_>,
) -> Result<Vec<u8>, Box<MeshingFailure>> {
    let codec = CodecControl::new(control);
    one_record(
        publication,
        MeshingStageKind::Serialization,
        MeshingChunkMediaType::SolverMeshArtifact,
        SOLVER_MESH_ARTIFACT_SCHEMA_VERSION,
        MeshingStageKind::Publication,
        &codec,
    )
}

fn one_record(
    publication: &PreparedMeshingResultPublication,
    producer_stage: MeshingStageKind,
    media_type: MeshingChunkMediaType,
    schema_version: u16,
    consumer_stage: MeshingStageKind,
    control: &CodecControl<'_>,
) -> Result<Vec<u8>, Box<MeshingFailure>> {
    let stage = publication.stage_objects();
    if stage.result_identity.stage != producer_stage
        || stage.result_identity.result_kind != MeshingStageResultKind::WholeStage
    {
        return Err(invalid_inputs(consumer_stage));
    }
    let streams =
        crate::controlled_streams::decode_admitted(stage, control, control).map_err(|error| {
            crate::budget::failure(consumer_stage, error.category, &error.detail, None)
        })?;
    let [stream] = streams.as_slice() else {
        return Err(invalid_inputs(consumer_stage));
    };
    if stream.media_type != media_type
        || stream.schema_version != schema_version
        || stream.records.len() != 1
    {
        return Err(invalid_inputs(consumer_stage));
    }
    control.checkpoint().map_err(|error| {
        crate::budget::failure(
            consumer_stage,
            crate::geometry_failure::category(error.kind),
            &error.reason,
            None,
        )
    })?;
    Ok(crate::controlled_streams::take_single_record(streams, 0)
        .expect("one admitted terminal record"))
}

pub(super) fn map_error(
    stage: MeshingStageKind,
    control: &CodecControl<'_>,
    error: MeshingContractError,
) -> Box<MeshingFailure> {
    match control.take_failure() {
        Some(original) => crate::budget::failure(
            stage,
            crate::geometry_failure::category(original.kind),
            &original.reason,
            None,
        ),
        None => terminal_failure(stage, &error.to_string()),
    }
}
