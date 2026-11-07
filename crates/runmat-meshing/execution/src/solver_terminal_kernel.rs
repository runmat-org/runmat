mod encoding;
mod source;
use source::{decode_projection, decode_validation, serialized_artifact_bytes};
use std::collections::BTreeMap;

use runmat_meshing_core::{
    CanonicalMeshingContract, MeshingChunkMediaType, MeshingChunkStream, MeshingDiagnosticEntry,
    MeshingDiagnosticValue, MeshingFailure, MeshingFailureCategory, MeshingPartitionKind,
    MeshingStageKind, MeshingStageResultKind, SolverMeshArtifact, SolverMeshProjection,
    SolverMeshValidation, StableDigest, MESHING_EVIDENCE_SCHEMA_VERSION,
    MESHING_FAILURE_SCHEMA_VERSION, SOLVER_MESH_ARTIFACT_SCHEMA_VERSION,
    SOLVER_MESH_PROJECTION_SCHEMA_VERSION, SOLVER_MESH_VALIDATION_SCHEMA_VERSION,
};

use crate::{
    MeshingHostWorkload, MeshingStageCheckpoint, MeshingStageInvocation, MeshingStageKernel,
    PreparedMeshingInput, PreparedMeshingResultPublication, ValidatedMeshingStageOutput,
};

#[derive(Clone, Copy, Debug, Default)]
pub struct SolverValidationKernel;

impl MeshingStageKernel for SolverValidationKernel {
    fn execute(
        &self,
        invocation: MeshingStageInvocation<'_, '_>,
    ) -> Result<ValidatedMeshingStageOutput, Box<MeshingFailure>> {
        require_stage(&invocation, MeshingStageKind::Validation)?;
        let [PreparedMeshingInput::StageArtifact(projection_publication)] = invocation.inputs
        else {
            return Err(invalid_inputs(MeshingStageKind::Validation));
        };
        invocation
            .control
            .checkpoint(MeshingStageCheckpoint::default())?;
        let input_control = invocation.control.geometry_evaluation_control();
        let (_projection_storage, projection) =
            decode_projection(projection_publication, invocation.host, &input_control)?;
        invocation
            .control
            .checkpoint(MeshingStageCheckpoint::default())?;
        let validation_work = encoding::projection_work(
            &projection,
            MeshingStageKind::Validation,
            &input_control,
            false,
        )?;
        let validation_codec = crate::codec_control::CodecControl::new(&input_control);
        let validation = SolverMeshValidation::from_validated_projection_with_cancellation(
            &projection,
            &validation_codec,
        )
        .map_err(|error| {
            source::map_error(MeshingStageKind::Validation, &validation_codec, error)
        })?;
        drop(validation_work);
        invocation
            .control
            .checkpoint(MeshingStageCheckpoint::default())?;
        let output_control = invocation.control.geometry_evaluation_control();
        let encoded = encoding::encode(
            &validation,
            MeshingStageKind::Validation,
            &output_control,
            0,
            |_| validation.canonical_encode(),
        )?;
        let invariant_summary_digest =
            encoding::digest(&encoded, MeshingStageKind::Validation, &output_control)?;
        let checkpoint = topology_checkpoint(&projection, encoded.len() as u64)?;
        invocation.control.checkpoint(checkpoint.clone())?;
        Ok(ValidatedMeshingStageOutput {
            invariant_summary_digest,
            streams: vec![MeshingChunkStream {
                media_type: MeshingChunkMediaType::ValidationEvidence,
                schema_version: SOLVER_MESH_VALIDATION_SCHEMA_VERSION,
                records: vec![encoded],
            }],
            final_checkpoint: checkpoint,
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SolverSerializationKernel;

impl MeshingStageKernel for SolverSerializationKernel {
    fn execute(
        &self,
        invocation: MeshingStageInvocation<'_, '_>,
    ) -> Result<ValidatedMeshingStageOutput, Box<MeshingFailure>> {
        require_stage(&invocation, MeshingStageKind::Serialization)?;
        let (projection_publication, validation_publication) = terminal_inputs(invocation.inputs)?;
        invocation
            .control
            .checkpoint(MeshingStageCheckpoint::default())?;
        let input_control = invocation.control.geometry_evaluation_control();
        let (_projection_storage, projection) =
            decode_projection(projection_publication, invocation.host, &input_control)?;
        invocation
            .control
            .checkpoint(MeshingStageCheckpoint::default())?;
        let validation = decode_validation(validation_publication, &input_control)?;
        invocation
            .control
            .checkpoint(MeshingStageCheckpoint::default())?;
        let validation_work = encoding::projection_work(
            &projection,
            MeshingStageKind::Serialization,
            &input_control,
            false,
        )?;
        let validation_codec = crate::codec_control::CodecControl::new(&input_control);
        validation
            .validate_against_with_cancellation(&projection, &validation_codec)
            .map_err(|error| {
                source::map_error(MeshingStageKind::Serialization, &validation_codec, error)
            })?;
        drop(validation_work);
        invocation
            .control
            .checkpoint(MeshingStageCheckpoint::default())?;
        let projection_digest =
            StableDigest::from_bytes(*projection_publication.stage_objects().root.digest.bytes());
        if validation_publication
            .stage_objects()
            .manifest
            .prerequisite_manifest_digests
            .binary_search(&projection_digest)
            .is_err()
        {
            return Err(invalid_inputs(MeshingStageKind::Serialization));
        }
        let validation_manifest_digest =
            StableDigest::from_bytes(*validation_publication.stage_objects().root.digest.bytes());
        let sealing_work = encoding::projection_work(
            &projection,
            MeshingStageKind::Serialization,
            &input_control,
            true,
        )?;
        let sealing_codec = crate::codec_control::CodecControl::new(&input_control);
        let artifact = projection
            .into_artifact_with_cancellation(validation_manifest_digest, &sealing_codec)
            .map_err(|error| {
                source::map_error(MeshingStageKind::Serialization, &sealing_codec, error)
            })?;
        drop(sealing_work);
        invocation
            .control
            .checkpoint(MeshingStageCheckpoint::default())?;
        let output_control = invocation.control.geometry_evaluation_control();
        let validation_scratch = artifact
            .topology
            .validation_scratch_bytes(artifact.resolved_request.element_order)
            .map_err(|error| {
                terminal_failure(MeshingStageKind::Serialization, &error.to_string())
            })?;
        let encoded = encoding::encode(
            &artifact,
            MeshingStageKind::Serialization,
            &output_control,
            validation_scratch,
            |codec| artifact.canonical_encode_with_cancellation(codec),
        )?;
        let checkpoint = artifact_checkpoint(&artifact, encoded.len() as u64)?;
        invocation.control.checkpoint(checkpoint.clone())?;
        Ok(ValidatedMeshingStageOutput {
            invariant_summary_digest: artifact.canonical_digest,
            streams: vec![MeshingChunkStream {
                media_type: MeshingChunkMediaType::SolverMeshArtifact,
                schema_version: SOLVER_MESH_ARTIFACT_SCHEMA_VERSION,
                records: vec![encoded],
            }],
            final_checkpoint: checkpoint,
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SolverPublicationKernel;

impl MeshingStageKernel for SolverPublicationKernel {
    fn execute(
        &self,
        invocation: MeshingStageInvocation<'_, '_>,
    ) -> Result<ValidatedMeshingStageOutput, Box<MeshingFailure>> {
        require_stage(&invocation, MeshingStageKind::Publication)?;
        let (serialization, evidence_input) = publication_inputs(invocation.inputs)?;
        invocation
            .control
            .checkpoint(MeshingStageCheckpoint::default())?;
        let artifact_control = invocation.control.geometry_evaluation_control();
        let artifact_bytes = serialized_artifact_bytes(serialization, &artifact_control)?;
        invocation
            .control
            .checkpoint(MeshingStageCheckpoint::default())?;
        let evidence = &evidence_input.evidence_objects().evidence;
        let artifact_byte_count = u64::try_from(artifact_bytes.len()).map_err(|_| {
            terminal_failure(
                MeshingStageKind::Publication,
                "serialized artifact byte inventory overflowed",
            )
        })?;
        let artifact_digest = serialization
            .stage_objects()
            .result_identity
            .invariant_summary_digest;
        evidence
            .validate_publication_closure(
                &invocation.host.stage_identity.geometry,
                &invocation.host.resolved_request,
                artifact_digest,
                artifact_byte_count,
            )
            .map_err(|error| terminal_failure(MeshingStageKind::Publication, &error.to_string()))?;
        invocation
            .control
            .checkpoint(MeshingStageCheckpoint::default())?;
        let serialization_digest =
            StableDigest::from_bytes(*serialization.stage_objects().root.digest.bytes());
        if evidence
            .stages
            .last()
            .map(|stage| stage.stage_result_digest)
            != Some(serialization_digest)
        {
            return Err(invalid_inputs(MeshingStageKind::Publication));
        }
        let output_control = invocation.control.geometry_evaluation_control();
        let evidence_bytes = encoding::encode(
            evidence,
            MeshingStageKind::Publication,
            &output_control,
            0,
            |_| evidence.canonical_encode(),
        )?;
        let invariant_summary_digest = encoding::digest(
            &evidence_bytes,
            MeshingStageKind::Publication,
            &output_control,
        )?;
        let encoded_length = u64::try_from(artifact_bytes.len())
            .ok()
            .and_then(|artifact_length| {
                u64::try_from(evidence_bytes.len())
                    .ok()
                    .and_then(|evidence_length| artifact_length.checked_add(evidence_length))
            })
            .ok_or_else(|| {
                terminal_failure(
                    MeshingStageKind::Publication,
                    "terminal publication byte inventory overflowed",
                )
            })?;
        let checkpoint = artifact_inventory_checkpoint(
            evidence.resources.generated_nodes,
            evidence.resources.generated_elements,
            evidence_entity_count(evidence, "solver_boundary_faces")?,
            evidence_entity_count(evidence, "solver_boundary_edges")?,
            encoded_length,
            MeshingStageKind::Publication,
        )?;
        invocation.control.checkpoint(checkpoint.clone())?;
        Ok(ValidatedMeshingStageOutput {
            invariant_summary_digest,
            streams: vec![
                MeshingChunkStream {
                    media_type: MeshingChunkMediaType::SolverMeshArtifact,
                    schema_version: SOLVER_MESH_ARTIFACT_SCHEMA_VERSION,
                    records: vec![artifact_bytes],
                },
                MeshingChunkStream {
                    media_type: MeshingChunkMediaType::MeshingEvidence,
                    schema_version: MESHING_EVIDENCE_SCHEMA_VERSION,
                    records: vec![evidence_bytes],
                },
            ],
            final_checkpoint: checkpoint,
        })
    }
}

fn require_stage(
    invocation: &MeshingStageInvocation<'_, '_>,
    stage: MeshingStageKind,
) -> Result<(), Box<MeshingFailure>> {
    if invocation.host.workload.stage != stage
        || invocation.host.workload.partition.kind != MeshingPartitionKind::WholeStage
    {
        return Err(terminal_failure(
            stage,
            "terminal solver stages require one whole-stage workload",
        ));
    }
    Ok(())
}

fn terminal_inputs(
    inputs: &[PreparedMeshingInput],
) -> Result<
    (
        &PreparedMeshingResultPublication,
        &PreparedMeshingResultPublication,
    ),
    Box<MeshingFailure>,
> {
    let mut projection = None;
    let mut validation = None;
    for input in inputs {
        let PreparedMeshingInput::StageArtifact(publication) = input else {
            return Err(invalid_inputs(MeshingStageKind::Serialization));
        };
        match publication.stage_objects().result_identity.stage {
            MeshingStageKind::OrderElevation if projection.is_none() => {
                projection = Some(publication.as_ref());
            }
            MeshingStageKind::Validation if validation.is_none() => {
                validation = Some(publication.as_ref());
            }
            _ => return Err(invalid_inputs(MeshingStageKind::Serialization)),
        }
    }
    Ok((
        projection.ok_or_else(|| invalid_inputs(MeshingStageKind::Serialization))?,
        validation.ok_or_else(|| invalid_inputs(MeshingStageKind::Serialization))?,
    ))
}

fn publication_inputs(
    inputs: &[PreparedMeshingInput],
) -> Result<
    (
        &PreparedMeshingResultPublication,
        &crate::PreparedEvidenceInput,
    ),
    Box<MeshingFailure>,
> {
    let mut serialization = None;
    let mut evidence = None;
    for input in inputs {
        match input {
            PreparedMeshingInput::StageArtifact(publication)
                if publication.stage_objects().result_identity.stage
                    == MeshingStageKind::Serialization
                    && serialization.is_none() =>
            {
                serialization = Some(publication.as_ref());
            }
            PreparedMeshingInput::Evidence(input) if evidence.is_none() => {
                evidence = Some(input.as_ref());
            }
            _ => return Err(invalid_inputs(MeshingStageKind::Publication)),
        }
    }
    Ok((
        serialization.ok_or_else(|| invalid_inputs(MeshingStageKind::Publication))?,
        evidence.ok_or_else(|| invalid_inputs(MeshingStageKind::Publication))?,
    ))
}

fn topology_checkpoint(
    projection: &SolverMeshProjection,
    encoded_length: u64,
) -> Result<MeshingStageCheckpoint, Box<MeshingFailure>> {
    checkpoint(
        projection.topology.nodes.len() as u64,
        projection.topology.volume_elements.len() as u64,
        projection.topology.boundary_faces.len() as u64,
        projection.topology.boundary_edges.len() as u64,
        encoded_length,
        MeshingStageKind::Validation,
    )
}

fn artifact_checkpoint(
    artifact: &SolverMeshArtifact,
    encoded_length: u64,
) -> Result<MeshingStageCheckpoint, Box<MeshingFailure>> {
    artifact_checkpoint_for_stage(artifact, encoded_length, MeshingStageKind::Serialization)
}

fn artifact_checkpoint_for_stage(
    artifact: &SolverMeshArtifact,
    encoded_length: u64,
    stage: MeshingStageKind,
) -> Result<MeshingStageCheckpoint, Box<MeshingFailure>> {
    checkpoint(
        artifact.topology.nodes.len() as u64,
        artifact.topology.volume_elements.len() as u64,
        artifact.topology.boundary_faces.len() as u64,
        artifact.topology.boundary_edges.len() as u64,
        encoded_length,
        stage,
    )
}

fn artifact_inventory_checkpoint(
    node_count: u64,
    element_count: u64,
    face_count: u64,
    edge_count: u64,
    encoded_length: u64,
    stage: MeshingStageKind,
) -> Result<MeshingStageCheckpoint, Box<MeshingFailure>> {
    checkpoint(
        node_count,
        element_count,
        face_count,
        edge_count,
        encoded_length,
        stage,
    )
}

fn evidence_entity_count(
    evidence: &runmat_meshing_core::MeshingEvidence,
    name: &str,
) -> Result<u64, Box<MeshingFailure>> {
    evidence
        .stages
        .last()
        .and_then(|stage| stage.entity_counts.get(name))
        .copied()
        .ok_or_else(|| {
            terminal_failure(
                MeshingStageKind::Publication,
                "serialization evidence is missing its solver entity inventory",
            )
        })
}

fn checkpoint(
    node_count: u64,
    element_count: u64,
    face_count: u64,
    edge_count: u64,
    encoded_length: u64,
    stage: MeshingStageKind,
) -> Result<MeshingStageCheckpoint, Box<MeshingFailure>> {
    let completed_work = node_count
        .checked_add(element_count)
        .ok_or_else(|| terminal_failure(stage, "terminal solver work inventory overflowed"))?;
    Ok(MeshingStageCheckpoint {
        completed_work,
        estimated_work: completed_work,
        node_count,
        element_count,
        peak_memory_bytes: encoded_length,
        entity_counts: BTreeMap::from([
            ("solver_nodes".into(), node_count),
            ("solver_volume_elements".into(), element_count),
            ("solver_boundary_faces".into(), face_count),
            ("solver_boundary_edges".into(), edge_count),
        ]),
        ..MeshingStageCheckpoint::default()
    })
}

fn invalid_inputs(stage: MeshingStageKind) -> Box<MeshingFailure> {
    terminal_failure(
        stage,
        "terminal solver inputs do not form the canonical projection-validation closure",
    )
}

fn terminal_failure(stage: MeshingStageKind, detail: &str) -> Box<MeshingFailure> {
    let detail = crate::diagnostic::bounded_diagnostic_text(detail, "terminal solver stage failed");
    Box::new(MeshingFailure {
        schema_version: MESHING_FAILURE_SCHEMA_VERSION,
        category: MeshingFailureCategory::InternalInvariantViolation,
        stage,
        operation: stage.operation(),
        entity_ids: Vec::new(),
        witnesses: Vec::new(),
        request_values: Vec::new(),
        achieved_values: vec![MeshingDiagnosticEntry {
            name: "terminal_solver_failure".into(),
            value: MeshingDiagnosticValue::Text(detail),
            unit: None,
        }],
        accounting_error: None,
        remediation: "regenerate the solver projection and validation from current inputs".into(),
    })
}
