use std::collections::BTreeMap;

use runmat_meshing_core::{
    CanonicalMeshingContract, MeshingChunkMediaType, MeshingChunkStream, MeshingDiagnosticEntry,
    MeshingDiagnosticValue, MeshingFailure, MeshingFailureCategory, MeshingPartitionKind,
    MeshingStageKind, MeshingStageResultKind, SolverMeshArtifact, SolverMeshProjection,
    SolverMeshValidation, StableDigest, ANALYSIS_MESH_ARTIFACT_SCHEMA_VERSION,
    MESHING_EVIDENCE_SCHEMA_VERSION, MESHING_FAILURE_SCHEMA_VERSION,
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
        let projection = decode_projection(projection_publication, invocation.host)?;
        let validation = SolverMeshValidation::from_validated_projection(&projection)
            .map_err(|error| terminal_failure(MeshingStageKind::Validation, &error.to_string()))?;
        validation
            .validate_against(&projection)
            .map_err(|error| terminal_failure(MeshingStageKind::Validation, &error.to_string()))?;
        let encoded = validation
            .canonical_encode()
            .map_err(|error| terminal_failure(MeshingStageKind::Validation, &error.to_string()))?;
        let checkpoint = topology_checkpoint(&projection, encoded.len() as u64)?;
        invocation.control.checkpoint(checkpoint.clone())?;
        Ok(ValidatedMeshingStageOutput {
            invariant_summary_digest: validation.canonical_digest().map_err(|error| {
                terminal_failure(MeshingStageKind::Validation, &error.to_string())
            })?,
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
        let projection = decode_projection(projection_publication, invocation.host)?;
        let validation = decode_validation(validation_publication)?;
        validation.validate_against(&projection).map_err(|error| {
            terminal_failure(MeshingStageKind::Serialization, &error.to_string())
        })?;
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
        let artifact = projection
            .into_artifact(validation_manifest_digest)
            .map_err(|error| {
                terminal_failure(MeshingStageKind::Serialization, &error.to_string())
            })?;
        artifact.validate().map_err(|error| {
            terminal_failure(MeshingStageKind::Serialization, &error.to_string())
        })?;
        let encoded = artifact.canonical_encode().map_err(|error| {
            terminal_failure(MeshingStageKind::Serialization, &error.to_string())
        })?;
        let checkpoint = artifact_checkpoint(&artifact, encoded.len() as u64)?;
        invocation.control.checkpoint(checkpoint.clone())?;
        Ok(ValidatedMeshingStageOutput {
            invariant_summary_digest: artifact.canonical_digest,
            streams: vec![MeshingChunkStream {
                media_type: MeshingChunkMediaType::AnalysisMeshArtifact,
                schema_version: ANALYSIS_MESH_ARTIFACT_SCHEMA_VERSION,
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
        let artifact = decode_artifact(serialization)?;
        let evidence = &evidence_input.evidence_objects().evidence;
        evidence
            .validate(&artifact)
            .map_err(|error| terminal_failure(MeshingStageKind::Publication, &error.to_string()))?;
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
        let artifact_bytes = artifact
            .canonical_encode()
            .map_err(|error| terminal_failure(MeshingStageKind::Publication, &error.to_string()))?;
        let evidence_bytes = evidence
            .canonical_encode()
            .map_err(|error| terminal_failure(MeshingStageKind::Publication, &error.to_string()))?;
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
        let checkpoint = artifact_checkpoint_for_stage(
            &artifact,
            encoded_length,
            MeshingStageKind::Publication,
        )?;
        invocation.control.checkpoint(checkpoint.clone())?;
        Ok(ValidatedMeshingStageOutput {
            invariant_summary_digest: evidence.canonical_digest().map_err(|error| {
                terminal_failure(MeshingStageKind::Publication, &error.to_string())
            })?,
            streams: vec![
                MeshingChunkStream {
                    media_type: MeshingChunkMediaType::AnalysisMeshArtifact,
                    schema_version: ANALYSIS_MESH_ARTIFACT_SCHEMA_VERSION,
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

fn decode_projection(
    publication: &PreparedMeshingResultPublication,
    host: &MeshingHostWorkload,
) -> Result<SolverMeshProjection, Box<MeshingFailure>> {
    let bytes = one_record(
        publication,
        MeshingStageKind::OrderElevation,
        MeshingChunkMediaType::SolverMeshProjection,
        SOLVER_MESH_PROJECTION_SCHEMA_VERSION,
        host.workload.stage,
    )?;
    let projection = SolverMeshProjection::canonical_decode(&bytes)
        .map_err(|error| terminal_failure(host.workload.stage, &error.to_string()))?;
    if projection.geometry != host.stage_identity.geometry
        || projection.resolved_request != host.resolved_request
    {
        return Err(invalid_inputs(host.workload.stage));
    }
    Ok(projection)
}

fn decode_validation(
    publication: &PreparedMeshingResultPublication,
) -> Result<SolverMeshValidation, Box<MeshingFailure>> {
    let bytes = one_record(
        publication,
        MeshingStageKind::Validation,
        MeshingChunkMediaType::ValidationEvidence,
        SOLVER_MESH_VALIDATION_SCHEMA_VERSION,
        MeshingStageKind::Serialization,
    )?;
    SolverMeshValidation::canonical_decode(&bytes)
        .map_err(|error| terminal_failure(MeshingStageKind::Serialization, &error.to_string()))
}

fn decode_artifact(
    publication: &PreparedMeshingResultPublication,
) -> Result<SolverMeshArtifact, Box<MeshingFailure>> {
    let bytes = one_record(
        publication,
        MeshingStageKind::Serialization,
        MeshingChunkMediaType::AnalysisMeshArtifact,
        ANALYSIS_MESH_ARTIFACT_SCHEMA_VERSION,
        MeshingStageKind::Publication,
    )?;
    SolverMeshArtifact::canonical_decode(&bytes)
        .map_err(|error| terminal_failure(MeshingStageKind::Publication, &error.to_string()))
}

fn one_record(
    publication: &PreparedMeshingResultPublication,
    producer_stage: MeshingStageKind,
    media_type: MeshingChunkMediaType,
    schema_version: u16,
    consumer_stage: MeshingStageKind,
) -> Result<Vec<u8>, Box<MeshingFailure>> {
    let stage = publication.stage_objects();
    if stage.result_identity.stage != producer_stage
        || stage.result_identity.result_kind != MeshingStageResultKind::WholeStage
    {
        return Err(invalid_inputs(consumer_stage));
    }
    let streams = stage
        .decoded_streams()
        .map_err(|error| terminal_failure(consumer_stage, &error.to_string()))?;
    let [stream] = streams.as_slice() else {
        return Err(invalid_inputs(consumer_stage));
    };
    if stream.media_type != media_type || stream.schema_version != schema_version {
        return Err(invalid_inputs(consumer_stage));
    }
    let [record] = stream.records.as_slice() else {
        return Err(invalid_inputs(consumer_stage));
    };
    Ok(record.clone())
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
        remediation: "regenerate the solver projection and validation from current inputs".into(),
    })
}
