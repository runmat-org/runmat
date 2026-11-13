use runmat_time::Instant;

use runmat_meshing_core::{CanonicalMeshingContract, MeshingChunkMediaType, MeshingEvidence};

use crate::{
    assemble_meshing_evidence, import_result_publication, prepare_evidence_input,
    prepare_evidence_objects, ExactMeshingDagPlanner, MeshingEvidenceContext,
};

use super::stage::{execute_planned_stage, ExactMeshingDagExecutor, ExecutedStage};
use super::{ExactMeshingDagRun, ExactMeshingDagRunError, ExactMeshingDagRunResult};

pub(super) async fn execute_terminal_dag<E: ExactMeshingDagExecutor>(
    executor: &mut E,
    planner: &ExactMeshingDagPlanner,
    run: &ExactMeshingDagRun<'_>,
    surface_root: runmat_execution::value::ValueRef,
    mut observations: Vec<runmat_meshing_core::MeshingStageEvidence>,
    started: Instant,
) -> Result<ExactMeshingDagRunResult, ExactMeshingDagRunError> {
    let projection_stage = if run.request.dimension.is_solid() {
        let volume = execute(
            executor,
            &planner.tetrahedralization(surface_root.clone())?,
            run,
            &mut observations,
        )
        .await?;
        planner.solver_projection(surface_root, volume.root)?
    } else {
        // Shell and beam meshes stop at the exact surface; no volume is generated.
        planner.surface_solver_projection(surface_root)?
    };
    let projection = execute(executor, &projection_stage, run, &mut observations).await?;
    let projection_root = projection.root;
    let validation = execute(
        executor,
        &planner.solver_validation(projection_root.clone())?,
        run,
        &mut observations,
    )
    .await?;
    let serialized = execute(
        executor,
        &planner.solver_serialization(projection_root, validation.root)?,
        run,
        &mut observations,
    )
    .await?;
    let (artifact, serialized_artifact_bytes) =
        decode_serialized_artifact(executor, run, &serialized.root)?;
    let wall_time_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    let evidence = assemble_meshing_evidence(
        &artifact,
        observations,
        MeshingEvidenceContext {
            platform: run.evidence.platform.clone(),
            sizing: {
                let (_, applied) = run
                    .request
                    .resolve_size_controls(
                        &run.named_regions,
                        &run.geometry.geometry_objects().topology,
                    )
                    .map_err(|error| ExactMeshingDagRunError::Invalid(error.to_string()))?;
                let mut sizing = run.evidence.sizing.clone();
                sizing.extend(runmat_meshing_core::size_control_evidence(
                    &artifact, &applied,
                ));
                sizing
            },
            cache_admission: run.evidence.cache_admission,
            wall_time_ms,
            ceilings: run.ceilings,
        },
    )?;
    let evidence_input = prepare_evidence_input(
        prepare_evidence_objects(evidence, run.inventory_limits)?,
        run.artifact_access.clone(),
        run.inventory_limits,
    )?;
    for object in &evidence_input.evidence_objects().objects {
        executor
            .write_verified(object)
            .map_err(crate::MeshingExecutionError::from)?;
    }
    let publication = execute_planned_stage(
        executor,
        &planner.solver_publication(serialized.root, evidence_input.root_input().clone())?,
        run.program_revision.clone(),
        run.inventory_limits,
    )
    .await?;
    decode_terminal_publication(
        executor,
        run,
        publication,
        artifact,
        &serialized_artifact_bytes,
    )
}

async fn execute<E: ExactMeshingDagExecutor>(
    executor: &mut E,
    stage: &crate::PlannedMeshingStage,
    run: &ExactMeshingDagRun<'_>,
    observations: &mut Vec<runmat_meshing_core::MeshingStageEvidence>,
) -> Result<ExecutedStage, ExactMeshingDagRunError> {
    let completed = execute_planned_stage(
        executor,
        stage,
        run.program_revision.clone(),
        run.inventory_limits,
    )
    .await?;
    observations.push(completed.evidence.clone());
    Ok(completed)
}

fn decode_serialized_artifact(
    source: &impl runmat_execution_artifact::cache::CacheImport,
    run: &ExactMeshingDagRun<'_>,
    root: &runmat_execution::value::ValueRef,
) -> Result<(runmat_meshing_core::SolverMeshArtifact, Vec<u8>), ExactMeshingDagRunError> {
    let publication = import_result_publication(
        source,
        root,
        run.artifact_access.clone(),
        run.inventory_limits,
    )?;
    let streams = publication.stage_objects().decoded_streams()?;
    let [stream] = streams.as_slice() else {
        return Err(invalid_terminal());
    };
    let [_record] = stream.records.as_slice() else {
        return Err(invalid_terminal());
    };
    if stream.media_type != MeshingChunkMediaType::SolverMeshArtifact {
        return Err(invalid_terminal());
    }
    let bytes =
        crate::controlled_streams::take_single_record(streams, 0).ok_or_else(invalid_terminal)?;
    drop(publication);
    let artifact = runmat_meshing_core::SolverMeshArtifact::canonical_decode(&bytes)?;
    Ok((artifact, bytes))
}

fn decode_terminal_publication(
    source: &impl runmat_execution_artifact::cache::CacheImport,
    run: &ExactMeshingDagRun<'_>,
    completed: ExecutedStage,
    artifact: runmat_meshing_core::SolverMeshArtifact,
    serialized_artifact_bytes: &[u8],
) -> Result<ExactMeshingDagRunResult, ExactMeshingDagRunError> {
    let publication = import_result_publication(
        source,
        &completed.root,
        run.artifact_access.clone(),
        run.inventory_limits,
    )?;
    let streams = publication.stage_objects().decoded_streams()?;
    let [artifact_stream, evidence_stream] = streams.as_slice() else {
        return Err(invalid_terminal());
    };
    let ([artifact_bytes], [evidence_bytes]) = (
        artifact_stream.records.as_slice(),
        evidence_stream.records.as_slice(),
    ) else {
        return Err(invalid_terminal());
    };
    if artifact_stream.media_type != MeshingChunkMediaType::SolverMeshArtifact
        || evidence_stream.media_type != MeshingChunkMediaType::MeshingEvidence
        || artifact_bytes.as_slice() != serialized_artifact_bytes
    {
        return Err(invalid_terminal());
    }
    let evidence = MeshingEvidence::canonical_decode(evidence_bytes)?;
    evidence.validate_publication_closure(
        &artifact.geometry,
        &artifact.resolved_request,
        artifact.canonical_digest,
        artifact_bytes.len() as u64,
    )?;
    Ok(ExactMeshingDagRunResult {
        root: completed.root,
        result_objects: completed.result_objects,
        artifact,
        evidence,
    })
}

fn invalid_terminal() -> ExactMeshingDagRunError {
    ExactMeshingDagRunError::Invalid("terminal publication has an invalid stream shape".into())
}
