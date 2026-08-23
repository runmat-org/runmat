use runmat_meshing_core::{
    CacheAdmissionDecision, CanonicalMeshingContract, MeshingEvidence, MeshingResourceUsage,
    MeshingStageEvidence, PlatformBuildIdentity, SizingResolutionEvidence, SolverMeshArtifact,
    MESHING_EVIDENCE_SCHEMA_VERSION,
};

use crate::{MeshingExecutionError, MeshingExecutionResult};

/// Driver-known facts that are not properties of any one meshing stage.
#[derive(Clone, Debug, PartialEq)]
pub struct MeshingEvidenceContext {
    pub platform: PlatformBuildIdentity,
    pub sizing: Vec<SizingResolutionEvidence>,
    pub cache_admission: CacheAdmissionDecision,
    /// Driver-observed elapsed time for the complete operation, including concurrent stages.
    pub wall_time_ms: u64,
}

/// Canonicalizes factual winning-stage observations and closes them over the final artifact.
///
/// Stage execution remains the authority for measurements. This function only performs checked
/// aggregation; it does not estimate or synthesize counters that a stage did not report.
pub fn assemble_meshing_evidence(
    artifact: &SolverMeshArtifact,
    mut stages: Vec<MeshingStageEvidence>,
    context: MeshingEvidenceContext,
) -> MeshingExecutionResult<MeshingEvidence> {
    artifact.validate()?;
    stages.sort_by(|left, right| {
        (&left.stage, &left.partition, &left.stage_result_digest).cmp(&(
            &right.stage,
            &right.partition,
            &right.stage_result_digest,
        ))
    });
    let artifact_bytes = u64::try_from(artifact.canonical_encode()?.len())
        .map_err(|_| invalid("canonical solver artifact length exceeds u64"))?;
    let resources = aggregate_resources(artifact, &stages, artifact_bytes, context.wall_time_ms)?;
    let evidence = MeshingEvidence {
        schema_version: MESHING_EVIDENCE_SCHEMA_VERSION,
        geometry: artifact.geometry.clone(),
        resolved_request_digest: artifact.resolved_request.canonical_digest()?,
        artifact_digest: artifact.canonical_digest,
        algorithms: artifact.resolved_request.algorithms.clone(),
        deterministic_seed: artifact.resolved_request.deterministic_seed,
        platform: context.platform,
        stages,
        sizing: context.sizing,
        resources,
        cache_admission: context.cache_admission,
    };
    evidence.validate(artifact)?;
    Ok(evidence)
}

fn aggregate_resources(
    artifact: &SolverMeshArtifact,
    stages: &[MeshingStageEvidence],
    artifact_bytes: u64,
    wall_time_ms: u64,
) -> MeshingExecutionResult<MeshingResourceUsage> {
    let mut search_work = 0_u64;
    let mut iterations = 0_u64;
    let mut peak_memory_bytes = 0_u64;
    let mut peak_scratch_bytes = 0_u64;
    let mut maximum_recursion_depth = 0_u32;
    for stage in stages {
        search_work = checked_sum(search_work, stage.search_work, "search work")?;
        iterations = checked_sum(iterations, stage.iterations, "iterations")?;
        peak_memory_bytes = peak_memory_bytes.max(stage.peak_memory_bytes);
        peak_scratch_bytes = peak_scratch_bytes.max(stage.peak_scratch_bytes);
        maximum_recursion_depth = maximum_recursion_depth.max(stage.maximum_recursion_depth);
    }
    Ok(MeshingResourceUsage {
        generated_nodes: u64::try_from(artifact.topology.nodes.len())
            .map_err(|_| invalid("solver node inventory exceeds u64"))?,
        generated_elements: u64::try_from(artifact.topology.volume_elements.len())
            .map_err(|_| invalid("solver element inventory exceeds u64"))?,
        peak_memory_bytes,
        peak_scratch_bytes,
        wall_time_ms,
        artifact_bytes,
        search_work,
        maximum_recursion_depth,
        iterations,
    })
}

fn checked_sum(left: u64, right: u64, counter: &str) -> MeshingExecutionResult<u64> {
    left.checked_add(right)
        .ok_or_else(|| invalid(format!("meshing evidence {counter} overflowed")))
}

fn invalid(reason: impl Into<String>) -> MeshingExecutionError {
    MeshingExecutionError::Invalid(reason.into())
}
