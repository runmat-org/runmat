//! Authenticated original workload, topology, controller, state and accuracy authorities.
use super::*;

pub(super) fn validate(
    completed: &NonlinearSolidStudyCompletion,
    source: &NonlinearSolidStudyObjectSource,
    mesh: &SolverMeshArtifact,
    evidence: &MeshingEvidence,
    definition: Digest,
    maximum_workspace_bytes: u64,
) -> Result<(), crate::analysis::storage::AnalysisPublicationError> {
    source.authenticate(completed, maximum_workspace_bytes)?;
    completed.compiled.validate_against(mesh).map_err(string)?;
    evidence.validate(mesh).map_err(string)?;
    completed.catalog.validate().map_err(string)?;
    let expected = runmat_analysis_fea::study::plan_nonlinear_solid_study_product(
        completed.compiled.clone(),
        mesh,
        completed.workload.controls.clone(),
    )
    .map_err(string)?;
    if expected.workload != completed.workload
        || completed.compiled.definition_digest().map_err(string)? != definition
        || !completed.catalog.roots.contains(&completed.root)
        || !completed.terminal_block.complete
        || completed.terminal_block.request.topology
            != Digest::from_bytes(*mesh.canonical_digest.bytes())
        || completed.block_count
            != completed
                .terminal_block
                .request
                .block_ordinal
                .checked_add(1)
                .ok_or("nonlinear block count overflow")?
    {
        return Err(
            "nonlinear publication differs from original compiled input/root/terminal block".into(),
        );
    }
    completed
        .numerical_receipt
        .validate_against(&completed.workload, &completed.terminal_block)
        .map_err(string)?;
    if completed.states.len()
        != completed.workload.model.sequence.steps[0]
            .state_outputs
            .len()
    {
        return Err("nonlinear publication omits original state declarations".into());
    }
    for state in &completed.states {
        state.validate().map_err(string)?;
        let runmat_analysis_fea::execution::CanonicalSemanticStatePayload::NonlinearSolid {
            state: original,
        } = &state.payload
        else {
            return Err("nonlinear original state replaced by a field projection".into());
        };
        if *original.workload != completed.workload || *original.block != completed.terminal_block {
            return Err("nonlinear original state differs from terminal history".into());
        }
    }
    Ok(())
}

pub(super) fn options(
    controls: &runmat_analysis_fea::solve::nonlinear::NonlinearSolveOptions,
) -> AnalysisNonlinearRunOptions {
    AnalysisNonlinearRunOptions {
        deterministic_mode: false,
        precision_mode: super::super::PrecisionMode::Fp64,
        quality_policy: super::super::QualityPolicy::Strict,
        increment_count: controls.increment_count,
        max_newton_iters: controls.max_newton_iters,
        tolerance: controls.tolerance,
        linear_tolerance: controls.linear_tolerance,
        max_load_cutbacks: controls.max_load_cutbacks,
        minimum_load_increment: controls.minimum_load_increment,
        max_accepted_increments: controls.max_accepted_increments,
        line_search: controls.line_search,
        max_line_search_backtracks: controls.max_line_search_backtracks,
        line_search_reduction: controls.line_search_reduction,
        contact_enforcement: None,
        solver_mesh_artifact_path: None,
    }
}
fn string(error: impl std::fmt::Display) -> String {
    error.to_string()
}
