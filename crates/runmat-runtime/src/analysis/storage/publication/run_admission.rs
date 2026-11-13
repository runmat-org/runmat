//! Bounded failure facts for the original run; no payload logging or admission bypass.
//! The run itself is admitted phase by phase by its preparation owner.
use super::*;

/// Bounded facts attached to a rejected run admission.
pub(super) fn run_failure_facts(run: &AnalysisRunResult, maximum: u64) -> String {
    let projected = |result: Result<u64, AnalysisPublicationError>| match result {
        Ok(bytes) => bytes.to_string(),
        Err(_) => "exceeds_profile_limits".into(),
    };
    let modes = run
        .modal_results
        .as_ref()
        .map_or(0, |modal| modal.basis.modes().len());
    let nodes = run
        .modal_results
        .as_ref()
        .map_or(0, |modal| modal.basis.node_count());
    format!(
        "run staging facts: numerically_acceptable={}, solver_convergence={:?}, result_quality={:?}, modal_nodes={nodes}, retained_modes={modes}, run_projection_bytes={}, modal_projection_bytes={}, render_projection_bytes={}",
        run.numerically_acceptable,
        run.solver_convergence,
        run.result_quality,
        projected(resident(&run.run, maximum)),
        projected(resident(&run.modal_results, maximum)),
        projected(resident(&run.render_topology, maximum)),
    )
}
