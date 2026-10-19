//! Bounded failure facts for the original run; no payload logging or admission bypass.
use super::*;

impl AnalysisPublicationBuilder {
    pub(super) fn admit_run(
        &self,
        run: &AnalysisRunResult,
    ) -> Result<(), AnalysisPublicationError> {
        let additional = resident(run, self.maximum_bytes)?
            .checked_mul(2)
            .ok_or_else(|| AnalysisPublicationError::ResourceLimit {
                message: "staging byte overflow".into(),
            })?;
        self.admit(additional).map_err(|error| {
            // Streaming profiles allocate neither a JSON tree nor an encoded copy.
            // They describe codec projections, not measured typed heap or process RSS.
            error.context(&run_failure_facts(run, self.maximum_bytes))
        })
    }
}

fn run_failure_facts(run: &AnalysisRunResult, maximum: u64) -> String {
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
