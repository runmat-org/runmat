//! Entry point without an original local source; shared preparation remains the authority.
use super::*;

pub fn solve_modal_system(
    summary: &AssemblySummary,
    requested_mode_count: usize,
    eigenspace_relative_tolerance: f64,
    convergence: crate::pipeline::modal_governance::ModalConvergencePolicy,
    backend: ComputeBackend,
) -> ModalSolveResult {
    solve_modal_system_reusing(
        summary,
        requested_mode_count,
        eigenspace_relative_tolerance,
        convergence,
        backend,
        &mut crate::algebra::LinearPreparationReuse::default(),
        SubspaceSolvePath::ShiftInvertFactorization,
        None,
    )
}
