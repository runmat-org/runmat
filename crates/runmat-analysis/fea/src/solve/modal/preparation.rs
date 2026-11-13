//! Entry point without an original local source; shared preparation remains the authority.
use super::*;

pub(crate) struct ModalSolveInput<'a> {
    pub summary: &'a AssemblySummary,
    pub requested_mode_count: usize,
    pub eigenspace_relative_tolerance: f64,
    pub convergence: crate::pipeline::modal_governance::ModalConvergencePolicy,
    pub backend: ComputeBackend,
    pub path: SubspaceSolvePath,
    pub node_coordinates_m: Option<&'a [[f64; 3]]>,
}

pub fn solve_modal_system(
    summary: &AssemblySummary,
    requested_mode_count: usize,
    eigenspace_relative_tolerance: f64,
    convergence: crate::pipeline::modal_governance::ModalConvergencePolicy,
    backend: ComputeBackend,
) -> ModalSolveResult {
    solve_modal_system_reusing(
        ModalSolveInput {
            summary,
            requested_mode_count,
            eigenspace_relative_tolerance,
            convergence,
            backend,
            path: SubspaceSolvePath::ShiftInvertFactorization,
            node_coordinates_m: None,
        },
        &mut crate::algebra::LinearPreparationReuse::default(),
    )
}
