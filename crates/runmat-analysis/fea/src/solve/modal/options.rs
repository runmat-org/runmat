use super::{AssemblySummary, MODAL_LANCZOS_BLOCK_SIZE};

/// Eigensolver options for one modal search of `subspace_dimension` vectors. The resource
/// planner sizes the search's working set from these same options.
pub(crate) fn modal_eigen_options(
    subspace_dimension: usize,
    convergence: crate::pipeline::modal_governance::ModalConvergencePolicy,
) -> runmat_sparse_solvers::eigen::EigenOptions {
    runmat_sparse_solvers::eigen::EigenOptions {
        count: subspace_dimension,
        shift: Some(0.0),
        block_size: MODAL_LANCZOS_BLOCK_SIZE,
        tolerance: modal_lanczos_tolerance(convergence),
        max_restarts: convergence.maximum_subspace_iterations().max(1),
        sturm_check: true,
        ..Default::default()
    }
}

/// Ritz-estimate tolerance of the shift-invert Lanczos: a tenth of the policy's tolerance,
/// capped at 1e-10. The Lanczos stops on its Ritz estimates, and the published gate is measured
/// on the returned pairs afterwards; a Ritz estimate converged to 1e-10 leaves backward errors
/// near 1e-12 on small models, so the stopping rule follows the gate's tolerance.
fn modal_lanczos_tolerance(
    convergence: crate::pipeline::modal_governance::ModalConvergencePolicy,
) -> f64 {
    (0.1 * convergence.algebraic_tolerance()).min(1.0e-10)
}
/// LOBPCG options for one modal search of `subspace_dimension` vectors on the factor-free
/// path. The resource planner sizes the path's working set from these same options.
///
/// The iteration stops at half the policy's tolerance, in the policy's measure, so the true
/// residuals it reports pass the policy; it may take up to four LOBPCG steps per admitted Lanczos restart
/// because each step applies the preconditioner once per unconverged vector instead of a
/// factor solve.
pub(crate) fn modal_lobpcg_options(
    subspace_dimension: usize,
    convergence: crate::pipeline::modal_governance::ModalConvergencePolicy,
) -> runmat_sparse_solvers::eigen::IterativeModalOptions {
    runmat_sparse_solvers::eigen::IterativeModalOptions {
        lobpcg: runmat_sparse_solvers::eigen::LobpcgOptions {
            count: subspace_dimension,
            guard: MODAL_LANCZOS_BLOCK_SIZE,
            tolerance: 0.5 * convergence.algebraic_tolerance(),
            measure: convergence.solver_measure(),
            max_iterations: convergence
                .maximum_subspace_iterations()
                .max(1)
                .saturating_mul(4),
        },
        ordering: runmat_sparse_solvers::ordering::Ordering::Auto,
    }
}

/// Whether the full space holds only translational solid DOFs, three per node in node-major
/// order, and the constraint reduction keeps each node's three DOFs together.
pub(super) fn keeps_whole_translational_nodes(
    summary: &AssemblySummary,
    free_dofs: &[usize],
) -> bool {
    summary.dof_count > 0
        && summary.structural_rotational_dof_count == 0
        && summary.structural_translational_dof_count == summary.dof_count
        && summary.dof_count == 3 * summary.structural_node_count
        && !free_dofs.is_empty()
        && free_dofs.len().is_multiple_of(3)
        && free_dofs.chunks_exact(3).all(|triple| {
            triple[0] % 3 == 0 && triple[1] == triple[0] + 1 && triple[2] == triple[0] + 2
        })
}
