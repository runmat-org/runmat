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
        tolerance: (0.1 * convergence.relative_residual_tolerance()).min(1.0e-10),
        max_restarts: convergence.maximum_subspace_iterations().max(1),
        sturm_check: true,
        ..Default::default()
    }
}
/// LOBPCG options for one modal search of `subspace_dimension` vectors on the factor-free
/// path. The resource planner sizes the path's working set from these same options.
///
/// The iteration stops at half the policy's relative residual tolerance so the true residuals
/// it reports pass the policy; it may take up to four LOBPCG steps per admitted Lanczos restart
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
            tolerance: 0.5 * convergence.relative_residual_tolerance(),
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
