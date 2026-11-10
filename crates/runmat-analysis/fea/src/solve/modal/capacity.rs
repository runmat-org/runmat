//! A capacity failure cannot produce a qualified modal result.
use super::*;

pub(super) fn failure(
    error: crate::contracts::FeaRunError,
    available_modes: usize,
) -> ModalSolveResult {
    ModalSolveResult {
        converged: false,
        frequencies_hz: Vec::new(),
        mode_shapes: Vec::new(),
        residual_norms: Vec::new(),
        diagnostics: vec![FeaDiagnostic {
            code: "FEA_MODAL_SUBSPACE_CAPACITY".into(),
            severity: FeaDiagnosticSeverity::Error,
            message: error.to_string(),
        }],
        solver_method: SolverMethod::ShiftInvertBlockLanczos,
        solver_backend: LinearAlgebraBackendKind::CpuReference,
        preconditioner: SolverPreconditioner::None,
        solver_host_sync_count: 0,
        device_apply_k_count: 0,
        device_apply_k_attempt_count: 0,
        accelerator_evidence: Vec::new(),
        available_mode_count: available_modes,
        boundary_probe_count: 0,
        boundary_probe_frequency_hz: None,
        boundary_unresolved_at_capacity: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{
        CanonicalRealOperators, MatrixProperties, OperatorRole, RealSparseAssembler,
    };
    use crate::pipeline::modal_governance::ModalConvergencePolicy;

    fn repeated_operator() -> AssemblySummary {
        let model =
            crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
        let excitation = crate::excitation::realize_step_excitations(
            &model,
            &model.sequence.steps[0],
            crate::excitation::ExcitationSamplePoint::Static,
        )
        .unwrap();
        let mut summary = crate::assembly::assemble_structural_system(
            &model,
            &excitation,
            None,
            crate::assembly::StructuralFormulationRequirement::SyntheticOperator,
        )
        .unwrap();
        let diagonal = |role, values: [f64; 6]| {
            let mut matrix =
                RealSparseAssembler::new(6, 6, role, MatrixProperties::SYMMETRIC_INDEFINITE);
            for (row, value) in values.into_iter().enumerate() {
                matrix.add(row, row, value);
            }
            matrix.finish().unwrap()
        };
        let operators = CanonicalRealOperators::try_new(
            diagonal(OperatorRole::Stiffness, [1., 1., 1., 1., 1., 4.]),
            diagonal(OperatorRole::Mass, [1.; 6]),
            diagonal(OperatorRole::Damping, [0.; 6]),
        )
        .unwrap();
        summary.operator =
            crate::algebra::AssembledRealSystem::try_new(operators, vec![false; 6], vec![0.; 6])
                .unwrap();
        summary.dof_count = 6;
        summary.constraint_matrix = None;
        summary.affine_constraint_reduction = None;
        summary
    }

    #[test]
    fn modal_capacity_never_certifies_a_truncated_repeated_eigenspace() {
        let summary = repeated_operator();
        let policy = ModalConvergencePolicy::production().with_maximum_subspace_dimension(4);
        let partial =
            super::super::solve_modal_system(&summary, 2, 1e-6, policy, ComputeBackend::Cpu);
        assert!(!partial.converged);
        assert!(partial.mode_shapes.len() <= 4);
        assert!(partial.boundary_probe_frequency_hz.is_none());
        // The five-fold cluster at the boundary needs more than the four admitted vectors: the
        // solver stops at the admitted size and names it for the typed capacity error.
        assert_eq!(partial.boundary_unresolved_at_capacity, Some(4));
        let complete = super::super::solve_modal_system(
            &summary,
            2,
            1e-6,
            policy.with_maximum_subspace_dimension(6),
            ComputeBackend::Cpu,
        );
        assert!(complete.converged, "{:?}", complete.diagnostics);
        assert_eq!(complete.mode_shapes.len(), 5);
        assert!(complete.boundary_probe_frequency_hz.is_some());
        assert_eq!(complete.boundary_unresolved_at_capacity, None);
    }
}
