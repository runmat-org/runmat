use super::*;
use crate::planning::resource::LocalThreadCount;

fn coupled_reduction() -> AffineConstraintReduction {
    let reduction = AffineConstraintReduction {
        full_dof_count: 8,
        reduced_dof_count: 3,
        free_dof_indices: vec![0, 1, 2],
        pivot_dof_indices: vec![3, 4, 5, 6, 7],
        recovery_row_offsets: vec![0, 1, 2, 3, 5, 5, 7, 8, 10],
        recovery_column_indices: vec![0, 1, 2, 0, 2, 0, 1, 2, 1, 2],
        recovery_coefficients: vec![1.0, 1.0, 1.0, 0.25, -0.75, 2.0, -0.5, 1.5, -1.0, 0.125],
        offset: vec![0.0, 0.0, 0.0, 0.001, -0.002, 0.0, 0.004, -0.001],
    };
    reduction.validate().unwrap();
    reduction
}

#[test]
fn scoped_affine_lift_preserves_serial_bits() {
    let reduction = coupled_reduction();
    let reduced = [0.75, -2.0, 1.25];
    let serial_linear = expand_linear(&reduction, &reduced, None).unwrap();
    let serial_full = expand(&reduction, &reduced, None).unwrap();
    for threads in [2, 3, 7, 8] {
        let plan = LocalExecutionPlan::fixed(LocalThreadCount::new(threads).unwrap()).unwrap();
        assert_eq!(
            expand_linear(&reduction, &reduced, Some(plan)).unwrap(),
            serial_linear
        );
        assert_eq!(
            expand(&reduction, &reduced, Some(plan)).unwrap(),
            serial_full
        );
    }
}

#[test]
fn repeated_affine_solves_rebind_lifting_and_invalidate_changed_congruence() {
    use crate::algebra::{
        AssembledRealSystem, CanonicalRealOperators, MatrixProperties, OperatorRole,
        RealSparseMatrix,
    };
    use crate::constraints::ResolvedConstraintMatrix;
    use crate::solve::backend::cpu_reference::CpuReferenceBackend;
    use runmat_execution::Digest;
    let diagonal = |values, role| {
        RealSparseMatrix::diagonal(values, role, MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE)
            .unwrap()
    };
    let operator = AssembledRealSystem::try_new(
        CanonicalRealOperators::try_new(
            diagonal(vec![2.0, 3.0, 5.0], OperatorRole::Stiffness),
            diagonal(vec![1.0; 3], OperatorRole::Mass),
            diagonal(vec![1.0; 3], OperatorRole::Damping),
        )
        .unwrap(),
        vec![false; 3],
        vec![1.0, 2.0, 3.0],
    )
    .unwrap();
    let mut reuse = crate::algebra::LinearPreparationReuse::default();
    // Independent constrained energy minimum for x0=x1+offset, x1=ratio*x2.
    for (offset, ratio) in [(0.25, 1.0), (1.0, 1.0), (1.0, 2.0)] {
        let matrix = ResolvedConstraintMatrix {
            full_dof_count: 3,
            row_offsets: vec![0, 2, 4],
            column_indices: vec![0, 1, 1, 2],
            coefficients: vec![1.0, -1.0, 1.0, -ratio],
            right_hand_side: vec![offset, 0.0],
            row_identities: vec![Digest::sha256("first"), Digest::sha256("second")],
            constraint_ids: vec!["first".into(), "second".into()],
        };
        let reduction = matrix.affine_reduction().unwrap();
        let layout = crate::assembly::dofs::StructuralDofLayout::reduced_translational_rows(3);
        let view = StructuralLinearSystemView {
            dof_count: 3,
            physical_layout: crate::assembly::physical_fields::StructuralSolveLayout::Full(&layout),
            operator: &operator,
            constraint_matrix: Some(&matrix),
            affine_constraint_reduction: Some(&reduction),
            planned_reduced_support: None,
        };
        let warm = solve_affine_constraint_system_reusing(
            &view,
            &reduction,
            LinearAlgebraBackendKind::CpuReference,
            &CpuReferenceBackend,
            None,
            &mut reuse,
        )
        .unwrap();
        let cold = solve_affine_constraint_system(
            &view,
            &reduction,
            LinearAlgebraBackendKind::CpuReference,
            &CpuReferenceBackend,
            None,
        )
        .unwrap();
        assert_eq!(warm, cold);
        assert!(warm.converged);
        let t = (3.0 * ratio + 3.0 - 2.0 * ratio * offset) / (5.0 * ratio * ratio + 5.0);
        for (value, expected) in warm.solution.iter().zip([ratio * t + offset, ratio * t, t]) {
            assert!((value - expected).abs() < 1e-14);
        }
        let mut foreign = reduction.clone();
        foreign.offset[0] += 0.25;
        assert!(
            !solve_affine_constraint_system_reusing(
                &view,
                &foreign,
                LinearAlgebraBackendKind::CpuReference,
                &CpuReferenceBackend,
                None,
                &mut reuse
            )
            .unwrap()
            .converged
        );
    }
}
