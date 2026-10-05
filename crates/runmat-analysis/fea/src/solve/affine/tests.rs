use super::*;
use crate::{
    planning::resource::{execute_scoped_chunks, LocalThreadCount, ScopedLocalExecutionError},
    progress::{replace_fea_progress_context, FeaCancellationToken, FeaProgressContext},
};

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
fn scoped_affine_forward_projection_and_lift_preserve_serial_bits() {
    let reduction = coupled_reduction();
    let reduced = [0.75, -2.0, 1.25];
    let serial_linear = expand_linear(&reduction, &reduced, None).unwrap();
    let serial_full = expand(&reduction, &reduced, None).unwrap();
    let transpose_index = AffineTransposeIndex::new(&reduction);
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
        assert_eq!(
            apply_reduced_operator(
                &reduction,
                &reduced,
                &|full| Ok(full.to_vec()),
                Some(plan),
                Some(&transpose_index),
            )
            .unwrap(),
            transpose_apply(&reduction, &serial_linear),
        );
        assert_eq!(
            transpose_index
                .apply(&reduction, &serial_linear, plan)
                .unwrap(),
            transpose_apply(&reduction, &serial_linear),
        );
    }
}

#[test]
fn scoped_affine_projection_rejects_nested_execution_and_cancellation() {
    let reduction = coupled_reduction();
    let reduced = [0.75, -2.0, 1.25];
    let plan = LocalExecutionPlan::fixed(LocalThreadCount::new(3).unwrap()).unwrap();
    let transpose_index = AffineTransposeIndex::new(&reduction);
    let full = expand_linear(&reduction, &reduced, None).unwrap();
    let nested = execute_scoped_chunks(LocalExecutionPlan::serial(), 1, |_| {
        Ok::<_, ()>(transpose_index.apply(&reduction, &full, plan))
    })
    .unwrap();
    assert!(matches!(
        &nested[0],
        Err(ScopedSpmvError::Execution(
            ScopedLocalExecutionError::NestedExecution
        ))
    ));

    let cancellation = FeaCancellationToken::new();
    cancellation.cancel();
    let _context = replace_fea_progress_context(FeaProgressContext {
        cancellation,
        ..FeaProgressContext::default()
    });
    assert!(matches!(
        transpose_index.apply(&reduction, &full, plan),
        Err(ScopedSpmvError::Execution(
            ScopedLocalExecutionError::Work {
                source: crate::algebra::SpmvWorkError::Cancelled,
                ..
            }
        ))
    ));
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
        let view = StructuralLinearSystemView {
            dof_count: 3,
            operator: &operator,
            constraint_matrix: Some(&matrix),
            affine_constraint_reduction: Some(&reduction),
        };
        let warm = solve_affine_constraint_system_reusing(
            &view,
            &reduction,
            LinearAlgebraBackendKind::CpuReference,
            &CpuReferenceBackend,
            None,
            &mut reuse,
        );
        let cold = solve_affine_constraint_system(
            &view,
            &reduction,
            LinearAlgebraBackendKind::CpuReference,
            &CpuReferenceBackend,
            None,
        );
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
            .converged
        );
    }
}
