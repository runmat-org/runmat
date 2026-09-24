use super::*;
use crate::{
    algebra::{
        CanonicalRealOperators, MatrixProperties, OperatorRole, RealSparseAssembler,
        RealSparseMatrix,
    },
    planning::resource::{execute_scoped_chunks, LocalThreadCount, ScopedLocalExecutionError},
    solve::backend::cpu_reference::CpuReferenceBackend,
};

fn system(stiffness_diagonal: [f64; 2]) -> AssembledRealSystem {
    let mut stiffness = RealSparseAssembler::square(
        2,
        OperatorRole::Stiffness,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    );
    stiffness.add(0, 0, stiffness_diagonal[0]);
    stiffness.add(0, 1, -1.0);
    stiffness.add(1, 0, -1.0);
    stiffness.add(1, 1, stiffness_diagonal[1]);
    let mass = RealSparseMatrix::diagonal(
        vec![1.0; 2],
        OperatorRole::Mass,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )
    .unwrap();
    let damping = RealSparseMatrix::diagonal(
        vec![0.0; 2],
        OperatorRole::Damping,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    )
    .unwrap();
    AssembledRealSystem::try_new(
        CanonicalRealOperators::try_new(stiffness.finish().unwrap(), mass, damping).unwrap(),
        vec![false; 2],
        vec![1.0, 2.0],
    )
    .unwrap()
}

#[test]
fn cpu_reference_solve_converges_with_small_canonical_residual() {
    let operator = system([4.0, 3.0]);
    let result = solve_operator_system(
        &operator,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        None,
    );
    assert!(result.converged);
    assert!(result.residual_norm <= 1.0e-8);
}

#[test]
fn scoped_pcg_matches_serial_and_rejects_nested_execution() {
    let operator = system([4.0, 3.0]);
    let serial = solve_operator_system(
        &operator,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        None,
    );
    let plan = LocalExecutionPlan::fixed(LocalThreadCount::new(2).unwrap()).unwrap();
    let parallel = solve_operator_system_scoped(
        &operator,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        None,
        plan,
    )
    .unwrap();
    let repeated = solve_operator_system_scoped(
        &operator,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        None,
        plan,
    )
    .unwrap();
    assert_eq!(parallel, repeated);
    assert_eq!(parallel, serial);
    assert!((parallel.solution[0] - 5.0 / 11.0).abs() < 1.0e-12);
    assert!((parallel.solution[1] - 9.0 / 11.0).abs() < 1.0e-12);

    let nested = execute_scoped_chunks(LocalExecutionPlan::serial(), 1, |_| {
        Ok::<_, ()>(solve_operator_system_scoped(
            &operator,
            SpdPreconditionerKind::Jacobi,
            LinearAlgebraBackendKind::CpuReference,
            &CpuReferenceBackend,
            None,
            plan,
        ))
    })
    .unwrap();
    assert!(matches!(
        &nested[0],
        Err(ScopedSpmvError::Execution(
            ScopedLocalExecutionError::NestedExecution
        ))
    ));
}

#[test]
fn scoped_pcg_reduction_failure_is_not_reported_as_numerical_breakdown() {
    let operator = system([4.0, 3.0]);
    let result = solve_operator_system_with_apply(
        &operator,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        None,
        (
            |prepared: &PreparedRealLinearSystem, x: &[f64]| {
                Ok(crate::algebra::apply_prepared(prepared, x))
            },
            |_: &[f64], _: &[f64]| Err(ScopedSpmvError::ReductionDimensionMismatch),
            None,
        ),
    );
    assert!(matches!(
        result,
        Err(ScopedSpmvError::ReductionDimensionMismatch)
    ));
}

#[test]
fn cpu_reference_warm_start_does_not_use_an_absolute_breakdown_threshold() {
    let stiffness = RealSparseMatrix::diagonal(
        vec![1.0e12, 2.0e12],
        OperatorRole::Stiffness,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )
    .unwrap();
    let mass = RealSparseMatrix::diagonal(
        vec![1.0; 2],
        OperatorRole::Mass,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )
    .unwrap();
    let damping = RealSparseMatrix::diagonal(
        vec![0.0; 2],
        OperatorRole::Damping,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    )
    .unwrap();
    let operator = AssembledRealSystem::try_new(
        CanonicalRealOperators::try_new(stiffness, mass, damping).unwrap(),
        vec![false; 2],
        vec![1.0, 2.0],
    )
    .unwrap();
    let initial_guess = [1.0e-12 - 1.0e-19, 1.0e-12 + 1.0e-19];

    let result = solve_operator_system(
        &operator,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        Some(&initial_guess),
    );

    assert!(result.converged, "diagnostics={:?}", result.diagnostics);
    assert!(result.residual_norm <= 1.0e-8);
}

#[test]
fn noncoercive_operator_is_rejected_before_preconditioner_setup() {
    let operator = system([0.0, 3.0]);
    let result = solve_operator_system(
        &operator,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        None,
    );
    assert!(!result.converged);
    assert_eq!(result.iterations, 0);
    assert_eq!(result.diagnostics[0].code, "FEA_INVALID_REAL_SYSTEM");
}

#[test]
fn cpu_and_accelerator_reject_unproved_semidefinite_system_before_iteration() {
    let stiffness = RealSparseMatrix::diagonal(
        vec![1.0, 0.0],
        OperatorRole::Stiffness,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    )
    .unwrap();
    let mass = RealSparseMatrix::diagonal(
        vec![1.0; 2],
        OperatorRole::Mass,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )
    .unwrap();
    let damping = RealSparseMatrix::diagonal(
        vec![0.0; 2],
        OperatorRole::Damping,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    )
    .unwrap();
    let operator = AssembledRealSystem::try_new(
        CanonicalRealOperators::try_new(stiffness, mass, damping).unwrap(),
        vec![false; 2],
        vec![1.0, 0.0],
    )
    .unwrap();

    for backend in [
        LinearAlgebraBackendKind::CpuReference,
        LinearAlgebraBackendKind::Accelerator,
    ] {
        let result = solve_operator_system(
            &operator,
            SpdPreconditionerKind::Jacobi,
            backend,
            &CpuReferenceBackend,
            None,
        );
        assert!(!result.converged);
        assert_eq!(result.iterations, 0);
        assert_eq!(result.device_apply_k_attempt_count, 0);
        assert_eq!(result.diagnostics[0].code, "FEA_INVALID_REAL_SYSTEM");
    }
}
