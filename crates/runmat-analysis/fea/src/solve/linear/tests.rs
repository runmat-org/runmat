use super::*;
use crate::{
    algebra::{
        CanonicalRealOperators, MatrixProperties, OperatorRole, RealSparseAssembler,
        RealSparseMatrix,
    },
    planning::resource::{ByteCount, Count, LocalThreadCount},
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
    )
    .unwrap();
    assert!(result.converged);
    assert!(result.residual_norm <= 1.0e-8);
}

#[test]
fn scoped_authored_tolerance_refines_a_warm_start_accepted_by_default() {
    let operator = system([4.0, 3.0]);
    let guess = [5.0 / 11.0 + 1e-10, 9.0 / 11.0 - 1e-10];
    for plan in [
        LocalExecutionPlan::serial(),
        LocalExecutionPlan::fixed(LocalThreadCount::new(2).unwrap()).unwrap(),
    ] {
        let solve = |relative_tolerance| {
            solve_operator_system_scoped(
                &operator,
                SpdPreconditionerKind::Jacobi,
                LinearAlgebraBackendKind::CpuReference,
                &CpuReferenceBackend,
                Some(&guess),
                ScopedLinearSolveSettings {
                    plan,
                    admission: None,
                    relative_tolerance,
                    preparation: None,
                },
            )
            .unwrap()
        };
        let loose = solve(settings::DEFAULT_RELATIVE_TOLERANCE);
        let strict = solve(1e-12);
        assert!(loose.converged && strict.converged);
        assert_eq!(loose.iterations, 0);
        assert!(strict.iterations > 0);
        assert!(loose.residual_norm > 1e-12 * 5.0_f64.sqrt());
        assert!(strict.residual_norm <= 1e-12 * 5.0_f64.sqrt());
    }
}

#[test]
fn invalid_scoped_tolerances_reject_before_cpu_or_provider_work() {
    let operator = system([4.0, 3.0]);
    for backend in [
        LinearAlgebraBackendKind::CpuReference,
        LinearAlgebraBackendKind::Accelerator,
    ] {
        for relative_tolerance in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let result = solve_operator_system_scoped(
                &operator,
                SpdPreconditionerKind::Jacobi,
                backend,
                &CpuReferenceBackend,
                None,
                ScopedLinearSolveSettings {
                    plan: LocalExecutionPlan::serial(),
                    admission: None,
                    relative_tolerance,
                    preparation: None,
                },
            )
            .unwrap();
            assert!(!result.converged);
            assert_eq!(result.iterations, 0);
            assert_eq!(result.device_apply_k_attempt_count, 0);
            assert!(result.accelerator_evidence.is_none());
            assert_eq!(result.diagnostics[0].code, "FEA_INVALID_SOLVER_TOLERANCE");
        }
    }
}

#[test]
fn exact_prepared_work_is_checked_before_pcg_iterations() {
    let operator = system([4.0, 3.0]);
    let prepared = operator.prepare_linear().unwrap();
    let (dimension, factor_nonzeros, updates, resident_bytes) =
        prepared.admissibility().unwrap().coercivity().work_counts();
    let admission = SolverAdmissionResourcePolicy {
        dimension: Count(dimension),
        factor_nonzero_upper_bound: Count(factor_nonzeros),
        update_count_upper_bound: Count(updates),
        resident_bytes_upper_bound: ByteCount(resident_bytes),
    };
    let plan = LocalExecutionPlan::serial();
    let accepted = solve_operator_system_scoped(
        &operator,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        None,
        ScopedLinearSolveSettings::with_default_tolerance(plan, Some(&admission)),
    )
    .unwrap();
    assert!(accepted.converged);

    for rejected in [
        SolverAdmissionResourcePolicy {
            factor_nonzero_upper_bound: Count(factor_nonzeros - 1),
            ..admission.clone()
        },
        SolverAdmissionResourcePolicy {
            update_count_upper_bound: Count(updates - 1),
            ..admission.clone()
        },
        SolverAdmissionResourcePolicy {
            resident_bytes_upper_bound: ByteCount(resident_bytes - 1),
            ..admission.clone()
        },
        SolverAdmissionResourcePolicy {
            dimension: Count(dimension - 1),
            ..admission.clone()
        },
    ] {
        let result = solve_operator_system_scoped(
            &operator,
            SpdPreconditionerKind::Jacobi,
            LinearAlgebraBackendKind::CpuReference,
            &CpuReferenceBackend,
            None,
            ScopedLinearSolveSettings::with_default_tolerance(plan, Some(&rejected)),
        )
        .unwrap();
        assert!(!result.converged);
        assert_eq!(result.iterations, 0);
        assert_eq!(
            result.diagnostics[0].code,
            "FEA_SOLVER_WORK_EXCEEDS_ADMISSION"
        );
    }
}

#[test]
fn static_pcg_preserves_a_manufactured_solution_under_operator_and_load_scaling() {
    for scale in [1.0, 1.0e-20] {
        let stiffness = RealSparseMatrix::diagonal(
            vec![2.0 * scale, 4.0 * scale],
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
            vec![2.0 * scale, -8.0 * scale],
        )
        .unwrap();
        let serial = solve_operator_system(
            &operator,
            SpdPreconditionerKind::Jacobi,
            LinearAlgebraBackendKind::CpuReference,
            &CpuReferenceBackend,
            None,
        )
        .unwrap();
        let plan = LocalExecutionPlan::fixed(LocalThreadCount::new(2).unwrap()).unwrap();
        let scoped = solve_operator_system_scoped(
            &operator,
            SpdPreconditionerKind::Jacobi,
            LinearAlgebraBackendKind::CpuReference,
            &CpuReferenceBackend,
            None,
            ScopedLinearSolveSettings::with_default_tolerance(plan, None),
        )
        .unwrap();
        assert!(serial.converged, "scale={scale}: {:?}", serial.diagnostics);
        assert_eq!(serial, scoped);
        assert!((serial.solution[0] - 1.0).abs() <= 1.0e-12);
        assert!((serial.solution[1] + 2.0).abs() <= 1.0e-12);
        assert!(serial.residual_norm <= 1.0e-8 * (68.0_f64.sqrt() * scale));
    }
}

#[test]
fn zero_load_warm_start_returns_the_exact_zero_solution() {
    let stiffness = RealSparseMatrix::diagonal(
        vec![2.0, 4.0],
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
        vec![0.0; 2],
    )
    .unwrap();
    let solved = solve_operator_system(
        &operator,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        Some(&[1.0, -1.0]),
    )
    .unwrap();
    assert!(solved.converged, "{:?}", solved.diagnostics);
    // A homogeneous nonsingular system has the exact solution zero, whatever the warm start.
    assert_eq!(solved.iterations, 0);
    assert!(solved.solution.iter().all(|value| *value == 0.0));
    assert_eq!(solved.relative_residual, Some(0.0));
}

// witness: FEA-SCALE-PARALLEL-001/shared_parallel_kernels @
//   shared_infrastructure/canonical_mixed_topology/parallel_cpu; coverage: partial;
//   boundary: A scoped PCG on a two-unknown SPD system equals the serial solution and repeats
//   exactly; nested scoped execution is rejected. Tiny system, one preconditioner.
// witness: FEA-SHARED-ACCELERATORS-001/resource_scoped_local_parallelism @
//   shared_infrastructure/canonical_mixed_topology/parallel_cpu; coverage: partial;
//   boundary: A scoped pool solve equals the serial solve and nested pools are rejected; global
//   pool independence and CPU resource accounting are covered in the resources requirement.
#[test]
fn scoped_pcg_matches_serial_and_rejects_nested_execution() {
    let operator = system([4.0, 3.0]);
    let serial = solve_operator_system(
        &operator,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        None,
    )
    .unwrap();
    let plan = LocalExecutionPlan::fixed(LocalThreadCount::new(2).unwrap()).unwrap();
    let parallel = solve_operator_system_scoped(
        &operator,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        None,
        ScopedLinearSolveSettings::with_default_tolerance(plan, None),
    )
    .unwrap();
    let repeated = solve_operator_system_scoped(
        &operator,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        None,
        ScopedLinearSolveSettings::with_default_tolerance(plan, None),
    )
    .unwrap();
    assert_eq!(parallel, repeated);
    assert_eq!(parallel, serial);
    assert!((parallel.solution[0] - 5.0 / 11.0).abs() < 1.0e-12);
    assert!((parallel.solution[1] - 9.0 / 11.0).abs() < 1.0e-12);
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
    )
    .unwrap();

    assert!(result.converged, "diagnostics={:?}", result.diagnostics);
    assert!(result.residual_norm <= 1.0e-8);
}

// witness: FEA-SHARED-OPERATORS-001/compatible_solver_and_preconditioner_selection @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: A noncoercive operator is rejected before preconditioner setup.
#[test]
fn noncoercive_operator_is_rejected_before_preconditioner_setup() {
    let operator = system([0.0, 3.0]);
    let result = solve_operator_system(
        &operator,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        None,
    )
    .unwrap();
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
        )
        .unwrap();
        assert!(!result.converged);
        assert_eq!(result.iterations, 0);
        assert_eq!(result.device_apply_k_attempt_count, 0);
        assert_eq!(result.diagnostics[0].code, "FEA_INVALID_REAL_SYSTEM");
    }
}

#[test]
fn warmed_preparation_keeps_exact_work_admission_and_original_pcg_result() {
    let original = system([4.0, 3.0]);
    let mut reuse = crate::algebra::LinearPreparationReuse::default();
    let prepared = original.prepare_linear_reusing(&mut reuse).unwrap();
    let (dimension, factors, updates, bytes) =
        prepared.admissibility().unwrap().coercivity().work_counts();
    let cold = solve_operator_system(
        &original,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        None,
    )
    .unwrap();
    let mut settings =
        ScopedLinearSolveSettings::with_default_tolerance(LocalExecutionPlan::serial(), None);
    settings.preparation = Some(&mut reuse);
    let warm = solve_operator_system_scoped(
        &original,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        None,
        settings,
    )
    .unwrap();
    assert_eq!(cold, warm);
    let policy = crate::planning::resource::SolverAdmissionResourcePolicy {
        dimension: Count(dimension),
        factor_nonzero_upper_bound: Count(factors - 1),
        update_count_upper_bound: Count(updates),
        resident_bytes_upper_bound: ByteCount(bytes),
    };
    let rejected = solve_operator_system_scoped(
        &original,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::Accelerator,
        &CpuReferenceBackend,
        None,
        ScopedLinearSolveSettings {
            plan: LocalExecutionPlan::serial(),
            admission: Some(&policy),
            relative_tolerance: 1e-8,
            preparation: Some(&mut reuse),
        },
    )
    .unwrap();
    assert!(!rejected.converged);
    assert_eq!(rejected.iterations, 0);
    assert_eq!(rejected.device_apply_k_attempt_count, 0);
    assert_eq!(
        rejected.diagnostics[0].code,
        "FEA_SOLVER_WORK_EXCEEDS_ADMISSION"
    );
}
