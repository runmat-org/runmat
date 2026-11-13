use futures::executor::block_on;
use runmat_accelerate_api::{
    AccelProvider, GpuTensorStorage, HostCsrPatternView, HostNumericDataOwned, HostNumericDataView,
    HostNumericTensorView, NumericElementType, SparseCostQuery, SparseFeasibility,
    SparseFeasibilityQuery, SparseIndexWidth, SparseKrylovMethod, SparseKrylovRequest,
    SparseKrylovTermination, SparseOperation, SparsePreconditionerApplyRequest,
    SparsePreconditionerKind, SparsePreconditionerRequest, SparseSpmvRequest, SparseTranspose,
};
use runmat_execution::CandidatePreparationState;

use super::super::InProcessProvider;

fn prepare_matrix(
    provider: &InProcessProvider,
) -> (
    runmat_accelerate_api::SparsePatternHandle,
    runmat_accelerate_api::SparseMatrixHandle,
) {
    let pattern = provider
        .upload_sparse_pattern(HostCsrPatternView {
            rows: 2,
            columns: 2,
            row_offsets: &[0, 2, 4],
            column_indices: &[0, 1, 0, 1],
            index_width: SparseIndexWidth::U32,
        })
        .unwrap();
    let matrix = provider
        .bind_sparse_values(
            &pattern,
            &HostNumericTensorView {
                data: HostNumericDataView::F64(&[4.0, 1.0, 1.0, 3.0]),
                shape: &[4],
                storage: GpuTensorStorage::Real,
            },
        )
        .unwrap();
    (pattern, matrix)
}

fn upload_vector(
    provider: &InProcessProvider,
    values: &[f64],
) -> runmat_accelerate_api::GpuTensorHandle {
    provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(values),
            shape: &[values.len()],
            storage: GpuTensorStorage::Real,
        })
        .unwrap()
}

fn download_vector(
    provider: &InProcessProvider,
    handle: &runmat_accelerate_api::GpuTensorHandle,
) -> Vec<f64> {
    match block_on(provider.download_numeric(handle)).unwrap().data {
        HostNumericDataOwned::F64(values) => values,
        _ => panic!("expected f64 vector"),
    }
}

fn execution_context<'a>(
    provider: &InProcessProvider,
    cancellation: &'a runmat_accelerate_api::ProviderCancellationToken,
) -> runmat_accelerate_api::SparseExecutionContext<'a> {
    let capability = provider.capability_snapshot();
    runmat_accelerate_api::SparseExecutionContext {
        attempt_id: runmat_execution::identity::AttemptId::derive(&[b"inprocess-sparse-test"]),
        operator_identity: runmat_execution::Digest::sha256(b"inprocess-sparse-operator"),
        provider_capability_schema_version: capability.schema_version,
        provider_capability_revision: capability.revision,
        cancellation,
    }
}

#[test]
fn inprocess_sparse_lifecycle_executes_spmv_jacobi_and_resident_cg() {
    let provider = InProcessProvider::new();
    let (pattern, matrix) = prepare_matrix(&provider);
    let input = upload_vector(&provider, &[1.0, 2.0]);

    let product = block_on(provider.sparse_spmv(&SparseSpmvRequest {
        matrix: &matrix,
        input: &input,
        transpose: SparseTranspose::None,
    }))
    .unwrap();
    assert_eq!(download_vector(&provider, &product), vec![6.0, 7.0]);

    let preconditioner = block_on(provider.prepare_sparse_preconditioner(
        &SparsePreconditionerRequest {
            matrix: &matrix,
            kind: SparsePreconditionerKind::Jacobi,
        },
    ))
    .unwrap();
    let diagonal_input = upload_vector(&provider, &[4.0, 6.0]);
    let diagonal_output = block_on(provider.apply_sparse_preconditioner(
        &SparsePreconditionerApplyRequest {
            preconditioner: &preconditioner,
            input: &diagonal_input,
        },
    ))
    .unwrap();
    assert_eq!(download_vector(&provider, &diagonal_output), vec![1.0, 2.0]);

    let rhs = upload_vector(&provider, &[1.0, 2.0]);
    let cancellation = runmat_accelerate_api::ProviderCancellationToken::new();
    let request = SparseKrylovRequest {
        matrix: &matrix,
        rhs: &rhs,
        initial_guess: None,
        preconditioner: Some(&preconditioner),
        method: SparseKrylovMethod::ConjugateGradient,
        relative_tolerance: 1.0e-12,
        absolute_tolerance: 0.0,
        max_iterations: 10,
        execution: execution_context(&provider, &cancellation),
    };
    let result = block_on(provider.sparse_krylov_solve(&request)).unwrap();
    assert_eq!(result.termination, SparseKrylovTermination::Converged);
    assert!(result.iterations <= 2);
    assert!(
        result
            .final_residual_norm
            .expect("converged solve residual")
            <= 1.0e-12 * 5.0_f64.sqrt()
    );
    let solution = download_vector(
        &provider,
        result.solution.as_ref().expect("converged CG solution"),
    );
    assert!((solution[0] - 1.0 / 11.0).abs() < 1.0e-12);
    assert!((solution[1] - 7.0 / 11.0).abs() < 1.0e-12);
    assert!(!result.evidence.used_host_fallback);
    assert_eq!(
        result.evidence.binding.attempt_id,
        request.execution.attempt_id
    );
    assert_eq!(
        result.evidence.binding.operator_identity,
        request.execution.operator_identity
    );
    assert!(result.evidence.cancellation_checks > 0);
    assert_eq!(result.evidence.host_synchronizations, 0);
    assert_eq!(result.evidence.device_readbacks, 0);

    let capability = provider.capability_snapshot();
    result.validate(request, &capability).unwrap();
    let mut substituted_evidence = result.clone();
    substituted_evidence.evidence.binding.operator_identity =
        runmat_execution::Digest::sha256(b"substituted-operator");
    assert!(substituted_evidence.validate(request, &capability).is_err());
    substituted_evidence = result.clone();
    substituted_evidence
        .evidence
        .binding
        .provider_capability_revision += 1;
    assert!(substituted_evidence.validate(request, &capability).is_err());
    substituted_evidence = result.clone();
    substituted_evidence.evidence.peak_resident_bytes = 0;
    assert!(substituted_evidence.validate(request, &capability).is_err());

    let mut stale_execution = request.execution;
    stale_execution.provider_capability_revision += 1;
    let stale_request = SparseKrylovRequest {
        execution: stale_execution,
        ..request
    };
    assert!(block_on(provider.sparse_krylov_solve(&stale_request)).is_err());

    provider.free_sparse_pattern(&pattern).unwrap();
    assert!(provider.free_sparse_pattern(&pattern).is_err());
    provider
        .free_sparse_preconditioner(&preconditioner)
        .unwrap();
    provider.free_sparse_matrix(&matrix).unwrap();
}

#[test]
fn inprocess_resident_cg_observes_typed_cancellation_before_iteration() {
    let provider = InProcessProvider::new();
    let (_pattern, matrix) = prepare_matrix(&provider);
    let rhs = upload_vector(&provider, &[1.0, 2.0]);
    let cancellation = runmat_accelerate_api::ProviderCancellationToken::new();
    cancellation.cancel();
    let request = SparseKrylovRequest {
        matrix: &matrix,
        rhs: &rhs,
        initial_guess: None,
        preconditioner: None,
        method: SparseKrylovMethod::ConjugateGradient,
        relative_tolerance: 1.0e-12,
        absolute_tolerance: 0.0,
        max_iterations: 10,
        execution: execution_context(&provider, &cancellation),
    };

    let result = block_on(provider.sparse_krylov_solve(&request)).unwrap();
    assert_eq!(result.termination, SparseKrylovTermination::Cancelled);
    assert_eq!(result.iterations, 0);
    assert_eq!(result.initial_residual_norm, None);
    assert_eq!(result.final_residual_norm, None);
    assert!(result.solution.is_none());
    assert_eq!(result.evidence.resident_solution_bytes, 0);
    assert!(result.evidence.cancellation_observed);
    assert_eq!(result.evidence.cancellation_checks, 1);
}

#[test]
fn inprocess_sparse_admission_is_exact_costed_and_side_effect_free() {
    let provider = InProcessProvider::new();
    let descriptor = HostCsrPatternView {
        rows: 2,
        columns: 2,
        row_offsets: &[0, 2, 4],
        column_indices: &[0, 1, 0, 1],
        index_width: SparseIndexWidth::U32,
    }
    .validate()
    .unwrap();
    let query = SparseFeasibilityQuery {
        operation: SparseOperation::KrylovSolve {
            method: SparseKrylovMethod::ConjugateGradient,
            preconditioner: Some(SparsePreconditionerKind::Jacobi),
        },
        pattern: descriptor,
        element_type: NumericElementType::F64,
        storage: GpuTensorStorage::Real,
        vector_count: 5,
        vector_length: 2,
        max_iterations: Some(10),
    };
    let SparseFeasibility::Supported { estimate } = provider.query_sparse_feasibility(&query)
    else {
        panic!("exact in-process sparse envelope should be supported");
    };
    assert_eq!(estimate.host_synchronization_upper_bound, 0);
    assert!(estimate.checked_peak_bytes().is_some());
    assert!(provider
        .estimate_sparse_cost(&SparseCostQuery {
            operation: query,
            preparation: CandidatePreparationState::Ready,
            required_upload_bytes: 0,
            required_download_bytes: 0,
            downstream_materialization: false,
        })
        .is_some());
    assert!(provider
        .estimate_sparse_cost(&SparseCostQuery {
            operation: query,
            preparation: CandidatePreparationState::Unknown,
            required_upload_bytes: 0,
            required_download_bytes: 0,
            downstream_materialization: false,
        })
        .is_none());
    assert!(matches!(
        provider.query_sparse_feasibility(&SparseFeasibilityQuery {
            element_type: NumericElementType::F32,
            ..query
        }),
        SparseFeasibility::Rejected { .. }
    ));
    assert!(matches!(
        provider.query_sparse_feasibility(&SparseFeasibilityQuery {
            operation: SparseOperation::KrylovSolve {
                method: SparseKrylovMethod::Gmres,
                preconditioner: None,
            },
            ..query
        }),
        SparseFeasibility::Rejected { .. }
    ));
    provider
        .capability_snapshot()
        .sparse
        .unwrap()
        .validate()
        .unwrap();
}

#[test]
fn inprocess_sparse_rejects_forged_and_cross_provider_handles_before_work() {
    let provider = InProcessProvider::new();
    let other = InProcessProvider::new();
    let (_pattern, matrix) = prepare_matrix(&provider);
    let other_input = upload_vector(&other, &[1.0, 2.0]);
    assert!(block_on(provider.sparse_spmv(&SparseSpmvRequest {
        matrix: &matrix,
        input: &other_input,
        transpose: SparseTranspose::None,
    }))
    .is_err());

    let mut forged = matrix.clone();
    forged.descriptor.pattern_id = forged.descriptor.pattern_id.wrapping_add(1);
    let input = upload_vector(&provider, &[1.0, 2.0]);
    assert!(block_on(provider.sparse_spmv(&SparseSpmvRequest {
        matrix: &forged,
        input: &input,
        transpose: SparseTranspose::None,
    }))
    .is_err());
}

#[test]
fn freeing_a_matrix_invalidates_its_derived_preconditioners() {
    let provider = InProcessProvider::new();
    let (_pattern, matrix) = prepare_matrix(&provider);
    let preconditioner = block_on(provider.prepare_sparse_preconditioner(
        &SparsePreconditionerRequest {
            matrix: &matrix,
            kind: SparsePreconditionerKind::Jacobi,
        },
    ))
    .unwrap();
    provider.free_sparse_matrix(&matrix).unwrap();

    let input = upload_vector(&provider, &[1.0, 2.0]);
    assert!(block_on(
        provider.apply_sparse_preconditioner(&SparsePreconditionerApplyRequest {
            preconditioner: &preconditioner,
            input: &input,
        },)
    )
    .is_err());
    assert!(provider
        .free_sparse_preconditioner(&preconditioner)
        .is_err());
}
