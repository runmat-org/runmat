use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};

use anyhow::anyhow;

use super::*;
use crate::{
    AccelDownloadFuture, AccelProvider, GpuTensorHandle, GpuTensorStorage, HostNumericDataView,
    HostNumericTensorView, NumericElementType, ProviderCancellationToken,
};

fn numeric_handle(device_id: u32, length: usize) -> GpuTensorHandle {
    crate::test_support::handle(
        device_id,
        vec![length],
        NumericElementType::F64,
        GpuTensorStorage::Real,
    )
}

#[test]
fn provider_cancellation_token_shares_one_atomic_state() {
    let shared = Arc::new(AtomicBool::new(false));
    let token = ProviderCancellationToken::from_shared(Arc::clone(&shared));
    assert!(!token.is_cancelled());
    shared.store(true, Ordering::Release);
    assert!(token.is_cancelled());

    let shared = Arc::new(AtomicBool::new(false));
    let token = ProviderCancellationToken::from_shared(Arc::clone(&shared));
    token.cancel();
    assert!(shared.load(Ordering::Acquire));
}

fn matrix_handle(device_id: u32, matrix_id: u64) -> SparseMatrixHandle {
    SparseMatrixHandle {
        device_id,
        matrix_id,
        pattern: SparsePatternDescriptor {
            format: SparseFormat::Csr,
            index_width: SparseIndexWidth::U32,
            rows: 2,
            columns: 2,
            nnz: 2,
        },
        descriptor: SparseMatrixDescriptor {
            pattern_id: 9,
            element_type: NumericElementType::F64,
            storage: GpuTensorStorage::Real,
        },
    }
}

fn execution_context(
    cancellation: &ProviderCancellationToken,
    provider_capability_revision: u64,
) -> SparseExecutionContext<'_> {
    SparseExecutionContext {
        attempt_id: runmat_execution::identity::AttemptId::derive(&[b"sparse-contract-test"]),
        operator_identity: runmat_execution::Digest::sha256(b"sparse-operator-test"),
        provider_capability_schema_version: crate::PROVIDER_CAPABILITY_SCHEMA_VERSION,
        provider_capability_revision,
        cancellation,
    }
}

struct TransferCountingProvider {
    uploads: AtomicUsize,
}

impl AccelProvider for TransferCountingProvider {
    fn upload(&self, _host: &crate::HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.uploads.fetch_add(1, Ordering::Relaxed);
        Err(anyhow!("test upload"))
    }

    fn download<'a>(&'a self, _handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { Err(anyhow!("test download")) })
    }

    fn device_info(&self) -> String {
        "sparse-contract-test".to_string()
    }
}

#[test]
fn csr_validation_rejects_noncanonical_and_narrow_indices() {
    assert!(HostCsrPatternView {
        rows: 2,
        columns: 2,
        row_offsets: &[0, 2, 2],
        column_indices: &[1, 0],
        index_width: SparseIndexWidth::U32,
    }
    .validate()
    .is_err());
    assert!(HostCsrPatternView {
        rows: 1,
        columns: u64::from(u32::MAX) + 2,
        row_offsets: &[0, 1],
        column_indices: &[u64::from(u32::MAX) + 1],
        index_width: SparseIndexWidth::U32,
    }
    .validate()
    .is_err());
}

#[test]
fn csr_descriptor_accounts_for_indices_without_dense_shape_math() {
    let descriptor = HostCsrPatternView {
        rows: 3,
        columns: 3,
        row_offsets: &[0, 2, 3, 5],
        column_indices: &[0, 2, 1, 0, 2],
        index_width: SparseIndexWidth::U32,
    }
    .validate()
    .unwrap();
    assert_eq!(descriptor.nnz, 5);
    assert_eq!(descriptor.checked_storage_bytes(), Some(36));
}

#[test]
fn matrix_values_must_match_pattern_shape_and_float_storage() {
    let pattern = SparsePatternHandle {
        device_id: 7,
        pattern_id: 11,
        descriptor: SparsePatternDescriptor {
            format: SparseFormat::Csr,
            index_width: SparseIndexWidth::U32,
            rows: 2,
            columns: 2,
            nnz: 2,
        },
    };
    let values = [1.0, 2.0];
    let descriptor = validate_sparse_values(
        &pattern,
        HostNumericTensorView {
            data: HostNumericDataView::F64(&values),
            shape: &[2],
            storage: GpuTensorStorage::Real,
        },
    )
    .unwrap();
    assert_eq!(descriptor.pattern_id, 11);
    assert_eq!(descriptor.element_type, NumericElementType::F64);

    let integer_values = [1_i32, 2];
    assert!(validate_sparse_values(
        &pattern,
        HostNumericTensorView {
            data: HostNumericDataView::I32(&integer_values),
            shape: &[2],
            storage: GpuTensorStorage::Real,
        },
    )
    .is_err());
    assert!(validate_sparse_values(
        &pattern,
        HostNumericTensorView {
            data: HostNumericDataView::F64(&[1.0, f64::NAN]),
            shape: &[2],
            storage: GpuTensorStorage::Real,
        },
    )
    .is_err());
}

#[test]
fn feasibility_rejects_impossible_constructed_pattern_descriptors() {
    let query = SparseFeasibilityQuery {
        operation: SparseOperation::UploadPattern,
        pattern: SparsePatternDescriptor {
            format: SparseFormat::Csr,
            index_width: SparseIndexWidth::U32,
            rows: 2,
            columns: 2,
            nnz: 5,
        },
        element_type: NumericElementType::F64,
        storage: GpuTensorStorage::Real,
        vector_count: 0,
        vector_length: 0,
        max_iterations: None,
    };
    assert!(query.validate().is_err());
}

#[test]
fn krylov_validation_rejects_cross_device_and_preconditioner_substitution() {
    let matrix = matrix_handle(7, 13);
    let rhs = numeric_handle(7, 2);
    let wrong_device_guess = numeric_handle(8, 2);
    let substituted = SparsePreconditionerHandle {
        device_id: 7,
        preconditioner_id: 31,
        descriptor: SparsePreconditionerDescriptor {
            matrix_id: 99,
            kind: SparsePreconditionerKind::Jacobi,
            dimension: 2,
            element_type: NumericElementType::F64,
            storage: GpuTensorStorage::Real,
        },
    };
    let cancellation = ProviderCancellationToken::new();
    let base = SparseKrylovRequest {
        matrix: &matrix,
        rhs: &rhs,
        initial_guess: Some(&wrong_device_guess),
        preconditioner: None,
        method: SparseKrylovMethod::ConjugateGradient,
        relative_tolerance: 1.0e-8,
        absolute_tolerance: 0.0,
        max_iterations: 20,
        execution: execution_context(&cancellation, 1),
    };
    assert!(base.validate(7).is_err());
    assert!(SparseKrylovRequest {
        initial_guess: None,
        preconditioner: Some(&substituted),
        ..base
    }
    .validate(7)
    .is_err());
}

#[test]
fn krylov_validation_requires_typed_provider_loss_to_match_the_terminal() {
    let provider = TransferCountingProvider {
        uploads: AtomicUsize::new(0),
    };
    let mut capability = crate::ProviderCapabilitySnapshot::conservative(&provider);
    capability.revision = 1;
    capability.sparse = Some(ProviderSparseCapabilities {
        formats: vec![SparseFormat::Csr],
        index_widths: vec![SparseIndexWidth::U32],
        element_types: vec![NumericElementType::F64],
        storage: vec![GpuTensorStorage::Real],
        transpose_modes: vec![SparseTranspose::None],
        preconditioners: vec![],
        krylov: vec![SparseKrylovCapability {
            method: SparseKrylovMethod::ConjugateGradient,
            preconditioner: None,
        }],
        max_host_synchronizations_per_solve: Some(1),
        cancellation: SparseCancellationCapability::Cooperative {
            maximum_work_units_between_checks: 1_024,
        },
        transactional_results: true,
        max_rows: Some(2),
        max_columns: Some(2),
        max_nnz: Some(2),
    });
    let matrix = matrix_handle(0, 13);
    let rhs = numeric_handle(0, 2);
    let cancellation = ProviderCancellationToken::new();
    let request = SparseKrylovRequest {
        matrix: &matrix,
        rhs: &rhs,
        initial_guess: None,
        preconditioner: None,
        method: SparseKrylovMethod::ConjugateGradient,
        relative_tolerance: 1.0e-8,
        absolute_tolerance: 0.0,
        max_iterations: 20,
        execution: execution_context(&cancellation, capability.revision),
    };
    let evidence = SparseOperationEvidence {
        schema_version: SPARSE_EVIDENCE_SCHEMA_VERSION,
        binding: SparseExecutionBinding {
            attempt_id: request.execution.attempt_id,
            operator_identity: request.execution.operator_identity,
            provider_device_id: 0,
            provider_capability_schema_version: capability.schema_version,
            provider_capability_revision: capability.revision,
            pattern_id: matrix.descriptor.pattern_id,
            matrix_id: matrix.matrix_id,
            preconditioner_id: None,
        },
        timings: SparseOperationTimings::default(),
        resident_pattern_bytes: 0,
        resident_matrix_bytes: 0,
        resident_preconditioner_bytes: 0,
        resident_solution_bytes: 0,
        host_upload_bytes: 0,
        host_download_bytes: 0,
        setup_dispatches: 0,
        spmv_dispatches: 0,
        preconditioner_dispatches: 0,
        vector_dispatches: 0,
        host_synchronizations: 1,
        device_readbacks: 1,
        peak_resident_bytes: 0,
        scratch_bytes: 0,
        cancellation_checks: 1,
        cancellation_observed: false,
        provider_loss: Some(SparseProviderLossReason::Destroyed),
        used_host_fallback: false,
    };
    let lost = SparseKrylovResult {
        solution: None,
        termination: SparseKrylovTermination::ProviderLost,
        iterations: 3,
        initial_residual_norm: None,
        final_residual_norm: None,
        evidence,
    };
    lost.validate(request, &capability)
        .expect("typed provider loss is a valid transactional terminal");

    let mut missing_reason = lost.clone();
    missing_reason.evidence.provider_loss = None;
    assert!(missing_reason.validate(request, &capability).is_err());

    let mut forged_completed = lost;
    forged_completed.termination = SparseKrylovTermination::Cancelled;
    forged_completed.evidence.cancellation_observed = true;
    assert!(forged_completed.validate(request, &capability).is_err());
}

#[test]
fn default_sparse_contract_rejects_without_dense_transfer() {
    let provider = TransferCountingProvider {
        uploads: AtomicUsize::new(0),
    };
    let pattern_view = HostCsrPatternView {
        rows: 2,
        columns: 2,
        row_offsets: &[0, 1, 2],
        column_indices: &[0, 1],
        index_width: SparseIndexWidth::U32,
    };
    assert!(provider.upload_sparse_pattern(pattern_view).is_err());
    let descriptor = pattern_view.validate().unwrap();
    let query = SparseFeasibilityQuery {
        operation: SparseOperation::KrylovSolve {
            method: SparseKrylovMethod::ConjugateGradient,
            preconditioner: Some(SparsePreconditionerKind::Jacobi),
        },
        pattern: descriptor,
        element_type: NumericElementType::F64,
        storage: GpuTensorStorage::Real,
        vector_count: 6,
        vector_length: 2,
        max_iterations: Some(20),
    };
    assert!(matches!(
        provider.query_sparse_feasibility(&query),
        SparseFeasibility::Rejected { .. }
    ));

    let matrix = matrix_handle(0, 13);
    let rhs = numeric_handle(0, 2);
    let cancellation = ProviderCancellationToken::new();
    let request = SparseKrylovRequest {
        matrix: &matrix,
        rhs: &rhs,
        initial_guess: None,
        preconditioner: None,
        method: SparseKrylovMethod::ConjugateGradient,
        relative_tolerance: 1.0e-8,
        absolute_tolerance: 0.0,
        max_iterations: 20,
        execution: execution_context(&cancellation, 1),
    };
    let _unsupported = provider.sparse_krylov_solve(&request);
    assert_eq!(provider.uploads.load(Ordering::Relaxed), 0);
    assert!(provider.capability_snapshot().sparse.is_none());
}

#[test]
fn sparse_capabilities_require_resident_krylov_and_exact_envelope_match() {
    let capabilities = ProviderSparseCapabilities {
        formats: vec![SparseFormat::Csr],
        index_widths: vec![SparseIndexWidth::U32],
        element_types: vec![NumericElementType::F64],
        storage: vec![GpuTensorStorage::Real],
        transpose_modes: vec![SparseTranspose::None],
        preconditioners: vec![SparsePreconditionerKind::Jacobi],
        krylov: vec![SparseKrylovCapability {
            method: SparseKrylovMethod::ConjugateGradient,
            preconditioner: Some(SparsePreconditionerKind::Jacobi),
        }],
        max_host_synchronizations_per_solve: Some(1),
        cancellation: SparseCancellationCapability::Cooperative {
            maximum_work_units_between_checks: 1_024,
        },
        transactional_results: true,
        max_rows: Some(100),
        max_columns: Some(100),
        max_nnz: Some(1_000),
    };
    let query = SparseFeasibilityQuery {
        operation: SparseOperation::KrylovSolve {
            method: SparseKrylovMethod::ConjugateGradient,
            preconditioner: Some(SparsePreconditionerKind::Jacobi),
        },
        pattern: SparsePatternDescriptor {
            format: SparseFormat::Csr,
            index_width: SparseIndexWidth::U32,
            rows: 20,
            columns: 20,
            nnz: 90,
        },
        element_type: NumericElementType::F64,
        storage: GpuTensorStorage::Real,
        vector_count: 6,
        vector_length: 20,
        max_iterations: Some(100),
    };
    assert!(capabilities.validate().is_ok());
    assert!(capabilities.supports(query));
    assert!(!capabilities.supports(SparseFeasibilityQuery {
        operation: SparseOperation::KrylovSolve {
            method: SparseKrylovMethod::ConjugateGradient,
            preconditioner: None,
        },
        ..query
    }));
    assert!(!capabilities.supports(SparseFeasibilityQuery {
        element_type: NumericElementType::F32,
        ..query
    }));

    let mut invalid = capabilities.clone();
    invalid.cancellation = SparseCancellationCapability::BeforeDispatch;
    assert!(invalid.validate().is_err());
    invalid = capabilities.clone();
    invalid.cancellation = SparseCancellationCapability::Cooperative {
        maximum_work_units_between_checks: 0,
    };
    assert!(invalid.validate().is_err());
    invalid = capabilities.clone();
    invalid.transactional_results = false;
    assert!(invalid.validate().is_err());

    let mut invalid_pairing = capabilities;
    invalid_pairing.krylov[0].preconditioner = Some(SparsePreconditionerKind::Ilu0);
    assert!(invalid_pairing.validate().is_err());
}

#[test]
fn sparse_resource_totals_are_checked() {
    assert_eq!(
        SparseResourceEstimate {
            persistent_pattern_bytes: 100,
            persistent_matrix_bytes: 200,
            persistent_preconditioner_bytes: 50,
            workspace_bytes: 400,
            output_bytes: 80,
            dispatch_upper_bound: 20,
            host_synchronization_upper_bound: 1,
        }
        .checked_peak_bytes(),
        Some(830)
    );
    assert!(SparseResourceEstimate {
        persistent_pattern_bytes: u64::MAX,
        persistent_matrix_bytes: 1,
        persistent_preconditioner_bytes: 0,
        workspace_bytes: 0,
        output_bytes: 0,
        dispatch_upper_bound: 0,
        host_synchronization_upper_bound: 0,
    }
    .checked_peak_bytes()
    .is_none());
}
