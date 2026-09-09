use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::anyhow;

use super::*;
use crate::{
    AccelDownloadFuture, AccelProvider, GpuTensorDescriptor, GpuTensorHandle, GpuTensorStorage,
    HostNumericDataView, HostNumericTensorView, NumericElementType,
};

fn numeric_handle(device_id: u32, buffer_id: u64, length: usize) -> GpuTensorHandle {
    GpuTensorHandle {
        shape: vec![length],
        device_id,
        buffer_id,
        descriptor: GpuTensorDescriptor::numeric(NumericElementType::F64, GpuTensorStorage::Real),
    }
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

    fn free(&self, _handle: &GpuTensorHandle) -> anyhow::Result<()> {
        Ok(())
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
}

#[test]
fn krylov_validation_rejects_cross_device_and_preconditioner_substitution() {
    let matrix = matrix_handle(7, 13);
    let rhs = numeric_handle(7, 21, 2);
    let wrong_device_guess = numeric_handle(8, 22, 2);
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
    let base = SparseKrylovRequest {
        matrix: &matrix,
        rhs: &rhs,
        initial_guess: Some(&wrong_device_guess),
        preconditioner: None,
        method: SparseKrylovMethod::ConjugateGradient,
        relative_tolerance: 1.0e-8,
        absolute_tolerance: 0.0,
        max_iterations: 20,
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
    let rhs = numeric_handle(0, 21, 2);
    let request = SparseKrylovRequest {
        matrix: &matrix,
        rhs: &rhs,
        initial_guess: None,
        preconditioner: None,
        method: SparseKrylovMethod::ConjugateGradient,
        relative_tolerance: 1.0e-8,
        absolute_tolerance: 0.0,
        max_iterations: 20,
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
        krylov_methods: vec![SparseKrylovMethod::ConjugateGradient],
        resident_krylov: true,
        max_host_synchronizations_per_solve: Some(1),
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
        element_type: NumericElementType::F32,
        ..query
    }));

    let mut nonresident = capabilities;
    nonresident.resident_krylov = false;
    assert!(nonresident.validate().is_err());
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
