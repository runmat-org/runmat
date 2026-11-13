use futures::executor::block_on;
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostCsrPatternView, HostNumericDataOwned,
    HostNumericDataView, HostNumericTensorView, NumericElementType, ProviderCancellationToken,
    ProviderRejectionCode, SparseExecutionContext, SparseFeasibility, SparseFeasibilityQuery,
    SparseIndexWidth, SparseKrylovMethod, SparseKrylovRequest, SparseKrylovTermination,
    SparseOperation, SparsePreconditionerApplyRequest, SparsePreconditionerKind,
    SparsePreconditionerRequest, SparseProviderLossReason, SparseSpmvRequest, SparseTranspose,
};
use runmat_execution::{identity::AttemptId, Digest};

use super::super::super::WgpuProvider;
use crate::backend::wgpu::provider::{
    register_test_wgpu_provider, WgpuPrecisionPolicy, WgpuProviderOptions, WgpuTestSession,
};
use crate::backend::wgpu::types::NumericPrecision;

const ROW_OFFSETS: [u64; 3] = [0, 2, 4];
const COLUMN_INDICES: [u64; 4] = [0, 1, 0, 1];

fn test_session() -> Option<WgpuTestSession> {
    match register_test_wgpu_provider(WgpuProviderOptions::default()) {
        Ok(session) => Some(session),
        Err(error) if std::env::var_os("RUNMAT_REQUIRE_WGPU_SPARSE").is_some() => {
            panic!("required WGPU sparse test device is unavailable: {error}")
        }
        Err(_) => None,
    }
}

fn upload_pattern(provider: &WgpuProvider) -> runmat_accelerate_api::SparsePatternHandle {
    provider
        .upload_sparse_pattern(HostCsrPatternView {
            rows: 2,
            columns: 2,
            row_offsets: &ROW_OFFSETS,
            column_indices: &COLUMN_INDICES,
            index_width: SparseIndexWidth::U32,
        })
        .expect("upload a valid CSR pattern")
}

fn bind_matrix(
    provider: &WgpuProvider,
    pattern: &runmat_accelerate_api::SparsePatternHandle,
) -> runmat_accelerate_api::SparseMatrixHandle {
    match provider.precision {
        NumericPrecision::F32 => provider
            .bind_sparse_values(
                pattern,
                &HostNumericTensorView {
                    data: HostNumericDataView::F32(&[4.0, 1.0, 1.0, 3.0]),
                    shape: &[4],
                    storage: GpuTensorStorage::Real,
                },
            )
            .expect("bind native f32 values"),
        NumericPrecision::F64 => provider
            .bind_sparse_values(
                pattern,
                &HostNumericTensorView {
                    data: HostNumericDataView::F64(&[4.0, 1.0, 1.0, 3.0]),
                    shape: &[4],
                    storage: GpuTensorStorage::Real,
                },
            )
            .expect("bind native f64 values"),
    }
}

fn upload_vector(provider: &WgpuProvider, values: [f64; 2]) -> GpuTensorHandle {
    upload_numeric_vector(provider, &values)
}

fn upload_numeric_vector(provider: &WgpuProvider, values: &[f64]) -> GpuTensorHandle {
    match provider.precision {
        NumericPrecision::F32 => {
            let values = values.iter().map(|value| *value as f32).collect::<Vec<_>>();
            provider
                .upload_numeric(&HostNumericTensorView {
                    data: HostNumericDataView::F32(&values),
                    shape: &[values.len()],
                    storage: GpuTensorStorage::Real,
                })
                .expect("upload native f32 vector")
        }
        NumericPrecision::F64 => provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F64(values),
                shape: &[values.len()],
                storage: GpuTensorStorage::Real,
            })
            .expect("upload native f64 vector"),
    }
}

fn upload_csr(
    provider: &WgpuProvider,
    rows: usize,
    row_offsets: &[u64],
    column_indices: &[u64],
    values: &[f64],
) -> (
    runmat_accelerate_api::SparsePatternHandle,
    runmat_accelerate_api::SparseMatrixHandle,
) {
    let pattern = provider
        .upload_sparse_pattern(HostCsrPatternView {
            rows: rows as u64,
            columns: rows as u64,
            row_offsets,
            column_indices,
            index_width: SparseIndexWidth::U32,
        })
        .expect("upload CSR pattern");
    let matrix = match provider.precision {
        NumericPrecision::F32 => {
            let values = values.iter().map(|value| *value as f32).collect::<Vec<_>>();
            provider
                .bind_sparse_values(
                    &pattern,
                    &HostNumericTensorView {
                        data: HostNumericDataView::F32(&values),
                        shape: &[values.len()],
                        storage: GpuTensorStorage::Real,
                    },
                )
                .expect("bind f32 CSR values")
        }
        NumericPrecision::F64 => provider
            .bind_sparse_values(
                &pattern,
                &HostNumericTensorView {
                    data: HostNumericDataView::F64(values),
                    shape: &[values.len()],
                    storage: GpuTensorStorage::Real,
                },
            )
            .expect("bind f64 CSR values"),
    };
    (pattern, matrix)
}

fn tridiagonal_system(rows: usize, diagonal: f64) -> (Vec<u64>, Vec<u64>, Vec<f64>) {
    let mut row_offsets = Vec::with_capacity(rows + 1);
    let mut columns = Vec::with_capacity(rows.saturating_mul(3).saturating_sub(2));
    let mut values = Vec::with_capacity(columns.capacity());
    row_offsets.push(0);
    for row in 0..rows {
        if row > 0 {
            columns.push((row - 1) as u64);
            values.push(-1.0);
        }
        columns.push(row as u64);
        values.push(diagonal);
        if row + 1 < rows {
            columns.push((row + 1) as u64);
            values.push(-1.0);
        }
        row_offsets.push(columns.len() as u64);
    }
    (row_offsets, columns, values)
}

fn download_vector(provider: &WgpuProvider, handle: &GpuTensorHandle) -> Vec<f64> {
    match block_on(provider.download_numeric(handle))
        .expect("download sparse SpMV output")
        .data
    {
        HostNumericDataOwned::F32(values) => values.into_iter().map(f64::from).collect(),
        HostNumericDataOwned::F64(values) => values,
        other => panic!("expected floating sparse output, got {other:?}"),
    }
}

/// Identity has an exact zero residual either initially or after one CG step;
/// zero rho is then convergence, not an attempted division/breakdown.
fn assert_exact_zero_residual_converges(provider: &WgpuProvider) {
    let (pattern, matrix) = upload_csr(
        provider,
        2,
        &ROW_OFFSETS,
        &COLUMN_INDICES,
        &[1.0, 0.0, 0.0, 1.0],
    );
    let preconditioner = block_on(provider.prepare_sparse_preconditioner(
        &SparsePreconditionerRequest {
            matrix: &matrix,
            kind: SparsePreconditionerKind::Jacobi,
        },
    ))
    .expect("prepare exact-zero Jacobi");
    let capability = provider.capability_snapshot();
    let cancellation = ProviderCancellationToken::new();
    for (rhs_values, expected_iterations) in [([0.0, 0.0], 0_u32), ([1.0, 2.0], 1_u32)] {
        let rhs = upload_vector(provider, rhs_values);
        let case = expected_iterations.to_le_bytes();
        let request = SparseKrylovRequest {
            matrix: &matrix,
            rhs: &rhs,
            initial_guess: None,
            preconditioner: Some(&preconditioner),
            method: SparseKrylovMethod::ConjugateGradient,
            relative_tolerance: 1.0e-12,
            absolute_tolerance: 0.0,
            max_iterations: 4,
            execution: SparseExecutionContext {
                attempt_id: AttemptId::derive(&[b"wgpu-sparse-cg-exact-zero".as_slice(), &case]),
                operator_identity: Digest::sha256(b"wgpu-sparse-cg-identity"),
                provider_capability_schema_version: capability.schema_version,
                provider_capability_revision: capability.revision,
                cancellation: &cancellation,
            },
        };
        let result = block_on(provider.sparse_krylov_solve(&request)).expect("solve exact-zero CG");
        result
            .validate(request, &capability)
            .expect("validate exact-zero CG evidence");
        assert_eq!(result.termination, SparseKrylovTermination::Converged);
        assert_eq!(result.iterations, expected_iterations);
        assert_eq!(result.final_residual_norm, Some(0.0));
        assert_eq!(
            download_vector(provider, result.solution.as_ref().unwrap()),
            rhs_values
        );
        assert!(!result.evidence.used_host_fallback);
        drop(result);
        drop(rhs);
    }
    provider
        .free_sparse_preconditioner(&preconditioner)
        .unwrap();
    provider.free_sparse_matrix(&matrix).unwrap();
    provider.free_sparse_pattern(&pattern).unwrap();
}

#[test]
fn sparse_admission_is_exact_and_side_effect_free() {
    let Some(session) = test_session() else {
        return;
    };
    let provider = session.provider();
    let descriptor = HostCsrPatternView {
        rows: 2,
        columns: 2,
        row_offsets: &ROW_OFFSETS,
        column_indices: &COLUMN_INDICES,
        index_width: SparseIndexWidth::U32,
    }
    .validate()
    .expect("valid query pattern");
    let element_type = match provider.precision {
        NumericPrecision::F32 => NumericElementType::F32,
        NumericPrecision::F64 => NumericElementType::F64,
    };
    let query = SparseFeasibilityQuery {
        operation: SparseOperation::Spmv(SparseTranspose::None),
        pattern: descriptor,
        element_type,
        storage: GpuTensorStorage::Real,
        vector_count: 1,
        vector_length: 2,
        max_iterations: None,
    };

    assert_eq!(provider.sparse.resident_bytes(), 0);
    let SparseFeasibility::Supported { estimate } = provider.query_sparse_feasibility(&query)
    else {
        panic!("native provider precision should admit CSR SpMV");
    };
    assert_eq!(estimate.dispatch_upper_bound, 1);
    assert_eq!(estimate.host_synchronization_upper_bound, 0);
    assert_eq!(provider.sparse.resident_bytes(), 0);
    assert!(matches!(
        provider.query_sparse_feasibility(&SparseFeasibilityQuery {
            element_type: match element_type {
                NumericElementType::F32 => NumericElementType::F64,
                _ => NumericElementType::F32,
            },
            ..query
        }),
        SparseFeasibility::Rejected {
            rejection: runmat_accelerate_api::ProviderRejection {
                code: ProviderRejectionCode::UnsupportedElementType,
                ..
            }
        }
    ));
    let snapshot = provider.capability_snapshot();
    // Revision 25 is the F64 provider without IEEE-word reductions.
    assert!(matches!(snapshot.revision, 24 | 25));
    snapshot
        .sparse
        .expect("WGPU sparse capability envelope")
        .validate()
        .expect("valid sparse capability envelope");
}

#[test]
fn sparse_storage_lifecycle_keeps_shared_pattern_until_last_matrix_release() {
    let Some(session) = test_session() else {
        return;
    };
    let provider = session.provider();
    let baseline_bytes = provider.placement_resources().live_bytes;
    let pattern = upload_pattern(provider);
    let pattern_bytes = provider.sparse.resident_bytes();
    assert_eq!(pattern_bytes, 28);
    assert_eq!(
        provider.placement_resources().live_bytes,
        baseline_bytes + pattern_bytes
    );
    let matrix = bind_matrix(provider, &pattern);
    let matrix_bytes = provider.sparse.resident_bytes();
    assert_eq!(
        matrix_bytes,
        pattern_bytes + 4 * provider.element_size as u64
    );

    provider
        .free_sparse_pattern(&pattern)
        .expect("release public pattern handle");
    assert_eq!(provider.sparse.resident_bytes(), matrix_bytes);
    assert!(provider.free_sparse_pattern(&pattern).is_err());
    provider
        .free_sparse_matrix(&matrix)
        .expect("release final matrix owner");
    assert_eq!(provider.sparse.resident_bytes(), 0);
    assert_eq!(provider.placement_resources().live_bytes, baseline_bytes);
    assert!(provider.free_sparse_matrix(&matrix).is_err());
}

#[test]
fn sparse_spmv_executes_on_device_and_rejects_stale_session_handles() {
    let stale_matrix = {
        let Some(session) = test_session() else {
            return;
        };
        let provider = session.provider();
        let pattern = upload_pattern(provider);
        let matrix = bind_matrix(provider, &pattern);
        let input = upload_vector(provider, [1.0, 2.0]);
        let output = block_on(provider.sparse_spmv(&SparseSpmvRequest {
            matrix: &matrix,
            input: &input,
            transpose: SparseTranspose::None,
        }))
        .expect("execute device-resident CSR SpMV");
        let values = download_vector(provider, &output);
        assert!((values[0] - 6.0).abs() < 1.0e-5);
        assert!((values[1] - 7.0).abs() < 1.0e-5);
        matrix
    };

    let session = register_test_wgpu_provider(WgpuProviderOptions::default())
        .expect("reuse the initialized test device");
    let provider = session.provider();
    let input = upload_vector(provider, [1.0, 2.0]);
    assert!(block_on(provider.sparse_spmv(&SparseSpmvRequest {
        matrix: &stale_matrix,
        input: &input,
        transpose: SparseTranspose::None,
    }))
    .is_err());
}

#[test]
fn sparse_spmv_covers_multiple_workgroups_without_host_staging() {
    let Some(session) = test_session() else {
        return;
    };
    let provider = session.provider();
    let rows = usize::try_from(provider.workgroup_config.scalar)
        .expect("workgroup size fits usize")
        .saturating_mul(2)
        .saturating_add(3);
    let row_offsets = (0..=rows).map(|index| index as u64).collect::<Vec<_>>();
    let column_indices = (0..rows).map(|index| index as u64).collect::<Vec<_>>();
    let pattern = provider
        .upload_sparse_pattern(HostCsrPatternView {
            rows: rows as u64,
            columns: rows as u64,
            row_offsets: &row_offsets,
            column_indices: &column_indices,
            index_width: SparseIndexWidth::U32,
        })
        .expect("upload a multi-workgroup identity pattern");

    let input_values = (0..rows)
        .map(|index| index as f64 * 0.25)
        .collect::<Vec<_>>();
    let (matrix, input) = match provider.precision {
        NumericPrecision::F32 => {
            let values = vec![1.0_f32; rows];
            let input = input_values
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>();
            (
                provider
                    .bind_sparse_values(
                        &pattern,
                        &HostNumericTensorView {
                            data: HostNumericDataView::F32(&values),
                            shape: &[rows],
                            storage: GpuTensorStorage::Real,
                        },
                    )
                    .expect("bind identity values"),
                provider
                    .upload_numeric(&HostNumericTensorView {
                        data: HostNumericDataView::F32(&input),
                        shape: &[rows],
                        storage: GpuTensorStorage::Real,
                    })
                    .expect("upload multi-workgroup input"),
            )
        }
        NumericPrecision::F64 => {
            let values = vec![1.0_f64; rows];
            (
                provider
                    .bind_sparse_values(
                        &pattern,
                        &HostNumericTensorView {
                            data: HostNumericDataView::F64(&values),
                            shape: &[rows],
                            storage: GpuTensorStorage::Real,
                        },
                    )
                    .expect("bind identity values"),
                provider
                    .upload_numeric(&HostNumericTensorView {
                        data: HostNumericDataView::F64(&input_values),
                        shape: &[rows],
                        storage: GpuTensorStorage::Real,
                    })
                    .expect("upload multi-workgroup input"),
            )
        }
    };
    let output = block_on(provider.sparse_spmv(&SparseSpmvRequest {
        matrix: &matrix,
        input: &input,
        transpose: SparseTranspose::None,
    }))
    .expect("execute multi-workgroup sparse SpMV");
    assert_eq!(output.shape(), [rows]);
    assert_eq!(download_vector(provider, &output), input_values);
}

#[test]
fn empty_sparse_matrix_preserves_zero_shape_and_physical_accounting() {
    let Some(session) = test_session() else {
        return;
    };
    let provider = session.provider();
    let pattern = provider
        .upload_sparse_pattern(HostCsrPatternView {
            rows: 0,
            columns: 0,
            row_offsets: &[0],
            column_indices: &[],
            index_width: SparseIndexWidth::U32,
        })
        .expect("upload an empty CSR pattern");
    let (matrix, input) = match provider.precision {
        NumericPrecision::F32 => (
            provider
                .bind_sparse_values(
                    &pattern,
                    &HostNumericTensorView {
                        data: HostNumericDataView::F32(&[]),
                        shape: &[0],
                        storage: GpuTensorStorage::Real,
                    },
                )
                .expect("bind empty f32 values"),
            provider
                .upload_numeric(&HostNumericTensorView {
                    data: HostNumericDataView::F32(&[]),
                    shape: &[0],
                    storage: GpuTensorStorage::Real,
                })
                .expect("upload empty f32 input"),
        ),
        NumericPrecision::F64 => (
            provider
                .bind_sparse_values(
                    &pattern,
                    &HostNumericTensorView {
                        data: HostNumericDataView::F64(&[]),
                        shape: &[0],
                        storage: GpuTensorStorage::Real,
                    },
                )
                .expect("bind empty f64 values"),
            provider
                .upload_numeric(&HostNumericTensorView {
                    data: HostNumericDataView::F64(&[]),
                    shape: &[0],
                    storage: GpuTensorStorage::Real,
                })
                .expect("upload empty f64 input"),
        ),
    };
    assert_eq!(
        provider.sparse.resident_bytes(),
        8 + provider.element_size as u64
    );
    let output = block_on(provider.sparse_spmv(&SparseSpmvRequest {
        matrix: &matrix,
        input: &input,
        transpose: SparseTranspose::None,
    }))
    .expect("execute empty sparse SpMV");
    assert_eq!(output.shape(), [0]);
    assert!(download_vector(provider, &output).is_empty());
}

#[test]
fn sparse_jacobi_is_prepared_and_applied_on_device() {
    let Some(session) = test_session() else {
        return;
    };
    let provider = session.provider();
    let pattern = upload_pattern(provider);
    let matrix = bind_matrix(provider, &pattern);
    let before_preconditioner = provider.sparse.resident_bytes();
    let preconditioner = block_on(provider.prepare_sparse_preconditioner(
        &SparsePreconditionerRequest {
            matrix: &matrix,
            kind: SparsePreconditionerKind::Jacobi,
        },
    ))
    .expect("prepare a resident Jacobi preconditioner");
    assert_eq!(
        provider.sparse.resident_bytes(),
        before_preconditioner + 2 * provider.element_size as u64
    );

    let input = upload_vector(provider, [8.0, 9.0]);
    let output = block_on(provider.apply_sparse_preconditioner(
        &SparsePreconditionerApplyRequest {
            preconditioner: &preconditioner,
            input: &input,
        },
    ))
    .expect("apply resident Jacobi preconditioner");
    let values = download_vector(provider, &output);
    assert!((values[0] - 2.0).abs() < 1.0e-5);
    assert!((values[1] - 3.0).abs() < 1.0e-5);

    drop(output);
    drop(input);
    provider
        .free_sparse_preconditioner(&preconditioner)
        .expect("release Jacobi preconditioner");
    provider
        .free_sparse_matrix(&matrix)
        .expect("release sparse matrix");
    provider
        .free_sparse_pattern(&pattern)
        .expect("release sparse pattern");
    assert_eq!(provider.sparse.resident_bytes(), 0);
}

#[test]
fn sparse_cg_keeps_iteration_work_resident_and_returns_bound_evidence() {
    let Some(session) = test_session() else {
        return;
    };
    let provider = session.provider();
    let pattern = upload_pattern(provider);
    let matrix = bind_matrix(provider, &pattern);
    let preconditioner = block_on(provider.prepare_sparse_preconditioner(
        &SparsePreconditionerRequest {
            matrix: &matrix,
            kind: SparsePreconditionerKind::Jacobi,
        },
    ))
    .expect("prepare Jacobi for CG");
    let rhs = upload_vector(provider, [1.0, 2.0]);
    let initial = upload_vector(provider, [0.1, 0.1]);
    let capability = provider.capability_snapshot();
    let cancellation = ProviderCancellationToken::new();
    let attempt_id = AttemptId::derive(&[b"wgpu-sparse-cg-test"]);
    let operator_identity = Digest::sha256(b"wgpu-sparse-cg-operator");
    let request = SparseKrylovRequest {
        matrix: &matrix,
        rhs: &rhs,
        initial_guess: Some(&initial),
        preconditioner: Some(&preconditioner),
        method: SparseKrylovMethod::ConjugateGradient,
        relative_tolerance: 1.0e-5,
        absolute_tolerance: 0.0,
        max_iterations: 64,
        execution: SparseExecutionContext {
            attempt_id,
            operator_identity,
            provider_capability_schema_version: capability.schema_version,
            provider_capability_revision: capability.revision,
            cancellation: &cancellation,
        },
    };
    let result = block_on(provider.sparse_krylov_solve(&request)).expect("solve resident CG");
    result
        .validate(request, &capability)
        .expect("validate resident CG result");
    let SparseFeasibility::Supported { estimate } =
        provider.query_sparse_feasibility(&SparseFeasibilityQuery {
            operation: SparseOperation::KrylovSolve {
                method: SparseKrylovMethod::ConjugateGradient,
                preconditioner: Some(SparsePreconditionerKind::Jacobi),
            },
            pattern: matrix.pattern,
            element_type: matrix.descriptor.element_type,
            storage: matrix.descriptor.storage,
            vector_count: 6,
            vector_length: matrix.pattern.rows,
            max_iterations: Some(request.max_iterations),
        })
    else {
        panic!("executed CG request must retain a feasible resource estimate");
    };
    assert_eq!(result.termination, SparseKrylovTermination::Converged);
    assert!(result.iterations > 0 && result.iterations <= 2);
    assert!(result.evidence.spmv_dispatches >= 2);
    assert!(result.evidence.vector_dispatches > result.evidence.spmv_dispatches);
    assert_eq!(
        result.evidence.setup_dispatches
            + result.evidence.spmv_dispatches
            + result.evidence.preconditioner_dispatches
            + result.evidence.vector_dispatches,
        7 + (result.evidence.spmv_dispatches - 1) * 12
    );
    assert!(result.evidence.host_upload_bytes > 0);
    assert_eq!(
        result.evidence.host_download_bytes,
        u64::from(result.evidence.device_readbacks) * result.evidence.host_upload_bytes
    );
    assert_eq!(estimate.workspace_bytes, result.evidence.scratch_bytes);
    assert_eq!(
        estimate.checked_peak_bytes(),
        Some(result.evidence.peak_resident_bytes)
    );
    assert!(!result.evidence.used_host_fallback);
    let solution = result.solution.as_ref().expect("resident CG solution");
    let values = download_vector(provider, solution);
    assert!((values[0] - 1.0 / 11.0).abs() < 1.0e-4);
    assert!((values[1] - 7.0 / 11.0).abs() < 1.0e-4);

    drop(result);
    drop(initial);
    drop(rhs);
    provider
        .free_sparse_preconditioner(&preconditioner)
        .expect("release preconditioner");
    provider
        .free_sparse_matrix(&matrix)
        .expect("release matrix");
    provider
        .free_sparse_pattern(&pattern)
        .expect("release pattern");
    assert_eq!(provider.sparse.resident_bytes(), 0);
}

#[test]
fn exact_f64_sparse_cg_is_executed_or_rejected_by_adapter_capability() {
    let provider = match WgpuProvider::new(WgpuProviderOptions {
        precision: WgpuPrecisionPolicy::RequireF64,
        ..WgpuProviderOptions::default()
    }) {
        Ok(provider) => provider,
        Err(error)
            if error
                .to_string()
                .contains("selected adapter does not expose SHADER_F64") =>
        {
            if std::env::var_os("RUNMAT_REQUIRE_WGPU_F64").is_some() {
                panic!("required exact-f64 WGPU adapter is unavailable: {error}");
            }
            eprintln!("exact-f64 WGPU qualification: {error}");
            return;
        }
        Err(error) if std::env::var_os("RUNMAT_REQUIRE_WGPU_SPARSE").is_some() => {
            panic!("required exact-f64 WGPU qualification could not inspect an adapter: {error}")
        }
        Err(_) => return,
    };
    assert_eq!(provider.precision, NumericPrecision::F64);
    assert_exact_zero_residual_converges(&provider);
    let pattern = upload_pattern(&provider);
    let matrix = bind_matrix(&provider, &pattern);
    let preconditioner = block_on(provider.prepare_sparse_preconditioner(
        &SparsePreconditionerRequest {
            matrix: &matrix,
            kind: SparsePreconditionerKind::Jacobi,
        },
    ))
    .expect("prepare exact-f64 Jacobi");
    let rhs = upload_vector(&provider, [1.0, 2.0]);
    let capability = provider.capability_snapshot();
    assert_eq!(
        capability
            .sparse
            .as_ref()
            .expect("exact-f64 sparse capability")
            .element_types,
        vec![NumericElementType::F64]
    );
    let cancellation = ProviderCancellationToken::new();
    let request = SparseKrylovRequest {
        matrix: &matrix,
        rhs: &rhs,
        initial_guess: None,
        preconditioner: Some(&preconditioner),
        method: SparseKrylovMethod::ConjugateGradient,
        relative_tolerance: 1.0e-12,
        absolute_tolerance: 0.0,
        max_iterations: 64,
        execution: SparseExecutionContext {
            attempt_id: AttemptId::derive(&[b"wgpu-exact-f64-sparse-cg"]),
            operator_identity: Digest::sha256(b"wgpu-exact-f64-sparse-cg-operator"),
            provider_capability_schema_version: capability.schema_version,
            provider_capability_revision: capability.revision,
            cancellation: &cancellation,
        },
    };
    let result = block_on(provider.sparse_krylov_solve(&request)).expect("solve exact-f64 CG");
    result
        .validate(request, &capability)
        .expect("validate exact-f64 CG evidence");
    assert_eq!(
        result.termination,
        SparseKrylovTermination::Converged,
        "iterations={}, initial residual={:?}, final residual={:?}",
        result.iterations,
        result.initial_residual_norm,
        result.final_residual_norm
    );
    let solution = result.solution.as_ref().expect("exact-f64 solution");
    let downloaded = block_on(provider.download_numeric(solution)).expect("download f64 solution");
    let HostNumericDataOwned::F64(values) = downloaded.data else {
        panic!("exact-f64 provider returned a narrowed solution")
    };
    assert!((values[0] - 1.0 / 11.0).abs() < 1.0e-12);
    assert!((values[1] - 7.0 / 11.0).abs() < 1.0e-12);

    drop(result);
    drop(rhs);
    provider
        .free_sparse_preconditioner(&preconditioner)
        .expect("release exact-f64 preconditioner");
    provider
        .free_sparse_matrix(&matrix)
        .expect("release exact-f64 matrix");
    provider
        .free_sparse_pattern(&pattern)
        .expect("release exact-f64 pattern");
    assert_eq!(provider.sparse.resident_bytes(), 0);
}

#[test]
fn sparse_cg_matches_a_representative_multi_workgroup_reference() {
    let Some(session) = test_session() else {
        return;
    };
    let provider = session.provider();
    let rows = 4_096;
    let (row_offsets, columns, values) = tridiagonal_system(rows, 4.0);
    let (pattern, matrix) = upload_csr(provider, rows, &row_offsets, &columns, &values);
    let preconditioner = block_on(provider.prepare_sparse_preconditioner(
        &SparsePreconditionerRequest {
            matrix: &matrix,
            kind: SparsePreconditionerKind::Jacobi,
        },
    ))
    .expect("prepare representative Jacobi");
    let expected = (0..rows)
        .map(|index| 1.0 + (index % 17) as f64 * 0.01)
        .collect::<Vec<_>>();
    let rhs_values = (0..rows)
        .map(|row| {
            4.0 * expected[row]
                - row.checked_sub(1).map_or(0.0, |left| expected[left])
                - expected.get(row + 1).copied().unwrap_or(0.0)
        })
        .collect::<Vec<_>>();
    let rhs = upload_numeric_vector(provider, &rhs_values);
    let capability = provider.capability_snapshot();
    let cancellation = ProviderCancellationToken::new();
    let request = SparseKrylovRequest {
        matrix: &matrix,
        rhs: &rhs,
        initial_guess: None,
        preconditioner: Some(&preconditioner),
        method: SparseKrylovMethod::ConjugateGradient,
        relative_tolerance: 1.0e-5,
        absolute_tolerance: 0.0,
        max_iterations: 128,
        execution: SparseExecutionContext {
            attempt_id: AttemptId::derive(&[b"wgpu-sparse-cg-scale"]),
            operator_identity: Digest::sha256(b"wgpu-sparse-cg-scale-operator"),
            provider_capability_schema_version: capability.schema_version,
            provider_capability_revision: capability.revision,
            cancellation: &cancellation,
        },
    };
    let result = block_on(provider.sparse_krylov_solve(&request)).expect("solve representative CG");
    assert_eq!(result.termination, SparseKrylovTermination::Converged);
    let solution = result.solution.as_ref().expect("representative solution");
    let actual = download_vector(provider, solution);
    let max_error = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_error < 5.0e-4, "maximum solution error {max_error}");
    let residual_squared = (0..rows)
        .map(|row| {
            let product = 4.0 * actual[row]
                - row.checked_sub(1).map_or(0.0, |left| actual[left])
                - actual.get(row + 1).copied().unwrap_or(0.0);
            (rhs_values[row] - product).powi(2)
        })
        .sum::<f64>();
    let rhs_squared = rhs_values.iter().map(|value| value * value).sum::<f64>();
    assert!((residual_squared / rhs_squared).sqrt() < 2.0e-5);

    drop(result);
    drop(rhs);
    provider
        .free_sparse_preconditioner(&preconditioner)
        .expect("release representative preconditioner");
    provider
        .free_sparse_matrix(&matrix)
        .expect("release representative matrix");
    provider
        .free_sparse_pattern(&pattern)
        .expect("release representative pattern");
    assert_eq!(provider.sparse.resident_bytes(), 0);
}

#[test]
fn sparse_cg_reports_iteration_limit_and_numerical_breakdown() {
    let Some(session) = test_session() else {
        return;
    };
    let provider = session.provider();
    assert_exact_zero_residual_converges(provider);
    let capability = provider.capability_snapshot();
    let cancellation = ProviderCancellationToken::new();

    let pattern = upload_pattern(provider);
    let matrix = bind_matrix(provider, &pattern);
    let preconditioner = block_on(provider.prepare_sparse_preconditioner(
        &SparsePreconditionerRequest {
            matrix: &matrix,
            kind: SparsePreconditionerKind::Jacobi,
        },
    ))
    .expect("prepare iteration-limit Jacobi");
    let rhs = upload_vector(provider, [1.0, 2.0]);
    let limited_request = SparseKrylovRequest {
        matrix: &matrix,
        rhs: &rhs,
        initial_guess: None,
        preconditioner: Some(&preconditioner),
        method: SparseKrylovMethod::ConjugateGradient,
        relative_tolerance: 1.0e-12,
        absolute_tolerance: 0.0,
        max_iterations: 1,
        execution: SparseExecutionContext {
            attempt_id: AttemptId::derive(&[b"wgpu-sparse-cg-iteration-limit"]),
            operator_identity: Digest::sha256(b"wgpu-sparse-cg-iteration-limit-operator"),
            provider_capability_schema_version: capability.schema_version,
            provider_capability_revision: capability.revision,
            cancellation: &cancellation,
        },
    };
    let limited = block_on(provider.sparse_krylov_solve(&limited_request))
        .expect("execute iteration-limited CG");
    assert_eq!(
        limited.termination,
        SparseKrylovTermination::MaximumIterations
    );
    assert_eq!(limited.iterations, 1);
    drop(limited.solution.expect("limited solution"));
    drop(rhs);
    provider
        .free_sparse_preconditioner(&preconditioner)
        .expect("release limited preconditioner");
    provider
        .free_sparse_matrix(&matrix)
        .expect("release limited matrix");
    provider
        .free_sparse_pattern(&pattern)
        .expect("release limited pattern");

    let (pattern, matrix) = upload_csr(
        provider,
        2,
        &ROW_OFFSETS,
        &COLUMN_INDICES,
        &[-1.0, 0.0, 0.0, -2.0],
    );
    let preconditioner = block_on(provider.prepare_sparse_preconditioner(
        &SparsePreconditionerRequest {
            matrix: &matrix,
            kind: SparsePreconditionerKind::Jacobi,
        },
    ))
    .expect("prepare breakdown Jacobi");
    let rhs = upload_vector(provider, [1.0, 1.0]);
    let breakdown_request = SparseKrylovRequest {
        matrix: &matrix,
        rhs: &rhs,
        initial_guess: None,
        preconditioner: Some(&preconditioner),
        method: SparseKrylovMethod::ConjugateGradient,
        relative_tolerance: 1.0e-5,
        absolute_tolerance: 0.0,
        max_iterations: 64,
        execution: SparseExecutionContext {
            attempt_id: AttemptId::derive(&[b"wgpu-sparse-cg-breakdown"]),
            operator_identity: Digest::sha256(b"wgpu-sparse-cg-breakdown-operator"),
            provider_capability_schema_version: capability.schema_version,
            provider_capability_revision: capability.revision,
            cancellation: &cancellation,
        },
    };
    let breakdown =
        block_on(provider.sparse_krylov_solve(&breakdown_request)).expect("execute breakdown CG");
    assert_eq!(
        breakdown.termination,
        SparseKrylovTermination::NumericalBreakdown
    );
    assert_eq!(breakdown.iterations, 0);
    drop(breakdown.solution.expect("breakdown solution"));
    drop(rhs);
    provider
        .free_sparse_preconditioner(&preconditioner)
        .expect("release breakdown preconditioner");
    provider
        .free_sparse_matrix(&matrix)
        .expect("release breakdown matrix");
    provider
        .free_sparse_pattern(&pattern)
        .expect("release breakdown pattern");
    assert_eq!(provider.sparse.resident_bytes(), 0);
}

#[test]
fn sparse_cg_observes_cancellation_after_resident_work_begins() {
    let Some(session) = test_session() else {
        return;
    };
    let provider = session.provider();
    let rows = 131_072;
    let (row_offsets, columns, values) = tridiagonal_system(rows, 2.0);
    let (pattern, matrix) = upload_csr(provider, rows, &row_offsets, &columns, &values);
    let preconditioner = block_on(provider.prepare_sparse_preconditioner(
        &SparsePreconditionerRequest {
            matrix: &matrix,
            kind: SparsePreconditionerKind::Jacobi,
        },
    ))
    .expect("prepare cancellation Jacobi");
    let rhs = upload_numeric_vector(provider, &vec![1.0; rows]);
    let capability = provider.capability_snapshot();
    let cancellation = ProviderCancellationToken::new();
    let cancellation_signal = cancellation.clone();
    let request = SparseKrylovRequest {
        matrix: &matrix,
        rhs: &rhs,
        initial_guess: None,
        preconditioner: Some(&preconditioner),
        method: SparseKrylovMethod::ConjugateGradient,
        relative_tolerance: 1.0e-12,
        absolute_tolerance: 0.0,
        max_iterations: super::cg::MAXIMUM_ITERATIONS,
        execution: SparseExecutionContext {
            attempt_id: AttemptId::derive(&[b"wgpu-sparse-cg-running-cancellation"]),
            operator_identity: Digest::sha256(b"wgpu-sparse-cg-running-cancellation-operator"),
            provider_capability_schema_version: capability.schema_version,
            provider_capability_revision: capability.revision,
            cancellation: &cancellation,
        },
    };
    let persistent_bytes = provider.sparse.resident_bytes();
    let result = std::thread::scope(|scope| {
        let canceller = scope.spawn(|| {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            while provider.sparse.resident_bytes() == persistent_bytes
                && std::time::Instant::now() < deadline
            {
                std::thread::yield_now();
            }
            assert!(
                provider.sparse.resident_bytes() > persistent_bytes,
                "CG workspace never became visible to placement accounting"
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
            cancellation_signal.cancel();
        });
        let result = block_on(provider.sparse_krylov_solve(&request));
        canceller.join().expect("cancellation observer");
        result
    })
    .expect("execute cancellable resident CG");
    assert_eq!(result.termination, SparseKrylovTermination::Cancelled);
    assert!(result.solution.is_none());
    assert!(result.evidence.cancellation_observed);
    assert!(result.evidence.host_synchronizations >= 1);
    assert!(result.evidence.host_synchronizations < 257);
    assert_eq!(provider.sparse.resident_bytes(), persistent_bytes);

    drop(rhs);
    provider
        .free_sparse_preconditioner(&preconditioner)
        .expect("release cancellation preconditioner");
    provider
        .free_sparse_matrix(&matrix)
        .expect("release cancellation matrix");
    provider
        .free_sparse_pattern(&pattern)
        .expect("release cancellation pattern");
    assert_eq!(provider.sparse.resident_bytes(), 0);
}

#[test]
fn sparse_cg_reports_provider_loss_after_resident_work_begins() {
    let Some(session) = test_session() else {
        return;
    };
    let provider = session.provider();
    let rows = 131_072;
    let (row_offsets, columns, values) = tridiagonal_system(rows, 2.0);
    let (pattern, matrix) = upload_csr(provider, rows, &row_offsets, &columns, &values);
    let preconditioner = block_on(provider.prepare_sparse_preconditioner(
        &SparsePreconditionerRequest {
            matrix: &matrix,
            kind: SparsePreconditionerKind::Jacobi,
        },
    ))
    .expect("prepare provider-loss Jacobi");
    let rhs = upload_numeric_vector(provider, &vec![1.0; rows]);
    let capability = provider.capability_snapshot();
    let cancellation = ProviderCancellationToken::new();
    let request = SparseKrylovRequest {
        matrix: &matrix,
        rhs: &rhs,
        initial_guess: None,
        preconditioner: Some(&preconditioner),
        method: SparseKrylovMethod::ConjugateGradient,
        relative_tolerance: 1.0e-12,
        absolute_tolerance: 0.0,
        max_iterations: super::cg::MAXIMUM_ITERATIONS,
        execution: SparseExecutionContext {
            attempt_id: AttemptId::derive(&[b"wgpu-sparse-cg-running-provider-loss"]),
            operator_identity: Digest::sha256(b"wgpu-sparse-cg-running-provider-loss-operator"),
            provider_capability_schema_version: capability.schema_version,
            provider_capability_revision: capability.revision,
            cancellation: &cancellation,
        },
    };
    let persistent_bytes = provider.sparse.resident_bytes();
    let result = std::thread::scope(|scope| {
        let (marked_tx, marked_rx) = std::sync::mpsc::sync_channel(0);
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
        let loss_observer = scope.spawn(move || {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            while provider.sparse.resident_bytes() == persistent_bytes
                && std::time::Instant::now() < deadline
            {
                std::thread::yield_now();
            }
            assert!(
                provider.sparse.resident_bytes() > persistent_bytes,
                "CG workspace never became visible to placement accounting"
            );
            let _loss = provider
                .device_health
                .inject_loss(SparseProviderLossReason::Destroyed);
            marked_tx.send(()).expect("publish injected provider loss");
            release_rx.recv().expect("release injected provider loss");
        });
        let result = block_on(provider.sparse_krylov_solve(&request));
        marked_rx.recv().expect("observe injected provider loss");
        assert!(provider.placement_resources().lost);
        release_tx.send(()).expect("release loss observer");
        loss_observer.join().expect("provider-loss observer");
        result
    })
    .expect("execute provider-loss resident CG");
    assert_eq!(result.termination, SparseKrylovTermination::ProviderLost);
    assert!(result.solution.is_none());
    assert!(!result.evidence.cancellation_observed);
    assert_eq!(
        result.evidence.provider_loss,
        Some(SparseProviderLossReason::Destroyed)
    );
    assert!(result.evidence.host_synchronizations >= 1);
    assert!(result.evidence.host_synchronizations < 257);
    assert_eq!(provider.sparse.resident_bytes(), persistent_bytes);
    assert!(!provider.placement_resources().lost);

    drop(rhs);
    provider
        .free_sparse_preconditioner(&preconditioner)
        .expect("release provider-loss preconditioner");
    provider
        .free_sparse_matrix(&matrix)
        .expect("release provider-loss matrix");
    provider
        .free_sparse_pattern(&pattern)
        .expect("release provider-loss pattern");
    assert_eq!(provider.sparse.resident_bytes(), 0);
}
