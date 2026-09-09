use futures::executor::block_on;
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostCsrPatternView, HostNumericDataOwned,
    HostNumericDataView, HostNumericTensorView, NumericElementType, ProviderRejectionCode,
    SparseFeasibility, SparseFeasibilityQuery, SparseIndexWidth, SparseOperation,
    SparseSpmvRequest, SparseTranspose,
};

use super::super::super::WgpuProvider;
use crate::backend::wgpu::provider::{
    register_test_wgpu_provider, WgpuProviderOptions, WgpuTestSession,
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
    match provider.precision {
        NumericPrecision::F32 => {
            let values = [values[0] as f32, values[1] as f32];
            provider
                .upload_numeric(&HostNumericTensorView {
                    data: HostNumericDataView::F32(&values),
                    shape: &[2],
                    storage: GpuTensorStorage::Real,
                })
                .expect("upload native f32 vector")
        }
        NumericPrecision::F64 => provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F64(&values),
                shape: &[2],
                storage: GpuTensorStorage::Real,
            })
            .expect("upload native f64 vector"),
    }
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
    assert_eq!(snapshot.revision, 2);
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
    assert_eq!(output.shape, vec![rows]);
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
    assert_eq!(output.shape, vec![0]);
    assert!(download_vector(provider, &output).is_empty());
}
