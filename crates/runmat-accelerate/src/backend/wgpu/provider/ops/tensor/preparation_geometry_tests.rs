use super::*;
use runmat_accelerate_api::{HostNumericDataOwned, HostNumericDataView, HostNumericTensorView};

#[test]
#[ignore = "requires actual F32 WGPU device; RUNMAT_WGPU_FORCE_PRECISION=f32; includes F32-only repmat and flip routes"]
fn inverse_permutation_of_lazy_transpose_restores_physical_order_without_dispatch() {
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU device required");
    // A buffer can retain a native class without this provider owning a
    // qualified reduction kernel for it. Optional absence must be pre-effect.
    let alternate = provider
        .upload_numeric_exec(&HostNumericTensorView {
            data: if provider.precision == NumericPrecision::F32 {
                HostNumericDataView::F64(&[1.0, 2.0])
            } else {
                HostNumericDataView::F32(&[1.0, 2.0])
            },
            shape: &[2, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let before = provider.telemetry_snapshot();
    assert!(
        futures::executor::block_on(provider.reduce_mean(&alternate))
            .unwrap()
            .is_none()
    );
    assert!(
        futures::executor::block_on(provider.reduce_mean_dim(&alternate, 0))
            .unwrap()
            .is_none()
    );
    if provider.precision == NumericPrecision::F32 {
        assert!(
            futures::executor::block_on(provider.reduce_mean_nd(&alternate, &[0]))
                .unwrap()
                .is_none()
        );
    }
    for axes in [&[][..], &[2][..]] {
        assert!(futures::executor::block_on(provider.reduce_mean_nd(&alternate, axes)).is_err());
    }
    assert!(futures::executor::block_on(provider.reduce_mean_dim(&alternate, 2)).is_err());
    assert_eq!(provider.telemetry_snapshot(), before);
    let shape = [u32::MAX as usize, u32::MAX as usize, 2, 0];
    let empty = provider
        .upload_numeric_exec(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[]),
            shape: &shape,
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let permuted = provider.permute(&empty, &[3, 2, 1, 0]).unwrap().unwrap();
    assert_eq!(
        permuted.shape(),
        [0, 2, u32::MAX as usize, u32::MAX as usize]
    );
    let tiled = provider.repmat(&empty, &[1, 1, 1, 1]).unwrap().unwrap();
    assert_eq!(tiled.shape(), shape);
    let flipped = provider.flip(&empty, &[0, usize::MAX]).unwrap().unwrap();
    assert!(flipped.shares_allocation_with(&empty));
    assert!(
        futures::executor::block_on(provider.reduce_mean_nd(&empty, &[0]))
            .unwrap()
            .is_none()
    );
    let source = provider
        .upload_numeric_exec(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            shape: &[3, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let lazy = provider.transpose_exec(&source).unwrap();
    assert_eq!(lazy.shape(), [2, 3]);
    assert!(runmat_accelerate_api::handle_is_transposed(&lazy));
    provider.reset_telemetry();
    let before = provider.telemetry_snapshot();
    assert!(provider.flip(&lazy, &[0]).unwrap().is_none());
    let unchanged = provider.flip(&source, &[usize::MAX]).unwrap().unwrap();
    assert!(unchanged.shares_allocation_with(&source));
    let inverse = provider.permute(&lazy, &[1, 0]).unwrap().unwrap();
    assert_eq!(inverse.shape(), [3, 2]);
    assert!(!runmat_accelerate_api::handle_is_transposed(&inverse));
    assert!(inverse.shares_allocation_with(&source));
    assert_eq!(provider.telemetry_snapshot(), before);
    let downloaded = futures::executor::block_on(provider.download_numeric(&inverse)).unwrap();
    assert_eq!(
        downloaded.data,
        HostNumericDataOwned::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    );
    // The public optional hooks reject unsupported geometry before allocation,
    // while the actual physical routine owns post-allocation failure cleanup.
    let before = provider.telemetry_snapshot();
    let order = (0..crate::backend::wgpu::params::PERMUTE_MAX_RANK + 1).collect::<Vec<_>>();
    assert!(provider.permute(&source, &order).unwrap().is_none());
    assert!(provider
        .repmat(
            &source,
            &vec![1; crate::backend::wgpu::params::REPMAT_MAX_RANK + 1]
        )
        .unwrap()
        .is_none());
    assert_eq!(provider.telemetry_snapshot(), before);
    for stage in 0..3 {
        let allocated = std::cell::Cell::new(false);
        let retired = std::cell::RefCell::new(None);
        let fault = |buffer: &runmat_accelerate_api::WgpuBufferOwner| -> Result<()> {
            *retired.borrow_mut() = Some(runmat_accelerate_api::WgpuBufferOwner::downgrade(buffer));
            Err(runmat_accelerate_api::unsupported_provider_operation(
                "request-local structural allocation fault",
            ))
        };
        let result = match stage {
            0 => provider.permute_optional_exec(&source, &[1, 0], &allocated, &fault),
            1 => provider.repmat_optional_exec(&source, &[2, 1], &allocated, &fault),
            _ => provider.flip_optional_exec(&source, &[0], &allocated, &fault),
        };
        let error = result
            .map_err(|source| {
                let source = runmat_accelerate_api::ProviderExecutionError::from(source);
                if allocated.get() {
                    source.after_effect()
                } else {
                    source.with_unproven_effects()
                }
            })
            .unwrap_err();
        assert!(allocated.get());
        assert!(error.is_after_effect());
        assert!(!error.is_unsupported());
        assert!(retired.borrow().as_ref().unwrap().upgrade().is_none());
        source.allocation_lease().validate_live().unwrap();
    }
}

#[test]
#[ignore = "requires actual WGPU device; execute explicitly for scalar lazy-view semantics"]
fn scalar_max_retains_lazy_transpose_coordinates() {
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU device required");
    let source = provider
        .upload_numeric_exec(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]),
            shape: &[3, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let lazy = provider.transpose_exec(&source).unwrap();
    let result = provider
        .scalar_op_exec(crate::backend::wgpu::types::ScalarOpCode::Max, &lazy, 0.0)
        .unwrap();
    assert_eq!(result.shape(), [2, 3]);
    assert!(runmat_accelerate_api::handle_is_transposed(&result));
    let downloaded = futures::executor::block_on(provider.download_numeric(&result)).unwrap();
    assert_eq!(
        downloaded.data,
        HostNumericDataOwned::F32(vec![0.0, 4.0, 2.0, 0.0, 0.0, 6.0])
    );
}

#[test]
#[ignore = "requires actual WGPU device and RUNMAT_WG=64; execute explicitly"]
fn non512_lazy_transpose_materialization_covers_chunk_boundary() {
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU device required");
    assert_eq!(
        provider.workgroup_config.scalar, 64,
        "run with RUNMAT_WG=64"
    );
    let capacity = crate::backend::wgpu::config::elementwise_chunk_capacity(64).unwrap() as usize;
    let columns = capacity / 2 + 1;
    let values: Vec<f32> = (0..2 * columns).map(|i| (i % 1024) as f32).collect();
    let source = provider
        .upload_numeric_exec(&HostNumericTensorView {
            data: HostNumericDataView::F32(&values),
            shape: &[2, columns],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let transposed = provider.transpose_exec(&source).unwrap();
    assert!(runmat_accelerate_api::handle_is_transposed(&transposed));
    let output = provider.reshape_exec(&transposed, &[columns, 2]).unwrap();
    assert!(!runmat_accelerate_api::handle_is_transposed(&output));
    assert_eq!(output.shape(), [columns, 2]);
    let downloaded = futures::executor::block_on(provider.download_numeric(&output)).unwrap();
    let HostNumericDataOwned::F32(actual) = downloaded.data else {
        panic!("materialization changed physical storage");
    };
    assert_eq!(actual.len(), values.len());
    for (index, actual) in actual.iter().enumerate() {
        assert_eq!(
            *actual,
            values[index / columns + 2 * (index % columns)],
            "logical index {index}"
        );
    }
    output.release().unwrap();
    if transposed.buffer_id() != source.buffer_id() {
        transposed.release().unwrap();
    }
    source.release().unwrap();
}
