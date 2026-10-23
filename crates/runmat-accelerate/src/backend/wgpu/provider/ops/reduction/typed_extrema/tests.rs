use super::*;
use runmat_accelerate_api::{
    AccelProvider, HostIntegerDataOwned, HostIntegerDataView, HostIntegerTensorView, HostTensorView,
};
use std::error::Error as _;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn route_class_admission_is_pre_effect_and_explicit() {
    assert!(admit_class(GpuTensorStorage::Real, false, false, ExtremaRoute::Global).is_ok());
    assert!(admit_class(
        GpuTensorStorage::Real,
        false,
        true,
        ExtremaRoute::Dimensional
    )
    .is_ok());
    for (storage, logical, integer, route) in [
        (GpuTensorStorage::Real, false, true, ExtremaRoute::Global),
        (GpuTensorStorage::Real, true, false, ExtremaRoute::Global),
        (
            GpuTensorStorage::ComplexInterleaved,
            false,
            false,
            ExtremaRoute::Dimensional,
        ),
    ] {
        let error = admit_class(storage, logical, integer, route).unwrap_err();
        let typed = runmat_accelerate_api::ProviderExecutionError::from(error);
        assert!(typed.is_unsupported());
        assert!(!typed.is_after_effect());
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn unknown_effect_boundary_never_authorizes_fallback() {
    let source = || unsupported_provider_operation("fault after uncertain upload or reshape");
    let unknown = terminal_error(source(), false);
    assert!(!unknown.is_unsupported());
    assert!(!unknown.is_after_effect());
    let observed_allocation = terminal_error(source(), true);
    assert!(!observed_allocation.is_unsupported());
    assert!(observed_allocation.is_after_effect());
}

async fn owner() -> &'static WgpuProvider {
    #[cfg(not(target_arch = "wasm32"))]
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU adapter");
    #[cfg(target_arch = "wasm32")]
    let provider = WgpuProvider::new_async(Default::default())
        .await
        .expect("actual browser WebGPU adapter");
    let provider = Box::leak(Box::new(provider));
    // SAFETY: the actual provider stays live for this process or browser worker.
    unsafe { runmat_accelerate_api::register_device_provider(provider) };
    provider
}

async fn actual_extrema_four_hooks_preserve_owner_and_words() {
    let provider = owner().await;
    let input = provider
        .upload(&HostTensorView {
            data: &[3.0, 1.0, 4.0, 2.0],
            shape: &[2, 2],
        })
        .unwrap();
    let global = ExtremaInputFrame::global(provider, &input).unwrap();
    assert_eq!(global.kind, ReductionExtremaPipelineKind::Global);
    let dim = ExtremaInputFrame::dimensional(provider, &input, 0).unwrap();
    assert_eq!(dim.kind, ReductionExtremaPipelineKind::Dimensional);
    let cache_before = provider.validated_fusion_cache.lock().unwrap().len();
    let min = provider.reduce_min(&input).await.unwrap();
    assert_eq!(provider.download(&min).await.unwrap().data, vec![1.0]);
    assert_eq!(
        provider.validated_fusion_cache.lock().unwrap().len(),
        cache_before + 1
    );
    let hits_before = provider.telemetry_snapshot().fusion_cache_hits;
    let max = provider.reduce_max(&input).await.unwrap();
    assert_eq!(provider.download(&max).await.unwrap().data, vec![4.0]);
    assert!(provider.telemetry_snapshot().fusion_cache_hits > hits_before);
    let minimum = provider.reduce_min_dim(&input, 0).await.unwrap();
    let maximum = provider.reduce_max_dim(&input, 0).await.unwrap();
    assert_eq!(
        provider.download(&minimum.values).await.unwrap().data,
        vec![1.0, 2.0]
    );
    assert_eq!(
        provider.download(&maximum.values).await.unwrap().data,
        vec![3.0, 4.0]
    );
    assert!(!minimum.values.shares_allocation_with(&input));
    assert!(!minimum.values.shares_allocation_with(&minimum.indices));
    assert_eq!(
        provider.download(&input).await.unwrap().data,
        vec![3.0, 1.0, 4.0, 2.0]
    );

    let integer = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::I64(&[i64::MAX, -1, 5, i64::MIN]),
            shape: &[2, 2],
        })
        .unwrap();
    let records_before_refusal = provider.allocations.record_count().unwrap();
    let cache_before_refusal = provider.validated_fusion_cache.lock().unwrap().len();
    let wrong_global = provider
        .typed_global_extrema_exec(&integer, crate::backend::wgpu::types::GlobalReduceOp::Min)
        .await;
    let wrong_global = match wrong_global {
        Ok(_) => panic!("global floating extrema admitted native integer storage"),
        Err(error) => error,
    };
    assert!(wrong_global.is_unsupported());
    assert!(!wrong_global.is_after_effect());
    let bad_dimension = provider
        .typed_dimensional_extrema_exec(
            &integer,
            2,
            crate::backend::wgpu::types::DimReduceExtrema::Min,
        )
        .await;
    let bad_dimension = match bad_dimension {
        Ok(_) => panic!("out-of-rank dimensional extrema passed preflight"),
        Err(error) => error,
    };
    assert!(bad_dimension.is_unsupported());
    assert!(!bad_dimension.is_after_effect());
    assert_eq!(
        provider.allocations.record_count().unwrap(),
        records_before_refusal
    );
    assert_eq!(
        provider.validated_fusion_cache.lock().unwrap().len(),
        cache_before_refusal
    );
    let integer_frame = ExtremaInputFrame::dimensional(provider, &integer, 0).unwrap();
    assert_eq!(
        integer_frame.kind,
        ReductionExtremaPipelineKind::IntegerDimensional
    );
    let before_integer = provider.validated_fusion_cache.lock().unwrap().len();
    let integer_min = provider.reduce_min_dim(&integer, 0).await.unwrap();
    let integer_max = provider.reduce_max_dim(&integer, 0).await.unwrap();
    assert_eq!(
        provider.validated_fusion_cache.lock().unwrap().len(),
        before_integer + 1
    );
    assert_eq!(
        provider
            .download_integer(&integer_min.values)
            .await
            .unwrap()
            .data,
        HostIntegerDataOwned::I64(vec![-1, i64::MIN])
    );
    assert_eq!(
        provider
            .download_integer(&integer_max.values)
            .await
            .unwrap()
            .data,
        HostIntegerDataOwned::I64(vec![i64::MAX, 5])
    );
    assert_eq!(
        provider
            .get_entry_raw(&integer_min.values)
            .unwrap()
            .allocated_bytes,
        16
    );
    assert_eq!(
        provider
            .get_entry_raw(&integer_min.values)
            .unwrap()
            .buffer
            .size(),
        16
    );
    assert_eq!(
        provider.download_integer(&integer).await.unwrap().data,
        HostIntegerDataOwned::I64(vec![i64::MAX, -1, 5, i64::MIN])
    );
    let empty = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::I64(&[]),
            shape: &[0, 3],
        })
        .unwrap();
    let empty_min = provider.reduce_min_dim(&empty, 0).await.unwrap();
    assert_eq!(empty_min.values.shape(), [0, 3]);
    assert_eq!(
        provider
            .get_entry_raw(&empty_min.values)
            .unwrap()
            .buffer
            .size(),
        4
    );
    assert_eq!(
        provider
            .get_entry_raw(&empty_min.values)
            .unwrap()
            .allocated_bytes,
        4
    );
    assert_eq!(
        provider
            .download_integer(&empty_min.values)
            .await
            .unwrap()
            .data,
        HostIntegerDataOwned::I64(vec![])
    );
    empty.allocation_lease().validate_live().unwrap();
    integer.allocation_lease().validate_live().unwrap();
    input.allocation_lease().validate_live().unwrap();
}

async fn actual_allocated_extrema_failures_are_terminal_and_keep_source() {
    let provider = owner().await;
    let input = provider
        .upload(&HostTensorView {
            data: &[3.0, 1.0, 4.0, 2.0],
            shape: &[2, 2],
        })
        .unwrap();
    let observed_bytes = std::cell::Cell::new(0u64);
    let observed_owner = std::cell::Cell::new(0usize);
    #[cfg(not(target_arch = "wasm32"))]
    let observed_weak: std::cell::RefCell<Option<std::sync::Weak<wgpu::Buffer>>> =
        std::cell::RefCell::new(None);
    #[cfg(target_arch = "wasm32")]
    let observed_weak: std::cell::RefCell<Option<std::rc::Weak<wgpu::Buffer>>> =
        std::cell::RefCell::new(None);
    let fault = |buffer: &runmat_accelerate_api::WgpuBufferOwner| -> Result<()> {
        observed_bytes.set(buffer.size());
        observed_owner.set(runmat_accelerate_api::WgpuBufferOwner::as_ptr(buffer) as usize);
        #[cfg(not(target_arch = "wasm32"))]
        let weak = runmat_accelerate_api::WgpuBufferOwner::downgrade(buffer);
        #[cfg(target_arch = "wasm32")]
        let weak = runmat_accelerate_api::WgpuBufferOwner::downgrade(buffer);
        *observed_weak.borrow_mut() = Some(weak);
        Err(anyhow::Error::new(std::io::Error::new(
            std::io::ErrorKind::BrokenPipe,
            "extrema output allocated",
        )))
    };
    let before_download = provider.telemetry_snapshot().download_bytes;
    let before_global_records = provider.allocations.record_count().unwrap();
    let global = provider
        .typed_global_extrema_with_fault(
            &input,
            crate::backend::wgpu::types::GlobalReduceOp::Min,
            Some(&fault),
        )
        .await;
    let global = match global {
        Ok(_) => panic!("injected global allocation fault was lost"),
        Err(error) => error,
    };
    assert!(global.is_after_effect());
    assert!(!global.is_unsupported());
    assert_eq!(observed_bytes.get(), provider.element_size as u64);
    assert_ne!(observed_owner.get(), 0);
    assert!(observed_weak
        .borrow()
        .as_ref()
        .and_then(|weak| weak.upgrade())
        .is_none());
    assert_eq!(
        provider.allocations.record_count().unwrap(),
        before_global_records
    );
    assert_eq!(
        global
            .source()
            .and_then(|source| source.downcast_ref::<std::io::Error>())
            .map(std::io::Error::kind),
        Some(std::io::ErrorKind::BrokenPipe)
    );
    observed_bytes.set(0);
    observed_owner.set(0);
    *observed_weak.borrow_mut() = None;
    let before_float_dim_records = provider.allocations.record_count().unwrap();
    let float_dim = provider
        .typed_dimensional_extrema_with_fault(
            &input,
            0,
            crate::backend::wgpu::types::DimReduceExtrema::Min,
            Some(&fault),
        )
        .await;
    let float_dim = match float_dim {
        Ok(_) => panic!("injected floating dimensional allocation fault was lost"),
        Err(error) => error,
    };
    assert!(float_dim.is_after_effect());
    assert!(!float_dim.is_unsupported());
    assert_eq!(observed_bytes.get(), 2 * provider.element_size as u64);
    assert_ne!(observed_owner.get(), 0);
    assert!(observed_weak
        .borrow()
        .as_ref()
        .and_then(|weak| weak.upgrade())
        .is_none());
    assert_eq!(
        provider.allocations.record_count().unwrap(),
        before_float_dim_records
    );
    assert_eq!(
        float_dim
            .source()
            .and_then(|source| source.downcast_ref::<std::io::Error>())
            .map(std::io::Error::kind),
        Some(std::io::ErrorKind::BrokenPipe)
    );
    let integer = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::I64(&[i64::MAX, -1, 5, i64::MIN]),
            shape: &[2, 2],
        })
        .unwrap();
    observed_bytes.set(0);
    observed_owner.set(0);
    *observed_weak.borrow_mut() = None;
    let before_integer_dim_records = provider.allocations.record_count().unwrap();
    let integer_dim = provider
        .typed_dimensional_extrema_with_fault(
            &integer,
            0,
            crate::backend::wgpu::types::DimReduceExtrema::Max,
            Some(&fault),
        )
        .await;
    let integer_dim = match integer_dim {
        Ok(_) => panic!("injected integer dimensional allocation fault was lost"),
        Err(error) => error,
    };
    assert!(integer_dim.is_after_effect());
    assert!(!integer_dim.is_unsupported());
    assert_eq!(observed_bytes.get(), 16);
    assert_ne!(observed_owner.get(), 0);
    assert!(observed_weak
        .borrow()
        .as_ref()
        .and_then(|weak| weak.upgrade())
        .is_none());
    assert_eq!(
        provider.allocations.record_count().unwrap(),
        before_integer_dim_records
    );
    assert_eq!(
        integer_dim
            .source()
            .and_then(|source| source.downcast_ref::<std::io::Error>())
            .map(std::io::Error::kind),
        Some(std::io::ErrorKind::BrokenPipe)
    );
    assert_eq!(
        provider.telemetry_snapshot().download_bytes,
        before_download
    );
    input.allocation_lease().validate_live().unwrap();
    integer.allocation_lease().validate_live().unwrap();
    assert_eq!(
        provider.download(&input).await.unwrap().data,
        vec![3.0, 1.0, 4.0, 2.0]
    );
    assert_eq!(
        provider.download_integer(&integer).await.unwrap().data,
        HostIntegerDataOwned::I64(vec![i64::MAX, -1, 5, i64::MIN])
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
#[ignore = "requires actual WGPU adapter; no adapter skip"]
async fn native_extrema_four_hooks_preserve_owner_and_words() {
    actual_extrema_four_hooks_preserve_owner_and_words().await;
    actual_allocated_extrema_failures_are_terminal_and_keep_source().await;
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn browser_extrema_four_hooks_preserve_owner_and_words() {
    actual_extrema_four_hooks_preserve_owner_and_words().await;
    actual_allocated_extrema_failures_are_terminal_and_keep_source().await;
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn browser_cold_extrema_cancellation_keeps_borrowed_input() {
    use std::future::Future;
    let provider = owner().await;
    let input = provider
        .upload(&HostTensorView {
            data: &[3.0, 1.0],
            shape: &[1, 2],
        })
        .unwrap();
    let before = provider.validated_fusion_cache.lock().unwrap().len();
    let mut future = Box::pin(provider.reduce_min(&input));
    let pending = std::future::poll_fn(|context| {
        std::task::Poll::Ready(future.as_mut().poll(context).is_pending())
    })
    .await;
    assert!(pending, "cold scoped validation Promise must suspend");
    drop(future);
    assert_eq!(
        provider.validated_fusion_cache.lock().unwrap().len(),
        before
    );
    input.allocation_lease().validate_live().unwrap();
    assert_eq!(
        provider.download(&input).await.unwrap().data,
        vec![3.0, 1.0]
    );
}
