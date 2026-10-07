use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::{
    AccelProvider, HostNumericDataOwned, HostNumericDataView, HostNumericTensorView,
    NumericElementType,
};

fn assert_values(provider: &WgpuProvider, handle: &GpuTensorHandle, expected: &[f64]) {
    let tensor = block_on(provider.download_numeric_exec(handle)).unwrap();
    let values = match tensor.data {
        HostNumericDataOwned::F32(values) => {
            assert_eq!(provider.precision, NumericPrecision::F32);
            values.into_iter().map(f64::from).collect::<Vec<_>>()
        }
        HostNumericDataOwned::F64(values) => {
            assert_eq!(provider.precision, NumericPrecision::F64);
            values
        }
        other => panic!("expected native real floating result, got {other:?}"),
    };
    assert_eq!(values.len(), expected.len());
    for (actual, expected) in values.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
    }
}

fn exercise_lazy_mean_and_moments(precision: NumericPrecision) {
    let provider = WgpuProvider::new(WgpuProviderOptions::default())
        .expect("qualification requires an actual WGPU adapter");
    assert_eq!(
        provider.precision, precision,
        "set RUNMAT_WGPU_FORCE_PRECISION to the test precision; F64 requires SHADER_F64"
    );
    let single = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let double = single.map(f64::from);
    let data = match precision {
        NumericPrecision::F32 => HostNumericDataView::F32(&single),
        NumericPrecision::F64 => HostNumericDataView::F64(&double),
    };
    let original = provider
        .upload_numeric_exec(&HostNumericTensorView {
            data,
            shape: &[2, 3],
            storage: runmat_accelerate_api::GpuTensorStorage::Real,
        })
        .unwrap();
    let lazy = provider.transpose_exec(&original).unwrap();
    assert!(runmat_accelerate_api::handle_is_transposed(&lazy));
    assert!(lazy.shares_allocation_with(&original));
    let original_view = original.view().clone();
    let lazy_view = lazy.view().clone();
    let identity = lazy.view_identity();
    let class = match precision {
        NumericPrecision::F32 => NumericElementType::F32,
        NumericPrecision::F64 => NumericElementType::F64,
    };
    let native_bytes = 6 * class.element_size() as u64;
    assert_eq!(
        provider.download_numeric_host_workspace_bound(&original),
        runmat_execution::ResourceByteBound::Known(native_bytes)
    );
    assert_eq!(
        provider.download_numeric_host_workspace_bound(&lazy),
        runmat_execution::ResourceByteBound::Known(native_bytes * 2)
    );

    for (axis, shape, means, squares) in [
        (0, vec![1, 2], vec![3.0, 4.0], vec![35.0 / 3.0, 56.0 / 3.0]),
        (1, vec![3, 1], vec![1.5, 3.5, 5.5], vec![2.5, 12.5, 30.5]),
    ] {
        let before = provider.telemetry_snapshot().download_bytes;
        let mean = block_on(provider.reduce_mean_nd(&lazy, &[axis])).unwrap();
        assert_eq!(provider.telemetry_snapshot().download_bytes, before);
        assert_eq!(mean.shape(), shape);
        assert_eq!(mean.physical().element_type(), class);
        assert!(!mean.shares_allocation_with(&lazy));
        assert_values(&provider, &mean, &means);

        // A second call must reuse the cache under the original view, not a
        // transient materialized view whose lease has already been released.
        let before = provider.telemetry_snapshot().download_bytes;
        let repeated = block_on(provider.reduce_mean_nd(&lazy, &[axis])).unwrap();
        assert_eq!(provider.telemetry_snapshot().download_bytes, before);
        assert!(repeated.shares_allocation_with(&mean));
        assert!(provider
            .reduction_reuse
            .lock()
            .unwrap()
            .moment_pair(&(identity, vec![axis]))
            .is_some());
        assert_values(&provider, &repeated, &means);

        // Direct moments is also used independently of mean's cache. Both
        // returned allocations must reflect logical order, including E[x²].
        let before = provider.telemetry_snapshot().download_bytes;
        let moments = block_on(provider.reduce_moments_nd(&lazy, &[axis])).unwrap();
        assert_eq!(provider.telemetry_snapshot().download_bytes, before);
        assert!(!moments.mean.shares_allocation_with(&moments.ex2));
        for output in [&moments.mean, &moments.ex2] {
            assert_eq!(output.shape(), shape);
            assert_eq!(output.physical().element_type(), class);
            assert!(!output.shares_allocation_with(&original));
            output.allocation_lease().validate_live().unwrap();
        }
        assert_values(&provider, &moments.mean, &means);
        assert_values(&provider, &moments.ex2, &squares);
        assert_eq!(lazy.view_identity(), identity);
        assert_eq!(lazy.view(), &lazy_view);
        assert_eq!(original.view(), &original_view);
        assert_values(&provider, &original, &double);
        assert_values(&provider, &lazy, &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }
    drop(lazy);
    assert_values(&provider, &original, &double);
}

#[test]
#[ignore = "requires real WGPU adapter; run with RUNMAT_WGPU_FORCE_PRECISION=f32"]
fn actual_lazy_mean_and_moments_f32() {
    exercise_lazy_mean_and_moments(NumericPrecision::F32);
}

#[test]
#[ignore = "requires SHADER_F64 adapter; run with RUNMAT_WGPU_FORCE_PRECISION=f64"]
fn actual_lazy_mean_and_moments_f64() {
    exercise_lazy_mean_and_moments(NumericPrecision::F64);
}

fn exercise_source_retirement_with_cached_pair(poison_cache: bool) {
    let provider = WgpuProvider::new(WgpuProviderOptions::default())
        .expect("qualification requires an actual WGPU adapter");
    assert_eq!(provider.precision, NumericPrecision::F32);
    let baseline_records = provider.allocations.record_count().unwrap();
    let input = provider
        .upload_numeric_exec(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.0, 2.0, 3.0, 4.0]),
            shape: &[2, 2],
            storage: runmat_accelerate_api::GpuTensorStorage::Real,
        })
        .unwrap();
    let input_weak = input.downgrade();
    let mean = block_on(provider.reduce_mean_nd(&input, &[0])).unwrap();
    let (cached_mean, cached_ex2) = provider
        .reduction_reuse
        .lock()
        .unwrap()
        .moment_pair(&(input.view_identity(), vec![0]))
        .expect("native moments pair retained for exact source view");
    let mean_weak = cached_mean.downgrade();
    let ex2_weak = cached_ex2.downgrade();
    drop(mean);
    drop(cached_mean);
    drop(cached_ex2);
    if poison_cache {
        let cache = provider.reduction_reuse.clone();
        assert!(std::thread::spawn(move || {
            let _guard = cache.lock().unwrap();
            panic!("injected optional cache bookkeeping poison");
        })
        .join()
        .is_err());
        assert!(provider.reduction_reuse.is_poisoned());
    }
    drop(input);
    assert!(input_weak.acquire().is_err());
    assert!(mean_weak.acquire().is_err());
    assert!(ex2_weak.acquire().is_err());
    assert_eq!(
        provider.allocations.record_count().unwrap(),
        baseline_records
    );
}

#[test]
#[ignore = "requires actual F32 WGPU adapter; verifies source-driven cache output retirement"]
fn actual_source_retirement_releases_cached_moment_output_owners() {
    exercise_source_retirement_with_cached_pair(false);
}

#[test]
#[ignore = "requires actual F32 WGPU adapter; verifies optional cache poison cannot block retirement"]
fn poisoned_cache_does_not_quarantine_actual_source_or_cached_output_owners() {
    exercise_source_retirement_with_cached_pair(true);
}

#[test]
#[ignore = "requires actual F32 WGPU adapter; verifies exact-view lazy pruning"]
fn dropped_source_view_is_pruned_while_physical_sibling_remains_live() {
    let provider = WgpuProvider::new(WgpuProviderOptions::default())
        .expect("qualification requires an actual WGPU adapter");
    assert_eq!(provider.precision, NumericPrecision::F32);
    let original = provider
        .upload_numeric_exec(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            shape: &[2, 3],
            storage: runmat_accelerate_api::GpuTensorStorage::Real,
        })
        .unwrap();
    let view = provider.transpose_exec(&original).unwrap();
    assert_ne!(view.view_identity(), original.view_identity());
    let view_identity = view.view_identity();
    let mean = block_on(provider.reduce_mean_nd(&view, &[0])).unwrap();
    let weak_mean = mean.downgrade();
    drop(mean);
    drop(view);
    assert!(original.allocation_lease().validate_live().is_ok());
    assert!(
        provider
            .reduction_reuse
            .lock()
            .unwrap()
            .moment_pair(&(view_identity, vec![0]))
            .is_some(),
        "bounded cache retains pair until next insertion or eviction"
    );
    let replacement = block_on(provider.reduce_mean_nd(&original, &[0])).unwrap();
    assert!(
        provider
            .reduction_reuse
            .lock()
            .unwrap()
            .moment_pair(&(view_identity, vec![0]))
            .is_none(),
        "dead exact view was lazily pruned"
    );
    assert!(weak_mean.acquire().is_err());
    assert!(original.allocation_lease().validate_live().is_ok());
    drop(replacement);
}
