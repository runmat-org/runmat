use super::*;
use crate::backend::wgpu::provider::WgpuProviderOptions;
use crate::backend::wgpu::types::NumericPrecision;
use futures::executor::block_on;
use runmat_accelerate_api::{
    AccelProvider, HostNumericDataOwned, HostNumericDataView, HostNumericTensorView,
    NumericElementType,
};

#[test]
fn successive_rank_n_axes_keep_each_call_boundary() {
    let first = NdReductionGeometry::new(&[2, 3, 4], &[1], "mean").unwrap();
    assert_eq!(first.out_shape, [2, 1, 4]);
    assert_eq!(first.rows, 3);
    let second = NdReductionGeometry::new(&first.out_shape, &[2], "mean").unwrap();
    assert_eq!(second.out_shape, [2, 1, 1]);
    assert_eq!(second.rows, 4);
    let reverse = NdReductionGeometry::new(&[2, 3, 4], &[2], "mean").unwrap();
    assert_eq!(reverse.out_shape, [2, 3, 1]);
    assert_eq!(reverse.rows, 4);
    let fourth = NdReductionGeometry::new(&[2, 3, 4, 5], &[3], "mean").unwrap();
    assert_eq!(fourth.out_shape, [2, 3, 4, 1]);
    assert_eq!(fourth.rows, 5);
    assert!(NdReductionGeometry::new(&[2, 0, 4], &[1], "mean").is_err());
    assert!(NdReductionGeometry::new(&[2, 3, 4], &[3], "mean").is_err());
}

fn actual_nested_rank_three(precision: NumericPrecision) {
    let provider = WgpuProvider::new(WgpuProviderOptions::default())
        .expect("qualification requires a real WGPU adapter");
    assert_eq!(
        provider.precision, precision,
        "set RUNMAT_WGPU_FORCE_PRECISION to the required precision"
    );
    let single: [f32; 24] = std::array::from_fn(|index| index as f32);
    let double = single.map(f64::from);
    let data = match precision {
        NumericPrecision::F32 => HostNumericDataView::F32(&single),
        NumericPrecision::F64 => HostNumericDataView::F64(&double),
    };
    let input = provider
        .upload_numeric_exec(&HostNumericTensorView {
            data,
            shape: &[2, 3, 4],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let input_weak = input.downgrade();
    let before_download = provider.telemetry_snapshot().download_bytes;
    let cold = provider
        .prepare_cold_mean_nd(&input, &[1])
        .unwrap()
        .expect("rank-N source and current limits have a selected cold route");
    assert_eq!(cold.output_shape(), [2, 1, 4]);
    assert_eq!(
        cold.requested_pair_bytes(),
        Some(
            2 * 8
                * match precision {
                    NumericPrecision::F32 => 4,
                    NumericPrecision::F64 => 8,
                }
        )
    );
    assert!(cold.requested_uniform_bytes() > 0);
    assert!(cold.validate_live_source(&input).unwrap());
    drop(cold);
    let mu_height = block_on(provider.reduce_mean_dim(&input, 1))
        .unwrap()
        .expect("admitted mean");
    assert_eq!(mu_height.shape(), [2, 1, 4]);
    let warm = block_on(provider.reduce_mean_dim(&input, 1))
        .unwrap()
        .expect("admitted mean");
    assert!(warm.shares_allocation_with(&mu_height));
    let mu = block_on(provider.reduce_mean_dim(&mu_height, 2))
        .unwrap()
        .expect("admitted mean");
    assert_eq!(mu.shape(), [2, 1, 1]);
    let width_first = block_on(provider.reduce_mean_dim(&input, 2))
        .unwrap()
        .expect("admitted mean");
    assert_eq!(width_first.shape(), [2, 3, 1]);
    let reverse = block_on(provider.reduce_mean_dim(&width_first, 1))
        .unwrap()
        .expect("admitted mean");
    assert_eq!(reverse.shape(), [2, 1, 1]);
    assert_eq!(
        provider.telemetry_snapshot().download_bytes,
        before_download
    );
    let element = match precision {
        NumericPrecision::F32 => NumericElementType::F32,
        NumericPrecision::F64 => NumericElementType::F64,
    };
    for result in [&mu_height, &mu, &width_first, &reverse] {
        assert_eq!(result.physical().element_type(), element);
        result.allocation_lease().validate_live().unwrap();
        assert!(!result.shares_allocation_with(&input));
    }
    assert!(!mu_height.shares_allocation_with(&mu));
    for result in [&mu, &reverse] {
        let values = block_on(provider.download_numeric_exec(result)).unwrap();
        match (precision, values.data) {
            (NumericPrecision::F32, HostNumericDataOwned::F32(values)) => {
                assert_eq!(values, [11.0, 12.0])
            }
            (NumericPrecision::F64, HostNumericDataOwned::F64(values)) => {
                assert_eq!(values, [11.0, 12.0])
            }
            other => panic!("expected native floating mean, got {other:?}"),
        }
    }
    let cached = provider
        .reserve_cached_mean_nd(&input, &[1])
        .unwrap()
        .expect("first dimensional call retains an exact-view pair");
    assert!(cached.mean().shares_allocation_with(&mu_height));
    assert!(cached.retained_pair_bytes() >= 2 * 8 * element.element_size() as u64);
    let foreign = WgpuProvider::new(WgpuProviderOptions::default()).unwrap();
    assert!(block_on(foreign.reduce_mean_dim(&input, 1)).is_err());
    assert!(block_on(provider.reduce_mean_dim(&input, 3)).is_err());
    input.allocation_lease().validate_live().unwrap();
    drop(cached);
    drop(warm);
    drop(reverse);
    drop(width_first);
    drop(mu);
    drop(mu_height);
    drop(input);
    assert!(input_weak.acquire().is_err());
}

fn actual_rank_four(precision: NumericPrecision) {
    let provider = WgpuProvider::new(WgpuProviderOptions::default())
        .expect("qualification requires a real WGPU adapter");
    assert_eq!(provider.precision, precision);
    let single: [f32; 48] = std::array::from_fn(|index| index as f32);
    let double = single.map(f64::from);
    let data = match precision {
        NumericPrecision::F32 => HostNumericDataView::F32(&single),
        NumericPrecision::F64 => HostNumericDataView::F64(&double),
    };
    let input = provider
        .upload_numeric_exec(&HostNumericTensorView {
            data,
            shape: &[2, 3, 4, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let result = block_on(provider.reduce_mean_dim(&input, 3))
        .unwrap()
        .expect("admitted mean");
    assert_eq!(result.shape(), [2, 3, 4, 1]);
    let downloaded = block_on(provider.download_numeric_exec(&result)).unwrap();
    match (precision, downloaded.data) {
        (NumericPrecision::F32, HostNumericDataOwned::F32(values)) => {
            assert_eq!(values, (12..36).map(|n| n as f32).collect::<Vec<_>>())
        }
        (NumericPrecision::F64, HostNumericDataOwned::F64(values)) => {
            assert_eq!(values, (12..36).map(f64::from).collect::<Vec<_>>())
        }
        other => panic!("expected native floating rank-four mean, got {other:?}"),
    }
    assert!(!result.shares_allocation_with(&input));
    input.allocation_lease().validate_live().unwrap();
}

#[test]
#[ignore = "requires real F32 WGPU adapter; RUNMAT_WGPU_FORCE_PRECISION=f32"]
fn actual_rank_three_nested_f32() {
    actual_nested_rank_three(NumericPrecision::F32);
    actual_rank_four(NumericPrecision::F32);
}

#[test]
#[ignore = "requires SHADER_F64 WGPU adapter; RUNMAT_WGPU_FORCE_PRECISION=f64"]
fn actual_rank_three_nested_f64() {
    actual_nested_rank_three(NumericPrecision::F64);
    actual_rank_four(NumericPrecision::F64);
}

#[test]
#[ignore = "requires a real shader-f64 WGPU adapter executing physical F32 N-D moments"]
fn actual_f64_owner_preserves_f32_rank_n_mean_and_hidden_pair() {
    let mut provider = WgpuProvider::new(WgpuProviderOptions::default())
        .expect("qualification requires a real shader-f64 WGPU adapter");
    assert_eq!(provider.precision, NumericPrecision::F64);
    let values: [f32; 24] = std::array::from_fn(|index| index as f32);
    let input = provider
        .upload_numeric_exec(&HostNumericTensorView {
            data: HostNumericDataView::F32(&values),
            shape: &[2, 3, 4],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    assert!(provider.get_entry(&input).is_err());
    let before_download = provider.telemetry_snapshot().download_bytes;
    let cold = provider
        .prepare_cold_mean_nd(&input, &[1])
        .unwrap()
        .expect("physical F32 cold route must be selected on shader-f64 owner");
    assert_eq!(cold.output_shape(), [2, 1, 4]);
    assert_eq!(cold.requested_pair_bytes(), Some(64));
    assert!(cold.validate_live_source(&input).unwrap());
    drop(cold);
    let first = block_on(provider.reduce_mean_dim(&input, 1))
        .unwrap()
        .expect("admitted mean");
    let pair = provider
        .reserve_cached_mean_nd(&input, &[1])
        .unwrap()
        .expect("one actual F32 moments dispatch retains mean and ex2");
    assert!(pair.mean().shares_allocation_with(&first));
    assert_eq!(
        pair.ex2().physical().element_type(),
        NumericElementType::F32
    );
    assert!(!pair.ex2().shares_allocation_with(&input));
    assert!(!pair.ex2().shares_allocation_with(&first));
    let second = block_on(provider.reduce_mean_dim(&first, 2))
        .unwrap()
        .expect("admitted mean");
    let combined = block_on(provider.reduce_mean_nd(&input, &[1, 2]))
        .unwrap()
        .expect("admitted mean");
    let unsorted_direct = block_on(provider.reduce_mean_nd(&input, &[2, 1]))
        .unwrap()
        .expect("admitted mean");
    assert!(provider
        .prepare_cold_mean_nd(&input, &[2, 1])
        .unwrap()
        .is_none());
    assert!(provider
        .reserve_cached_mean_nd(&input, &[2, 1])
        .unwrap()
        .is_none());
    assert_eq!(second.shape(), [2, 1, 1]);
    assert_eq!(combined.shape(), [2, 1, 1]);
    assert_eq!(
        provider.telemetry_snapshot().download_bytes,
        before_download
    );
    for output in [&first, &second, &combined, &unsorted_direct] {
        assert_eq!(output.physical().element_type(), NumericElementType::F32);
        output.allocation_lease().validate_live().unwrap();
        assert!(!output.shares_allocation_with(&input));
    }
    for output in [&second, &combined, &unsorted_direct] {
        assert_eq!(
            block_on(provider.download_numeric_exec(output))
                .unwrap()
                .data,
            HostNumericDataOwned::F32(vec![11.0, 12.0])
        );
    }
    let foreign = WgpuProvider::new(WgpuProviderOptions::default()).unwrap();
    assert!(block_on(foreign.reduce_mean_dim(&input, 1)).is_err());
    let original_limit = provider.adapter_limits.max_buffer_size;
    provider.adapter_limits.max_buffer_size = 16;
    assert!(provider
        .prepare_cold_mean_nd(&input, &[1])
        .unwrap()
        .is_none());
    provider.adapter_limits.max_buffer_size = original_limit;
    input.allocation_lease().validate_live().unwrap();
    drop(pair);
}
