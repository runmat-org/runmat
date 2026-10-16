//! Scalar owner qualification only: Runtime public Max can select binary Max,
//! whose include-NaN/positive-zero policy is still a separate prerequisite.
use super::*;
use crate::backend::wgpu::provider::WgpuProviderOptions;
use crate::backend::wgpu::types::ScalarOpCode;
use runmat_accelerate_api::{
    AccelProvider, HostNumericDataOwned, HostNumericDataView, HostNumericTensorView,
    NumericElementType, ProviderPrecision,
};
use runmat_value::{Tensor, Value};

async fn owner() -> &'static WgpuProvider {
    #[cfg(not(target_arch = "wasm32"))]
    let provider = WgpuProvider::new(WgpuProviderOptions::default()).expect("actual WGPU adapter");
    #[cfg(target_arch = "wasm32")]
    let provider = WgpuProvider::new_async(WgpuProviderOptions::default())
        .await
        .expect("actual WebGPU adapter");
    let provider = Box::leak(Box::new(provider));
    // SAFETY: the uniquely created actual device facade lives for the process.
    unsafe { runmat_accelerate_api::register_device_provider(provider) };
    provider
}

async fn host_max(values: &[f64], scalar: f64, single: bool, scalar_left: bool) -> Vec<f64> {
    #[cfg(target_arch = "wasm32")]
    runmat_runtime::builtins::wasm_registry::register_all();
    let tensor = |values: Vec<f64>, shape| {
        Value::Tensor(if single {
            Tensor::from_f32(values.into_iter().map(|v| v as f32).collect(), shape).unwrap()
        } else {
            Tensor::new(values, shape).unwrap()
        })
    };
    let array = tensor(values.to_vec(), vec![values.len(), 1]);
    let control = tensor(vec![scalar], vec![1, 1]);
    let args = if scalar_left {
        [control, array]
    } else {
        [array, control]
    };
    let expected = runmat_runtime::call_builtin_async("max", &args)
        .await
        .unwrap();
    let Value::Tensor(expected) = expected.value() else {
        panic!("host Max tensor oracle")
    };
    if single {
        expected
            .as_f32_slice()
            .unwrap()
            .iter()
            .copied()
            .map(f64::from)
            .collect()
    } else {
        expected.as_f64_slice().unwrap().to_vec()
    }
}

fn assert_values(actual: &[f64], expected: &[f64], single: bool) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &e) in actual.iter().zip(expected) {
        if e.is_nan() {
            assert!(
                a.is_nan(),
                "include-NaN policy must propagate either operand"
            );
        } else if single {
            assert_eq!((a as f32).to_bits(), (e as f32).to_bits());
        } else {
            assert_eq!(a.to_bits(), e.to_bits());
        }
    }
}

async fn qualify(provider: &'static WgpuProvider, single: bool) {
    let values = [
        f64::NAN,
        -0.0,
        0.0,
        f64::NEG_INFINITY,
        f64::INFINITY,
        -3.0,
        3.0,
        1.0,
        f64::from(f32::from_bits(1)),
        -f64::from(f32::from_bits(1)),
        f64::from_bits(1),
        -f64::from_bits(1),
    ];
    let singles: Vec<f32> = values.iter().map(|v| *v as f32).collect();
    let input = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: if single {
                HostNumericDataView::F32(&singles)
            } else {
                HostNumericDataView::F64(&values)
            },
            shape: &[values.len(), 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let precision = if single {
        NumericPrecision::F32
    } else {
        NumericPrecision::F64
    };
    let key = provider
        .real_scalar_pipeline_for_operation(ScalarOpCode::Max, precision)
        .unwrap()
        .scalar_launch()
        .unwrap()
        .key();
    assert!(!provider
        .real_scalar_pipeline_for_operation(ScalarOpCode::Max, precision)
        .unwrap()
        .is_initialized());
    assert_ne!(
        key,
        provider
            .real_scalar_pipeline(precision)
            .unwrap()
            .scalar_launch()
            .unwrap()
            .key()
    );
    let smallest = if single {
        f64::from(f32::from_bits(1))
    } else {
        f64::from_bits(1)
    };
    for scalar in [
        3.0,
        -0.0,
        0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        smallest,
        -smallest,
    ] {
        let expected = host_max(&values, scalar, single, false).await;
        let reversed = host_max(&values, scalar, single, true).await;
        assert_values(&reversed, &expected, single);
        // First call uses a fresh lazy Max bundle; the second reuses it.
        let mut previous = None;
        for _ in 0..2 {
            let observed = RefCell::new(RealObserved::default());
            let output = provider
                .scalar_op_exec_recorded(ScalarOpCode::Max, &input, scalar, Some(&observed), None)
                .unwrap();
            let observed = observed.into_inner();
            assert!(observed.effect);
            assert_eq!(
                observed.resources.output_buffer_bytes,
                Some(output.physical().requested_bytes())
            );
            assert_eq!(observed.resources.uniform_allocations, 1);
            assert_eq!(observed.resources.uniform_buffer_bytes, 256);
            assert_eq!(observed.resources.dispatches, 1);
            assert_eq!(observed.resources.queue_submissions, 1);
            if let Some(previous) = &previous {
                assert_eq!(previous, &observed.resources);
            }
            previous = Some(observed.resources);
            assert_eq!(output.shape(), input.shape());
            assert!(!output.shares_allocation_with(&input));
            assert_eq!(output.physical().storage(), GpuTensorStorage::Real);
            assert_eq!(
                output.physical().element_type(),
                if single {
                    NumericElementType::F32
                } else {
                    NumericElementType::F64
                }
            );
            let downloaded = provider.download_numeric(&output).await.unwrap();
            let actual = match downloaded.data {
                HostNumericDataOwned::F32(v) => v.into_iter().map(f64::from).collect::<Vec<_>>(),
                HostNumericDataOwned::F64(v) => v,
                _ => panic!("real floating Max output"),
            };
            assert_values(&actual, &expected, single);
            let retired = output.downgrade();
            drop(output);
            assert!(retired.acquire().is_err());
            input.allocation_lease().validate_live().unwrap();
        }
        let direct_handle = provider.scalar_max(&input, scalar).unwrap();
        let direct = provider
            .download_numeric(&direct_handle)
            .await
            .unwrap()
            .data;
        let direct = match direct {
            HostNumericDataOwned::F32(v) => v.into_iter().map(f64::from).collect::<Vec<_>>(),
            HostNumericDataOwned::F64(v) => v,
            _ => panic!("ordinary scalar Max changed class"),
        };
        assert_values(&direct, &expected, single);
        let retired = direct_handle.downgrade();
        drop(direct_handle);
        assert!(retired.acquire().is_err());
        input.allocation_lease().validate_live().unwrap();
        assert_eq!(
            key,
            provider
                .real_scalar_pipeline_for_operation(ScalarOpCode::Max, precision)
                .unwrap()
                .scalar_launch()
                .unwrap()
                .key()
        );
    }
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(
    not(target_arch = "wasm32"),
    ignore = "requires actual WGPU adapter; tests actual physical F32 on either provider precision"
)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn actual_scalar_max_f32_cold_warm_matches_host_nan_zero_infinity_and_ties() {
    let provider = owner().await;
    qualify(provider, true).await;
    let input = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.0]),
            shape: &[1, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let before = provider.telemetry_snapshot();
    let allocations = provider.allocations.record_count().unwrap();
    let error = provider.scalar_max(&input, 1.0 + f64::EPSILON).unwrap_err();
    assert!(runmat_accelerate_api::is_unsupported_provider_operation(
        &error
    ));
    assert_eq!(before, provider.telemetry_snapshot());
    assert_eq!(allocations, provider.allocations.record_count().unwrap());
    input.allocation_lease().validate_live().unwrap();
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
#[ignore = "requires actual enabled SHADER_F64 and SHADER_INT64; not qualified on F32-only hardware"]
async fn actual_scalar_max_f64_cold_warm_matches_host_ieee_policy() {
    let provider = owner().await;
    assert_eq!(provider.precision(), ProviderPrecision::F64);
    assert!(provider
        .enabled_features
        .contains(wgpu::Features::SHADER_F64 | wgpu::Features::SHADER_INT64));
    qualify(provider, false).await;
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(
    not(target_arch = "wasm32"),
    ignore = "requires actual F32-only WGPU adapter; RUNMAT_WGPU_FORCE_PRECISION=f32"
)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn unavailable_f64_max_policy_declines_before_allocation_or_submission() {
    let provider = owner().await;
    assert_eq!(provider.precision(), ProviderPrecision::F32);
    let before = provider.telemetry_snapshot();
    let allocations = provider.allocations.record_count().unwrap();
    let error = provider
        .real_scalar_pipeline_for_operation(ScalarOpCode::Max, NumericPrecision::F64)
        .err()
        .unwrap();
    assert!(runmat_accelerate_api::is_unsupported_provider_operation(
        &error
    ));
    assert_eq!(before, provider.telemetry_snapshot());
    assert_eq!(allocations, provider.allocations.record_count().unwrap());
}
