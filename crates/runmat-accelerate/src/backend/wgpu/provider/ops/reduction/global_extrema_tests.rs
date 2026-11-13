//! Direct public global reduction hooks; ordinary `max(A)` uses a different
//! dimensional route and would not exercise this shader.

use super::*;
use crate::backend::wgpu::provider::WgpuProviderOptions;
use crate::backend::wgpu::types::GlobalReduceOp;
use crate::backend::wgpu::types::NumericPrecision;
use runmat_accelerate_api::{
    AccelProvider, HostNumericDataOwned, HostNumericDataView, HostNumericTensorView,
    NumericElementType, ProviderPrecision,
};

async fn owner() -> &'static WgpuProvider {
    #[cfg(not(target_arch = "wasm32"))]
    let provider = WgpuProvider::new(WgpuProviderOptions::default()).expect("actual WGPU adapter");
    #[cfg(target_arch = "wasm32")]
    let provider = WgpuProvider::new_async(WgpuProviderOptions::default())
        .await
        .expect("actual WebGPU adapter");
    let provider = Box::leak(Box::new(provider));
    // SAFETY: this unique actual device facade lives for the process/worker.
    unsafe { runmat_accelerate_api::register_device_provider(provider) };
    provider
}

fn upload(provider: &WgpuProvider, values: &[f64]) -> GpuTensorHandle {
    let represented = values.iter().map(|value| *value as f32).collect::<Vec<_>>();
    provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: if provider.precision == NumericPrecision::F32 {
                HostNumericDataView::F32(&represented)
            } else {
                HostNumericDataView::F64(values)
            },
            shape: &[1, values.len()],
            storage: GpuTensorStorage::Real,
        })
        .expect("actual typed global input")
}

async fn check_case(provider: &WgpuProvider, values: &[f64], minimum: bool, expected: f64) {
    let input = upload(provider, values);
    for _ in 0..2 {
        let output = if minimum {
            provider.reduce_min(&input).await
        } else {
            provider.reduce_max(&input).await
        }
        .expect("global extrema hook");
        assert_eq!(output.shape(), &[1, 1]);
        assert!(!output.shares_allocation_with(&input));
        let result = provider.download_numeric(&output).await.unwrap();
        match (provider.precision, result.data) {
            (NumericPrecision::F32, HostNumericDataOwned::F32(actual)) => {
                assert_eq!(output.physical().element_type(), NumericElementType::F32);
                assert_eq!(actual.len(), 1);
                if expected.is_nan() {
                    assert!(actual[0].is_nan());
                } else {
                    assert_eq!(actual[0].to_bits(), (expected as f32).to_bits());
                }
            }
            (NumericPrecision::F64, HostNumericDataOwned::F64(actual)) => {
                assert_eq!(output.physical().element_type(), NumericElementType::F64);
                assert_eq!(actual.len(), 1);
                if expected.is_nan() {
                    assert!(actual[0].is_nan());
                } else {
                    assert_eq!(actual[0].to_bits(), expected.to_bits());
                }
            }
            _ => panic!("global extrema changed result class"),
        }
        input.allocation_lease().validate_live().unwrap();
    }
}

async fn check_count(provider: &WgpuProvider, values: &[f64], expected: f64) {
    let input = upload(provider, values);
    for _ in 0..2 {
        let output = provider
            .reduce_nnz(&input)
            .await
            .expect("public CountNonZero")
            .expect("provider accepted numerical operation");
        assert_eq!(output.shape(), &[1, 1]);
        assert!(!output.shares_allocation_with(&input));
        let result = provider.download_numeric(&output).await.unwrap();
        match (provider.precision, result.data) {
            (NumericPrecision::F32, HostNumericDataOwned::F32(actual)) => {
                assert_eq!(output.physical().element_type(), NumericElementType::F32);
                assert_eq!(actual, vec![expected as f32]);
            }
            (NumericPrecision::F64, HostNumericDataOwned::F64(actual)) => {
                assert_eq!(output.physical().element_type(), NumericElementType::F64);
                assert_eq!(actual, vec![expected]);
            }
            _ => panic!("global CountNonZero changed result class"),
        }
        input.allocation_lease().validate_live().unwrap();
    }
}

async fn qualify(provider: &'static WgpuProvider) {
    let workgroup = provider.global_reduce_launch.workgroup() as usize;
    let len = workgroup * 5 + 13;
    assert!(len > workgroup * 4, "exercise more than two workgroups");

    // The admitted tree must use the same actual native allocator and launch
    // as execution. Singleton results are lawful borrowed scalar views.
    for size in [1, len] {
        let input = upload(provider, &vec![7.0; size]);
        for kind in [
            runmat_accelerate_api::ProviderExtremaKind::Minimum,
            runmat_accelerate_api::ProviderExtremaKind::Maximum,
        ] {
            let recipe =
                runmat_accelerate_api::ProviderGlobalExtremaRecipe::plan(provider, &input, kind)
                    .unwrap()
                    .expect("actual selected global recipe");
            let expected = recipe.physical().expected_observations().unwrap();
            let result = recipe
                .execute()
                .await
                .expect("actual selected global execution");
            assert_eq!(result.observed, expected);
            assert_eq!(result.output.shares_allocation_with(&input), size == 1);
            assert_eq!(
                result.output.physical().element_type(),
                input.physical().element_type()
            );
            let host = provider.download_numeric(&result.output).await.unwrap();
            match host.data {
                HostNumericDataOwned::F32(values) => assert_eq!(values, vec![7.0_f32]),
                HostNumericDataOwned::F64(values) => assert_eq!(values, vec![7.0_f64]),
                _ => panic!("selected global tree changed native class"),
            }
            drop(result);
            input.allocation_lease().validate_live().unwrap();
        }
    }

    let mut values = vec![7.0; len];
    values[workgroup + 3] = f64::NAN;
    values[len - 2] = f64::NAN;
    check_case(provider, &values, false, f64::NAN).await;
    check_case(provider, &values, true, f64::NAN).await;
    if provider.precision == NumericPrecision::F64 {
        values[workgroup + 3] = 7.0;
        // The only NaN mantissa bit is in the low word. Both reduction
        // directions must observe it after an independent workgroup pass.
        values[len - 2] = f64::from_bits(0x7ff0_0000_0000_0001);
        check_case(provider, &values, false, f64::NAN).await;
        check_case(provider, &values, true, f64::NAN).await;
    }

    let mut zeros = vec![-0.0; len];
    zeros[len - 1] = 0.0;
    check_case(provider, &zeros, false, 0.0).await;
    zeros.fill(0.0);
    zeros[len - 1] = -0.0;
    check_case(provider, &zeros, true, -0.0).await;

    let mut infinities = vec![1.0; len];
    infinities[len - 1] = f64::INFINITY;
    check_case(provider, &infinities, false, f64::INFINITY).await;
    infinities[len - 1] = f64::NEG_INFINITY;
    check_case(provider, &infinities, true, f64::NEG_INFINITY).await;

    let smallest_positive = if provider.precision == NumericPrecision::F32 {
        f32::from_bits(1) as f64
    } else {
        f64::from_bits(1)
    };
    let mut subnormal = vec![-1.0; len];
    subnormal[len - 1] = smallest_positive;
    check_case(provider, &subnormal, false, smallest_positive).await;
    subnormal.fill(1.0);
    subnormal[len - 1] = smallest_positive;
    check_case(provider, &subnormal, true, smallest_positive).await;
    subnormal.fill(-1.0);
    subnormal[len - 1] = -smallest_positive;
    check_case(provider, &subnormal, false, -smallest_positive).await;
    subnormal.fill(1.0);
    subnormal[len - 1] = -smallest_positive;
    check_case(provider, &subnormal, true, -smallest_positive).await;
    check_case(provider, &vec![7.0; len], false, 7.0).await;
    check_case(provider, &vec![7.0; len], true, 7.0).await;

    check_case(provider, &[], false, f64::NEG_INFINITY).await;
    check_case(provider, &[], true, f64::INFINITY).await;
    for (source, expected) in [
        (vec![], 0.0),
        (vec![0.0], 0.0),
        (vec![-0.0], 0.0),
        (vec![42.0], 1.0),
        (vec![f64::NAN], 1.0),
    ] {
        check_count(provider, &source, expected).await;
    }

    // The same frozen launch also executes the untouched arithmetic branches.
    let mut arithmetic = vec![1.0; len];
    arithmetic[0] = 0.0;
    arithmetic[len - 1] = 2.0;
    let input = upload(provider, &arithmetic);
    if provider.precision == NumericPrecision::F32 {
        assert!(
            provider
                .reduce_sum(&input)
                .await
                .unwrap_err()
                .is_unsupported(),
            "public F32 Sum retains ordinary Double-accumulation policy"
        );
    }
    for _ in 0..2 {
        for (output, expected) in [
            (
                provider
                    .reduce_global_exec(&input, GlobalReduceOp::Sum)
                    .unwrap(),
                len as f64,
            ),
            (
                provider
                    .reduce_prod(&input)
                    .await
                    .unwrap()
                    .expect("provider accepted numerical operation"),
                0.0,
            ),
            (
                provider
                    .reduce_nnz(&input)
                    .await
                    .unwrap()
                    .expect("provider accepted numerical operation"),
                (len - 1) as f64,
            ),
        ] {
            assert_eq!(output.shape(), &[1, 1]);
            let downloaded = provider.download_numeric(&output).await.unwrap();
            let actual = match downloaded.data {
                HostNumericDataOwned::F32(values) => {
                    assert_eq!(output.physical().element_type(), NumericElementType::F32);
                    assert_eq!(values.len(), 1);
                    f64::from(values[0])
                }
                HostNumericDataOwned::F64(values) => {
                    assert_eq!(output.physical().element_type(), NumericElementType::F64);
                    assert_eq!(values.len(), 1);
                    values[0]
                }
                _ => panic!("global arithmetic reduction changed result class"),
            };
            assert_eq!(actual, expected);
            input.allocation_lease().validate_live().unwrap();
        }
    }
    let mut many = vec![0.0; len];
    many[0] = -1.0;
    many[workgroup + 1] = f64::NAN;
    many[len - 1] = 2.0;
    check_count(provider, &many, 3.0).await;
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(
    not(target_arch = "wasm32"),
    ignore = "requires an actual F32 WGPU adapter"
)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn actual_global_f32_minmax_include_nan_signed_zero_subnormal_and_ties() {
    let provider = owner().await;
    assert_eq!(provider.precision(), ProviderPrecision::F32);
    qualify(provider).await;
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
#[ignore = "requires actual enabled F64 and SHADER_INT64 WGPU hardware; RUNMAT_WGPU_FORCE_PRECISION=f64"]
async fn actual_global_f64_minmax_include_nan_signed_zero_subnormal_and_ties() {
    assert_eq!(
        std::env::var("RUNMAT_WGPU_FORCE_PRECISION").as_deref(),
        Ok("f64"),
        "this required hardware gate must explicitly select F64"
    );
    let provider = owner().await;
    assert_eq!(provider.precision(), ProviderPrecision::F64);
    qualify(provider).await;
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
#[ignore = "requires actual F64 WGPU hardware and RUNMAT_WGPU_FORCE_PRECISION=f64; explicit capability rejection"]
async fn global_double_minmax_reject_missing_enabled_ieee_words_before_effects() {
    assert_eq!(
        std::env::var("RUNMAT_WGPU_FORCE_PRECISION").as_deref(),
        Ok("f64"),
        "this required capability gate must explicitly select F64"
    );
    let mut provider = Box::new(WgpuProvider::new(WgpuProviderOptions::default()).unwrap());
    assert_eq!(provider.precision(), ProviderPrecision::F64);
    provider
        .enabled_features
        .remove(wgpu::Features::SHADER_INT64);
    let provider = Box::leak(provider);
    // SAFETY: this actual device facade lives for the process.
    unsafe { runmat_accelerate_api::register_device_provider(provider) };
    let input = upload(provider, &[1.0, 2.0]);
    for minimum in [true, false] {
        let before = provider.telemetry_snapshot();
        let records = provider.allocations.record_count().unwrap();
        let error = if minimum {
            provider.reduce_min(&input).await
        } else {
            provider.reduce_max(&input).await
        }
        .unwrap_err();
        assert!(error.is_unsupported());
        assert_eq!(provider.telemetry_snapshot(), before);
        assert_eq!(provider.allocations.record_count().unwrap(), records);
        input.allocation_lease().validate_live().unwrap();
    }
}
