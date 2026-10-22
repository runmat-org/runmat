#![cfg(feature = "wgpu")]

use runmat_accelerate::backend::wgpu::provider::{WgpuProvider, WgpuProviderOptions};
use runmat_accelerate_api::{
    allocation::AllocationReleaseOutcome, AccelProvider, GpuTensorHandle, GpuTensorStorage,
    HostNumericDataOwned, HostNumericDataView, HostNumericTensorView, NumericElementType,
    ProviderNanMode, ProviderPrecision, ProviderScanDirection,
};

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

fn upload_edge_values(provider: &WgpuProvider, precision: ProviderPrecision) -> GpuTensorHandle {
    match precision {
        ProviderPrecision::F32 => {
            let values = [
                2.0f32,
                f32::from_bits(0x7fc0_1234),
                f32::from_bits(0x7f80_0001),
                f32::from_bits(0xffc0_5678),
                f32::from_bits(0xff80_0001),
                1.0,
            ];
            provider
                .upload_numeric(&HostNumericTensorView {
                    data: HostNumericDataView::F32(&values),
                    shape: &[6, 1],
                    storage: GpuTensorStorage::Real,
                })
                .expect("upload real F32 NaN payloads")
        }
        ProviderPrecision::F64 => {
            let values = [
                2.0f64,
                f64::from_bits(0x7ff8_0000_0000_0001),
                f64::from_bits(0x7ff0_0000_0000_0001),
                f64::from_bits(0xfff8_0000_0000_0001),
                f64::from_bits(0xfff0_0000_0000_0001),
                1.0,
            ];
            provider
                .upload_numeric(&HostNumericTensorView {
                    data: HostNumericDataView::F64(&values),
                    shape: &[6, 1],
                    storage: GpuTensorStorage::Real,
                })
                .expect("upload real F64 low-word NaN payloads")
        }
    }
}

fn upload_finite_ties(provider: &WgpuProvider, precision: ProviderPrecision) -> GpuTensorHandle {
    match precision {
        ProviderPrecision::F32 => {
            let values = [2.0f32, 2.0, f32::from_bits(0x7fc0_1234), 1.0, 2.0];
            provider
                .upload_numeric(&HostNumericTensorView {
                    data: HostNumericDataView::F32(&values),
                    shape: &[5, 1],
                    storage: GpuTensorStorage::Real,
                })
                .expect("upload real F32 repeated finite values")
        }
        ProviderPrecision::F64 => {
            let values = [2.0f64, 2.0, f64::from_bits(0x7ff8_0000_0000_0001), 1.0, 2.0];
            provider
                .upload_numeric(&HostNumericTensorView {
                    data: HostNumericDataView::F64(&values),
                    shape: &[5, 1],
                    storage: GpuTensorStorage::Real,
                })
                .expect("upload real F64 repeated finite values")
        }
    }
}

fn upload_nan_and_zero(provider: &WgpuProvider, precision: ProviderPrecision) -> GpuTensorHandle {
    match precision {
        ProviderPrecision::F32 => {
            let values = [f32::from_bits(0xffc0_5678), 0.0f32];
            provider
                .upload_numeric(&HostNumericTensorView {
                    data: HostNumericDataView::F32(&values),
                    shape: &[2, 1],
                    storage: GpuTensorStorage::Real,
                })
                .unwrap()
        }
        ProviderPrecision::F64 => {
            let values = [f64::from_bits(0xfff8_0000_0000_0001), 0.0f64];
            provider
                .upload_numeric(&HostNumericTensorView {
                    data: HostNumericDataView::F64(&values),
                    shape: &[2, 1],
                    storage: GpuTensorStorage::Real,
                })
                .unwrap()
        }
    }
}

async fn read_values(provider: &WgpuProvider, handle: &GpuTensorHandle) -> Vec<f64> {
    let expected_class = match provider.precision() {
        ProviderPrecision::F32 => NumericElementType::F32,
        ProviderPrecision::F64 => NumericElementType::F64,
    };
    assert_eq!(handle.descriptor().element_type, Some(expected_class));
    assert_eq!(handle.descriptor().storage, Some(GpuTensorStorage::Real));
    match provider.download_numeric(handle).await.unwrap().data {
        HostNumericDataOwned::F32(values) => values.into_iter().map(f64::from).collect(),
        HostNumericDataOwned::F64(values) => values,
        other => panic!("unexpected reduction/scan numeric output: {other:?}"),
    }
}

fn assert_series(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        if expected.is_nan() {
            assert!(
                actual.is_nan(),
                "element {index}: expected NaN, got {actual}"
            );
        } else {
            assert_eq!(actual.to_bits(), expected.to_bits(), "element {index}");
        }
    }
}

fn retire(handle: GpuTensorHandle) {
    let weak = handle.allocation_lease().downgrade();
    assert_eq!(handle.release().unwrap(), AllocationReleaseOutcome::Retired);
    assert!(weak.acquire().is_none());
}

async fn verify_logical_nan_mask_global_any_all_ranks(
    provider: &WgpuProvider,
    precision: ProviderPrecision,
) {
    for (shape, values, expected_any, expected_all) in [
        (vec![], vec![0.0], 0.0, 0.0),
        (vec![], vec![f64::NAN], 1.0, 1.0),
        (vec![4], vec![0.0, 1.0, f64::NAN, 2.0], 1.0, 0.0),
        (vec![2, 1, 2], vec![0.0, 1.0, 2.0, 3.0], 0.0, 0.0),
        (vec![2, 1, 2], vec![1.0, 2.0, 3.0, 4.0], 0.0, 1.0),
        (vec![2, 1, 2], vec![0.0, 1.0, f64::NAN, 3.0], 1.0, 0.0),
        (vec![0, 2, 3], vec![], 0.0, 1.0),
    ] {
        let input = match precision {
            ProviderPrecision::F32 => {
                let values = values.iter().map(|value| *value as f32).collect::<Vec<_>>();
                provider
                    .upload_numeric(&HostNumericTensorView {
                        data: HostNumericDataView::F32(&values),
                        shape: &shape,
                        storage: GpuTensorStorage::Real,
                    })
                    .unwrap()
            }
            ProviderPrecision::F64 => provider
                .upload_numeric(&HostNumericTensorView {
                    data: HostNumericDataView::F64(&values),
                    shape: &shape,
                    storage: GpuTensorStorage::Real,
                })
                .unwrap(),
        };
        let mask = provider.logical_isnan(&input).unwrap();
        assert!(mask.view().is_logical());
        assert_eq!(mask.shape(), shape);
        let any = provider.reduce_any(&mask, false).await.unwrap();
        assert_eq!(any.shape(), [1, 1]);
        assert!(any.view().is_logical());
        assert!(!any.shares_allocation_with(&mask));
        assert!(!any.shares_allocation_with(&input));
        assert_series(&read_values(provider, &any).await, &[expected_any]);
        let all_include = provider.reduce_all(&input, false).await.unwrap();
        let all_omit = provider.reduce_all(&input, true).await.unwrap();
        for all in [&all_include, &all_omit] {
            assert_eq!(all.shape(), [1, 1]);
            assert!(all.view().is_logical());
            assert!(!all.shares_allocation_with(&input));
            assert_series(&read_values(provider, all).await, &[expected_all]);
        }
        retire(any);
        retire(all_include);
        retire(all_omit);
        retire(mask);
        input.allocation_lease().validate_live().unwrap();
        retire(input);
    }
    let nan_and_zero = match precision {
        ProviderPrecision::F32 => provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&[f32::NAN, 0.0]),
                shape: &[2],
                storage: GpuTensorStorage::Real,
            })
            .unwrap(),
        ProviderPrecision::F64 => provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F64(&[f64::NAN, 0.0]),
                shape: &[2],
                storage: GpuTensorStorage::Real,
            })
            .unwrap(),
    };
    let include = provider.reduce_any(&nan_and_zero, false).await.unwrap();
    let omit = provider.reduce_any(&nan_and_zero, true).await.unwrap();
    assert!(include.view().is_logical());
    assert!(omit.view().is_logical());
    assert_series(&read_values(provider, &include).await, &[1.0]);
    assert_series(&read_values(provider, &omit).await, &[0.0]);
    retire(include);
    retire(omit);
    retire(nan_and_zero);

    let input = match precision {
        ProviderPrecision::F32 => provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&[0.0_f32, 1.0, 2.0, 3.0]),
                shape: &[2, 2],
                storage: GpuTensorStorage::Real,
            })
            .unwrap(),
        ProviderPrecision::F64 => provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F64(&[0.0_f64, 1.0, 2.0, 3.0]),
                shape: &[2, 2],
                storage: GpuTensorStorage::Real,
            })
            .unwrap(),
    };
    let any = provider.reduce_any_dim(&input, 0, false).await.unwrap();
    let all = provider.reduce_all_dim(&input, 0, false).await.unwrap();
    let product = provider
        .reduce_prod_dim(&input, 0)
        .await
        .unwrap()
        .expect("provider accepted numerical operation");
    assert_eq!(any.shape(), [1, 2]);
    assert_eq!(all.shape(), [1, 2]);
    assert!(any.view().is_logical());
    assert!(all.view().is_logical());
    assert!(!product.view().is_logical());
    assert_series(&read_values(provider, &product).await, &[0.0, 6.0]);
    // Sum has a double-accumulation contract even for Single inputs. Its
    // unsupported F32-owner route must remain a pre-effect refusal; product
    // independently exercises numeric registration on both owner precisions.
    match precision {
        ProviderPrecision::F32 => {
            let error = provider.reduce_sum_dim(&input, 0).await.unwrap_err();
            assert!(error.is_unsupported());
            assert!(!error.is_after_effect());
        }
        ProviderPrecision::F64 => {
            let sum = provider.reduce_sum_dim(&input, 0).await.unwrap();
            assert!(!sum.view().is_logical());
            assert_series(&read_values(provider, &sum).await, &[1.0, 5.0]);
            retire(sum);
        }
    }
    assert_series(&read_values(provider, &any).await, &[1.0, 1.0]);
    assert_series(&read_values(provider, &all).await, &[0.0, 1.0]);
    retire(any);
    retire(all);
    retire(product);
    retire(input);
}

async fn verify_reduction_and_scan_nan_policy(
    provider: &WgpuProvider,
    precision: ProviderPrecision,
) {
    assert_eq!(provider.precision(), precision);
    verify_logical_nan_mask_global_any_all_ranks(provider, precision).await;
    let input = upload_edge_values(provider, precision);
    let direction = ProviderScanDirection::Forward;
    let include = ProviderNanMode::Include;
    let omit = ProviderNanMode::Omit;

    if precision == ProviderPrecision::F32 {
        // Native single values remain legal, but F32 storage cannot satisfy
        // the public double-index demand. Decline before either scan executes.
        assert!(provider
            .cummin_scan(
                &input,
                0,
                direction,
                include,
                runmat_accelerate_api::ProviderExtremaDemand::ValuesAndIndices
            )
            .await
            .unwrap()
            .is_none());
        assert!(provider
            .cummax_scan(
                &input,
                0,
                direction,
                include,
                runmat_accelerate_api::ProviderExtremaDemand::ValuesAndIndices
            )
            .await
            .unwrap()
            .is_none());
        input.allocation_lease().validate_live().unwrap();
    }

    let sum_include = provider
        .cumsum_scan(&input, 0, direction, include)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(sum_include.shape(), input.shape());
    assert_series(
        &read_values(provider, &sum_include).await,
        &[2.0, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN],
    );
    let sum_omit = provider
        .cumsum_scan(&input, 0, direction, omit)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(sum_omit.shape(), input.shape());
    assert_series(
        &read_values(provider, &sum_omit).await,
        &[2.0, 2.0, 2.0, 2.0, 2.0, 3.0],
    );
    let prod_include = provider
        .cumprod_scan(&input, 0, direction, include)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(prod_include.shape(), input.shape());
    assert_series(
        &read_values(provider, &prod_include).await,
        &[2.0, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN],
    );
    let prod_omit = provider
        .cumprod_scan(&input, 0, direction, omit)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(prod_omit.shape(), input.shape());
    assert_series(
        &read_values(provider, &prod_omit).await,
        &[2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
    );
    let reverse_include = provider
        .cumsum_scan(&input, 0, ProviderScanDirection::Reverse, include)
        .await
        .unwrap()
        .unwrap();
    assert_series(
        &read_values(provider, &reverse_include).await,
        &[f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, 1.0],
    );
    let reverse_omit = provider
        .cumsum_scan(&input, 0, ProviderScanDirection::Reverse, omit)
        .await
        .unwrap()
        .unwrap();
    assert_series(
        &read_values(provider, &reverse_omit).await,
        &[3.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    );

    for (mode, expected_values, expected_indices) in [
        (
            include,
            [2.0, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN],
            [1.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        ),
        (
            omit,
            [2.0, 2.0, 2.0, 2.0, 2.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 6.0],
        ),
    ] {
        let minimum = provider
            .cummin_scan(
                &input,
                0,
                direction,
                mode,
                runmat_accelerate_api::ProviderExtremaDemand::ValuesOnly,
            )
            .await
            .unwrap()
            .unwrap();
        assert_series(
            &read_values(provider, &minimum.values).await,
            &expected_values,
        );
        assert_series(
            &read_values(provider, &minimum.indices).await,
            &expected_indices,
        );
        retire(minimum.values);
        retire(minimum.indices);
    }
    for (mode, expected_values, expected_indices) in [
        (
            include,
            [2.0, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN],
            [1.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        ),
        (
            omit,
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
    ] {
        let maximum = provider
            .cummax_scan(
                &input,
                0,
                direction,
                mode,
                runmat_accelerate_api::ProviderExtremaDemand::ValuesOnly,
            )
            .await
            .unwrap()
            .unwrap();
        assert_series(
            &read_values(provider, &maximum.values).await,
            &expected_values,
        );
        assert_series(
            &read_values(provider, &maximum.indices).await,
            &expected_indices,
        );
        retire(maximum.values);
        retire(maximum.indices);
    }

    let reverse = ProviderScanDirection::Reverse;
    for (mode, expected_values, expected_indices) in [
        (
            include,
            [f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, 1.0],
            [5.0, 5.0, 5.0, 5.0, 5.0, 6.0],
        ),
        (
            omit,
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [6.0, 6.0, 6.0, 6.0, 6.0, 6.0],
        ),
    ] {
        let minimum = provider
            .cummin_scan(
                &input,
                0,
                reverse,
                mode,
                runmat_accelerate_api::ProviderExtremaDemand::ValuesOnly,
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(minimum.values.shape(), input.shape());
        assert_eq!(minimum.indices.shape(), input.shape());
        assert_series(
            &read_values(provider, &minimum.values).await,
            &expected_values,
        );
        assert_series(
            &read_values(provider, &minimum.indices).await,
            &expected_indices,
        );
        retire(minimum.values);
        retire(minimum.indices);
    }
    for (mode, expected_values, expected_indices) in [
        (
            include,
            [f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, 1.0],
            [5.0, 5.0, 5.0, 5.0, 5.0, 6.0],
        ),
        (
            omit,
            [2.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 6.0, 6.0, 6.0, 6.0, 6.0],
        ),
    ] {
        let maximum = provider
            .cummax_scan(
                &input,
                0,
                reverse,
                mode,
                runmat_accelerate_api::ProviderExtremaDemand::ValuesOnly,
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(maximum.values.shape(), input.shape());
        assert_eq!(maximum.indices.shape(), input.shape());
        assert_series(
            &read_values(provider, &maximum.values).await,
            &expected_values,
        );
        assert_series(
            &read_values(provider, &maximum.indices).await,
            &expected_indices,
        );
        retire(maximum.values);
        retire(maximum.indices);
    }

    let ties = upload_finite_ties(provider, precision);
    for (
        mode,
        expected_min_values,
        expected_min_indices,
        expected_max_values,
        expected_max_indices,
    ) in [
        (
            include,
            [2.0, 2.0, f64::NAN, f64::NAN, f64::NAN],
            [1.0, 1.0, 3.0, 3.0, 3.0],
            [2.0, 2.0, f64::NAN, f64::NAN, f64::NAN],
            [1.0, 1.0, 3.0, 3.0, 3.0],
        ),
        (
            omit,
            [2.0, 2.0, 2.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 4.0, 4.0],
            [2.0, 2.0, 2.0, 2.0, 2.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ),
    ] {
        let minimum = provider
            .cummin_scan(
                &ties,
                0,
                direction,
                mode,
                runmat_accelerate_api::ProviderExtremaDemand::ValuesOnly,
            )
            .await
            .unwrap()
            .unwrap();
        let maximum = provider
            .cummax_scan(
                &ties,
                0,
                direction,
                mode,
                runmat_accelerate_api::ProviderExtremaDemand::ValuesOnly,
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(minimum.values.shape(), ties.shape());
        assert_eq!(maximum.values.shape(), ties.shape());
        assert_series(
            &read_values(provider, &minimum.values).await,
            &expected_min_values,
        );
        assert_series(
            &read_values(provider, &minimum.indices).await,
            &expected_min_indices,
        );
        assert_series(
            &read_values(provider, &maximum.values).await,
            &expected_max_values,
        );
        assert_series(
            &read_values(provider, &maximum.indices).await,
            &expected_max_indices,
        );
        retire(minimum.values);
        retire(minimum.indices);
        retire(maximum.values);
        retire(maximum.indices);
    }
    ties.allocation_lease().validate_live().unwrap();
    retire(ties);

    for (omit_nan, expected_any, expected_all) in [(false, 1.0, 1.0), (true, 1.0, 1.0)] {
        let any = provider.reduce_any(&input, omit_nan).await.unwrap();
        let all = provider.reduce_all(&input, omit_nan).await.unwrap();
        assert_series(&read_values(provider, &any).await, &[expected_any]);
        assert_series(&read_values(provider, &all).await, &[expected_all]);
        retire(any);
        retire(all);
    }
    let nan_and_zero = upload_nan_and_zero(provider, precision);
    for (omit_nan, expected_any) in [(false, 1.0), (true, 0.0)] {
        let any = provider.reduce_any(&nan_and_zero, omit_nan).await.unwrap();
        let all = provider.reduce_all(&nan_and_zero, omit_nan).await.unwrap();
        assert_series(&read_values(provider, &any).await, &[expected_any]);
        assert_series(&read_values(provider, &all).await, &[0.0]);
        retire(any);
        retire(all);
    }
    retire(nan_and_zero);

    for handle in [sum_include, sum_omit, prod_include, prod_omit] {
        assert!(!handle.shares_allocation_with(&input));
        retire(handle);
    }
    retire(reverse_include);
    retire(reverse_omit);
    input.allocation_lease().validate_live().unwrap();
    retire(input);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires an actual F32 WGPU adapter; run in the device qualification gate"]
fn f32_reduction_and_scan_nan_policy_on_real_device() {
    let provider =
        futures::executor::block_on(WgpuProvider::new_async(WgpuProviderOptions::default()))
            .expect("actual WGPU adapter required");
    futures::executor::block_on(verify_reduction_and_scan_nan_policy(
        &provider,
        ProviderPrecision::F32,
    ));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires an actual shader-f64 WGPU adapter; run with RUNMAT_WGPU_FORCE_PRECISION=f64"]
fn f64_reduction_and_scan_nan_policy_on_real_device() {
    let provider =
        futures::executor::block_on(WgpuProvider::new_async(WgpuProviderOptions::default()))
            .expect("actual shader-f64 WGPU adapter required");
    futures::executor::block_on(verify_reduction_and_scan_nan_policy(
        &provider,
        ProviderPrecision::F64,
    ));
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn browser_reduction_and_scan_nan_policy_on_real_webgpu() {
    let provider = WgpuProvider::new_async(WgpuProviderOptions::default())
        .await
        .expect("actual browser WebGPU device required");
    verify_reduction_and_scan_nan_policy(&provider, ProviderPrecision::F32).await;
}
