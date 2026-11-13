//! Actual ordinary binary Max routes. Native tests are explicit adapter gates;
//! browser tests run against WebGPU. These do not grant image-stage admission.
use super::*;
use crate::backend::wgpu::provider::WgpuProviderOptions;
use crate::backend::wgpu::types::BinaryOpCode;
use runmat_accelerate_api::{
    AccelProvider, HostNumericDataOwned, HostNumericDataView, HostNumericTensorView,
    NumericElementType, ProviderPrecision, ThreadProviderGuard,
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

fn host_value(values: &[f64], shape: &[usize], single: bool) -> Value {
    Value::Tensor(if single {
        Tensor::from_f32(
            values.iter().map(|value| *value as f32).collect(),
            shape.to_vec(),
        )
        .unwrap()
    } else {
        Tensor::new(values.to_vec(), shape.to_vec()).unwrap()
    })
}

async fn host_max(
    left: &[f64],
    left_shape: &[usize],
    right: &[f64],
    right_shape: &[usize],
    single: bool,
) -> Tensor {
    #[cfg(target_arch = "wasm32")]
    runmat_runtime::builtins::wasm_registry::register_all();
    let args = [
        host_value(left, left_shape, single),
        host_value(right, right_shape, single),
    ];
    let result = runmat_runtime::call_builtin_async("max", &args)
        .await
        .expect("ordinary host Max oracle");
    let Value::Tensor(tensor) = result.value() else {
        panic!("nonscalar host Max result")
    };
    tensor.clone()
}

fn upload(
    provider: &'static WgpuProvider,
    values: &[f64],
    shape: &[usize],
    single: bool,
) -> GpuTensorHandle {
    let represented = values.iter().map(|value| *value as f32).collect::<Vec<_>>();
    provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: if single {
                HostNumericDataView::F32(&represented)
            } else {
                HostNumericDataView::F64(values)
            },
            shape,
            storage: GpuTensorStorage::Real,
        })
        .unwrap()
}

async fn assert_output(
    provider: &WgpuProvider,
    output: &GpuTensorHandle,
    expected: &Tensor,
    single: bool,
) {
    assert_eq!(output.shape(), expected.shape());
    assert_eq!(
        output.physical().element_type(),
        if single {
            NumericElementType::F32
        } else {
            NumericElementType::F64
        }
    );
    let actual = provider.download_numeric(output).await.unwrap().data;
    if single {
        let HostNumericDataOwned::F32(actual) = actual else {
            panic!("Single Max output class")
        };
        let expected = expected.as_f32_slice().unwrap();
        assert_eq!(actual.len(), expected.len(), "Single payload length");
        for (&actual, &expected) in actual.iter().zip(expected) {
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
    } else {
        let HostNumericDataOwned::F64(actual) = actual else {
            panic!("Double Max output class")
        };
        let expected = expected.as_f64_slice().unwrap();
        assert_eq!(actual.len(), expected.len(), "Double payload length");
        for (&actual, &expected) in actual.iter().zip(expected) {
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
    }
}

async fn qualify(provider: &'static WgpuProvider, single: bool) {
    let tiny = if single {
        f64::from(f32::from_bits(1))
    } else {
        f64::from_bits(1)
    };
    let left = [f64::NAN, -0.0, 0.0, -tiny, tiny, f64::INFINITY];
    let right = [3.0, 0.0, -0.0, tiny, -tiny, f64::NAN];
    let precision = if single {
        NumericPrecision::F32
    } else {
        NumericPrecision::F64
    };
    let ordinary = provider
        .real_binary_pipeline(precision, false, BinaryOpCode::Add)
        .unwrap();
    let max = provider
        .real_binary_pipeline(precision, false, BinaryOpCode::Max)
        .unwrap();
    let broadcast_max = provider
        .real_binary_pipeline(precision, true, BinaryOpCode::Max)
        .unwrap();
    assert!(!std::ptr::eq(max, ordinary));
    assert!(!std::ptr::eq(max, broadcast_max));
    let lhs = upload(provider, &left, &[6, 1], single);
    let rhs = upload(provider, &right, &[6, 1], single);
    let mut prior = None;
    for (a, b, a_values, b_values) in [
        (&lhs, &rhs, &left[..], &right[..]),
        (&rhs, &lhs, &right[..], &left[..]),
    ] {
        let expected = host_max(a_values, &[6, 1], b_values, &[6, 1], single).await;
        for _ in 0..2 {
            let observed = RefCell::new(RealBinaryObserved::default());
            let output = provider
                .binary_op_exec_recorded(BinaryOpCode::Max, a, b, Some(&observed), None, None)
                .unwrap();
            assert_output(provider, &output, &expected, single).await;
            assert!(!output.shares_allocation_with(&lhs));
            assert!(!output.shares_allocation_with(&rhs));
            let observation = observed.into_inner().into_resources();
            assert_eq!(
                observation.output_buffer_bytes,
                Some(output.physical().requested_bytes())
            );
            assert!(observation.dispatches > 0);
            if let Some(prior) = &prior {
                assert_eq!(prior, &observation, "cold/warm Max resources");
            }
            prior = Some(observation);
        }
    }
    for (left_values, right_values) in [
        (&left[..], &[0.0, f64::NAN][..]),
        (&right[..], &[-0.0, tiny][..]),
    ] {
        let left_handle = upload(provider, left_values, &[6, 1], single);
        let right_handle = upload(provider, right_values, &[1, 2], single);
        for (a, b, a_values, a_shape, b_values, b_shape) in [
            (
                &left_handle,
                &right_handle,
                left_values,
                &[6, 1][..],
                right_values,
                &[1, 2][..],
            ),
            (
                &right_handle,
                &left_handle,
                right_values,
                &[1, 2][..],
                left_values,
                &[6, 1][..],
            ),
        ] {
            let expected = host_max(a_values, a_shape, b_values, b_shape, single).await;
            let output = provider.elem_max(a, b).await.unwrap();
            assert_output(provider, &output, &expected, single).await;
        }
        for handle in [&left_handle, &right_handle] {
            handle.allocation_lease().validate_live().unwrap();
        }
    }
    if (single && provider.precision() == ProviderPrecision::F32)
        || (!single && provider.precision() == ProviderPrecision::F64)
    {
        // Exercise the provider hooks and the actual public Runtime caller,
        // which chooses fill_like then elem_max for this native-class form.
        // The alternate-class fill owner is a separate typed capability.
        for scalar in [f64::NAN, -0.0, 0.0, tiny] {
            let fill = provider
                .fill_like(&lhs, scalar)
                .unwrap()
                .expect("admitted optional constructor");
            let expected = host_max(&left, &[6, 1], &[scalar], &[1, 1], single).await;
            for (a, b) in [(&lhs, &fill), (&fill, &lhs)] {
                let output = provider.elem_max(a, b).await.unwrap();
                assert_output(provider, &output, &expected, single).await;
            }
            let control = host_value(&[scalar], &[1, 1], single);
            for args in [
                [Value::GpuTensor(lhs.clone()), control.clone()],
                [control.clone(), Value::GpuTensor(lhs.clone())],
            ] {
                let _selected = ThreadProviderGuard::set(Some(provider));
                let result = runmat_runtime::call_builtin_async("max", &args)
                    .await
                    .expect("public resident Max");
                let Value::GpuTensor(output) = result.value() else {
                    panic!("public native-class scalar Max must remain resident")
                };
                assert_output(provider, output, &expected, single).await;
            }
        }
    }
    lhs.allocation_lease().validate_live().unwrap();
    rhs.allocation_lease().validate_live().unwrap();
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(
    not(target_arch = "wasm32"),
    ignore = "requires an actual WGPU adapter"
)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn actual_f32_binary_broadcast_and_public_scalar_max_match_host_bits() {
    qualify(owner().await, true).await;
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
#[ignore = "requires enabled SHADER_F64 and SHADER_INT64 on an actual adapter"]
async fn actual_f64_binary_broadcast_and_public_scalar_max_match_host_bits() {
    let provider = owner().await;
    assert_eq!(provider.precision(), ProviderPrecision::F64);
    qualify(provider, false).await;
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
#[ignore = "requires actual F64 hardware; removes logical INT64 admission before effects"]
async fn absent_enabled_int64_rejects_every_double_max_geometry_without_effects() {
    let mut provider = Box::new(WgpuProvider::new(WgpuProviderOptions::default()).unwrap());
    assert_eq!(provider.precision(), ProviderPrecision::F64);
    provider
        .enabled_features
        .remove(wgpu::Features::SHADER_INT64);
    let provider = Box::leak(provider);
    // SAFETY: this actual device owner is leaked for the process.
    unsafe { runmat_accelerate_api::register_device_provider(provider) };
    let nonempty = upload(provider, &[1.0, -0.0], &[2, 1], false);
    let scalar = upload(provider, &[0.0], &[1, 1], false);
    let empty = upload(provider, &[], &[0, 1], false);
    for (lhs, rhs) in [
        (&nonempty, &nonempty),
        (&nonempty, &scalar),
        (&empty, &empty),
    ] {
        let before = provider.telemetry_snapshot();
        let records = provider.allocations.record_count().unwrap();
        let error = provider.elem_max(lhs, rhs).await.unwrap_err();
        assert!(error.is_unsupported());
        assert_eq!(provider.telemetry_snapshot(), before);
        assert_eq!(provider.allocations.record_count().unwrap(), records);
        lhs.allocation_lease().validate_live().unwrap();
        rhs.allocation_lease().validate_live().unwrap();
    }
    // No INT64 feature tax leaks to ordinary Add on the same F64 owner.
    let add = provider.elem_add(&nonempty, &nonempty).await.unwrap();
    assert_eq!(add.physical().element_type(), NumericElementType::F64);
    let HostNumericDataOwned::F64(values) = provider.download_numeric(&add).await.unwrap().data
    else {
        panic!("unrelated Add output class")
    };
    assert_eq!(values[0], 2.0);
}
