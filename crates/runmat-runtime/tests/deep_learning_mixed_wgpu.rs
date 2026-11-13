//! Public forward witness for a composed resident normalization/dense route.
#![cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]

use futures::executor::block_on;
use runmat_accelerate_api::{
    handle_precision, handle_provenance, handle_storage, AccelProvider, GpuHandleProvenance,
    GpuTensorStorage, HostNumericDataOwned, HostNumericDataView, HostNumericTensorView,
    ProviderPrecision, ThreadProviderGuard,
};
use runmat_runtime::call_builtin;
use runmat_runtime::execution::RootedValuesBuilder;
use runmat_value::{CellArray, Tensor, Value};

#[path = "deep_learning_mixed_wgpu/routes.rs"]
mod routes;
#[path = "deep_learning_mixed_wgpu/fixture.rs"]
mod fixture;

#[test]
#[ignore = "requires an actual WGPU device; run explicitly for both native precisions"]
fn public_nonsquare_cb_normalization_then_dense_stays_resident_and_matches_oracle() {
    let provider = fixture::device();
    let _selected = ThreadProviderGuard::set(Some(provider));
    for explicit_parameters in [false, true] {
        for lazy_source in [false, true] {
            let source = [1.0, 3.0, 2.0, 5.0, 5.0, 7.0];
            let backing = if lazy_source {
                [1.0, 2.0, 5.0, 3.0, 5.0, 7.0]
            } else {
                source
            };
            let backing_shape: &[usize] = if lazy_source { &[3, 2] } else { &[2, 3] };
            let single = backing.map(|value| value as f32);
            let uploaded = provider
                .upload_numeric(&HostNumericTensorView {
                    data: match provider.precision() {
                        ProviderPrecision::F32 => HostNumericDataView::F32(&single),
                        ProviderPrecision::F64 => HostNumericDataView::F64(&backing),
                    },
                    shape: backing_shape,
                    storage: GpuTensorStorage::Real,
                })
                .unwrap();
            let handle = (if lazy_source {
                provider.transpose(&uploaded)
            } else {
                Ok(uploaded)
            })
            .unwrap()
            .with_provenance(if explicit_parameters {
                GpuHandleProvenance::Automatic
            } else {
                GpuHandleProvenance::Explicit
            })
            .unwrap();
            let retained_input = handle.clone();
            let upload_parameter = |values: &[f64], shape: &[usize]| {
                let single = values.iter().map(|value| *value as f32).collect::<Vec<_>>();
                provider
                    .upload_numeric(&HostNumericTensorView {
                        data: match provider.precision() {
                            ProviderPrecision::F32 => HostNumericDataView::F32(&single),
                            ProviderPrecision::F64 => HostNumericDataView::F64(values),
                        },
                        shape,
                        storage: GpuTensorStorage::Real,
                    })
                    .unwrap()
                    .with_provenance(GpuHandleProvenance::Explicit)
                    .unwrap()
            };
            let parameter_handles = if explicit_parameters {
                vec![
                    upload_parameter(&[1.0, 1.0], &[2, 1]),
                    upload_parameter(&[0.0, 0.0], &[2, 1]),
                    upload_parameter(&[2.0, -1.0], &[1, 2]),
                    upload_parameter(&[0.5], &[1, 1]),
                ]
            } else {
                Vec::new()
            };
            let input = call_builtin(
                "dlarray",
                &[Value::GpuTensor(handle), Value::String("CB".into())],
            )
            .unwrap();
            let normalization_options = if explicit_parameters {
                vec![
                    Value::String("Scale".into()),
                    Value::GpuTensor(parameter_handles[0].clone()),
                    Value::String("Offset".into()),
                    Value::GpuTensor(parameter_handles[1].clone()),
                ]
            } else {
                Vec::new()
            };
            let weights = if explicit_parameters {
                Value::GpuTensor(parameter_handles[2].clone())
            } else {
                Value::Tensor(Tensor::new(vec![2.0, -1.0], vec![1, 2]).unwrap())
            };
            let bias = if explicit_parameters {
                Value::GpuTensor(parameter_handles[3].clone())
            } else {
                Value::Tensor(Tensor::new(vec![0.5], vec![1, 1]).unwrap())
            };
            let mut layers = RootedValuesBuilder::with_capacity(3);
            layers.push_owned(call_builtin("featureInputLayer", &[Value::Num(2.0)]).unwrap());
            layers.push_owned(
                call_builtin("layerNormalizationLayer", &normalization_options).unwrap(),
            );
            layers.push_owned(
                call_builtin(
                    "fullyConnectedLayer",
                    &[
                        Value::Num(1.0),
                        Value::String("Weights".into()),
                        weights,
                        Value::String("Bias".into()),
                        bias,
                    ],
                )
                .unwrap(),
            );
            let layers = layers.finish().unwrap();
            let network = layers
                .with_unbounded_values(|layers| {
                    call_builtin(
                        "dlnetwork",
                        &[Value::Cell(CellArray::new(layers, 3, 1).unwrap())],
                    )
                    .unwrap()
                })
                .expect("fixture layer batch is unbounded");
            provider.reset_telemetry();
            let output = network.with_owned(|network| {
                input.with_owned(|input| call_builtin("forward", &[network, input]).unwrap())
            });
            let Value::Object(output) = output.value() else {
                panic!("formatted dlarray output expected")
            };
            let Some(Value::GpuTensor(data)) = output.properties().get("Data") else {
                panic!("explicit input must retain resident output")
            };
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
            assert_eq!(data.device_id(), retained_input.device_id());
            assert_eq!(handle_precision(data), Some(provider.precision()));
            assert_eq!(handle_storage(data), GpuTensorStorage::Real);
            assert_eq!(handle_provenance(data), Some(GpuHandleProvenance::Explicit),
                "explicit_parameters={explicit_parameters}, lazy_source={lazy_source}, input_intent={:?}",
                handle_provenance(&retained_input));
            assert_eq!(
                handle_provenance(&retained_input),
                Some(if explicit_parameters {
                    GpuHandleProvenance::Automatic
                } else {
                    GpuHandleProvenance::Explicit
                })
            );
            assert_eq!(data.shape(), [1, 3]);
            assert_eq!(
                output.properties().get("Format"),
                Some(&Value::String("CB".into()))
            );
            let gathered = block_on(provider.download_numeric(data)).unwrap();
            let actual = match (provider.precision(), gathered.data) {
                (ProviderPrecision::F32, HostNumericDataOwned::F32(values)) => {
                    values.into_iter().map(f64::from).collect::<Vec<_>>()
                }
                (ProviderPrecision::F64, HostNumericDataOwned::F64(values)) => values,
                _ => panic!("resident route changed numeric class"),
            };
            let expected = [1.0_f64, 1.5, 1.0].map(|difference| {
                let centered = difference / (difference * difference + 1.0e-5).sqrt();
                0.5 - 3.0 * centered
            });
            assert_eq!(actual.len(), expected.len());
            let tolerance = if provider.precision() == ProviderPrecision::F32 {
                2.0e-4
            } else {
                1.0e-9
            };
            for (actual, expected) in actual.iter().zip(expected) {
                assert!(
                    (actual - expected).abs() < tolerance,
                    "{actual} vs {expected}"
                );
            }
            let retained = block_on(provider.download_numeric(&retained_input)).unwrap();
            let retained = match (provider.precision(), retained.data) {
                (ProviderPrecision::F32, HostNumericDataOwned::F32(values)) => {
                    values.into_iter().map(f64::from).collect::<Vec<_>>()
                }
                (ProviderPrecision::F64, HostNumericDataOwned::F64(values)) => values,
                _ => panic!("retained input changed numeric class"),
            };
            assert_eq!(retained, source);
            for (parameter, expected) in parameter_handles.iter().zip([
                &[1.0, 1.0][..],
                &[0.0, 0.0][..],
                &[2.0, -1.0][..],
                &[0.5][..],
            ]) {
                let retained = block_on(provider.download_numeric(parameter)).unwrap();
                let actual = match (provider.precision(), retained.data) {
                    (ProviderPrecision::F32, HostNumericDataOwned::F32(values)) => {
                        values.into_iter().map(f64::from).collect::<Vec<_>>()
                    }
                    (ProviderPrecision::F64, HostNumericDataOwned::F64(values)) => values,
                    _ => panic!("retained parameter changed numeric class"),
                };
                assert_eq!(actual, expected);
            }
        }
    }
}
