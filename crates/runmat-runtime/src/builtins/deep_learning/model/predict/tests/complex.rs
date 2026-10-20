use super::*;
use crate::builtins::deep_learning::{layers::*, model::forward_builtin, object};
use runmat_accelerate_api::{GpuTensorStorage, HostNumericDataView, HostNumericTensorView};
use runmat_value::{ComplexElement, ComplexTensor, NumericDType};

fn pooling_network() -> crate::execution::RootedValue {
    let input = block_on(sequence_input_layer_builtin(
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let pooling = block_on(global_average_pooling_1d_layer_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    block_on(dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(vec![input, pooling], 2, 1).unwrap(),
    )]))
    .unwrap()
}

fn formatted(data: Value) -> Value {
    object(
        "dlarray",
        vec![
            ("Data", data),
            ("Format", Value::String("CBT".into())),
            ("Labels", Value::String("CBT".into())),
        ],
    )
    .expect("fixture object")
}

fn predict(input: Value) -> Result<Value, crate::RuntimeError> {
    let network = pooling_network();
    block_on(crate::builtins::stats::ml::linear_model::predict_builtin(
        network.value().clone(),
        input,
        vec![],
    ))
}

#[test]
fn public_complex_predict_preserves_precision_and_presents_raw_and_formatted_outputs() {
    for dtype in [NumericDType::F32, NumericDType::F64] {
        let input = match dtype {
            NumericDType::F32 => ComplexTensor::from_f32(
                (1..=12).map(|i| (i as f32, i as f32 * 10.0)).collect(),
                vec![2, 2, 3],
            ),
            NumericDType::F64 => ComplexTensor::new(
                (1..=12).map(|i| (i as f64, i as f64 * 10.0)).collect(),
                vec![2, 2, 3],
            ),
            _ => unreachable!("test precision"),
        }
        .unwrap();
        let raw = predict(Value::ComplexTensor(input.clone())).unwrap();
        let wrapped = predict(formatted(Value::ComplexTensor(input.clone()))).unwrap();
        let Value::ComplexTensor(raw) = raw else {
            panic!("raw complex output")
        };
        let Value::Object(wrapped) = wrapped else {
            panic!("formatted dlarray output")
        };
        let Some(Value::ComplexTensor(data)) = wrapped.properties().get("Data") else {
            panic!("complex dlarray Data")
        };
        assert!(wrapped.class_name().is(super::super::DLARRAY_CLASS));
        assert_eq!(
            wrapped.properties().get("Format"),
            Some(&Value::String("CB".into()))
        );
        assert_eq!(
            wrapped.properties().get("Labels"),
            Some(&Value::String("CB".into()))
        );
        assert_eq!(raw.shape(), &[2, 2]);
        assert_eq!(data.shape(), &[2, 2]);
        assert_eq!(raw.numeric_dtype(), dtype);
        assert_eq!(data.numeric_dtype(), dtype);
        match dtype {
            NumericDType::F32 => {
                assert_eq!(
                    raw.as_f32_slice().unwrap(),
                    &[
                        ComplexElement(5., 50.),
                        ComplexElement(7., 70.),
                        ComplexElement(6., 60.),
                        ComplexElement(8., 80.)
                    ]
                );
                assert_eq!(
                    data.as_f32_slice().unwrap(),
                    &[
                        ComplexElement(5., 50.),
                        ComplexElement(6., 60.),
                        ComplexElement(7., 70.),
                        ComplexElement(8., 80.)
                    ]
                );
                assert_eq!(input.as_f32_slice().unwrap()[0], ComplexElement(1., 10.));
            }
            NumericDType::F64 => {
                assert_eq!(
                    raw.as_f64_slice().unwrap(),
                    &[
                        ComplexElement(5., 50.),
                        ComplexElement(7., 70.),
                        ComplexElement(6., 60.),
                        ComplexElement(8., 80.)
                    ]
                );
                assert_eq!(
                    data.as_f64_slice().unwrap(),
                    &[
                        ComplexElement(5., 50.),
                        ComplexElement(6., 60.),
                        ComplexElement(7., 70.),
                        ComplexElement(8., 80.)
                    ]
                );
                assert_eq!(input.as_f64_slice().unwrap()[0], ComplexElement(1., 10.));
            }
            _ => unreachable!("test precision"),
        }
    }
}

#[test]
fn public_complex_device_input_rejects_before_effects_even_for_host_compatible_topology() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let data = [1., 10., 2., 20., 3., 30., 4., 40.];
        let handle = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F64(&data),
                shape: &[2, 1, 2],
                storage: GpuTensorStorage::ComplexInterleaved,
            })
            .unwrap()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
        for wrapped in [false, true] {
            for caller in [
                super::super::NetworkCaller::Forward,
                super::super::NetworkCaller::Predict,
            ] {
                let input = Value::GpuTensor(handle.clone());
                let input = if wrapped { formatted(input) } else { input };
                provider.reset_telemetry();
                let result = match caller {
                    super::super::NetworkCaller::Forward => {
                        let network = pooling_network();
                        block_on(forward_builtin(
                            network.value().clone(),
                            input,
                            crate::builtins::common::test_support::builtin_arguments(vec![]),
                        ))
                    }
                    super::super::NetworkCaller::Predict => predict(input),
                };
                let error = result.expect_err("resident complex is not an admitted route");
                assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
                let telemetry = provider.telemetry_snapshot();
                assert_eq!(telemetry.upload_bytes, 0);
                assert_eq!(telemetry.download_bytes, 0);
                assert!(telemetry.kernel_launches.is_empty());
            }
        }
    });
}
