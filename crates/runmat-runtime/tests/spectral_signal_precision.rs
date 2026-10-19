//! RunMat signal-class policy, independent of analysis-control precision.
use futures::executor::block_on;
use runmat_value::{ComplexTensor, NumericDType, Tensor, Value};

fn real(values: Vec<f64>, shape: Vec<usize>, single: bool) -> Value {
    Value::Tensor(if single {
        Tensor::from_f32(
            values.into_iter().map(|value| value as f32).collect(),
            shape,
        )
        .unwrap()
    } else {
        Tensor::new(values, shape).unwrap()
    })
}

fn values(value: &Value, dtype: NumericDType) -> Vec<f64> {
    let Value::Tensor(value) = value else {
        panic!("expected real spectral output")
    };
    assert_eq!(value.numeric_dtype(), dtype);
    value.materialize_f64()
}

#[test]
fn surplus_host_output_slots_retain_shared_double_scalar_padding() {
    for name in ["periodogram", "spectrogram"] {
        for single in [false, true] {
            let named = if name == "periodogram" { 2 } else { 4 };
            let mut args = vec![
                real(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1], single),
                real(vec![1.0; 4], vec![4, 1], false),
            ];
            if name == "spectrogram" {
                args.push(Value::Num(0.0));
            }
            args.push(Value::Num(4.0));
            let result = block_on(runmat_runtime::call_builtin_async_with_outputs(
                name,
                &args,
                named + 2,
            ))
            .unwrap();
            let Value::OutputList(values) = result.value() else {
                panic!("padded outputs")
            };
            for value in &values[named..] {
                assert!(matches!(value, Value::Num(number) if *number == 0.0));
            }
        }
    }
}

#[test]
fn all_host_spectral_outputs_follow_signal_not_window_or_sampling_class() {
    for name in ["periodogram", "spectrogram"] {
        for single in [false, true] {
            for single_controls in [false, true] {
                for complex in [false, true] {
                    let dtype = if single {
                        NumericDType::F32
                    } else {
                        NumericDType::F64
                    };
                    let signal = if complex {
                        Value::ComplexTensor(if single {
                            ComplexTensor::from_f32(
                                vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)],
                                vec![4, 1],
                            )
                            .unwrap()
                        } else {
                            ComplexTensor::new(
                                vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)],
                                vec![4, 1],
                            )
                            .unwrap()
                        })
                    } else {
                        real(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1], single)
                    };
                    let mut args = vec![signal, real(vec![1.0; 4], vec![4, 1], single_controls)];
                    if name == "spectrogram" {
                        args.push(Value::Num(0.0));
                    }
                    args.extend([
                        Value::Num(4.0),
                        real(vec![4.0], vec![1, 1], single_controls),
                        Value::String("twosided".into()),
                    ]);
                    let result = block_on(runmat_runtime::call_builtin_async_with_outputs(
                        name,
                        &args,
                        if name == "periodogram" { 2 } else { 4 },
                    ))
                    .unwrap();
                    let Value::OutputList(outputs) = result.value() else {
                        panic!("multiple spectral outputs")
                    };
                    assert_eq!(values(&outputs[1], dtype), vec![0.0, 1.0, 2.0, 3.0]);
                    if name == "periodogram" {
                        assert_eq!(values(&outputs[0], dtype), vec![6.25, 0.5, 0.25, 0.5]);
                    } else {
                        assert_eq!(values(&outputs[2], dtype), vec![0.5]);
                        assert_eq!(values(&outputs[3], dtype), vec![6.25, 0.5, 0.25, 0.5]);
                        let Value::ComplexTensor(stft) = &outputs[0] else {
                            panic!("STFT remains complex storage")
                        };
                        assert_eq!(stft.numeric_dtype(), dtype);
                        let actual = stft.materialize_f64();
                        assert_eq!(
                            actual,
                            vec![(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0), (-2.0, -2.0)]
                        );
                    }
                }
            }
        }
    }
}
