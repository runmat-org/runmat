//! Compare descriptor admission with the actual owning host evaluators. These
//! are not provider-kernel numerical tests: no selected provider is required.
use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, HostNumericDataOwned, HostNumericTensorOwned,
    ProviderPrecision,
};
use runmat_value::{NumericDType, Value};

#[derive(Clone, Copy, Debug)]
struct Class {
    element: NumericElementType,
    complex: bool,
    logical: bool,
}

impl Class {
    fn payload(self, count: usize, imaginary: f64) -> HostNumericTensorOwned {
        let count = count * if self.complex { 2 } else { 1 };
        let floating: Vec<f64> = (0..count)
            .map(|index| {
                if self.complex && index % 2 == 1 {
                    imaginary
                } else {
                    1.0
                }
            })
            .collect();
        let data = match self.element {
            NumericElementType::F32 => {
                HostNumericDataOwned::F32(floating.iter().map(|x| *x as f32).collect())
            }
            NumericElementType::F64 => HostNumericDataOwned::F64(floating),
            NumericElementType::I8 => HostNumericDataOwned::I8(vec![1; count]),
            NumericElementType::I16 => HostNumericDataOwned::I16(vec![1; count]),
            NumericElementType::I32 => HostNumericDataOwned::I32(vec![1; count]),
            NumericElementType::I64 => HostNumericDataOwned::I64(vec![1; count]),
            NumericElementType::U8 => HostNumericDataOwned::U8(vec![1; count]),
            NumericElementType::U16 => HostNumericDataOwned::U16(vec![1; count]),
            NumericElementType::U32 => HostNumericDataOwned::U32(vec![1; count]),
            NumericElementType::U64 => HostNumericDataOwned::U64(vec![1; count]),
        };
        HostNumericTensorOwned {
            data,
            storage: if self.complex {
                GpuTensorStorage::ComplexInterleaved
            } else {
                GpuTensorStorage::Real
            },
            shape: vec![count / if self.complex { 2 } else { 1 }, 1],
        }
    }
}

fn host_result(kind: ExtremaKind, left: Value, right: Value) -> BuiltinResult<Value> {
    futures::executor::block_on(async {
        match kind {
            ExtremaKind::Min => crate::builtins::math::reduction::min::evaluate(
                left,
                crate::call::FunctionArgumentView::from_slice(&[right]),
            )
            .await
            .map(|x| x.into_value()),
            ExtremaKind::Max => crate::builtins::math::reduction::max::evaluate(
                left,
                crate::call::FunctionArgumentView::from_slice(&[right]),
            )
            .await
            .map(|x| x.into_value()),
        }
    })
}

fn facts(value: &Value) -> (NumericDType, bool) {
    match value {
        Value::Num(_) => (NumericDType::F64, false),
        Value::Complex(..) => (NumericDType::F64, true),
        Value::Int(value) => (value.numeric_dtype(), false),
        Value::Tensor(value) => (value.numeric_dtype(), false),
        Value::ComplexTensor(value) => (value.numeric_dtype(), true),
        other => panic!("unexpected extrema numerical result {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn full_pair_class_matrix_matches_owning_host_evaluators() {
    let owner = TestAllocationOwner::new();
    let mut classes: Vec<_> = [
        NumericElementType::F32,
        NumericElementType::F64,
        NumericElementType::I8,
        NumericElementType::I16,
        NumericElementType::I32,
        NumericElementType::I64,
        NumericElementType::U8,
        NumericElementType::U16,
        NumericElementType::U32,
        NumericElementType::U64,
    ]
    .into_iter()
    .map(|element| Class {
        element,
        complex: false,
        logical: false,
    })
    .collect();
    for element in [NumericElementType::F32, NumericElementType::F64] {
        classes.push(Class {
            element,
            complex: true,
            logical: false,
        });
        classes.push(Class {
            element,
            complex: false,
            logical: true,
        });
    }
    classes.push(Class {
        element: NumericElementType::U8,
        complex: false,
        logical: true,
    });
    for &left_class in &classes {
        for &right_class in &classes {
            for (left_count, right_count) in [(1, 1), (1, 2), (2, 1), (2, 2)] {
                for imaginary in [0.0, 2.0] {
                    let make = |class: Class, count| {
                        let payload = class.payload(count, imaginary);
                        let host = gpu_helpers::value_from_numeric_download(
                            payload.clone(),
                            class.logical,
                        )
                        .unwrap();
                        let allocation = if class.logical && class.element == NumericElementType::U8
                        {
                            owner
                                .allocate_logical(vec![1; count], payload.shape.clone())
                                .unwrap()
                        } else if class.logical {
                            owner
                                .allocate_logical_zeroed(
                                    payload.shape.clone(),
                                    class.element.precision().unwrap(),
                                )
                                .unwrap()
                        } else {
                            owner.allocate_numeric(payload).unwrap()
                        };
                        (host, allocation)
                    };
                    let (left, left_gpu) = make(left_class, left_count);
                    let (right, right_gpu) = make(right_class, right_count);
                    let contract = pair_contract(
                        &left_gpu.handle,
                        &right_gpu.handle,
                        vec![left_count.max(right_count), 1],
                    );
                    for kind in [ExtremaKind::Min, ExtremaKind::Max] {
                        let host = host_result(kind, left.clone(), right.clone());
                        match (&contract, host) {
                            (Some(contract), Ok(value)) => {
                                let (dtype, complex) = facts(&value);
                                assert_eq!(contract.integer, dtype.integer_class().map(Into::into), "{left_class:?} / {right_class:?}");
                                let precision = match dtype { NumericDType::F32 => Some(ProviderPrecision::F32), NumericDType::F64 => Some(ProviderPrecision::F64), _ => None };
                                assert_eq!(contract.precision, precision, "{left_class:?} / {right_class:?}");
                                assert_eq!(contract.storage != GpuTensorStorage::Real, complex, "{left_class:?} / {right_class:?}");
                            }
                            (None, Ok(value)) => {
                                // Sole supported but descriptor-inconclusive domain:
                                // Double complex singleton output complexity depends
                                // on actual imaginary values, not its carrier tag.
                                assert_eq!((left_count, right_count), (1, 1));
                                assert!(left_class.complex || right_class.complex);
                                assert!(left_class.logical || left_class.element == NumericElementType::F64);
                                assert!(right_class.logical || right_class.element == NumericElementType::F64);
                                assert_eq!(facts(&value), (NumericDType::F64, imaginary != 0.0));
                            }
                            (None, Err(_)) => assert!(left_class.element.integer_type().is_some() || right_class.element.integer_type().is_some()),
                            (Some(_), Err(error)) => panic!("descriptor admitted host-invalid {left_class:?} / {right_class:?}: {error}"),
                        }
                    }
                    drop((left_gpu.handle, right_gpu.handle));
                    assert_eq!(left_gpu.probe.retirements(), 1);
                    assert_eq!(right_gpu.probe.retirements(), 1);
                }
            }
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn immediate_and_tensor_double_complex_scalar_complexity_is_value_dependent() {
    for kind in [ExtremaKind::Min, ExtremaKind::Max] {
        for imaginary in [0.0, 2.0] {
            let tensor = gpu_helpers::value_from_numeric_download(
                Class {
                    element: NumericElementType::F64,
                    complex: true,
                    logical: false,
                }
                .payload(1, imaginary),
                false,
            )
            .unwrap();
            for complex in [Value::Complex(1.0, imaginary), tensor] {
                for reverse in [false, true] {
                    let (left, right) = if reverse {
                        (Value::Num(0.0), complex.clone())
                    } else {
                        (complex.clone(), Value::Num(0.0))
                    };
                    let value = host_result(kind, left, right).unwrap();
                    assert_eq!(facts(&value), (NumericDType::F64, imaginary != 0.0));
                }
            }
        }
    }
}
