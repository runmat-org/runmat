use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::{HostNumericDataOwned, NumericElementType};

const OPERATIONS: [Unary; 6] = [
    Unary::Sin,
    Unary::Cos,
    Unary::Abs,
    Unary::Exp,
    Unary::Log,
    Unary::Sqrt,
];

#[test]
fn every_native_integer_class_is_explicitly_unsupported_not_missing() {
    use runmat_accelerate_api::HostIntegerDataOwned as Integer;
    let provider = InProcessProvider::new();
    for data in [
        Integer::I8(vec![1]),
        Integer::I16(vec![1]),
        Integer::I32(vec![1]),
        Integer::I64(vec![1]),
        Integer::U8(vec![1]),
        Integer::U16(vec![1]),
        Integer::U32(vec![1]),
        Integer::U64(vec![1]),
    ] {
        let input = provider
            .allocate_numeric_data(
                InProcessNumericData::Integer(data),
                vec![1, 1],
                GpuTensorStorage::Real,
            )
            .unwrap();
        for operation in OPERATIONS {
            assert!(runmat_accelerate_api::is_unsupported_provider_operation(
                &execute(&provider, &input, operation).unwrap_err()
            ));
        }
        for result in [
            block_on(provider.reduce_min(&input)),
            block_on(provider.reduce_max(&input)),
        ] {
            assert!(runmat_accelerate_api::is_unsupported_provider_operation(
                &result.unwrap_err()
            ));
        }
    }
}

#[test]
fn admitted_reshape_view_preserves_shape_without_changing_source() {
    let provider = InProcessProvider::new();
    let input = provider
        .allocate_numeric_data(
            InProcessNumericData::F32(vec![0.0, 0.5, 1.0, 2.0]),
            vec![2, 2],
            GpuTensorStorage::Real,
        )
        .unwrap();
    let reshaped = input
        .with_view(runmat_accelerate_api::allocation::GpuTensorView::contiguous(vec![1, 2, 2]))
        .unwrap();
    let output = execute(&provider, &reshaped, Unary::Sin).unwrap();
    assert_eq!(output.shape(), &[1, 2, 2]);
    assert_eq!(input.shape(), &[2, 2]);
    assert!(input.shares_allocation_with(&reshaped));
    assert!(!output.shares_allocation_with(&reshaped));
    let HostNumericDataOwned::F32(values) =
        block_on(provider.download_numeric(&output)).unwrap().data
    else {
        panic!("Single widened")
    };
    assert_eq!(values, vec![0.0, 0.5f32.sin(), 1.0f32.sin(), 2.0f32.sin()]);
}

fn execute(
    provider: &InProcessProvider,
    input: &GpuTensorHandle,
    operation: Unary,
) -> Result<GpuTensorHandle> {
    block_on(match operation {
        Unary::Sin => provider.unary_sin(input),
        Unary::Cos => provider.unary_cos(input),
        Unary::Abs => provider.unary_abs(input),
        Unary::Exp => provider.unary_exp(input),
        Unary::Log => provider.unary_log(input),
        Unary::Sqrt => provider.unary_sqrt(input),
    })
}

#[test]
fn real_hooks_keep_native_precision_shape_and_exceptional_values() {
    let provider = InProcessProvider::new();
    let singles = [
        -0.0f32,
        f32::from_bits(1),
        0.25,
        -1.0,
        f32::INFINITY,
        f32::NAN,
    ];
    let doubles = [
        -0.0f64,
        f64::from_bits(1),
        0.25,
        -1.0,
        f64::INFINITY,
        f64::NAN,
    ];
    let single = provider
        .allocate_numeric_data(
            InProcessNumericData::F32(singles.to_vec()),
            vec![1, 2, 3],
            GpuTensorStorage::Real,
        )
        .unwrap();
    let double = provider
        .allocate_numeric_data(
            InProcessNumericData::F64(doubles.to_vec()),
            vec![1, 2, 3],
            GpuTensorStorage::Real,
        )
        .unwrap();
    for operation in OPERATIONS {
        let result = execute(&provider, &single, operation).unwrap();
        assert_eq!(result.shape(), &[1, 2, 3]);
        assert_eq!(result.physical().element_type(), NumericElementType::F32);
        assert!(!result.shares_allocation_with(&single));
        let HostNumericDataOwned::F32(actual) =
            block_on(provider.download_numeric(&result)).unwrap().data
        else {
            panic!("Single widened")
        };
        for (&x, actual) in singles.iter().zip(actual) {
            let expected = match operation {
                Unary::Sin => x.sin(),
                Unary::Cos => x.cos(),
                Unary::Abs => x.abs(),
                Unary::Exp => x.exp(),
                Unary::Log => x.ln(),
                Unary::Sqrt => {
                    if x == 0.0 {
                        0.0
                    } else {
                        x.sqrt()
                    }
                }
            };
            assert!(
                actual.to_bits() == expected.to_bits() || actual.is_nan() && expected.is_nan(),
                "{operation:?}: {x} => {actual}, expected {expected}"
            );
        }
        let result = execute(&provider, &double, operation).unwrap();
        assert_eq!(result.shape(), &[1, 2, 3]);
        assert_eq!(result.physical().element_type(), NumericElementType::F64);
        let HostNumericDataOwned::F64(actual) =
            block_on(provider.download_numeric(&result)).unwrap().data
        else {
            panic!("Double narrowed")
        };
        for (&x, actual) in doubles.iter().zip(actual) {
            let expected = match operation {
                Unary::Sin => x.sin(),
                Unary::Cos => x.cos(),
                Unary::Abs => x.abs(),
                Unary::Exp => x.exp(),
                Unary::Log => x.ln(),
                Unary::Sqrt => {
                    if x == 0.0 {
                        0.0
                    } else {
                        x.sqrt()
                    }
                }
            };
            assert!(actual.to_bits() == expected.to_bits() || actual.is_nan() && expected.is_nan());
        }
    }
}

#[test]
fn complex_trig_and_magnitude_preserve_native_class() {
    let provider = InProcessProvider::new();
    for single in [true, false] {
        let data = if single {
            InProcessNumericData::F32(vec![0.25, -0.5, 3.0, 4.0])
        } else {
            InProcessNumericData::F64(vec![0.25, -0.5, 3.0, 4.0])
        };
        let input = provider
            .allocate_numeric_data(data, vec![1, 1, 2], GpuTensorStorage::ComplexInterleaved)
            .unwrap();
        for operation in [Unary::Sin, Unary::Cos, Unary::Abs] {
            let result = execute(&provider, &input, operation).unwrap();
            assert_eq!(result.shape(), input.shape());
            assert_eq!(
                result.physical().element_type(),
                input.physical().element_type()
            );
            assert_eq!(
                result.physical().storage(),
                if matches!(operation, Unary::Abs) {
                    GpuTensorStorage::Real
                } else {
                    GpuTensorStorage::ComplexInterleaved
                }
            );
            let expected: Vec<f64> = [(0.25f64, -0.5f64), (3.0, 4.0)]
                .into_iter()
                .flat_map(|(re, im)| match operation {
                    Unary::Sin => vec![re.sin() * im.cosh(), re.cos() * im.sinh()],
                    Unary::Cos => vec![re.cos() * im.cosh(), -re.sin() * im.sinh()],
                    Unary::Abs => vec![if single {
                        f64::from((re as f32).hypot(im as f32))
                    } else {
                        re.hypot(im)
                    }],
                    _ => unreachable!(),
                })
                .collect();
            match block_on(provider.download_numeric(&result)).unwrap().data {
                HostNumericDataOwned::F32(actual) => assert_eq!(
                    actual,
                    expected.iter().map(|&x| x as f32).collect::<Vec<_>>()
                ),
                HostNumericDataOwned::F64(actual) => assert_eq!(actual, expected),
                _ => panic!("unexpected native class"),
            }
        }
        for operation in [Unary::Exp, Unary::Log, Unary::Sqrt] {
            assert!(runmat_accelerate_api::is_unsupported_provider_operation(
                &execute(&provider, &input, operation).unwrap_err()
            ));
        }
    }
}

#[test]
fn extrema_and_empty_arrays_use_native_typed_storage() {
    let provider = InProcessProvider::new();
    for single in [true, false] {
        for empty in [false, true] {
            let values = if empty {
                vec![]
            } else {
                vec![f64::NAN, -2.0, 3.0]
            };
            let data = if single {
                InProcessNumericData::F32(values.iter().map(|&x| x as f32).collect())
            } else {
                InProcessNumericData::F64(values)
            };
            let shape = if empty {
                vec![usize::MAX, usize::MAX, 0]
            } else {
                vec![1, 3]
            };
            let input = provider
                .allocate_numeric_data(data, shape.clone(), GpuTensorStorage::Real)
                .unwrap();
            for minimum in [true, false] {
                let result = block_on(if minimum {
                    provider.reduce_min(&input)
                } else {
                    provider.reduce_max(&input)
                })
                .unwrap();
                assert_eq!(result.shape(), &[1, 1]);
                assert_eq!(
                    result.physical().element_type(),
                    input.physical().element_type()
                );
                let expected = if empty {
                    if minimum {
                        f64::INFINITY
                    } else {
                        f64::NEG_INFINITY
                    }
                } else if minimum {
                    -2.0
                } else {
                    3.0
                };
                match block_on(provider.download_numeric(&result)).unwrap().data {
                    HostNumericDataOwned::F32(actual) => assert_eq!(actual, vec![expected as f32]),
                    HostNumericDataOwned::F64(actual) => assert_eq!(actual, vec![expected]),
                    _ => panic!("unexpected native class"),
                }
            }
            if empty {
                for operation in OPERATIONS {
                    let result = execute(&provider, &input, operation).unwrap();
                    assert_eq!(result.shape(), shape);
                    assert_eq!(
                        result.physical().element_type(),
                        input.physical().element_type()
                    );
                }
            }
        }
    }
}

#[test]
fn authentication_precedes_unsupported_and_future_drop_preserves_borrowed_input() {
    let provider = InProcessProvider::new();
    let other = InProcessProvider::new();
    let input = provider
        .allocate_numeric_data(
            InProcessNumericData::F32(vec![0.5]),
            vec![1, 1],
            GpuTensorStorage::Real,
        )
        .unwrap();
    let input_wire = input.wire_descriptor();
    for operation in OPERATIONS {
        assert!(!runmat_accelerate_api::is_unsupported_provider_operation(
            &execute(&other, &input, operation).unwrap_err()
        ));
    }
    let future = provider.unary_sin(&input);
    drop(future);
    assert!(provider.acquire_live(&input_wire).is_ok());
    let result = execute(&provider, &input, Unary::Sin).unwrap();
    let result_wire = result.wire_descriptor();
    drop(result);
    assert!(provider.acquire_live(&result_wire).is_err());
    assert!(provider.acquire_live(&input_wire).is_ok());
    drop(input);
    assert!(provider.acquire_live(&input_wire).is_err());

    let logical = provider
        .allocate_produced(
            InProcessNumericData::F32(vec![1.0]),
            vec![1, 1],
            GpuTensorStorage::Real,
            true,
        )
        .unwrap();
    for operation in OPERATIONS {
        assert!(runmat_accelerate_api::is_unsupported_provider_operation(
            &execute(&provider, &logical, operation).unwrap_err()
        ));
    }
    assert!(runmat_accelerate_api::is_unsupported_provider_operation(
        &block_on(provider.reduce_min(&logical)).unwrap_err()
    ));
}
