use super::*;
use futures::executor::block_on;

#[test]
fn native_include_contract_matches_actual_host_domains() {
    runmat_gc::gc_test_context(|| {
        let provider = InProcessProvider::new();
        for single in [true, false] {
            for (shape, values) in [
                (
                    vec![3, 2],
                    vec![f64::INFINITY, f64::NAN, 2.0, -0.0, 0.0, -0.0],
                ),
                (vec![2, 2], vec![f64::NAN; 4]),
                (vec![3, 2], vec![0.0, -0.0, -0.0, -0.0, 0.0, 0.0]),
                (vec![0, 3], vec![]),
                (vec![3, 0], vec![]),
            ] {
                let data = if single {
                    InProcessNumericData::F32(values.iter().map(|&x| x as f32).collect())
                } else {
                    InProcessNumericData::F64(values.clone())
                };
                let host_data = if single {
                    runmat_value::NumericStorage::F32(values.iter().map(|&x| x as f32).collect())
                } else {
                    runmat_value::NumericStorage::F64(values)
                };
                let host =
                    runmat_value::Tensor::from_numeric_storage(host_data, shape.clone()).unwrap();
                let input = provider
                    .allocate_numeric_data(data, shape, GpuTensorStorage::Real)
                    .unwrap();
                for (name, minimum) in [("min", true), ("max", false)] {
                    for dim in [0, 1] {
                        let host_result =
                            block_on(runmat_runtime::call_builtin_async_with_outputs(
                                name,
                                &[
                                    Value::Tensor(host.clone()),
                                    Value::Tensor(
                                        runmat_value::Tensor::new(Vec::new(), vec![0, 0]).unwrap(),
                                    ),
                                    Value::Num((dim + 1) as f64),
                                    Value::from("includenan"),
                                ],
                                2,
                            ))
                            .unwrap();
                        let Value::OutputList(expected) = host_result.value() else {
                            panic!("two host outputs required")
                        };
                        let actual = result(&provider, &input, dim, minimum).unwrap();
                        for (output_index, (actual, expected)) in [&actual.values, &actual.indices]
                            .into_iter()
                            .zip(expected.iter())
                            .enumerate()
                        {
                            let expected = match expected {
                                Value::Tensor(t) => t.clone(),
                                Value::Num(x) => {
                                    runmat_value::Tensor::new(vec![*x], vec![1, 1]).unwrap()
                                }
                                _ => panic!("real host extrema expected"),
                            };
                            assert_eq!(actual.shape(), expected.shape());
                            let downloaded = block_on(provider.download(actual)).unwrap();
                            let values =
                                runmat_runtime::builtins::common::tensor::tensor_values_f64_cow(
                                    &expected,
                                );
                            assert_eq!(downloaded.data.len(), values.len());
                            for (index, (a, e)) in
                                downloaded.data.iter().zip(values.iter()).enumerate()
                            {
                                assert!(a.to_bits() == e.to_bits() || a.is_nan() && e.is_nan(),
                                    "{name}: single={single}, dim={dim}, input_shape={:?}, output={output_index}, element={index}: actual={a:?} ({:#x}), host={e:?} ({:#x})",
                                    input.shape(), a.to_bits(), e.to_bits());
                            }
                        }
                    }
                }
            }
        }
    });
}

fn result(
    provider: &InProcessProvider,
    input: &GpuTensorHandle,
    dim: usize,
    minimum: bool,
) -> runmat_accelerate_api::ProviderExecutionResult<runmat_accelerate_api::ReduceDimResult> {
    block_on(if minimum {
        provider.reduce_min_dim(input, dim)
    } else {
        provider.reduce_max_dim(input, dim)
    })
}

#[test]
fn native_dimensions_select_first_nan_and_first_tie() {
    let provider = InProcessProvider::new();
    for single in [true, false] {
        let source = [
            f64::INFINITY,
            f64::NAN,
            3.0,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            -0.0,
            0.0,
            -0.0,
        ];
        let data = if single {
            InProcessNumericData::F32(source.iter().map(|&x| x as f32).collect())
        } else {
            InProcessNumericData::F64(source.to_vec())
        };
        let input = provider
            .allocate_numeric_data(data, vec![3, 3], GpuTensorStorage::Real)
            .unwrap();
        for minimum in [true, false] {
            for dim in [0, 1] {
                let out = result(&provider, &input, dim, minimum).unwrap();
                assert_eq!(
                    out.values.physical().element_type(),
                    input.physical().element_type()
                );
                assert_eq!(
                    out.indices.physical().element_type(),
                    NumericElementType::F64
                );
                let expected_indices = if dim == 0 {
                    vec![2.0, 1.0, if minimum { 1.0 } else { 2.0 }]
                } else {
                    vec![2.0, 1.0, 2.0]
                };
                assert_eq!(
                    block_on(provider.download(&out.indices)).unwrap().data,
                    expected_indices
                );
                let values = block_on(provider.download_numeric(&out.values)).unwrap();
                assert_eq!(values.shape, if dim == 0 { vec![1, 3] } else { vec![3, 1] });
                let actual: Vec<f64> = match values.data {
                    HostNumericDataOwned::F32(v) => v.into_iter().map(f64::from).collect(),
                    HostNumericDataOwned::F64(v) => v,
                    _ => panic!("wrong class"),
                };
                assert!(actual[0].is_nan() && actual[1].is_nan());
                if dim == 0 {
                    assert_eq!(
                        actual[2].to_bits(),
                        (if minimum { -0.0f64 } else { 0.0f64 }).to_bits()
                    );
                } else {
                    assert!(actual[2].is_nan());
                }
                assert!(!out.values.shares_allocation_with(&input));
                assert!(!out.indices.shares_allocation_with(&out.values));
                let wire = out.values.wire_descriptor();
                drop(out);
                assert!(provider.acquire_live(&wire).is_err());
                input.allocation_lease().validate_live().unwrap();
            }
        }
    }
}

#[test]
fn asymmetric_native_dimensions_preserve_values_after_transpose() {
    let provider = InProcessProvider::new();
    for single in [true, false] {
        let data = if single {
            InProcessNumericData::F32(vec![3.0, 1.0, -2.0, 7.0, 6.0, 4.0])
        } else {
            InProcessNumericData::F64(vec![3.0, 1.0, -2.0, 7.0, 6.0, 4.0])
        };
        let input = provider
            .allocate_numeric_data(data, vec![2, 3], GpuTensorStorage::Real)
            .unwrap();
        let input = provider.transpose(&input).unwrap();
        for (minimum, dim, values, indices) in [
            (true, 0, vec![-2.0, 1.0], vec![2.0, 1.0]),
            (false, 0, vec![6.0, 7.0], vec![3.0, 2.0]),
            (true, 1, vec![1.0, -2.0, 4.0], vec![2.0, 1.0, 2.0]),
            (false, 1, vec![3.0, 7.0, 6.0], vec![1.0, 2.0, 1.0]),
        ] {
            let output = result(&provider, &input, dim, minimum).unwrap();
            assert_eq!(
                block_on(provider.download(&output.values)).unwrap().data,
                values
            );
            assert_eq!(
                block_on(provider.download(&output.indices)).unwrap().data,
                indices
            );
        }
    }
}

#[test]
fn empty_native_extrema_preserve_shape_and_integer_class() {
    let provider = InProcessProvider::new();
    for shape in [vec![0, 3], vec![3, 0], vec![0, 0]] {
        for data in [
            InProcessNumericData::F32(vec![]),
            InProcessNumericData::F64(vec![]),
            InProcessNumericData::Integer(HostIntegerDataOwned::U64(vec![])),
        ] {
            let input = provider
                .allocate_numeric_data(data, shape.clone(), GpuTensorStorage::Real)
                .unwrap();
            for dim in [0, 1] {
                for minimum in [true, false] {
                    let output = result(&provider, &input, dim, minimum).unwrap();
                    assert_eq!(output.values.shape(), shape);
                    assert_eq!(output.indices.shape(), shape);
                    assert_eq!(
                        output.values.physical().element_type(),
                        input.physical().element_type()
                    );
                    assert_eq!(
                        output.indices.physical().element_type(),
                        NumericElementType::F64
                    );
                }
            }
        }
    }
}

#[test]
fn unsupported_shapes_are_not_foreign_owner_errors() {
    let provider = InProcessProvider::new();
    let other = InProcessProvider::new();
    let nd = provider
        .allocate_numeric_data(
            InProcessNumericData::F32(vec![1.0, 2.0]),
            vec![1, 1, 2],
            GpuTensorStorage::Real,
        )
        .unwrap();
    let unsupported_shape = result(&provider, &nd, 0, true).unwrap_err();
    assert!(!unsupported_shape.is_unsupported());
    assert!(!unsupported_shape.is_after_effect());
    assert!(std::error::Error::source(&unsupported_shape)
        .unwrap()
        .is::<runmat_accelerate_api::UnsupportedProviderOperation>());
    let foreign_owner = result(&other, &nd, 0, true).unwrap_err();
    assert!(!std::error::Error::source(&foreign_owner)
        .unwrap()
        .is::<runmat_accelerate_api::UnsupportedProviderOperation>());
    let matrix = nd
        .with_view(runmat_accelerate_api::allocation::GpuTensorView::contiguous(vec![1, 2]))
        .unwrap();
    let unsupported_dimension = result(&provider, &matrix, 2, false).unwrap_err();
    assert!(!unsupported_dimension.is_unsupported());
    assert!(!unsupported_dimension.is_after_effect());
    assert!(std::error::Error::source(&unsupported_dimension)
        .unwrap()
        .is::<runmat_accelerate_api::UnsupportedProviderOperation>());
    let pending = provider.reduce_min_dim(&matrix, 0);
    drop(pending);
    matrix.allocation_lease().validate_live().unwrap();
}
