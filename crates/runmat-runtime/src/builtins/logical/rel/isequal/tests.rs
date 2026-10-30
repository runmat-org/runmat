use super::*;

#[test]
fn string_missing_equality_is_operation_owned_for_scalars_arrays_and_nested_values() {
    use runmat_builtins::RelationalOperator;
    use runmat_value::StringElement;
    let assert_relational_missing = |a: &Value, b: &Value| {
        for operator in [
            RelationalOperator::Equal,
            RelationalOperator::NotEqual,
            RelationalOperator::LessThan,
            RelationalOperator::LessThanOrEqual,
            RelationalOperator::GreaterThan,
            RelationalOperator::GreaterThanOrEqual,
        ] {
            assert_eq!(
                crate::builtins::logical::rel::comparison::evaluate_host(
                    a.clone(),
                    b.clone(),
                    operator
                )
                .unwrap(),
                Value::Bool(operator == RelationalOperator::NotEqual),
            );
        }
    };
    let carriers = |element: StringElement| {
        vec![
            Value::String(element.clone()),
            Value::StringArray(
                StringArray::from_elements_unbounded(vec![element], vec![1, 1]).unwrap(),
            ),
        ]
    };
    for a in carriers(StringElement::Missing) {
        for b in carriers(StringElement::Missing) {
            assert_relational_missing(&a, &b);
            assert_eq!(
                run_isequal(vec![a.clone(), b.clone()]).unwrap(),
                Value::Bool(false)
            );
            assert_eq!(
                run_isequaln(vec![a.clone(), b.clone()]).unwrap(),
                Value::Bool(true)
            );
            let a = Value::Cell(CellArray::new(vec![a.clone()], 1, 1).unwrap());
            let b = Value::Cell(CellArray::new(vec![b], 1, 1).unwrap());
            assert_eq!(
                run_isequal(vec![a.clone(), b.clone()]).unwrap(),
                Value::Bool(false)
            );
            assert_eq!(run_isequaln(vec![a, b]).unwrap(), Value::Bool(true));
        }
        for literal in ["<missing>", "<MISSING>", "", "text"] {
            for b in carriers(literal.into()) {
                assert_relational_missing(&a, &b);
                assert_eq!(
                    run_isequal(vec![a.clone(), b.clone()]).unwrap(),
                    Value::Bool(false)
                );
                assert_eq!(
                    run_isequaln(vec![a.clone(), b.clone()]).unwrap(),
                    Value::Bool(false)
                );
                assert_eq!(
                    run_isequal(vec![b.clone(), b.clone()]).unwrap(),
                    Value::Bool(true)
                );
                assert_eq!(run_isequaln(vec![b.clone(), b]).unwrap(), Value::Bool(true));
            }
        }
    }
}
use crate::builtins::common::gpu_helpers;
use futures::executor::block_on;
use runmat_value::{CellArray, ComplexTensor, IntegerComplexStorage, IntegerStorage};

fn run_isequal(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    block_on(isequal_builtin(
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
}

fn run_isequaln(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    block_on(isequaln_builtin(
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
}

fn typed_complex_u64(real: u64, imag: u64) -> ComplexTensor {
    ComplexTensor::new_integer(
        IntegerComplexStorage::new(
            IntegerStorage::U64(vec![real]),
            IntegerStorage::U64(vec![imag]),
        )
        .expect("matching components"),
        vec![1, 1],
    )
    .expect("typed complex")
}

#[test]
fn numeric_equality_is_class_insensitive_without_losing_wide_integers() {
    let signed = Tensor::new_integer(IntegerStorage::I64(vec![255]), vec![1, 1]).unwrap();
    let unsigned = Tensor::new_integer(IntegerStorage::U8(vec![255]), vec![1, 1]).unwrap();
    assert_eq!(
        run_isequal(vec![Value::Tensor(signed), Value::Tensor(unsigned)]).unwrap(),
        Value::Bool(true)
    );

    let wide = (1_u64 << 53) + 1;
    let exact = Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).unwrap();
    let rounded = Tensor::new(vec![wide as f64], vec![1, 1]).unwrap();
    assert_eq!(
        run_isequal(vec![Value::Tensor(exact), Value::Tensor(rounded)]).unwrap(),
        Value::Bool(false)
    );
    assert_eq!(
        run_isequal(vec![Value::Bool(true), Value::Num(1.0)]).unwrap(),
        Value::Bool(true)
    );
}

#[test]
fn real_complex_and_sparse_numeric_values_compare_across_classes() {
    let wide = u64::MAX;
    let real = Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).unwrap();
    let complex = typed_complex_u64(wide, 0);
    assert_eq!(
        run_isequal(vec![Value::Tensor(real), Value::ComplexTensor(complex)]).unwrap(),
        Value::Bool(true)
    );
    assert_eq!(
        run_isequal(vec![Value::Int(IntValue::I32(1)), Value::Complex(1.0, 0.0)]).unwrap(),
        Value::Bool(true)
    );
    let sparse =
        SparseTensor::new_integer(2, 1, vec![0, 1], vec![1], IntegerStorage::U64(vec![wide]))
            .unwrap();
    let dense = Tensor::new_integer(IntegerStorage::U64(vec![0, wide]), vec![2, 1]).unwrap();
    assert_eq!(
        run_isequal(vec![Value::SparseTensor(sparse), Value::Tensor(dense)]).unwrap(),
        Value::Bool(true)
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn isequal_two_scalars_equal() {
    let result = run_isequal(vec![Value::Num(5.0), Value::Num(5.0)]).expect("isequal");
    assert_eq!(result, Value::Bool(true));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn isequal_two_scalars_not_equal() {
    let result = run_isequal(vec![Value::Num(5.0), Value::Num(4.0)]).expect("isequal");
    assert_eq!(result, Value::Bool(false));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn isequal_three_args_all_equal() {
    let result =
        run_isequal(vec![Value::Num(3.0), Value::Num(3.0), Value::Num(3.0)]).expect("isequal");
    assert_eq!(result, Value::Bool(true));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn isequal_three_args_one_different() {
    let result =
        run_isequal(vec![Value::Num(3.0), Value::Num(3.0), Value::Num(4.0)]).expect("isequal");
    assert_eq!(result, Value::Bool(false));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn isequal_tensors_equal() {
    let t1 = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let t2 = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let result = run_isequal(vec![Value::Tensor(t1), Value::Tensor(t2)]).expect("isequal");
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn equality_ignores_only_trailing_singleton_axes_for_numeric_arrays() {
    for extra_axes in [vec![2, 1, 1], vec![2, 1, 1, 1]] {
        let complex = |shape| {
            Value::ComplexTensor(ComplexTensor::new(vec![(1.0, 2.0), (3.0, -4.0)], shape).unwrap())
        };
        let real = |shape| Value::Tensor(Tensor::new(vec![1.0, 3.0], shape).unwrap());
        for inputs in [
            vec![complex(extra_axes.clone()), complex(vec![2, 1])],
            vec![real(extra_axes), real(vec![2, 1])],
        ] {
            assert_eq!(run_isequal(inputs.clone()).unwrap(), Value::Bool(true));
            assert_eq!(run_isequaln(inputs).unwrap(), Value::Bool(true));
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn isequal_tensors_different_shape() {
    let t1 = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let t2 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let result = run_isequal(vec![Value::Tensor(t1), Value::Tensor(t2)]).expect("isequal");
    assert_eq!(result, Value::Bool(false));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn isequal_preserves_exact_rank_identity_and_scalar_bridges() {
    let rank_zero = || Value::Tensor(Tensor::new(vec![7.0], Vec::new()).unwrap());
    let rank_one = || Value::Tensor(Tensor::new(vec![7.0], vec![1]).unwrap());
    let matrix_scalar = || Value::Tensor(Tensor::new(vec![7.0], vec![1, 1]).unwrap());

    assert_eq!(
        run_isequal(vec![rank_zero(), rank_zero()]).unwrap(),
        Value::Bool(true)
    );
    assert_eq!(
        run_isequal(vec![rank_zero(), rank_one()]).unwrap(),
        Value::Bool(false)
    );
    assert_eq!(
        run_isequal(vec![rank_zero(), matrix_scalar()]).unwrap(),
        Value::Bool(false)
    );
    assert_eq!(
        run_isequal(vec![Value::Num(7.0), rank_zero()]).unwrap(),
        Value::Bool(false)
    );
    assert_eq!(
        run_isequal(vec![Value::Num(7.0), rank_one()]).unwrap(),
        Value::Bool(false)
    );
    assert_eq!(
        run_isequal(vec![Value::Num(7.0), matrix_scalar()]).unwrap(),
        Value::Bool(true)
    );

    let rank_zero_complex =
        || Value::ComplexTensor(ComplexTensor::new(vec![(7.0, 0.0)], Vec::new()).unwrap());
    assert_eq!(
        run_isequal(vec![rank_zero(), rank_zero_complex()]).unwrap(),
        Value::Bool(true)
    );
    assert_eq!(
        run_isequal(vec![Value::Complex(7.0, 0.0), rank_zero_complex()]).unwrap(),
        Value::Bool(false)
    );
    assert_eq!(
        run_isequal(vec![
            Value::Complex(7.0, 0.0),
            Value::ComplexTensor(ComplexTensor::new(vec![(7.0, 0.0)], vec![1, 1]).unwrap()),
        ])
        .unwrap(),
        Value::Bool(true)
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn isequal_tensors_different_values() {
    let t1 = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let t2 = Tensor::new(vec![1.0, 2.0, 4.0], vec![1, 3]).unwrap();
    let result = run_isequal(vec![Value::Tensor(t1), Value::Tensor(t2)]).expect("isequal");
    assert_eq!(result, Value::Bool(false));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn isequal_empty_arrays() {
    // Test that empty arrays are equal
    let empty_a = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
    let empty_b = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
    let result =
        run_isequal(vec![Value::Tensor(empty_a), Value::Tensor(empty_b)]).expect("isequal");
    assert_eq!(result, Value::Bool(true));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn isequal_empty_cell_arrays() {
    // Test the cell example from the failing test: cell(2,2) elements should be []
    let empty = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
    let c1 = CellArray::new(vec![Value::Tensor(empty.clone()); 4], 2, 2).unwrap();
    let c2 = CellArray::new(vec![Value::Tensor(empty); 4], 2, 2).unwrap();
    let result = run_isequal(vec![Value::Cell(c1), Value::Cell(c2)]).expect("isequal");
    assert_eq!(result, Value::Bool(true));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn isequal_cell_element_with_empty() {
    // Test isequal(C{1,1}, [], C{2,2}, []) pattern
    let empty_a = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
    let empty_b = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
    let empty_c = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
    let empty_d = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
    let result = run_isequal(vec![
        Value::Tensor(empty_a),
        Value::Tensor(empty_b),
        Value::Tensor(empty_c),
        Value::Tensor(empty_d),
    ])
    .expect("isequal");
    assert_eq!(result, Value::Bool(true));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn isequal_nan_not_equal() {
    // In isequal, NaN != NaN (use isequaln for NaN equality)
    let result = run_isequal(vec![Value::Num(f64::NAN), Value::Num(f64::NAN)]).expect("isequal");
    assert_eq!(result, Value::Bool(false));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn isequaln_nan_values_are_equal_recursively() {
    let t1 = Tensor::new(vec![1.0, f64::NAN], vec![1, 2]).unwrap();
    let t2 = Tensor::new(vec![1.0, f64::NAN], vec![1, 2]).unwrap();
    let cells = CellArray::new(vec![Value::Tensor(t1), Value::Num(f64::NAN)], 1, 2).unwrap();
    let other = CellArray::new(vec![Value::Tensor(t2), Value::Num(f64::NAN)], 1, 2).unwrap();
    let result = run_isequaln(vec![Value::Cell(cells), Value::Cell(other)]).expect("isequaln");
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn isequal_and_isequaln_compare_single_storage_without_widening_class() {
    let left = Tensor::from_f32(vec![0.1, f32::NAN], vec![1, 2]).unwrap();
    let right = Tensor::from_f32(vec![0.1, f32::NAN], vec![1, 2]).unwrap();

    assert_eq!(
        run_isequal(vec![
            Value::Tensor(left.clone()),
            Value::Tensor(right.clone())
        ])
        .expect("isequal"),
        Value::Bool(false)
    );
    assert_eq!(
        run_isequaln(vec![Value::Tensor(left), Value::Tensor(right)]).expect("isequaln"),
        Value::Bool(true)
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn isequal_strings() {
    let result = run_isequal(vec![
        Value::String("hello".into()),
        Value::String("hello".into()),
    ])
    .expect("isequal");
    assert_eq!(result, Value::Bool(true));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn isequal_different_types() {
    let result = run_isequal(vec![Value::Num(5.0), Value::String("5".into())]).expect("isequal");
    assert_eq!(result, Value::Bool(false));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn isequal_not_enough_args() {
    let err = run_isequal(vec![Value::Num(5.0)]).unwrap_err();
    assert!(err.message().contains("at least two"));
    assert_eq!(err.identifier(), ISEQUAL_ERROR_NOT_ENOUGH_INPUTS.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn isequal_bool_and_num_compare_by_numeric_value() {
    let result = run_isequal(vec![Value::Bool(true), Value::Num(1.0)]).expect("isequal");
    assert_eq!(result, Value::Bool(true));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn isequal_tensor_numeric_classes_compare_by_value() {
    let double_tensor =
        Tensor::new_with_dtype(vec![1.0], vec![1, 1], runmat_value::NumericDType::F64).unwrap();
    let uint32_tensor =
        Tensor::new_with_dtype(vec![1.0], vec![1, 1], runmat_value::NumericDType::U32).unwrap();
    let result = run_isequal(vec![
        Value::Tensor(double_tensor),
        Value::Tensor(uint32_tensor),
    ])
    .expect("isequal");
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn isequal_integer_tensor_values_are_compared_exactly() {
    let same_left = Tensor::new_integer(
        IntegerStorage::U64(vec![(1_u64 << 53) + 1, u64::MAX]),
        vec![1, 2],
    )
    .expect("left integer tensor");
    let same_right = Tensor::new_integer(
        IntegerStorage::U64(vec![(1_u64 << 53) + 1, u64::MAX]),
        vec![1, 2],
    )
    .expect("right integer tensor");
    let different = Tensor::new_integer(
        IntegerStorage::U64(vec![(1_u64 << 53), u64::MAX]),
        vec![1, 2],
    )
    .expect("different integer tensor");

    assert_eq!(
        run_isequal(vec![Value::Tensor(same_left), Value::Tensor(same_right)])
            .expect("isequal same"),
        Value::Bool(true)
    );
    assert_eq!(
        run_isequal(vec![
            Value::Tensor(different),
            Value::Tensor(
                Tensor::new_integer(
                    IntegerStorage::U64(vec![(1_u64 << 53) + 1, u64::MAX]),
                    vec![1, 2],
                )
                .expect("comparison tensor"),
            ),
        ])
        .expect("isequal different"),
        Value::Bool(false)
    );
}

#[test]
fn isequal_integer_tensor_classes_compare_exact_values() {
    let unsigned = Tensor::new_integer(IntegerStorage::U64(vec![255]), vec![1, 1])
        .expect("unsigned integer tensor");
    let signed = Tensor::new_integer(IntegerStorage::I64(vec![255]), vec![1, 1])
        .expect("signed integer tensor");

    assert_eq!(
        run_isequal(vec![Value::Tensor(unsigned), Value::Tensor(signed)]).expect("isequal"),
        Value::Bool(true)
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn isequal_complex() {
    let result =
        run_isequal(vec![Value::Complex(1.0, 2.0), Value::Complex(1.0, 2.0)]).expect("isequal");
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn isequal_typed_complex_integer_compares_exact_components() {
    let left = typed_complex_u64(u64::MAX, 1_u64 << 63);
    let same = left.clone();
    let different = typed_complex_u64(u64::MAX - 1, 1_u64 << 63);

    assert_eq!(
        run_isequal(vec![Value::ComplexTensor(left), Value::ComplexTensor(same)]).expect("isequal"),
        Value::Bool(true)
    );
    assert_eq!(
        run_isequal(vec![
            Value::ComplexTensor(different),
            Value::ComplexTensor(typed_complex_u64(u64::MAX, 1_u64 << 63)),
        ])
        .expect("isequal"),
        Value::Bool(false)
    );
}

#[test]
fn errors_are_identity_specific_and_validate_before_materialization() {
    for nan in [false, true] {
        let name = if nan { "isequaln" } else { "isequal" };
        let resident = Value::GpuTensor(
            runmat_accelerate_api::test_support::TestAllocationOwner::new()
                .allocate_zeroed(
                    vec![1, 1],
                    runmat_accelerate_api::NumericElementType::F64,
                    runmat_accelerate_api::GpuTensorStorage::Real,
                )
                .unwrap()
                .handle,
        );
        let error = block_on(equality_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![resident.clone()]),
            nan,
            name,
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some(format!("RunMat:{name}:NotEnoughInputs").as_str())
        );
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let error = block_on(equality_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                resident.clone(),
                resident,
            ]),
            nan,
            name,
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some(format!("RunMat:{name}:TooManyOutputs").as_str())
        );
    }
}

#[test]
fn discarded_outputs_still_validate_and_compare() {
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    assert_eq!(
        run_isequal(vec![Value::Num(2.0), Value::Num(2.0)]).unwrap(),
        Value::Bool(true)
    );
    assert!(run_isequaln(vec![]).is_err());
}

#[test]
fn nested_resident_values_gather_exactly_without_consuming_inputs() {
    use crate::builtins::common::test_support;
    use runmat_value::StructArray;
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new_integer(
            IntegerStorage::U64(vec![(1_u64 << 53) + 1, u64::MAX]),
            vec![1, 2],
        )
        .unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &tensor).unwrap();
        let mut resident = runmat_value::UnboundedStructValueBuilder::new();
        resident.insert(
            "payload",
            Value::Cell(CellArray::new(vec![Value::GpuTensor(handle.clone())], 1, 1).unwrap()),
        );
        let mut host = runmat_value::UnboundedStructValueBuilder::new();
        host.insert(
            "payload",
            Value::Cell(CellArray::new(vec![Value::Tensor(tensor.clone())], 1, 1).unwrap()),
        );
        let host = host.finish();
        let resident = resident.finish();
        for nan in [false, true] {
            let name = if nan { "isequaln" } else { "isequal" };
            assert_eq!(
                block_on(equality_builtin(
                    crate::builtins::common::test_support::builtin_arguments(vec![
                        Value::Struct(resident.clone()),
                        Value::Struct(host.clone())
                    ]),
                    nan,
                    name
                ))
                .unwrap(),
                Value::Bool(true)
            );
            let left =
                StructArray::new(vec![resident.clone(), resident.clone()], vec![1, 2]).unwrap();
            let right = StructArray::new(vec![host.clone(), host.clone()], vec![1, 2]).unwrap();
            assert_eq!(
                block_on(equality_builtin(
                    crate::builtins::common::test_support::builtin_arguments(vec![
                        Value::StructArray(left),
                        Value::StructArray(right)
                    ]),
                    nan,
                    name
                ))
                .unwrap(),
                Value::Bool(true)
            );
        }
        let after = block_on(gpu_helpers::download_value_preserving_residency_async(
            provider, &handle,
        ))
        .unwrap();
        assert!(values_equal(&after, &Value::Tensor(tensor), false));
        handle.release().unwrap();
    });
}

#[test]
fn nested_unowned_resident_input_is_an_error_not_false() {
    let nested = Value::Cell(
        CellArray::new(
            vec![Value::GpuTensor(
                runmat_accelerate_api::test_support::TestAllocationOwner::new()
                    .allocate_zeroed(
                        vec![1, 1],
                        runmat_accelerate_api::NumericElementType::F64,
                        runmat_accelerate_api::GpuTensorStorage::Real,
                    )
                    .unwrap()
                    .handle,
            )],
            1,
            1,
        )
        .unwrap(),
    );
    let error = run_isequaln(vec![nested.clone(), nested]).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:isequaln:Internal"));
}

#[test]
fn empty_nd_and_complex_nan_boundaries() {
    let empty = |shape| Value::Tensor(Tensor::new(vec![], shape).unwrap());
    assert_eq!(
        run_isequal(vec![empty(vec![0, 2, 3]), empty(vec![0, 2, 3])]).unwrap(),
        Value::Bool(true)
    );
    assert_eq!(
        run_isequaln(vec![empty(vec![0, 2, 3]), empty(vec![0, 3, 2])]).unwrap(),
        Value::Bool(false)
    );
    let value = Value::Complex(f64::NAN, 2.0);
    assert_eq!(
        run_isequal(vec![value.clone(), value.clone()]).unwrap(),
        Value::Bool(false)
    );
    assert_eq!(
        run_isequaln(vec![value.clone(), value]).unwrap(),
        Value::Bool(true)
    );
}

#[test]
fn structs_compare_by_field_name_not_insertion_order() {
    use runmat_value::StructArray;
    let record = |reverse: bool, nan: bool| {
        let mut value = runmat_value::UnboundedStructValueBuilder::new();
        let a = Value::Num(if nan { f64::NAN } else { 1.0 });
        if reverse {
            value.insert("b", Value::Bool(true));
            value.insert("a", a);
        } else {
            value.insert("a", a);
            value.insert("b", Value::Bool(true));
        }
        value.finish()
    };
    for nan in [false, true] {
        for nan_equal in [false, true] {
            let left = record(false, nan);
            let right = record(true, nan);
            let expected = !nan || nan_equal;
            assert_eq!(
                values_equal(
                    &Value::Struct(left.clone()),
                    &Value::Struct(right.clone()),
                    nan_equal
                ),
                expected
            );
            let left_array = StructArray::new(vec![left.clone(), left], vec![1, 2]).unwrap();
            let right_array = StructArray::new(vec![right.clone(), right], vec![1, 2]).unwrap();
            assert_eq!(
                values_equal(
                    &Value::StructArray(left_array),
                    &Value::StructArray(right_array),
                    nan_equal
                ),
                expected
            );
        }
    }
    let left = record(false, false);
    let mut renamed = record(true, false);
    crate::builtins::common::test_support::remove_struct_field_unbounded(&mut renamed, "a");
    crate::builtins::common::test_support::replace_struct_field_unbounded(
        &mut renamed,
        "c",
        Value::Num(1.0),
    );
    assert!(!values_equal(
        &Value::Struct(left.clone()),
        &Value::Struct(renamed.clone()),
        false
    ));
    let left_array = StructArray::new(vec![left.clone(), left], vec![1, 2]).unwrap();
    let right_array = StructArray::new(vec![renamed.clone(), renamed], vec![1, 2]).unwrap();
    assert!(!values_equal(
        &Value::StructArray(left_array),
        &Value::StructArray(right_array),
        true
    ));
    let empty_left = StructArray::empty(vec!["a".into(), "b".into()], vec![0, 2]).unwrap();
    let empty_right = StructArray::empty(vec!["b".into(), "a".into()], vec![0, 2]).unwrap();
    assert!(values_equal(
        &Value::StructArray(empty_left),
        &Value::StructArray(empty_right),
        false
    ));
}

#[test]
fn complex_scalar_tensor_bridge_is_symmetric_and_exact() {
    let scalar = Value::Complex(3.0, 2.0);
    for tensor in [
        ComplexTensor::new(vec![(3.0, 2.0)], vec![1, 1]).unwrap(),
        ComplexTensor::from_f32(vec![(3.0, 2.0)], vec![1, 1]).unwrap(),
        typed_complex_u64(3, 2),
    ] {
        let array = Value::ComplexTensor(tensor);
        assert_eq!(
            run_isequal(vec![scalar.clone(), array.clone()]).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            run_isequaln(vec![array, scalar.clone()]).unwrap(),
            Value::Bool(true)
        );
    }
    let wide = (1_u64 << 53) + 1;
    assert!(!values_equal(
        &Value::Complex(wide as f64, 0.0),
        &Value::ComplexTensor(typed_complex_u64(wide, 0)),
        false
    ));
    let nonscalar =
        Value::ComplexTensor(ComplexTensor::new(vec![(3.0, 2.0); 2], vec![1, 2]).unwrap());
    assert!(!values_equal(&scalar, &nonscalar, false));
    for (real, imag) in [(f64::NAN, 2.0), (3.0, f64::NAN)] {
        let scalar = Value::Complex(real, imag);
        let array =
            Value::ComplexTensor(ComplexTensor::new(vec![(real, imag)], vec![1, 1]).unwrap());
        assert_eq!(
            run_isequal(vec![scalar.clone(), array.clone()]).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            run_isequaln(vec![scalar.clone(), array.clone()]).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            run_isequaln(vec![array, scalar]).unwrap(),
            Value::Bool(true)
        );
    }
}
