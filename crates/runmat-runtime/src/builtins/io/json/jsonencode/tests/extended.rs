use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn jsonencode_preserves_rank_zero_and_rank_one_numeric_serialization() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    for shape in [Vec::new(), vec![1]] {
        let real = Value::Tensor(Tensor::new(vec![7.0], shape.clone()).unwrap());
        assert_eq!(
            as_string(block_on(jsonencode_builtin(None, real, Vec::new())).unwrap()),
            "7"
        );
        let integer = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), shape.clone()).unwrap(),
        );
        assert_eq!(
            as_string(block_on(jsonencode_builtin(None, integer, Vec::new())).unwrap()),
            u64::MAX.to_string()
        );
        let complex = Value::ComplexTensor(ComplexTensor::new(vec![(1.0, 2.0)], shape).unwrap());
        assert_eq!(
            as_string(block_on(jsonencode_builtin(None, complex, Vec::new())).unwrap()),
            "{\"real\":1,\"imag\":2}"
        );
    }
    let vector = Value::Tensor(Tensor::new(vec![7.0, 8.0], vec![2]).unwrap());
    assert_eq!(
        as_string(block_on(jsonencode_builtin(None, vector, Vec::new())).unwrap()),
        "[7,8]"
    );
    let vector =
        Value::ComplexTensor(ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![2]).unwrap());
    assert_eq!(
        as_string(block_on(jsonencode_builtin(None, vector, Vec::new())).unwrap()),
        "[{\"real\":1,\"imag\":2},{\"real\":3,\"imag\":4}]"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsonencode_options_reject_non_scalar_tensor_bool() {
    let tensor = Tensor::new(vec![1.0, 0.0], vec![1, 2]).expect("tensor");
    let err = block_on(jsonencode_builtin(
        None,
        Value::Num(1.0),
        vec![Value::from("PrettyPrint"), Value::Tensor(tensor)],
    ))
    .expect_err("expected failure");
    assert_eq!(error_message(err), JSONENCODE_ERROR_OPTION_VALUE.message);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsonencode_options_accept_scalar_logical_array() {
    let logical = LogicalArray::new(vec![1], vec![1]).expect("logical");
    let args = vec![Value::from("PrettyPrint"), Value::LogicalArray(logical)];
    let encoded = block_on(jsonencode_builtin(None, Value::Num(7.0), args)).expect("jsonencode");
    assert_eq!(as_string(encoded), "7");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsonencode_convert_inf_and_nan_controls_null_output() {
    let tensor = Tensor::new(vec![1.0, f64::NAN], vec![1, 2]).expect("tensor");
    let encoded = block_on(jsonencode_builtin(
        None,
        Value::Tensor(tensor.clone()),
        Vec::new(),
    ))
    .expect("jsonencode");
    assert_eq!(as_string(encoded), "[1,null]");

    let err = block_on(jsonencode_builtin(
        None,
        Value::Tensor(tensor),
        vec![Value::from("ConvertInfAndNaN"), Value::Bool(false)],
    ))
    .expect_err("expected failure");
    assert_eq!(error_message(err), JSONENCODE_ERROR_INF_NAN.message);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsonencode_cell_array() {
    let elements = vec![Value::from(1.0), Value::from("two")];
    let cell = CellArray::new(elements, 1, 2).expect("cell");
    let encoded =
        block_on(jsonencode_builtin(None, Value::Cell(cell), Vec::new())).expect("jsonencode");
    assert_eq!(as_string(encoded), "[1,\"two\"]");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsonencode_char_array_zero_rows_is_empty_array() {
    let chars = CharArray::from_code_units(Vec::new(), 0, 3).expect("char array");
    let encoded = block_on(jsonencode_builtin(
        None,
        Value::CharArray(chars),
        Vec::new(),
    ))
    .expect("jsonencode");
    assert_eq!(as_string(encoded), "[]");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsonencode_char_array_empty_strings_per_row() {
    let chars = CharArray::from_code_units(Vec::new(), 2, 0).expect("char array");
    let encoded = block_on(jsonencode_builtin(
        None,
        Value::CharArray(chars),
        Vec::new(),
    ))
    .expect("jsonencode");
    let encoded_str = as_string(encoded);
    assert_eq!(encoded_str, "[\"\",\"\"]");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsonencode_string_array_matrix() {
    let sa = StringArray::new(vec!["alpha".to_string(), "beta".to_string()], vec![2, 1])
        .expect("string array");
    let encoded =
        block_on(jsonencode_builtin(None, Value::StringArray(sa), Vec::new())).expect("jsonencode");
    assert_eq!(as_string(encoded), "[\"alpha\",\"beta\"]");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsonencode_complex_tensor_outputs_objects() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let ct = ComplexTensor::new(vec![(1.0, 2.0), (3.5, -4.0)], vec![2, 1]).expect("complex");
    let encoded = block_on(jsonencode_builtin(
        None,
        Value::ComplexTensor(ct),
        Vec::new(),
    ))
    .expect("jsonencode");
    assert_eq!(
        as_string(encoded),
        "[{\"real\":1,\"imag\":2},{\"real\":3.5,\"imag\":-4}]"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsonencode_typed_complex_integers_preserves_every_class_exactly() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let cases = [
        (
            "int8",
            IntegerStorage::I8(vec![-1]),
            IntegerStorage::I8(vec![2]),
            "{\"real\":-1,\"imag\":2}",
        ),
        (
            "int16",
            IntegerStorage::I16(vec![-3]),
            IntegerStorage::I16(vec![4]),
            "{\"real\":-3,\"imag\":4}",
        ),
        (
            "int32",
            IntegerStorage::I32(vec![-5]),
            IntegerStorage::I32(vec![6]),
            "{\"real\":-5,\"imag\":6}",
        ),
        (
            "int64",
            IntegerStorage::I64(vec![-9223372036854775808]),
            IntegerStorage::I64(vec![9223372036854775807]),
            "{\"real\":-9223372036854775808,\"imag\":9223372036854775807}",
        ),
        (
            "uint8",
            IntegerStorage::U8(vec![1]),
            IntegerStorage::U8(vec![2]),
            "{\"real\":1,\"imag\":2}",
        ),
        (
            "uint16",
            IntegerStorage::U16(vec![3]),
            IntegerStorage::U16(vec![4]),
            "{\"real\":3,\"imag\":4}",
        ),
        (
            "uint32",
            IntegerStorage::U32(vec![5]),
            IntegerStorage::U32(vec![6]),
            "{\"real\":5,\"imag\":6}",
        ),
        (
            "uint64",
            IntegerStorage::U64(vec![u64::MAX]),
            IntegerStorage::U64(vec![1_u64 << 63]),
            "{\"real\":18446744073709551615,\"imag\":9223372036854775808}",
        ),
    ];

    for (class, real, imag, expected) in cases {
        let storage = runmat_value::IntegerComplexStorage::new(real, imag)
            .expect("matching integer components");
        let tensor = ComplexTensor::new_integer(storage, vec![1, 1]).expect("typed complex");
        let encoded = block_on(jsonencode_builtin(
            None,
            Value::ComplexTensor(tensor),
            Vec::new(),
        ))
        .expect("jsonencode typed complex integer");
        assert_eq!(as_string(encoded), expected, "{class}");
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsonencode_complex_input_is_rejected_by_compatibility_gate() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(jsonencode_builtin(
        None,
        Value::Complex(1.0, 2.0),
        Vec::new(),
    ))
    .expect_err("complex encoding is a RunMat extension");
    assert_eq!(
        error.identifier(),
        JSONENCODE_COMPLEX_INPUT_EXTENSION.error_identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsonencode_nested_complex_and_sparse_inputs_are_compatibility_gated() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let complex_cell = CellArray::new(vec![Value::Complex(1.0, 2.0)], 1, 1).expect("cell");
    let complex_error = block_on(jsonencode_builtin(
        None,
        Value::Cell(complex_cell),
        Vec::new(),
    ))
    .expect_err("nested complex input is an extension");
    assert_eq!(
        complex_error.identifier(),
        JSONENCODE_COMPLEX_INPUT_EXTENSION.error_identifier
    );

    let sparse =
        runmat_value::SparseTensor::new(1, 1, vec![0, 1], vec![0], vec![1.0]).expect("sparse");
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    fields
        .fields
        .insert("value".to_string(), Value::SparseTensor(sparse));
    let sparse_error = block_on(jsonencode_builtin(
        None,
        Value::Struct(fields.finish()),
        Vec::new(),
    ))
    .expect_err("nested sparse input is an extension");
    assert_eq!(
        sparse_error.identifier(),
        JSONENCODE_SPARSE_INPUT_EXTENSION.error_identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsonencode_resident_gate_precedes_provider_access() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    let original = allocation.handle.wire_descriptor();
    assert!(
        crate::builtins::common::gpu_helpers::exact_provider_for_handle(&allocation.handle)
            .is_none()
    );
    let handle = allocation
        .handle
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .expect("explicit immutable view");
    let error = block_on(jsonencode_builtin(
        None,
        Value::GpuTensor(handle.clone()),
        Vec::new(),
    ))
    .expect_err("resident extension must gate before owner lookup");
    drop(handle);
    assert_eq!(
        error.identifier(),
        JSONENCODE_RESIDENT_INPUT_EXTENSION.error_identifier
    );

    assert_eq!(allocation.handle.wire_descriptor(), original);
    assert_eq!(allocation.probe.retirements(), 0);
    drop(allocation.handle);
    assert_eq!(allocation.probe.retirements(), 1);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsonencode_integer_metadata_distinguishes_public_real_and_extended_complex() {
    assert_eq!(JSONENCODE_INTEGER_CAPABILITIES.len(), 3);
    assert_eq!(
        JSONENCODE_INTEGER_CAPABILITIES[0].inputs[0].availability,
        BuiltinIntegerInputAvailability::Documented
    );
    assert_eq!(
        JSONENCODE_INTEGER_CAPABILITIES[1].inputs[0].availability,
        BuiltinIntegerInputAvailability::RunMatOnly
    );
    assert_eq!(
        JSONENCODE_INTEGER_CAPABILITIES[2].inputs[0].availability,
        BuiltinIntegerInputAvailability::RunMatOnly
    );
    assert_eq!(JSONENCODE_EXTENSIONS.len(), 6);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn jsonencode_gpu_tensor_gathers_host_data() {
    test_support::with_test_provider(|provider| {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).expect("tensor");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = provider.upload(&view).expect("upload");
        let encoded = block_on(jsonencode_builtin(
            None,
            Value::GpuTensor(handle),
            Vec::new(),
        ))
        .expect("jsonencode");
        assert_eq!(as_string(encoded), "[[1,0],[0,1]]");
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg(feature = "wgpu")]
fn jsonencode_gpu_tensor_wgpu_gathers_host_data() {
    let ensure = runmat_accelerate::backend::wgpu::provider::ensure_wgpu_provider();
    let Some(_) = ensure.ok().flatten() else {
        // No WGPU device available on this host; skip.
        return;
    };
    let provider = runmat_accelerate_api::provider().expect("wgpu provider");
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).expect("tensor");
    let view = runmat_accelerate_api::HostTensorView {
        data: &tensor.materialize_f64(),
        shape: tensor.shape(),
    };
    let handle = provider.upload(&view).expect("upload");
    let encoded = block_on(jsonencode_builtin(
        None,
        Value::GpuTensor(handle),
        Vec::new(),
    ))
    .expect("jsonencode");
    assert_eq!(as_string(encoded), "[1,2,3]");
}
