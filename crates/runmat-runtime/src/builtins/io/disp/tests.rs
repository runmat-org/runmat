use super::*;
use crate::builtins::common::test_support;
use crate::make_cell;
use runmat_value::{ComplexTensor, IntValue, StringArray, SymbolicExpr, Tensor};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn disp_preserves_rank_zero_scalar_and_rank_one_row_display() {
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let cases = [
        (
            Value::Tensor(Tensor::new(vec![7.0], Vec::new()).unwrap()),
            "7\n",
        ),
        (
            Value::Tensor(Tensor::new(vec![7.0, 8.0], vec![2]).unwrap()),
            "     7       8\n",
        ),
        (
            Value::ComplexTensor(ComplexTensor::new(vec![(1.0, 2.0)], Vec::new()).unwrap()),
            "1+2i\n",
        ),
        (
            Value::ComplexTensor(
                ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![2]).unwrap(),
            ),
            "  1+2i    3+4i\n",
        ),
    ];
    for (value, expected) in cases {
        crate::console::reset_thread_buffer();
        futures::executor::block_on(disp_builtin(
            value,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("display exact-rank tensor");
        let entries = crate::console::take_thread_buffer();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].text, expected);
    }
}

#[test]
fn disp_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = DISP_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"disp(X)"));
    assert!(DISP_DESCRIPTOR.signatures[0].outputs.is_empty());
}

#[test]
fn disp_rejects_outputs_and_prints_nothing_for_empty_builtin_values() {
    let _outputs = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    let err = futures::executor::block_on(disp_builtin(
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:disp:TooManyOutputs"));
    drop(_outputs);
    crate::console::reset_thread_buffer();
    futures::executor::block_on(disp_builtin(
        Value::Tensor(Tensor::zeros(vec![0, 0])),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    assert!(crate::console::take_thread_buffer().is_empty());
    for value in [
        Value::ComplexTensor(ComplexTensor::new(Vec::new(), vec![0, 0]).unwrap()),
        Value::StringArray(StringArray::new(Vec::new(), vec![0, 1, 2]).unwrap()),
    ] {
        crate::console::reset_thread_buffer();
        futures::executor::block_on(disp_builtin(
            value,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap();
        assert!(crate::console::take_thread_buffer().is_empty());
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn string_scalar_without_quotes() {
    let lines = format_for_disp(&Value::String("Simulation complete.".into()));
    assert_eq!(lines, vec!["Simulation complete.".to_string()]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn symbolic_scalar_displays_expression() {
    let lines = format_for_disp(&Value::Symbolic(SymbolicExpr::variable("x")));

    assert_eq!(lines, vec!["x".to_string()]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn struct_field_string_has_quotes() {
    let mut sv = runmat_value::UnboundedStructValueBuilder::new();
    sv.insert("msg", Value::String("ok".into()));
    let lines = render_value(&Value::Struct(sv.finish()), RenderMode::TopLevel);
    assert_eq!(lines, vec!["    msg: \"ok\"".to_string()]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn numeric_matrix_right_aligned() {
    let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).expect("tensor");
    let lines = format_for_disp(&Value::Tensor(tensor));
    assert_eq!(
        lines,
        vec!["     1       2".to_string(), "     3       4".to_string()]
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn numeric_tensor_three_dimensional_pages() {
    let tensor = Tensor::new((1..=8).map(|v| v as f64).collect::<Vec<_>>(), vec![2, 2, 2]).unwrap();
    let lines = format_for_disp(&Value::Tensor(tensor));
    assert_eq!(
        lines,
        vec![
            "(:,:,1) =".to_string(),
            "     1       3".to_string(),
            "     2       4".to_string(),
            String::new(),
            "(:,:,2) =".to_string(),
            "     5       7".to_string(),
            "     6       8".to_string()
        ]
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn complex_tensor_three_dimensional_pages() {
    let data: Vec<(f64, f64)> = vec![
        (1.0, 0.0),
        (2.0, 0.0),
        (3.0, 0.0),
        (4.0, 0.0),
        (5.0, 0.5),
        (6.0, 0.5),
        (7.0, 0.5),
        (8.0, 0.5),
    ];
    let tensor = ComplexTensor::new(data, vec![2, 2, 2]).unwrap();
    let lines = format_for_disp(&Value::ComplexTensor(tensor));
    assert_eq!(
        lines,
        vec![
            "(:,:,1) =".to_string(),
            "     1       3".to_string(),
            "     2       4".to_string(),
            String::new(),
            "(:,:,2) =".to_string(),
            "5+0.5000i  7+0.5000i".to_string(),
            "6+0.5000i  8+0.5000i".to_string()
        ]
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn string_array_left_aligned() {
    let array = StringArray::new(
        vec![
            "alpha".into(),
            "gamma".into(),
            "beta".into(),
            "delta".into(),
        ],
        vec![2, 2],
    )
    .expect("string array");
    let lines = format_for_disp(&Value::StringArray(array));
    assert_eq!(
        lines,
        vec![
            "\"alpha\"  \"beta\"".to_string(),
            "\"gamma\"  \"delta\"".to_string()
        ]
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cell_array_summaries() {
    let cell = make_cell(vec![Value::Num(1.0), Value::String("alpha".into())], 1, 2).expect("cell");
    let lines = format_for_disp(&cell);
    assert_eq!(lines, vec!["    [1]  \"alpha\"".to_string()]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn struct_field_matrix_summarised() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("tensor");
    let mut sv = runmat_value::UnboundedStructValueBuilder::new();
    sv.insert("A", Value::Tensor(tensor));
    let lines = render_value(&Value::Struct(sv.finish()), RenderMode::TopLevel);
    assert_eq!(lines, vec!["    A: [2x2 double]".to_string()]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn integer_64_bit_display() {
    let lines = format_for_disp(&Value::Int(IntValue::U64(u64::MAX)));
    assert_eq!(lines, vec![u64::MAX.to_string()]);
}

#[test]
fn every_integer_scalar_class_formats_exactly() {
    for (value, expected) in [
        (IntValue::I8(i8::MIN), i8::MIN.to_string()),
        (IntValue::I16(i16::MIN), i16::MIN.to_string()),
        (IntValue::I32(i32::MIN), i32::MIN.to_string()),
        (IntValue::I64(i64::MIN), i64::MIN.to_string()),
        (IntValue::U8(u8::MAX), u8::MAX.to_string()),
        (IntValue::U16(u16::MAX), u16::MAX.to_string()),
        (IntValue::U32(u32::MAX), u32::MAX.to_string()),
        (IntValue::U64(u64::MAX), u64::MAX.to_string()),
    ] {
        assert_eq!(format_for_disp(&Value::Int(value)), vec![expected]);
    }
}

#[test]
fn integer_hex_format_uses_each_class_bit_width() {
    set_display_format(FormatMode::Hex);
    for (value, expected) in [
        (IntValue::I8(-1), "ff"),
        (IntValue::I16(-1), "ffff"),
        (IntValue::I32(-1), "ffffffff"),
        (IntValue::I64(-1), "ffffffffffffffff"),
        (IntValue::U8(1), "01"),
        (IntValue::U16(1), "0001"),
        (IntValue::U32(1), "00000001"),
        (IntValue::U64(1), "0000000000000001"),
    ] {
        assert_eq!(format_for_disp(&Value::Int(value)), vec![expected]);
    }
    set_display_format(FormatMode::Short);
}

#[test]
fn integer_short_and_long_formats_remain_exact_decimal() {
    for mode in [FormatMode::Short, FormatMode::Long] {
        set_display_format(mode);
        assert_eq!(
            format_for_disp(&Value::Int(IntValue::U64(u64::MAX))),
            vec![u64::MAX.to_string()]
        );
        assert_eq!(
            format_for_disp(&Value::Int(IntValue::I64(i64::MIN))),
            vec![i64::MIN.to_string()]
        );
    }
    set_display_format(FormatMode::Short);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn integer_tensor_display_uses_exact_backing_storage() {
    let tensor = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]), vec![1, 2])
        .expect("integer tensor");

    let lines = format_for_disp(&Value::Tensor(tensor));

    assert_eq!(lines, vec![format!("{}  {}", u64::MAX, 1_u64 << 63)]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn integer_tensor_nested_summary_preserves_class() {
    let tensor =
        Tensor::new_integer(IntegerStorage::I16(vec![1, 2]), vec![1, 2]).expect("integer tensor");
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    fields.insert("values", Value::Tensor(tensor));

    let lines = render_value(&Value::Struct(fields.finish()), RenderMode::TopLevel);

    assert_eq!(lines, vec!["    values: [1x2 int16]".to_string()]);
}

#[test]
fn implicit_numeric_empty_summary_preserves_class_and_shape() {
    let integer =
        Tensor::new_integer(IntegerStorage::I16(Vec::new()), vec![0, 5]).expect("integer empty");
    assert_eq!(
        format_for_display(&Value::Tensor(integer)),
        vec!["0×5 empty int16 matrix"]
    );

    let single = Tensor::from_f32(Vec::new(), vec![0, 0, 6]).expect("single empty");
    assert_eq!(
        format_for_display(&Value::Tensor(single)),
        vec!["0×0×6 empty single array"]
    );

    assert_eq!(
        format_for_display(&Value::Tensor(Tensor::zeros(vec![0, 0]))),
        vec!["[]"]
    );
}

#[test]
fn explicit_disp_of_numeric_empty_remains_silent() {
    let integer =
        Tensor::new_integer(IntegerStorage::U8(Vec::new()), vec![0, 5]).expect("integer empty");
    assert!(format_for_disp(&Value::Tensor(integer)).is_empty());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn native_single_tensor_nested_summary_preserves_class() {
    let tensor = Tensor::from_numeric_storage(
        runmat_value::NumericStorage::F32(vec![1.0, 2.0]),
        vec![1, 2],
    )
    .expect("single tensor");
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    fields.insert("values", Value::Tensor(tensor));

    let lines = render_value(&Value::Struct(fields.finish()), RenderMode::TopLevel);

    assert_eq!(lines, vec!["    values: [1x2 single]".to_string()]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn integer_sparse_cell_summary_preserves_class() {
    let sparse = runmat_value::SparseTensor::new_integer(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        IntegerStorage::U64(vec![(1_u64 << 53) + 1, u64::MAX]),
    )
    .expect("integer sparse");
    let cell = make_cell(vec![Value::SparseTensor(sparse.clone())], 1, 1).expect("cell");

    assert_eq!(
        format_for_disp(&cell),
        vec!["    [2x2 sparse uint64]".to_string()]
    );

    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    fields.insert("values", Value::SparseTensor(sparse));
    assert_eq!(
        format_for_disp(&Value::Struct(fields.finish())),
        vec!["    values: <sparse 2x2 uint64 nnz=2>".to_string()]
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn typed_complex_integer_display_uses_exact_components_and_class() {
    let storage = runmat_value::IntegerComplexStorage::new(
        IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]),
        IntegerStorage::U64(vec![7, 0]),
    )
    .expect("matching components");
    let tensor = ComplexTensor::new_integer(storage, vec![1, 2]).expect("typed complex");
    assert_eq!(
        format_for_disp(&Value::ComplexTensor(tensor.clone())),
        vec![format!("{}+7i  {}", u64::MAX, 1_u64 << 63)]
    );

    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    fields.insert("values", Value::ComplexTensor(tensor.clone()));
    assert_eq!(
        render_value(&Value::Struct(fields.finish()), RenderMode::TopLevel),
        vec!["    values: [1x2 complex uint64]".to_string()]
    );

    let cell = make_cell(vec![Value::ComplexTensor(tensor)], 1, 1).expect("cell");
    assert_eq!(
        format_for_disp(&cell),
        vec!["    [1x2 complex uint64]".to_string()]
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn disp_accepts_gpu_tensor() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = provider.upload(&view).expect("upload");
        let result = futures::executor::block_on(disp_builtin(
            Value::GpuTensor(handle),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("disp should succeed");
        assert_eq!(result, Value::Tensor(Tensor::zeros(vec![0, 0])));
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn disp_gathers_wgpu_integer_tensor_exactly() {
    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("register wgpu provider");
    let provider = runmat_accelerate_api::provider().expect("wgpu provider");
    let values = [(1_u64 << 53) + 1, u64::MAX];
    let handle = provider
        .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
            data: runmat_accelerate_api::HostIntegerDataView::U64(&values),
            shape: &[1, 2],
        })
        .expect("upload integer");
    crate::console::reset_thread_buffer();
    futures::executor::block_on(disp_builtin(
        Value::GpuTensor(handle),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("disp");
    let text =
        crate::console::take_thread_buffer()
            .into_iter()
            .fold(String::new(), |mut text, entry| {
                text.push_str(entry.text.as_str());
                text
            });
    assert!(text.contains("9007199254740993"));
    assert!(text.contains("18446744073709551615"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn disp_rejects_extra_arguments() {
    let err = futures::executor::block_on(disp_builtin(
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(IntValue::I32(
            2,
        ))]),
    ))
    .expect_err("expected error");
    assert!(err.contains("too many input arguments"));
}
