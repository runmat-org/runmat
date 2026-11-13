use super::*;
use futures::executor::block_on;
use runmat_value::{
    ComplexStorage, IntValue, IntegerComplexStorage, IntegerStorage, NumericDType, Tensor,
};

fn call(value: Value, shift: Option<Value>) -> crate::BuiltinResult<Value> {
    block_on(shiftdim_builtin(
        value,
        crate::builtins::common::test_support::builtin_arguments(shift.into_iter().collect()),
    ))
}

fn tensor_shape(value: &Value) -> &[usize] {
    match value {
        Value::Tensor(tensor) => tensor.shape(),
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn plan_matches_positive_negative_and_default_semantics() {
    assert_eq!(
        make_plan(&[4, 2, 3, 5], Some(2)).unwrap(),
        ShiftPlan::Rotate {
            order: vec![3, 4, 1, 2],
            permuted_shape: vec![3, 5, 4, 2],
            shape: vec![3, 5, 4, 2],
        }
    );
    assert_eq!(
        make_plan(&[4, 2, 3, 5], Some(-2)).unwrap(),
        ShiftPlan::Prepend {
            shape: vec![1, 1, 4, 2, 3, 5],
        }
    );
    assert_eq!(
        make_plan(&[1, 1, 3, 2, 4], None).unwrap(),
        ShiftPlan::RemoveLeading {
            shape: vec![3, 2, 4],
            removed: 2,
        }
    );
    assert_eq!(
        make_plan(&[1, 7], None).unwrap(),
        ShiftPlan::RemoveLeading {
            shape: vec![7, 1],
            removed: 1,
        }
    );
    assert_eq!(
        make_plan(&[1, 1, 1], None).unwrap(),
        ShiftPlan::RemoveLeading {
            shape: vec![1, 1],
            removed: 0,
        }
    );
    assert_eq!(
        make_plan(&[7], Some(1)).unwrap(),
        ShiftPlan::Rotate {
            order: vec![2, 1],
            permuted_shape: vec![7, 1],
            shape: vec![7, 1],
        }
    );
}

#[test]
fn positive_shift_reorders_column_major_data_and_wraps_large_n() {
    let input = Tensor::new((1..=24).map(f64::from).collect(), vec![2, 3, 4]).unwrap();
    let output = call(Value::Tensor(input), Some(Value::Num(4.0))).unwrap();
    let Value::Tensor(output) = output else {
        panic!("expected tensor")
    };
    assert_eq!(output.shape(), &[3, 4, 2]);
    assert_eq!(
        output.materialize_f64(),
        vec![
            1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 2.0, 4.0, 6.0, 8.0,
            10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0,
        ]
    );
}

#[test]
fn negative_shift_and_default_shift_only_change_shape_metadata() {
    let input = Tensor::from_numeric_storage(
        runmat_value::NumericStorage::F32(vec![1.0, 2.0, 3.0, 4.0]),
        vec![1, 1, 2, 2],
    )
    .unwrap();
    let output = call(Value::Tensor(input), None).unwrap();
    assert_eq!(tensor_shape(&output), &[2, 2]);
    let Value::Tensor(output) = output else {
        unreachable!()
    };
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
    let output = call(Value::Tensor(output), Some(Value::Int(IntValue::I8(-3)))).unwrap();
    assert_eq!(tensor_shape(&output), &[1, 1, 1, 2, 2]);
}

#[test]
fn empty_dimensions_shift_without_synthesizing_elements() {
    let input = Tensor::new(Vec::new(), vec![1, 0, 2]).unwrap();
    let Value::Tensor(output) = call(Value::Tensor(input.clone()), Some(Value::Num(1.0))).unwrap()
    else {
        panic!("expected tensor")
    };
    assert_eq!(output.shape(), &[0, 2]);
    assert!(output.is_empty());

    let Value::Tensor(output) = call(Value::Tensor(input), Some(Value::Num(-2.0))).unwrap() else {
        panic!("expected tensor")
    };
    assert_eq!(output.shape(), &[1, 1, 1, 0, 2]);
    assert!(output.is_empty());
}

#[test]
fn zero_and_full_cycle_shifts_keep_sparse_shape_without_provider_permutation() {
    let sparse = runmat_value::SparseTensor::new(2, 2, vec![0, 1, 1], vec![0], vec![7.0])
        .expect("sparse input");
    for shift in [0.0, 2.0] {
        let Value::SparseTensor(output) =
            call(Value::SparseTensor(sparse.clone()), Some(Value::Num(shift)))
                .expect("unchanged sparse shift")
        else {
            panic!("expected sparse output")
        };
        assert_eq!(output, sparse);
    }
    let error = call(
        Value::SparseTensor(sparse),
        Some(Value::Int(IntValue::I64(-1_000_000_000))),
    )
    .expect_err("negative sparse rank change must reject before allocation");
    assert_eq!(
        error.identifier.as_deref(),
        Some("RunMat:shiftdim:UnsupportedInput")
    );
}

macro_rules! integer_case {
    ($variant:ident, $ty:ty) => {{
        let storage = IntegerStorage::$variant(vec![1 as $ty, 2 as $ty, 3 as $ty, 4 as $ty]);
        let class_name = storage.class_name();
        let input = Tensor::new_integer(storage, vec![1, 2, 2]).unwrap();
        let output = call(Value::Tensor(input), Some(Value::Num(1.0))).unwrap();
        let Value::Tensor(output) = output else {
            panic!("expected tensor")
        };
        assert_eq!(output.shape(), &[2, 2]);
        assert_eq!(output.integer_storage().unwrap().class_name(), class_name);
    }};
}

#[test]
fn all_eight_integer_classes_preserve_native_storage() {
    integer_case!(I8, i8);
    integer_case!(I16, i16);
    integer_case!(I32, i32);
    integer_case!(I64, i64);
    integer_case!(U8, u8);
    integer_case!(U16, u16);
    integer_case!(U32, u32);
    integer_case!(U64, u64);
}

#[test]
fn uint64_and_complex_integer_values_remain_exact() {
    let wide = vec![9_007_199_254_740_993, u64::MAX, 1, 2];
    let input = Tensor::new_integer(IntegerStorage::U64(wide.clone()), vec![1, 2, 2]).unwrap();
    let Value::Tensor(output) =
        call(Value::Tensor(input), Some(Value::Int(IntValue::U8(1)))).unwrap()
    else {
        panic!("expected tensor")
    };
    assert_eq!(output.integer_storage(), Some(&IntegerStorage::U64(wide)));

    let storage = IntegerComplexStorage::new(
        IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
        IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993]),
    )
    .unwrap();
    let input = ComplexTensor::new_integer(storage.clone(), vec![1, 2, 1]).unwrap();
    let Value::ComplexTensor(output) =
        call(Value::ComplexTensor(input), Some(Value::Num(1.0))).unwrap()
    else {
        panic!("expected complex tensor")
    };
    assert_eq!(output.integer_storage(), Some(&storage));
    assert_eq!(output.shape(), &[2, 1]);
}

#[test]
fn floating_complex_single_preserves_component_class() {
    let input = ComplexTensor::from_f32(vec![(1.0, -1.0), (2.0, -2.0)], vec![1, 2, 1]).unwrap();
    let Value::ComplexTensor(output) =
        call(Value::ComplexTensor(input), Some(Value::Num(1.0))).unwrap()
    else {
        panic!("expected complex tensor")
    };
    assert!(matches!(output.complex_storage(), ComplexStorage::F32(_)));
    assert_eq!(output.shape(), &[2, 1]);
}

#[test]
fn two_output_form_reports_removed_leading_dimensions() {
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 1, 3]).unwrap();
    let output = call(Value::Tensor(input), None).unwrap();
    let Value::OutputList(values) = output else {
        panic!("expected output list")
    };
    assert_eq!(values.len(), 2);
    assert_eq!(tensor_shape(&values[0]), &[3, 1]);
    assert_eq!(values[1], Value::Num(2.0));
}

#[test]
fn output_and_shift_validation_are_stable() {
    for scalar in [
        Value::Num(3.0),
        Value::Bool(true),
        Value::String("x".into()),
    ] {
        let shifted = call(
            scalar.clone(),
            Some(Value::Int(IntValue::I64(isize::MIN as i64))),
        )
        .expect("scalar negative shift is identity without rank allocation");
        assert_eq!(shifted, scalar);
    }
    for invalid in [
        Value::Num(1.5),
        Value::Num(f64::NAN),
        Value::Num(f64::INFINITY),
        Value::Int(IntValue::U64(u64::MAX)),
        Value::String("1".into()),
        Value::Complex(1.0, 0.0),
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
    ] {
        let error = call(Value::Num(1.0), Some(invalid)).unwrap_err();
        assert_eq!(
            error.identifier.as_deref(),
            Some("RunMat:shiftdim:InvalidShift")
        );
    }
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let error = call(Value::Num(1.0), Some(Value::Num(1.0))).unwrap_err();
    assert_eq!(
        error.identifier.as_deref(),
        Some("RunMat:shiftdim:TooManyOutputs")
    );
}

#[test]
fn logical_string_cell_and_char_arrays_follow_shape_contract() {
    let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![1, 2, 2]).unwrap();
    let Value::LogicalArray(logical) =
        call(Value::LogicalArray(logical), Some(Value::Num(1.0))).unwrap()
    else {
        panic!("expected logical")
    };
    assert_eq!(logical.shape, vec![2, 2]);

    let strings = StringArray::new(
        vec!["a".into(), "b".into(), "c".into(), "d".into()],
        vec![1, 2, 2],
    )
    .unwrap();
    let Value::StringArray(strings) =
        call(Value::StringArray(strings), Some(Value::Num(1.0))).unwrap()
    else {
        panic!("expected strings")
    };
    assert_eq!(strings.shape, vec![2, 2]);

    let cell = CellArray::new_with_shape(
        vec![
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(3.0),
            Value::Num(4.0),
        ],
        vec![1, 2, 2],
    )
    .unwrap();
    let Value::Cell(cell) = call(Value::Cell(cell), Some(Value::Num(1.0))).unwrap() else {
        panic!("expected cell")
    };
    assert_eq!(cell.shape, vec![2, 2]);

    let chars = CharArray::new_row("abc");
    let Value::CharArray(chars) = call(Value::CharArray(chars), None).unwrap() else {
        panic!("expected chars")
    };
    assert_eq!((chars.rows, chars.cols), (3, 1));
    let Value::CharArray(chars) = call(
        Value::CharArray(CharArray::new_row("abc")),
        Some(Value::Num(-1.0)),
    )
    .expect("N-D char shiftdim") else {
        panic!("expected chars")
    };
    assert_eq!(chars.shape, vec![1, 1, 3]);
    assert_eq!(
        runmat_value::decode_utf16_code_units(chars.data()).expect("valid UTF-16 fixture"),
        "abc"
    );
}

#[test]
fn descriptor_exposes_both_output_forms() {
    assert_eq!(SHIFTDIM_DESCRIPTOR.signatures.len(), 3);
    assert_eq!(
        SHIFTDIM_DESCRIPTOR.output_mode,
        BuiltinOutputMode::ByRequestedOutputCount
    );
}

#[test]
fn simple_provider_keeps_integer_gpu_values_resident_and_exact() {
    runmat_accelerate::initialize_acceleration_provider();
    runmat_accelerate::simple_provider::register_inprocess_provider();
    let provider = runmat_accelerate_api::provider().expect("provider");
    let tensor = Tensor::new_integer(
        IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX, 3, 4, 5, 6]),
        vec![1, 2, 3],
    )
    .unwrap();
    let handle = gpu_helpers::upload_tensor(provider, &tensor).unwrap();
    let Value::GpuTensor(output) = call(Value::GpuTensor(handle), Some(Value::Num(1.0))).unwrap()
    else {
        panic!("expected GPU tensor")
    };
    assert_eq!(output.shape(), vec![2, 3]);
    assert_eq!(
        runmat_accelerate_api::handle_integer_type(&output),
        Some(runmat_accelerate_api::IntegerElementType::U64)
    );
    assert!(runmat_accelerate::fusion_residency::is_resident(&output));
    let gathered = block_on(gpu_helpers::gather_tensor_async(&output)).unwrap();
    assert_eq!(gathered.integer_storage(), tensor.integer_storage());

    let Value::GpuTensor(output) = call(Value::GpuTensor(output), Some(Value::Num(-2.0))).unwrap()
    else {
        panic!("expected GPU tensor")
    };
    assert_eq!(output.shape(), vec![1, 1, 2, 3]);
    assert_eq!(
        runmat_accelerate_api::handle_integer_type(&output),
        Some(runmat_accelerate_api::IntegerElementType::U64)
    );
    assert!(runmat_accelerate::fusion_residency::is_resident(&output));
}

#[test]
fn simple_provider_keeps_complex_gpu_values_resident() {
    runmat_accelerate::initialize_acceleration_provider();
    runmat_accelerate::simple_provider::register_inprocess_provider();
    let provider = runmat_accelerate_api::provider().expect("provider");
    let tensor = ComplexTensor::new(
        vec![(1.0, -1.0), (2.0, -2.0), (3.0, -3.0), (4.0, -4.0)],
        vec![1, 2, 2],
    )
    .unwrap();
    let handle = gpu_helpers::upload_complex_tensor(provider, &tensor).unwrap();
    let Value::GpuTensor(output) = call(Value::GpuTensor(handle), Some(Value::Num(1.0))).unwrap()
    else {
        panic!("expected GPU tensor")
    };
    assert_eq!(output.shape(), vec![2, 2]);
    assert_eq!(
        runmat_accelerate_api::handle_storage(&output),
        GpuTensorStorage::ComplexInterleaved
    );
    assert!(runmat_accelerate::fusion_residency::is_resident(&output));
    let gathered = block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(output))).unwrap();
    assert!(matches!(
        gathered,
        Value::ComplexTensor(value)
            if value.shape() == [2, 2]
                && value.materialize_f64() == tensor.materialize_f64()
    ));
}

#[test]
fn simple_provider_keeps_paired_complex_integer_gpu_values_resident_and_exact() {
    runmat_accelerate::initialize_acceleration_provider();
    runmat_accelerate::simple_provider::register_inprocess_provider();
    let provider = runmat_accelerate_api::provider().expect("provider");
    let wide = (1_u64 << 53) + 1;
    let tensor = ComplexTensor::new_integer(
        IntegerComplexStorage::new(
            IntegerStorage::U64(vec![wide, u64::MAX, 3, 4]),
            IntegerStorage::U64(vec![u64::MAX, wide, 5, 6]),
        )
        .expect("paired storage"),
        vec![1, 2, 2],
    )
    .expect("paired tensor");
    let handle = gpu_helpers::upload_complex_tensor(provider, &tensor).expect("upload paired");
    let Value::GpuTensor(output) =
        call(Value::GpuTensor(handle.clone()), Some(Value::Num(1.0))).expect("shift paired")
    else {
        panic!("expected resident paired output")
    };
    assert_eq!(output.shape(), vec![2, 2]);
    assert_eq!(
        runmat_accelerate_api::handle_storage(&output),
        GpuTensorStorage::ComplexInterleaved
    );
    assert_eq!(
        runmat_accelerate_api::handle_integer_type(&output),
        Some(runmat_accelerate_api::IntegerElementType::U64)
    );
    let gathered = block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(
        output.clone(),
    )))
    .expect("gather paired");
    let Value::ComplexTensor(gathered) = gathered else {
        panic!("expected paired complex tensor")
    };
    assert_eq!(gathered.integer_storage(), tensor.integer_storage());
    drop(handle);
    drop(output);
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_positive_integer_shift_preserves_class_when_adapter_is_available() {
    let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    ) else {
        return;
    };
    let tensor = Tensor::new_integer(IntegerStorage::I32(vec![1, 2, 3, 4]), vec![1, 2, 2]).unwrap();
    let handle = gpu_helpers::upload_tensor(provider, &tensor).unwrap();
    let Value::GpuTensor(output) = call(Value::GpuTensor(handle), Some(Value::Num(1.0))).unwrap()
    else {
        panic!("expected GPU tensor")
    };
    assert_eq!(output.shape(), vec![2, 2]);
    assert_eq!(
        runmat_accelerate_api::handle_integer_type(&output),
        Some(runmat_accelerate_api::IntegerElementType::I32)
    );
    let Value::GpuTensor(output) = call(Value::GpuTensor(output), Some(Value::Num(-1.0))).unwrap()
    else {
        panic!("expected GPU tensor")
    };
    assert_eq!(output.shape(), vec![1, 2, 2]);
    assert_eq!(
        runmat_accelerate_api::handle_integer_type(&output),
        Some(runmat_accelerate_api::IntegerElementType::I32)
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_paired_complex_integer_shift_stays_resident_and_exact_when_available() {
    let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    ) else {
        return;
    };
    let wide = (1_u64 << 53) + 1;
    let tensor = ComplexTensor::new_integer(
        IntegerComplexStorage::new(
            IntegerStorage::U64(vec![wide, u64::MAX, 3, 4]),
            IntegerStorage::U64(vec![u64::MAX, wide, 5, 6]),
        )
        .expect("paired storage"),
        vec![1, 2, 2],
    )
    .expect("paired tensor");
    let handle = gpu_helpers::upload_complex_tensor(provider, &tensor).expect("upload paired");
    let Value::GpuTensor(output) =
        call(Value::GpuTensor(handle.clone()), Some(Value::Num(1.0))).expect("shift paired")
    else {
        panic!("expected resident paired output")
    };
    assert_eq!(output.shape(), vec![2, 2]);
    assert_eq!(
        runmat_accelerate_api::handle_storage(&output),
        GpuTensorStorage::ComplexInterleaved
    );
    assert_eq!(
        runmat_accelerate_api::handle_integer_type(&output),
        Some(runmat_accelerate_api::IntegerElementType::U64)
    );
    let gathered = block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(
        output.clone(),
    )))
    .expect("gather paired");
    let Value::ComplexTensor(gathered) = gathered else {
        panic!("expected paired tensor")
    };
    assert_eq!(gathered.integer_storage(), tensor.integer_storage());
    drop(handle);
    drop(output);
}
