use super::*;
use futures::executor::block_on;
use runmat_builtins::{STATGET_DESCRIPTOR, STATSET_DESCRIPTOR};
use runmat_value::{CharArray, IntegerStorage};

fn struct_value(value: Value) -> StructValue {
    let Value::Struct(st) = value else {
        panic!("expected struct, got {value:?}");
    };
    st
}

fn num_field(options: &StructValue, name: &str) -> f64 {
    match options.fields.get(name).unwrap() {
        Value::Num(value) => *value,
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            tensor::tensor_value_f64(tensor, 0)
        }
        other => panic!("expected numeric field {name}, got {other:?}"),
    }
}

#[test]
fn statset_builds_custom_options() {
    let options = struct_value(
        block_on(statset_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("FunValCheck"),
                Value::from("on"),
                Value::from("TolX"),
                Value::Num(1.0e-8),
                Value::from("UseParallel"),
                Value::from("off"),
            ]),
        ))
        .unwrap(),
    );
    assert!(matches!(options.fields.get("FunValCheck"), Some(Value::String(s)) if s == "on"));
    assert_eq!(num_field(&options, "TolX"), 1.0e-8);
    assert!(matches!(
        options.fields.get("UseParallel"),
        Some(Value::Bool(false))
    ));
    assert!(
        matches!(options.fields.get("Streams"), Some(Value::Cell(cell)) if cell.data().is_empty())
    );
}

#[test]
fn statset_character_row_option_values_normalize_to_scalar_text() {
    let value =
        Value::CharArray(CharArray::from_unicode_scalars("off".chars().collect(), 1, 3).unwrap());
    let options = struct_value(
        block_on(statset_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("Display"),
                value,
            ]),
        ))
        .unwrap(),
    );
    assert!(matches!(options.fields.get("Display"), Some(Value::String(text)) if text == "off"));
}

#[test]
fn statset_empty_single_tolerance_retains_class_and_shape() {
    let value = Value::Tensor(Tensor::from_f32(Vec::new(), vec![1, 0]).unwrap());
    let options = struct_value(
        block_on(statset_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("TolX"),
                value,
            ]),
        ))
        .unwrap(),
    );
    let Some(Value::Tensor(tolerance)) = options.fields.get("TolX") else {
        panic!("expected empty numeric tolerance")
    };
    assert_eq!(tolerance.numeric_dtype(), runmat_value::NumericDType::F32);
    assert_eq!(tolerance.shape(), &[1, 0]);
}

#[test]
fn statset_scalar_logical_arrays_follow_scalar_logical_option_rules() {
    for value in [
        Value::Bool(true),
        Value::LogicalArray(runmat_value::LogicalArray::new(vec![1], vec![1, 1]).unwrap()),
    ] {
        let options = struct_value(
            block_on(statset_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::from("MaxIter"),
                    value.clone(),
                    Value::from("TolX"),
                    value.clone(),
                    Value::from("DerivStep"),
                    value.clone(),
                    Value::from("UseParallel"),
                    value,
                ]),
            ))
            .unwrap(),
        );
        assert_eq!(num_field(&options, "MaxIter"), 1.0);
        assert_eq!(num_field(&options, "TolX"), 1.0);
        assert_eq!(num_field(&options, "DerivStep"), 1.0);
        assert!(matches!(
            options.fields.get("UseParallel"),
            Some(Value::Bool(true))
        ));
    }
}

#[test]
fn statset_integer_derivstep_keeps_tensor_storage_boundary_explicit() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let scalar = struct_value(
        block_on(statset_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("DerivStep"),
                Value::Int(runmat_value::IntValue::U64(7)),
            ]),
        ))
        .unwrap(),
    );
    assert!(matches!(
        scalar.fields.get("DerivStep"),
        Some(Value::Num(7.0))
    ));
    let tensor = Tensor::new_integer(IntegerStorage::U64(vec![7]), vec![1, 1]).unwrap();
    let array = struct_value(
        block_on(statset_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("DerivStep"),
                Value::Tensor(tensor),
            ]),
        ))
        .unwrap(),
    );
    let Some(Value::Tensor(step)) = array.fields.get("DerivStep") else {
        panic!("expected integer tensor")
    };
    assert_eq!(step.numeric_dtype(), runmat_value::NumericDType::U64);
    assert_eq!(step.integer_storage(), Some(&IntegerStorage::U64(vec![7])));
}

#[test]
fn statset_reads_typed_integer_tensor_options_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let max_iter = Tensor::new_integer(IntegerStorage::U16(vec![7]), vec![1, 1]).expect("MaxIter");
    let tol_fun = Tensor::new_integer(IntegerStorage::U16(vec![2]), vec![1, 1]).expect("TolFun");
    let deriv_step =
        Tensor::new_integer(IntegerStorage::U16(vec![3, 4]), vec![1, 2]).expect("DerivStep");
    let use_parallel =
        Tensor::new_integer(IntegerStorage::U16(vec![1]), vec![1, 1]).expect("UseParallel");

    let options = struct_value(
        block_on(statset_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("MaxIter"),
                Value::Tensor(max_iter),
                Value::from("TolFun"),
                Value::Tensor(tol_fun),
                Value::from("DerivStep"),
                Value::Tensor(deriv_step.clone()),
                Value::from("UseParallel"),
                Value::Tensor(use_parallel),
            ]),
        ))
        .expect("statset"),
    );

    assert_eq!(num_field(&options, "MaxIter"), 7.0);
    assert_eq!(num_field(&options, "TolFun"), 2.0);
    let Some(Value::Tensor(parsed_deriv_step)) = options.fields.get("DerivStep") else {
        panic!("expected DerivStep tensor");
    };
    assert_eq!(parsed_deriv_step.shape(), deriv_step.shape());
    assert_eq!(
        parsed_deriv_step.integer_storage(),
        deriv_step.integer_storage()
    );
    assert_eq!(options.fields.get("UseParallel"), Some(&Value::Bool(true)));
}

#[test]
fn statset_typed_integer_option_is_mode_gated() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let value =
        Value::Tensor(Tensor::new_integer(IntegerStorage::U16(vec![7]), vec![1, 1]).unwrap());
    let error = block_on(statset_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("MaxIter"),
            value,
        ]),
    ))
    .expect_err("MATLAB-compatible mode must reject typed integer option values");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:StatsetIntegerOptionValueExtension")
    );
}

#[test]
fn statset_invalid_integer_statfun_is_not_misclassified_as_an_option_extension() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(statset_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
            runmat_value::IntValue::U8(7),
        )]),
    ))
    .expect_err("an integer is not a statistics function name");
    assert_eq!(error.identifier(), Some("RunMat:statset:InvalidArgument"));
}

#[test]
fn statset_scalar_options_read_all_integer_storage_variants() {
    let storages = [
        IntegerStorage::I8(vec![1]),
        IntegerStorage::I16(vec![1]),
        IntegerStorage::I32(vec![1]),
        IntegerStorage::I64(vec![1]),
        IntegerStorage::U8(vec![1]),
        IntegerStorage::U16(vec![1]),
        IntegerStorage::U32(vec![1]),
        IntegerStorage::U64(vec![1]),
    ];
    for storage in storages {
        let scalar = Tensor::new_integer(storage, vec![1, 1]).unwrap();
        assert!(matches!(
            bool_or_on_off_value("UseParallel", &Value::Tensor(scalar)),
            Ok(Value::Bool(true))
        ));
    }
}

#[test]
fn statset_integer_counts_read_all_integer_storage_variants() {
    for storage in [
        IntegerStorage::I8(vec![7]),
        IntegerStorage::I16(vec![7]),
        IntegerStorage::I32(vec![7]),
        IntegerStorage::I64(vec![7]),
        IntegerStorage::U8(vec![7]),
        IntegerStorage::U16(vec![7]),
        IntegerStorage::U32(vec![7]),
        IntegerStorage::U64(vec![7]),
    ] {
        let value = Tensor::new_integer(storage, vec![1, 1]).unwrap();
        assert_eq!(
            positive_integer_value("MaxIter", &Value::Tensor(value)).unwrap(),
            Value::Num(7.0)
        );
    }
    let wide = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).unwrap();
    assert!(positive_integer_value("MaxIter", &Value::Tensor(wide)).is_err());
}

#[test]
fn statset_applies_function_defaults_and_updates() {
    let base = block_on(statset_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("nbinfit")]),
    ))
    .unwrap();
    let options = struct_value(
        block_on(statset_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                base,
                Value::from("TolX"),
                Value::Num(1.0e-8),
            ]),
        ))
        .unwrap(),
    );
    assert_eq!(num_field(&options, "MaxIter"), 200.0);
    assert_eq!(num_field(&options, "TolX"), 1.0e-8);
    assert_eq!(num_field(&options, "TolFun"), 1.0e-6);
}

#[test]
fn statset_old_new_merge_prefers_nonempty_new_fields() {
    let oldopts = struct_value(
        block_on(statset_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("TolX"),
                Value::Num(1.0e-6),
                Value::from("MaxIter"),
                Value::Num(20.0),
            ]),
        ))
        .unwrap(),
    );
    let newopts = struct_value(
        block_on(statset_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("TolX"),
                Value::Num(1.0e-9),
            ]),
        ))
        .unwrap(),
    );
    let merged = struct_value(
        block_on(statset_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Struct(oldopts),
                Value::Struct(newopts),
            ]),
        ))
        .unwrap(),
    );
    assert_eq!(num_field(&merged, "TolX"), 1.0e-9);
    assert_eq!(num_field(&merged, "MaxIter"), 20.0);
}

#[test]
fn statget_supports_unique_prefix_and_default_for_empty() {
    let options = block_on(statset_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("TolX"),
            Value::Num(1.0e-8),
            Value::from("MaxIter"),
            Value::Num(15.0),
        ]),
    ))
    .unwrap();
    let value = block_on(statget_builtin(
        options.clone(),
        Value::from("TolX"),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    assert_eq!(value, Value::Num(1.0e-8));

    let value = block_on(statget_builtin(
        options.clone(),
        Value::from("MaxI"),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    assert_eq!(value, Value::Num(15.0));

    let value = block_on(statget_builtin(
        options,
        Value::from("TolFun"),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(3.0)]),
    ))
    .unwrap();
    assert_eq!(value, Value::Num(3.0));
}

#[test]
fn statget_returns_integer_default_data_without_a_floating_mirror() {
    let options = Value::Struct(empty_options(None).unwrap());
    let expected = IntegerStorage::U64(vec![9_007_199_254_740_993]);
    let default =
        Value::Tensor(Tensor::new_integer(expected.clone(), vec![1, 1]).expect("integer default"));
    let value = block_on(statget_builtin(
        options,
        Value::from("TolFun"),
        crate::builtins::common::test_support::builtin_arguments(vec![default]),
    ))
    .expect("statget");
    let Value::Tensor(tensor) = value else {
        panic!("expected tensor default");
    };
    assert_eq!(tensor.integer_storage(), Some(&expected));
}

#[test]
fn statset_rejects_invalid_values() {
    let err = block_on(statset_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("MaxIter"),
            Value::Num(2.5),
        ]),
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:statset:InvalidOption"));

    let err = block_on(statset_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Display"),
            Value::from("verbose"),
        ]),
    ))
    .unwrap_err();
    assert!(err.message.contains("Display"));
}

#[test]
fn descriptors_cover_public_forms() {
    let statset_labels = STATSET_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect::<Vec<_>>();
    assert!(statset_labels.contains(&"options = statset(statfun)"));
    assert!(statset_labels.contains(&"options = statset(oldopts, newopts)"));

    let statget_labels = STATGET_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect::<Vec<_>>();
    assert_eq!(
        statget_labels,
        vec![
            "val = statget(options, field)",
            "val = statget(options, field, defaultData)",
        ]
    );
}

#[test]
fn statset_invalid_pair_grammar_precedes_resident_materialization() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    let error = block_on(statset_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("MaxIter"),
            Value::GpuTensor(allocation.handle.clone()),
            Value::from("TolX"),
        ]),
    ))
    .expect_err("trailing field name must fail before an upload or gather");
    assert_eq!(error.identifier(), Some("RunMat:statset:InvalidArgument"));
    assert_eq!(allocation.probe.retirements(), 0);
    allocation.handle.release().unwrap();
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn statset_unknown_field_precedes_resident_value_materialization() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    let error = block_on(statset_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Struct(empty_options(None).unwrap()),
            Value::from("UnknownOption"),
            Value::GpuTensor(allocation.handle.clone()),
        ]),
    ))
    .expect_err("invalid field must fail before gathering its value");
    assert_eq!(error.identifier(), Some("RunMat:statset:InvalidOption"));
    assert_eq!(allocation.probe.retirements(), 0);
    allocation.handle.release().unwrap();
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn statget_unselected_resident_fields_and_default_are_not_materialized() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    let options = replace_option(
        None,
        &empty_options(None).unwrap(),
        StatOptionField::TolX,
        Value::GpuTensor(allocation.handle.clone()),
    )
    .unwrap();
    let result = block_on(statget_builtin(
        Value::Struct(options),
        Value::from("MaxIter"),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(7.0)]),
    ))
    .expect("selected empty field uses host default");
    assert_eq!(result, Value::Num(7.0));
    assert_eq!(allocation.probe.retirements(), 0);
    allocation.handle.release().unwrap();
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn statget_unknown_field_ignores_resident_default() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    let result = block_on(statget_builtin(
        Value::Struct(empty_options(None).unwrap()),
        Value::from("NoSuchField"),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
            allocation.handle.clone(),
        )]),
    ))
    .expect("unknown field returns empty without touching default");
    assert!(matches!(result, Value::Tensor(tensor) if tensor.shape() == [0, 0]));
    assert_eq!(allocation.probe.retirements(), 0);
    allocation.handle.release().unwrap();
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn statset_stores_callback_handles_without_running_them() {
    let callback = Value::FunctionHandle("undefined_callback_for_statset".into());
    let options = struct_value(
        block_on(statset_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("OutputFcn"),
                callback.clone(),
                Value::from("RobustWgtFun"),
                callback.clone(),
            ]),
        ))
        .expect("option construction stores callbacks"),
    );
    assert_eq!(options.fields.get("OutputFcn"), Some(&callback));
    assert_eq!(options.fields.get("RobustWgtFun"), Some(&callback));
}

#[test]
fn statset_and_statget_reject_extra_outputs_before_value_access() {
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let set_error = block_on(statset_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect_err("one output maximum");
    assert_eq!(
        set_error.identifier(),
        Some("RunMat:statset:InvalidArgument")
    );
    let get_error = block_on(statget_builtin(
        Value::Struct(empty_options(None).unwrap()),
        Value::from("TolX"),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect_err("one output maximum");
    assert_eq!(
        get_error.identifier(),
        Some("RunMat:statget:InvalidArgument")
    );
}
