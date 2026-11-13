use super::builtins::*;
use super::*;
use runmat_builtins::{BuiltinParamArity, BuiltinParamType};

async fn duration_subsref(
    obj: Value,
    kind: String,
    payload: Value,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    let subscript = crate::object::indexing::standard_substruct_fixture_from_parts(&kind, payload)?;
    super::builtins::duration_subsref(obj, subscript).await
}

async fn duration_subsasgn(
    obj: Value,
    kind: String,
    payload: Value,
    rhs: Value,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    let subscript = crate::object::indexing::standard_substruct_fixture_from_parts(&kind, payload)?;
    super::builtins::duration_subsasgn(None, obj, subscript, rhs).await
}

fn run_duration(args: Vec<Value>) -> Value {
    futures::executor::block_on(duration_builtin_source(None, args)).expect("duration")
}

fn integer_tensor(storage: runmat_value::IntegerStorage, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new_integer(storage, shape).expect("integer tensor"))
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn duration_low_rank_components_and_indexing_preserve_admitted_geometry() {
    let scalar = run_duration(vec![
        Value::Tensor(Tensor::new(vec![1.0], Vec::new()).unwrap()),
        Value::Num(0.0),
        Value::Num(0.0),
    ]);
    let scalar_days = duration_tensor_from_duration_value(&scalar).unwrap();
    assert_eq!(scalar_days.shape(), &[1, 1]);
    assert_eq!(scalar_days.materialize_f64(), vec![1.0 / 24.0]);

    let hours = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2]).unwrap());
    let vector = run_duration(vec![hours.clone(), Value::Num(0.0), Value::Num(0.0)]);
    let days = duration_tensor_from_duration_value(&vector).unwrap();
    assert_eq!(days.shape(), &[2]);
    let strings = duration_string_array(&vector).unwrap().unwrap();
    assert_eq!(strings.shape, vec![2]);
    assert_eq!(strings.data().len(), 2);
    assert_eq!(
        duration_summary(&vector).unwrap(),
        Some("[2 duration]".into())
    );
    let selectors = Value::Cell(
        runmat_value::CellArray::new(
            vec![integer_tensor(
                runmat_value::IntegerStorage::U64(vec![2, 1]),
                vec![2],
            )],
            1,
            1,
        )
        .unwrap(),
    );
    let selected = futures::executor::block_on(duration_subsref(vector, "()".into(), selectors))
        .expect("rank-one duration selection");
    let selected = duration_tensor_from_duration_value(&selected).unwrap();
    // Shared numeric linear indexing canonicalizes rank-one storage to its
    // minimum two-dimensional column result; the original remains rank one.
    assert_eq!(selected.shape(), &[2, 1]);
    let values = days.materialize_f64();
    assert_eq!(selected.materialize_f64(), vec![values[1], values[0]]);

    let mismatch = Value::Tensor(Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap());
    let error = futures::executor::block_on(duration_builtin_source(
        None,
        vec![hours, mismatch, Value::Num(0.0)],
    ))
    .expect_err("non-scalar components require exact shape, not equal cardinality");
    assert_eq!(
        error.identifier(),
        Some(DurationErrorKind::ShapeMismatch.identifier())
    );
}

#[test]
fn duration_descriptor_signatures_cover_constructor_and_methods() {
    let labels: Vec<&str> = DURATION_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"t = duration(X)"));
    assert!(labels.contains(&"t = duration(hours, minutes, seconds)"));
    assert!(labels.contains(&"t = duration(hours, minutes, seconds, milliseconds)"));
    assert!(labels.contains(&"t = duration(___, \"Format\", format)"));

    let four_component = DURATION_DESCRIPTOR
        .signatures
        .iter()
        .find(|signature| signature.label == "t = duration(hours, minutes, seconds, milliseconds)")
        .expect("four-component duration signature");
    assert_eq!(
        four_component
            .inputs
            .iter()
            .map(|input| input.name)
            .collect::<Vec<_>>(),
        ["hours", "minutes", "seconds", "milliseconds"]
    );
    assert!(four_component.inputs.iter().all(|input| {
        matches!(input.ty, BuiltinParamType::NumericArray)
            && matches!(input.arity, BuiltinParamArity::Required)
    }));

    assert_eq!(
        DURATION_SUBSREF_DESCRIPTOR.signatures[0].label,
        "out = duration.subsref(obj, S)"
    );
    assert_eq!(
        DURATION_BINARY_DESCRIPTOR.signatures[0].label,
        "out = duration.op(lhs, rhs)"
    );
}

#[test]
fn duration_builds_from_components() {
    let value = run_duration(vec![Value::Num(1.0), Value::Num(30.0), Value::Num(45.0)]);
    let rendered = duration_display_text(&value)
        .expect("display")
        .expect("duration text");
    assert_eq!(rendered, "01:30:45");
}

#[test]
fn typed_duration_diagnostics_preserve_messages_and_context() {
    let invalid = futures::executor::block_on(duration_builtin_source(
        None,
        vec![Value::String("bad".into())],
    ))
    .expect_err("text is not a numeric duration component");
    assert_eq!(
        invalid.identifier(),
        Some(DurationErrorKind::InvalidArgument.identifier())
    );
    assert!(invalid.message().contains("expected a numeric"));

    let invalid_format = futures::executor::block_on(duration_builtin_source(
        None,
        vec![
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(3.0),
            Value::String("Format".into()),
            Value::String("bogus".into()),
        ],
    ))
    .expect_err("unsupported Format value");
    assert_eq!(
        invalid_format.identifier(),
        Some(DurationErrorKind::InvalidFormat.identifier())
    );
    assert!(invalid_format
        .message()
        .contains("unsupported Format value"));

    let lhs = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
    let rhs = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap());
    let mismatch = futures::executor::block_on(duration_builtin_source(
        None,
        vec![lhs, rhs, Value::Num(0.0)],
    ))
    .expect_err("incompatible components");
    assert_eq!(
        mismatch.identifier(),
        Some(DurationErrorKind::ShapeMismatch.identifier())
    );
    assert!(mismatch.message().contains("matching sizes"));
}

#[test]
fn delegated_duration_failure_retains_identifier_cause_and_retry_policy() {
    let source = crate::build_runtime_error("index must be positive")
        .with_identifier("RunMat:index:InvalidIndex")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build();
    let wrapped = duration_error_from(
        DurationErrorKind::InvalidIndex,
        "duration.subsref: ",
        source,
    );
    assert_eq!(wrapped.identifier(), Some("RunMat:index:InvalidIndex"));
    assert_eq!(
        wrapped.message(),
        "duration.subsref: index must be positive"
    );
    assert_eq!(wrapped.context.builtin.as_deref(), Some("duration"));
    assert_eq!(wrapped.gpu_gather_retry(), crate::GpuGatherRetry::Never);
    assert!(wrapped.source.is_some());
}

#[test]
fn duration_formats_arrays() {
    let hours = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
    let minutes = Value::Tensor(Tensor::new(vec![15.0, 45.0], vec![1, 2]).unwrap());
    let value = run_duration(vec![hours, minutes, Value::Num(0.0)]);
    let rendered = duration_display_text(&value)
        .expect("display")
        .expect("duration text");
    assert!(rendered.contains("01:15:00"));
    assert!(rendered.contains("02:45:00"));
}

#[test]
fn duration_typed_integer_components_cross_double_boundary_exactly() {
    let hours = integer_tensor(runmat_value::IntegerStorage::U8(vec![1, 2]), vec![1, 2]);
    let minutes = integer_tensor(runmat_value::IntegerStorage::U16(vec![15, 45]), vec![1, 2]);
    let seconds = integer_tensor(runmat_value::IntegerStorage::I16(vec![0, 30]), vec![1, 2]);
    let value = run_duration(vec![hours, minutes, seconds]);
    let rendered = duration_display_text(&value)
        .expect("display")
        .expect("duration text");
    assert!(rendered.contains("01:15:00"));
    assert!(rendered.contains("02:45:30"));
}

#[test]
fn duration_integer_matrix_form_supports_all_classes_and_returns_column() {
    let storages = [
        runmat_value::IntegerStorage::I8(vec![1, 2, 15, 45, 0, 30]),
        runmat_value::IntegerStorage::I16(vec![1, 2, 15, 45, 0, 30]),
        runmat_value::IntegerStorage::I32(vec![1, 2, 15, 45, 0, 30]),
        runmat_value::IntegerStorage::I64(vec![1, 2, 15, 45, 0, 30]),
        runmat_value::IntegerStorage::U8(vec![1, 2, 15, 45, 0, 30]),
        runmat_value::IntegerStorage::U16(vec![1, 2, 15, 45, 0, 30]),
        runmat_value::IntegerStorage::U32(vec![1, 2, 15, 45, 0, 30]),
        runmat_value::IntegerStorage::U64(vec![1, 2, 15, 45, 0, 30]),
    ];
    for storage in storages {
        let value = run_duration(vec![integer_tensor(storage, vec![2, 3])]);
        let days = duration_tensor_from_duration_value(&value).expect("duration days");
        assert_eq!(days.shape(), &[2, 1]);
        let rendered = duration_display_text(&value)
            .expect("display")
            .expect("duration text");
        assert!(rendered.contains("01:15:00"));
        assert!(rendered.contains("02:45:30"));
    }
}

#[test]
fn duration_four_component_form_adds_milliseconds() {
    let value = run_duration(vec![
        Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap()),
        Value::Num(0.0),
        Value::Num(1.0),
        Value::Int(runmat_value::IntValue::U16(250)),
    ]);
    let days = duration_tensor_from_duration_value(&value).expect("duration days");
    assert_eq!(days.shape(), &[1, 2]);
    let seconds: Vec<f64> = days
        .materialize_f64()
        .into_iter()
        .map(|days| days * SECONDS_PER_DAY)
        .collect();
    assert!((seconds[0] - 1.25).abs() < 1.0e-12);
    assert!((seconds[1] - 3601.25).abs() < 1.0e-9);
}

#[test]
fn duration_public_components_preserve_nan_and_infinity() {
    let value = run_duration(vec![Value::Num(f64::NAN), Value::Num(0.0), Value::Num(0.0)]);
    assert!(duration_tensor_from_duration_value(&value)
        .unwrap()
        .materialize_f64()[0]
        .is_nan());
    for infinite in [f64::INFINITY, f64::NEG_INFINITY] {
        let value = run_duration(vec![Value::Num(infinite), Value::Num(0.0), Value::Num(0.0)]);
        assert_eq!(
            duration_tensor_from_duration_value(&value)
                .unwrap()
                .materialize_f64()[0],
            infinite
        );
        let expected = if infinite.is_sign_negative() {
            "-Inf"
        } else {
            "Inf"
        };
        assert_eq!(
            duration_display_text(&value).expect("display"),
            Some(expected.to_string())
        );
        assert_eq!(
            duration_string_array(&value)
                .expect("string conversion")
                .expect("duration string array")
                .data(),
            vec![expected.to_string()]
        );
        let chars = duration_char_array(&value)
            .expect("char conversion")
            .expect("duration char array");
        assert_eq!(
            runmat_value::decode_utf16_code_units(chars.data()).expect("valid UTF-16 fixture"),
            expected
        );
    }
}

#[test]
fn duration_short_numeric_forms_are_extension_gated_but_matrix_is_public() {
    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = futures::executor::block_on(duration_builtin_source(None, vec![Value::Num(1.0)]))
        .expect_err("one-component hour extension");
    assert_eq!(
        error.identifier(),
        DURATION_SHORT_COMPONENT_EXTENSION.error_identifier
    );
    let matrix = integer_tensor(runmat_value::IntegerStorage::U8(vec![1, 30, 0]), vec![1, 3]);
    futures::executor::block_on(duration_builtin_source(None, vec![matrix]))
        .expect("documented matrix form remains public");
    drop(strict);

    let extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    futures::executor::block_on(duration_builtin_source(None, vec![Value::Num(1.0)]))
        .expect("one-component extension in RunMat mode");
    futures::executor::block_on(duration_builtin_source(
        None,
        vec![Value::Num(1.0), Value::Num(30.0)],
    ))
    .expect("two-component extension in RunMat mode");
    drop(extensions);
}

#[test]
fn duration_gpu_extension_rejects_before_provider_access() {
    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 3],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    assert!(
        crate::builtins::common::gpu_helpers::exact_provider_for_handle(&allocation.handle)
            .is_none()
    );
    let resident = Value::GpuTensor(allocation.handle.clone());
    let error = futures::executor::block_on(duration_builtin_source(None, vec![resident]))
        .expect_err("GPU extension gate");
    assert_eq!(
        error.identifier(),
        DURATION_GPU_INPUT_EXTENSION.error_identifier
    );
    assert_eq!(allocation.probe.retirements(), 0);
    allocation.handle.release().unwrap();
    assert_eq!(allocation.probe.retirements(), 1);
    drop(strict);
}

#[test]
fn duration_missing_days_render_without_error() {
    let value = duration_object_from_days_tensor(
        None,
        Tensor::new(vec![f64::NAN], vec![1, 1]).unwrap(),
        DEFAULT_DURATION_FORMAT,
    )
    .expect("duration object");
    let rendered = duration_string_array(&value)
        .expect("string array")
        .expect("duration strings");
    assert_eq!(rendered.data(), vec!["NaN".to_string()]);
    assert_eq!(
        duration_display_text(&value).expect("display"),
        Some("NaN".to_string())
    );
}

#[test]
fn duration_unit_helpers_create_and_convert_values() {
    let one_day = futures::executor::block_on(days_builtin(Value::Num(1.0))).expect("days");
    assert!(is_duration_object(&one_day));
    let as_hours = futures::executor::block_on(hours_builtin(one_day.clone())).expect("hours");
    assert_eq!(as_hours, Value::Num(24.0));
    let as_minutes =
        futures::executor::block_on(minutes_builtin(one_day.clone())).expect("minutes");
    assert_eq!(as_minutes, Value::Num(1440.0));
    let as_seconds =
        futures::executor::block_on(seconds_builtin(one_day.clone())).expect("seconds");
    assert_eq!(as_seconds, Value::Num(86_400.0));
    let as_millis =
        futures::executor::block_on(milliseconds_builtin(one_day.clone())).expect("millis");
    assert_eq!(as_millis, Value::Num(86_400_000.0));

    let two_hours = futures::executor::block_on(hours_builtin(Value::Num(2.0))).expect("hours");
    let rendered = duration_display_text(&two_hours)
        .expect("display")
        .expect("duration text");
    assert_eq!(rendered, "02:00:00");

    let year = futures::executor::block_on(years_builtin(Value::Num(1.0))).expect("years");
    let year_days = duration_tensor_from_duration_value(&year).expect("duration tensor");
    assert!((tensor::tensor_value_f64(&year_days, 0) - 365.2425).abs() < 1e-9);
    assert_eq!(
        isduration_builtin(year).expect("isduration"),
        Value::Bool(true)
    );
    assert_eq!(
        isduration_builtin(Value::Num(1.0)).expect("isduration"),
        Value::Bool(false)
    );
    assert!(futures::executor::block_on(years_builtin(Value::Num(f64::MAX))).is_err());
}

#[test]
fn duration_unit_helpers_read_typed_integer_days_exactly() {
    let days = Tensor::new_integer(runmat_value::IntegerStorage::I16(vec![1, 2]), vec![1, 2])
        .expect("integer tensor");
    let value = duration_object_from_days_tensor(None, days, DEFAULT_DURATION_FORMAT)
        .expect("duration object");

    let hours = futures::executor::block_on(hours_builtin(value.clone())).expect("hours");
    assert_eq!(
        hours,
        Value::Tensor(Tensor::new(vec![24.0, 48.0], vec![1, 2]).unwrap())
    );

    let rendered = duration_display_text(&value)
        .expect("display")
        .expect("duration text");
    assert!(rendered.contains("24:00:00"));
    assert!(rendered.contains("48:00:00"));
    assert_eq!(
        duration_summary(&value).expect("summary"),
        Some("[1x2 duration]".to_string())
    );
}

#[test]
fn duration_supports_format_assignment_and_indexing() {
    let value = run_duration(vec![Value::Num(1.0), Value::Num(5.0), Value::Num(0.0)]);
    let updated = futures::executor::block_on(duration_subsasgn(
        value.clone(),
        ".".to_string(),
        Value::String(FORMAT_FIELD.to_string().into()),
        Value::String("hh:mm".to_string().into()),
    ))
    .expect("subsasgn");
    let rendered = duration_display_text(&updated)
        .expect("display")
        .expect("duration text");
    assert_eq!(rendered, "01:05");

    let array = run_duration(vec![
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
        Value::Num(0.0),
        Value::Num(0.0),
    ]);
    let payload = Value::Cell(runmat_value::CellArray::new(vec![Value::Num(2.0)], 1, 1).unwrap());
    let indexed = futures::executor::block_on(duration_subsref(array, "()".to_string(), payload))
        .expect("subsref");
    let text = duration_display_text(&indexed)
        .expect("display")
        .expect("duration text");
    assert_eq!(text, "02:00:00");
}

#[test]
fn duration_typed_integer_index_selectors_are_exact() {
    let array = run_duration(vec![
        integer_tensor(runmat_value::IntegerStorage::U8(vec![1, 2]), vec![1, 2]),
        Value::Num(0.0),
        Value::Num(0.0),
    ]);
    let payload = Value::Cell(
        runmat_value::CellArray::new(
            vec![integer_tensor(
                runmat_value::IntegerStorage::U64(vec![2]),
                vec![1, 1],
            )],
            1,
            1,
        )
        .unwrap(),
    );
    let indexed = futures::executor::block_on(duration_subsref(array, "()".to_string(), payload))
        .expect("subsref");
    let text = duration_display_text(&indexed)
        .expect("display")
        .expect("duration text");
    assert_eq!(text, "02:00:00");
}

#[test]
fn duration_wide_index_does_not_round_through_double() {
    let array = run_duration(vec![
        integer_tensor(runmat_value::IntegerStorage::U8(vec![1, 2]), vec![1, 2]),
        Value::Num(0.0),
        Value::Num(0.0),
    ]);
    let payload = Value::Cell(
        runmat_value::CellArray::new(
            vec![integer_tensor(
                runmat_value::IntegerStorage::U64(vec![u64::MAX]),
                vec![1, 1],
            )],
            1,
            1,
        )
        .unwrap(),
    );
    futures::executor::block_on(duration_subsref(array, "()".to_string(), payload))
        .expect_err("wide out-of-range index must not round to a valid double index");
}

#[test]
fn duration_comparisons_return_logical_values_without_tolerance() {
    let zero = duration_object_from_days(None, vec![0.0], vec![1, 1], DEFAULT_DURATION_FORMAT)
        .expect("zero duration");
    let tiny = duration_object_from_days(None, vec![1.0e-13], vec![1, 1], DEFAULT_DURATION_FORMAT)
        .expect("tiny duration");
    assert_eq!(
        futures::executor::block_on(super::builtins::duration_eq(zero.clone(), tiny.clone()))
            .expect("equality"),
        Value::Bool(false)
    );
    assert_eq!(
        futures::executor::block_on(super::builtins::duration_ne(zero.clone(), tiny))
            .expect("inequality"),
        Value::Bool(true)
    );
    assert_eq!(
        futures::executor::block_on(super::builtins::duration_le(zero, Value::Num(0.0)))
            .expect("numeric day comparison"),
        Value::Bool(true)
    );
}

#[test]
fn comparison_method_requires_a_duration_operand() {
    for result in [
        futures::executor::block_on(super::builtins::duration_eq(
            Value::Num(1.0),
            Value::Num(1.0),
        )),
        futures::executor::block_on(super::builtins::duration_lt(
            Value::Num(1.0),
            Value::Num(2.0),
        )),
    ] {
        assert!(result
            .expect_err("duration method receiver")
            .message()
            .contains("at least one operand"));
    }
}

#[test]
fn member_indexing_rejects_unrelated_object_class() {
    let other = Value::Object(ObjectInstance::new("OtherClass".to_string()));
    let read = futures::executor::block_on(duration_subsref(
        other.clone(),
        ".".to_string(),
        Value::from("Format"),
    ));
    assert!(read
        .expect_err("foreign read")
        .message()
        .contains("duration object"));
    let write = futures::executor::block_on(duration_subsasgn(
        other,
        ".".to_string(),
        Value::from("Format"),
        Value::from("hh:mm"),
    ));
    assert!(write
        .expect_err("foreign write")
        .message()
        .contains("duration object"));
}

#[test]
fn duration_arithmetic_and_comparison_broadcast_dimensions() {
    let column =
        duration_object_from_days(None, vec![1.0, 2.0], vec![2, 1], DEFAULT_DURATION_FORMAT)
            .expect("column duration");
    let row = duration_object_from_days(None, vec![1.0, 3.0], vec![1, 2], DEFAULT_DURATION_FORMAT)
        .expect("row duration");
    let equal = futures::executor::block_on(super::builtins::duration_eq(column.clone(), row))
        .expect("broadcast equality");
    let Value::LogicalArray(equal) = equal else {
        panic!("comparison must return a logical array");
    };
    assert_eq!(equal.shape, vec![2, 2]);
    let shifted = futures::executor::block_on(super::builtins::duration_plus(
        None,
        column,
        Value::Num(1.0),
    ))
    .expect("numeric days addition");
    let days = duration_tensor_from_duration_value(&shifted).expect("duration days");
    assert_eq!(days.shape(), &[2, 1]);
    assert_eq!(days.materialize_f64(), vec![2.0, 3.0]);
}

#[test]
fn duration_empty_component_and_unit_shapes_are_retained() {
    let empty = Value::Tensor(Tensor::new(vec![], vec![0, 2]).expect("empty tensor"));
    let value = run_duration(vec![empty, Value::Num(0.0), Value::Num(0.0)]);
    let days = duration_tensor_from_duration_value(&value).expect("duration days");
    assert_eq!(days.shape(), &[0, 2]);
    let hours = futures::executor::block_on(hours_builtin(value)).expect("hours");
    let Value::Tensor(hours) = hours else {
        panic!("empty unit counts remain an array")
    };
    assert_eq!(hours.shape(), &[0, 2]);
}

#[test]
fn duration_nd_and_empty_operator_shapes_are_retained() {
    let cube =
        duration_object_from_days(None, vec![1.0; 8], vec![2, 2, 2], DEFAULT_DURATION_FORMAT)
            .expect("N-D duration");
    let shifted = futures::executor::block_on(duration_plus(None, cube, Value::Num(1.0)))
        .expect("N-D addition");
    assert_eq!(
        duration_tensor_from_duration_value(&shifted)
            .unwrap()
            .shape(),
        &[2, 2, 2]
    );

    let empty = duration_object_from_days(None, vec![], vec![0, 2], DEFAULT_DURATION_FORMAT)
        .expect("empty duration");
    let compared =
        futures::executor::block_on(duration_eq(empty, Value::Num(1.0))).expect("empty comparison");
    let Value::LogicalArray(compared) = compared else {
        panic!("empty logical array")
    };
    assert_eq!(compared.shape, vec![0, 2]);
}

#[test]
fn duration_does_not_download_unrelated_resident_object_property() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("resident allocation");
    let mut value = duration_object_from_days(None, vec![1.0], vec![1, 1], DEFAULT_DURATION_FORMAT)
        .expect("duration");
    let Value::Object(ref mut object) = value else {
        unreachable!()
    };
    object.replace_property_unbounded(
        "unrelated".into(),
        Value::GpuTensor(allocation.handle.clone()),
    );
    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
    assert_eq!(
        futures::executor::block_on(hours_builtin(value)).expect("host day count"),
        Value::Num(24.0)
    );
    assert_eq!(allocation.probe.retirements(), 0);
    drop(strict);
    allocation
        .handle
        .release()
        .expect("release resident allocation");
}

#[test]
fn duration_nested_resident_day_count_rejects_before_download() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("resident allocation");
    let mut value = duration_object_from_days(None, vec![1.0], vec![1, 1], DEFAULT_DURATION_FORMAT)
        .expect("duration");
    let Value::Object(ref mut object) = value else {
        unreachable!()
    };
    object.replace_property_unbounded(
        DAYS_FIELD.into(),
        Value::GpuTensor(allocation.handle.clone()),
    );
    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = futures::executor::block_on(hours_builtin(value)).expect_err("strict mode");
    assert_eq!(
        error.identifier(),
        DURATION_GPU_INPUT_EXTENSION.error_identifier
    );
    assert_eq!(allocation.probe.retirements(), 0);
    drop(strict);
    allocation
        .handle
        .release()
        .expect("release resident allocation");
}

#[test]
fn duration_constructor_rejects_host_shape_conflict_before_resident_download() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 2],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("resident allocation");
    let host = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap());
    let error = futures::executor::block_on(duration_builtin_source(
        None,
        vec![
            host,
            Value::GpuTensor(allocation.handle.clone()),
            Value::Num(0.0),
        ],
    ))
    .expect_err("shape mismatch must preflight before provider access");
    assert!(error.message().contains("matching sizes"));
    assert_eq!(allocation.probe.retirements(), 0);
    allocation
        .handle
        .release()
        .expect("release resident allocation");
}

#[test]
fn logical_component_shape_conflict_precedes_resident_download() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let resident = crate::builtins::common::gpu_helpers::upload_tensor(
            provider,
            &Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap(),
        )
        .unwrap();
        let logical = Value::LogicalArray(
            runmat_value::LogicalArray::new(vec![1, 0, 1], vec![1, 3]).unwrap(),
        );
        provider.reset_telemetry();
        let error = futures::executor::block_on(duration_builtin_source(
            None,
            vec![Value::GpuTensor(resident), logical, Value::Num(0.0)],
        ))
        .expect_err("known logical shape conflict");
        assert!(error.message().contains("matching sizes"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn operator_shape_conflicts_precede_resident_download_and_valid_input_gathers_once() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let resident = crate::builtins::common::gpu_helpers::upload_tensor(
            provider,
            &Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap(),
        )
        .unwrap();
        let mut lhs =
            duration_object_from_days(None, vec![0.0], vec![1, 1], DEFAULT_DURATION_FORMAT)
                .unwrap();
        let Value::Object(object) = &mut lhs else {
            unreachable!()
        };
        object.replace_property_unbounded(DAYS_FIELD.into(), Value::GpuTensor(resident));
        provider.reset_telemetry();
        let bad = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap());
        for result in [
            futures::executor::block_on(duration_plus(None, lhs.clone(), bad.clone())),
            futures::executor::block_on(duration_minus(None, lhs.clone(), bad.clone())),
            futures::executor::block_on(duration_eq(lhs.clone(), bad)),
        ] {
            assert!(result
                .expect_err("shape conflict")
                .message()
                .contains("broadcast"));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        }
        let datetime = crate::builtins::datetime::datetime_object_from_serial_tensor(
            None,
            Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap(),
            "yyyy-MM-dd",
        )
        .unwrap();
        let error = futures::executor::block_on(duration_plus(None, lhs.clone(), datetime))
            .expect_err("datetime shape conflict");
        assert!(error.message().contains("broadcast"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

        let valid = futures::executor::block_on(duration_plus(None, lhs, Value::Num(1.0)))
            .expect("resident duration addition");
        assert_eq!(
            duration_tensor_from_duration_value(&valid)
                .unwrap()
                .materialize_f64(),
            vec![2.0, 3.0]
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 16);
    });
}
