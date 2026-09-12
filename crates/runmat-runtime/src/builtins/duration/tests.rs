use super::builtins::*;
use super::*;
use runmat_builtins::{BuiltinParamArity, BuiltinParamType};

async fn duration_subsref(obj: Value, kind: String, payload: Value) -> crate::BuiltinResult<Value> {
    let subscript = crate::object::indexing::standard_substruct_fixture_from_parts(&kind, payload)?;
    super::builtins::duration_subsref(obj, subscript).await
}

async fn duration_subsasgn(
    obj: Value,
    kind: String,
    payload: Value,
    rhs: Value,
) -> crate::BuiltinResult<Value> {
    let subscript = crate::object::indexing::standard_substruct_fixture_from_parts(&kind, payload)?;
    super::builtins::duration_subsasgn(obj, subscript, rhs).await
}

fn run_duration(args: Vec<Value>) -> Value {
    futures::executor::block_on(duration_builtin(args)).expect("duration")
}

fn integer_tensor(storage: runmat_value::IntegerStorage, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new_integer(storage, shape).expect("integer tensor"))
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
        assert_eq!(days.shape, vec![2, 1]);
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
    assert_eq!(days.shape, vec![1, 2]);
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
                .data,
            vec![expected.to_string()]
        );
        let chars = duration_char_array(&value)
            .expect("char conversion")
            .expect("duration char array");
        assert_eq!(chars.data.iter().collect::<String>(), expected);
    }
}

#[test]
fn duration_short_numeric_forms_are_extension_gated_but_matrix_is_public() {
    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = futures::executor::block_on(duration_builtin(vec![Value::Num(1.0)]))
        .expect_err("one-component hour extension");
    assert_eq!(
        error.identifier(),
        DURATION_SHORT_COMPONENT_EXTENSION.error_identifier
    );
    let matrix = integer_tensor(runmat_value::IntegerStorage::U8(vec![1, 30, 0]), vec![1, 3]);
    futures::executor::block_on(duration_builtin(vec![matrix]))
        .expect("documented matrix form remains public");
    drop(strict);

    let extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    futures::executor::block_on(duration_builtin(vec![Value::Num(1.0)]))
        .expect("one-component extension in RunMat mode");
    futures::executor::block_on(duration_builtin(vec![Value::Num(1.0), Value::Num(30.0)]))
        .expect("two-component extension in RunMat mode");
    drop(extensions);
}

#[test]
fn duration_gpu_extension_rejects_before_provider_access() {
    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
        shape: vec![1, 3],
        device_id: 0,
        buffer_id: 9_399_002,
        descriptor: Default::default(),
    });
    let error = futures::executor::block_on(duration_builtin(vec![resident]))
        .expect_err("GPU extension gate");
    assert_eq!(
        error.identifier(),
        DURATION_GPU_INPUT_EXTENSION.error_identifier
    );
    drop(strict);
}

#[test]
fn duration_missing_days_render_without_error() {
    let value = duration_object_from_days_tensor(
        Tensor::new(vec![f64::NAN], vec![1, 1]).unwrap(),
        DEFAULT_DURATION_FORMAT,
    )
    .expect("duration object");
    let rendered = duration_string_array(&value)
        .expect("string array")
        .expect("duration strings");
    assert_eq!(rendered.data, vec!["NaN".to_string()]);
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
    let value =
        duration_object_from_days_tensor(days, DEFAULT_DURATION_FORMAT).expect("duration object");

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
        Value::String(FORMAT_FIELD.to_string()),
        Value::String("hh:mm".to_string()),
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
