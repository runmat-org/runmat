use super::*;
use futures::executor::block_on;
use runmat_builtins::{
    builtin_function_by_name, BuiltinIntegerOutputClassRule, GAUSPULS_INTEGER_CAPABILITIES,
};
use runmat_value::{CharArray, IntValue, IntegerStorage, NumericStorage};

#[test]
fn extreme_finite_db_levels_do_not_underflow_before_logarithm() {
    let cutoff = gauspuls_cutoff(default_params(), -10000.);
    assert!(cutoff.is_finite() && cutoff > 0.);
    let params = GauspulsParams {
        fc: 1.,
        bw: 1.,
        bwr: -10000.,
    };
    let value = gauspuls_scalar(1., params);
    assert!(value.is_finite() && value > 0. && value < 1.);
    let ordinary = gauspuls_cutoff(default_params(), -60.);
    assert!((cutoff / ordinary - (10000_f64 / 60.).sqrt()).abs() < 1e-12);
}

#[test]
fn discarded_samples_still_validate_representation_and_exactness() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    for value in [
        Value::Complex(1.0, 2.0),
        Value::String("unknown-mode".into()),
        Value::Struct(runmat_value::StructValue::new()),
        Value::Int(IntValue::U64(u64::MAX)),
    ] {
        let error = call(value, vec![]).unwrap_err();
        assert_eq!(error.identifier(), GAUSPULS_ERROR_INVALID_INPUT.identifier);
    }
    let handle = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap()
        .handle;
    assert!(call(Value::GpuTensor(handle), vec![]).is_err());
    assert_eq!(
        call(Value::Num(0.0), vec![]).unwrap(),
        Value::output_list(vec![])
    );
}

fn call(t: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    block_on(gauspuls_builtin(
        t,
        crate::builtins::common::test_support::builtin_arguments(rest),
    ))
}

fn expect_tensor(value: Value) -> Tensor {
    match value {
        Value::Tensor(tensor) => tensor,
        other => panic!("expected tensor, got {other:?}"),
    }
}

fn integer_tensor(values: Vec<i16>, shape: Vec<usize>) -> Tensor {
    Tensor::new_integer(IntegerStorage::I16(values), shape).expect("typed integer tensor")
}

fn all_integer_time_tensors() -> Vec<Tensor> {
    vec![
        Tensor::new_integer(IntegerStorage::I8(vec![0]), vec![1, 1]).unwrap(),
        Tensor::new_integer(IntegerStorage::I16(vec![0]), vec![1, 1]).unwrap(),
        Tensor::new_integer(IntegerStorage::I32(vec![0]), vec![1, 1]).unwrap(),
        Tensor::new_integer(IntegerStorage::I64(vec![0]), vec![1, 1]).unwrap(),
        Tensor::new_integer(IntegerStorage::U8(vec![0]), vec![1, 1]).unwrap(),
        Tensor::new_integer(IntegerStorage::U16(vec![0]), vec![1, 1]).unwrap(),
        Tensor::new_integer(IntegerStorage::U32(vec![0]), vec![1, 1]).unwrap(),
        Tensor::new_integer(IntegerStorage::U64(vec![0]), vec![1, 1]).unwrap(),
    ]
}

fn all_integer_controls() -> Vec<Value> {
    vec![
        Value::Int(IntValue::I8(1)),
        Value::Int(IntValue::I16(1)),
        Value::Int(IntValue::I32(1)),
        Value::Int(IntValue::I64(1)),
        Value::Int(IntValue::U8(1)),
        Value::Int(IntValue::U16(1)),
        Value::Int(IntValue::U32(1)),
        Value::Int(IntValue::U64(1)),
    ]
}

#[test]
fn gauspuls_defaults_peak_at_one() {
    let out = call(Value::Num(0.0), Vec::new()).expect("gauspuls");
    assert!(matches!(out, Value::Num(value) if (value - 1.0).abs() <= 1e-12));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn gauspuls_single_samples_keep_rank_zero_and_rank_one_shapes() {
    for (shape, samples) in [(Vec::new(), vec![0.0_f32]), (vec![2], vec![0.0_f32, 0.0])] {
        let input = Tensor::from_f32(samples, shape.clone()).expect("single samples");
        let output = expect_tensor(call(Value::Tensor(input), Vec::new()).expect("gauspuls"));
        assert_eq!(output.shape(), shape);
        assert_eq!(output.numeric_dtype(), NumericDType::F32);
        assert_eq!(
            output.as_f32_slice(),
            Some(vec![1.0_f32; output.len()].as_slice())
        );
    }
}

#[test]
fn gauspuls_custom_parameters_preserve_shape() {
    let input = Tensor::new(vec![-0.001, 0.0, 0.001], vec![1, 3]).unwrap();
    let out = expect_tensor(
        call(
            Value::Tensor(input),
            vec![Value::Num(1000.0), Value::Num(0.5), Value::Num(-6.0)],
        )
        .expect("gauspuls"),
    );
    assert_eq!(out.shape(), &[1, 3]);
    assert!(out.materialize_f64()[1] > out.materialize_f64()[0]);
    assert!(out.materialize_f64()[1] > out.materialize_f64()[2]);
}

#[test]
fn gauspuls_reads_typed_integer_storage_exactly() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let input = integer_tensor(vec![0, 1], vec![1, 2]);
    let out = expect_tensor(call(Value::Tensor(input), Vec::new()).expect("gauspuls"));
    assert_eq!(out.shape(), &[1, 2]);
    assert!((out.materialize_f64()[0] - 1.0).abs() <= 1e-12);
    assert!(out.materialize_f64()[1].is_finite());
}

#[test]
fn gauspuls_reads_all_integer_time_classes_from_authoritative_storage() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    for input in all_integer_time_tensors() {
        let output = call(Value::Tensor(input), Vec::new()).expect("gauspuls");
        assert!(matches!(output, Value::Num(value) if value == 1.0));
    }
}

#[test]
fn gauspuls_accepts_all_integer_scalar_control_classes() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    for control in all_integer_controls() {
        let output = call(Value::Num(0.0), vec![control]).expect("integer FC");
        assert!(matches!(output, Value::Num(value) if value == 1.0));
    }
}

#[test]
fn gauspuls_rejects_wide_integers_at_binary64_boundaries() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let wide = (1_u64 << 53) + 1;
    let input = Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).unwrap();
    let error = call(Value::Tensor(input), Vec::new()).expect_err("wide sample time");
    assert_eq!(error.identifier(), GAUSPULS_ERROR_INVALID_INPUT.identifier);
    assert!(error.message().contains("exactly representable as double"));

    let error = call(Value::Num(0.0), vec![Value::Int(IntValue::U64(wide))])
        .expect_err("wide scalar control");
    assert_eq!(
        error.identifier(),
        GAUSPULS_ERROR_INVALID_PARAMETER.identifier
    );
    assert!(error.message().contains("exactly representable as double"));
}

#[test]
fn gauspuls_preserves_native_single_time_shape_and_output_class() {
    let input =
        Tensor::from_numeric_storage(NumericStorage::F32(vec![-0.001, 0.0, 0.001]), vec![1, 3])
            .unwrap();
    let output = expect_tensor(call(Value::Tensor(input), Vec::new()).expect("single T"));
    assert_eq!(output.shape(), &[1, 3]);
    assert_eq!(output.numeric_dtype(), NumericDType::F32);

    let _count = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let input =
        Tensor::from_numeric_storage(NumericStorage::F32(vec![0.0, 0.00025]), vec![2, 1]).unwrap();
    let Value::OutputList(outputs) = call(Value::Tensor(input), Vec::new()).expect("outputs")
    else {
        panic!("expected output list");
    };
    for output in outputs.try_into_owned().expect("unique output list") {
        let output = expect_tensor(output);
        assert_eq!(output.shape(), &[2, 1]);
        assert_eq!(output.numeric_dtype(), NumericDType::F32);
    }
}

#[test]
fn gauspuls_extension_roles_and_compatibility_order_are_stable() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);

    let integer_time =
        call(Value::Int(IntValue::U8(0)), Vec::new()).expect_err("integer time extension");
    assert_eq!(
        integer_time.identifier(),
        GAUSPULS_INTEGER_TIME_EXTENSION.error_identifier
    );

    let logical_time = call(Value::Bool(false), Vec::new()).expect_err("logical time extension");
    assert_eq!(
        logical_time.identifier(),
        GAUSPULS_LOGICAL_TIME_EXTENSION.error_identifier
    );

    let integer_control = call(Value::Num(0.0), vec![Value::Int(IntValue::U8(1))])
        .expect_err("integer control extension");
    assert_eq!(
        integer_control.identifier(),
        GAUSPULS_INTEGER_CONTROL_EXTENSION.error_identifier
    );

    let logical_control =
        call(Value::Num(0.0), vec![Value::Bool(true)]).expect_err("logical control extension");
    assert_eq!(
        logical_control.identifier(),
        GAUSPULS_LOGICAL_CONTROL_EXTENSION.error_identifier
    );

    let single_control =
        Tensor::from_numeric_storage(NumericStorage::F32(vec![1.0]), vec![1, 1]).unwrap();
    let single_control = call(Value::Num(0.0), vec![Value::Tensor(single_control)])
        .expect_err("single control extension");
    assert_eq!(
        single_control.identifier(),
        GAUSPULS_SINGLE_CONTROL_EXTENSION.error_identifier
    );

    let resident = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("admit owned fixture")
        .handle;
    let resident_error = call(Value::GpuTensor(resident.clone()), Vec::new())
        .expect_err("resident extension before provider access");
    assert_eq!(
        resident_error.identifier(),
        GAUSPULS_RESIDENT_INPUT_EXTENSION.error_identifier
    );
    let resident_control_error = call(Value::Num(0.0), vec![Value::GpuTensor(resident.clone())])
        .expect_err("resident control extension before provider access");
    assert_eq!(
        resident_control_error.identifier(),
        GAUSPULS_RESIDENT_INPUT_EXTENSION.error_identifier
    );

    let resident = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::U64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("admit owned fixture")
        .handle;
    let typed_resident_error = call(Value::GpuTensor(resident.clone()), Vec::new())
        .expect_err("integer role precedes residency");
    assert_eq!(
        typed_resident_error.identifier(),
        GAUSPULS_INTEGER_TIME_EXTENSION.error_identifier
    );
}

#[test]
fn gauspuls_integer_capabilities_cover_time_and_scalar_controls() {
    assert_eq!(GAUSPULS_INTEGER_CAPABILITIES.len(), 2);
    assert_eq!(GAUSPULS_INTEGER_CAPABILITIES[0].inputs[0].classes.len(), 8);
    assert_eq!(GAUSPULS_INTEGER_CAPABILITIES[1].inputs[0].classes.len(), 8);
    assert_eq!(
        GAUSPULS_INTEGER_CAPABILITIES[0].output_class,
        BuiltinIntegerOutputClassRule::Double
    );
}

#[test]
fn gauspuls_multi_output_returns_quadrature_and_envelope() {
    let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let input = Tensor::new(vec![0.0, 0.00025], vec![1, 2]).unwrap();
    let out = call(
        Value::Tensor(input),
        vec![Value::Num(1000.0), Value::Num(0.5), Value::Num(-6.0)],
    )
    .expect("gauspuls");
    let Value::OutputList(outputs) = out else {
        panic!("expected output list");
    };
    assert_eq!(outputs.len(), 3);
    let in_phase = expect_tensor(outputs[0].clone());
    let quadrature = expect_tensor(outputs[1].clone());
    let envelope = expect_tensor(outputs[2].clone());
    assert_eq!(in_phase.shape(), &[1, 2]);
    assert_eq!(quadrature.shape(), &[1, 2]);
    assert_eq!(envelope.shape(), &[1, 2]);
    assert!((in_phase.materialize_f64()[0] - 1.0).abs() <= 1e-12);
    assert!(quadrature.materialize_f64()[0].abs() <= 1e-12);
    assert!((envelope.materialize_f64()[0] - 1.0).abs() <= 1e-12);
    assert!(in_phase.materialize_f64()[1].abs() <= 1e-12);
    assert!((quadrature.materialize_f64()[1] - envelope.materialize_f64()[1]).abs() <= 1e-12);
}

#[test]
fn gauspuls_multi_output_reads_typed_integer_storage_exactly() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let input = integer_tensor(vec![0, 1], vec![1, 2]);
    let out = call(Value::Tensor(input), Vec::new()).expect("gauspuls");
    let Value::OutputList(outputs) = out else {
        panic!("expected output list");
    };
    assert_eq!(outputs.len(), 3);
    let in_phase = expect_tensor(outputs[0].clone());
    let quadrature = expect_tensor(outputs[1].clone());
    let envelope = expect_tensor(outputs[2].clone());
    assert_eq!(in_phase.shape(), &[1, 2]);
    assert_eq!(quadrature.shape(), &[1, 2]);
    assert_eq!(envelope.shape(), &[1, 2]);
    assert!((in_phase.materialize_f64()[0] - 1.0).abs() <= 1e-12);
    assert!(quadrature.materialize_f64()[0].abs() <= 1e-12);
    assert!((envelope.materialize_f64()[0] - 1.0).abs() <= 1e-12);
}

#[test]
fn gauspuls_cutoff_mode_returns_positive_time() {
    let out = call(
        Value::CharArray(CharArray::new_row("cutoff")),
        vec![
            Value::Num(1000.0),
            Value::Num(0.5),
            Value::Num(-6.0),
            Value::Num(-60.0),
        ],
    )
    .expect("cutoff");
    assert!(matches!(out, Value::Num(value) if value > 0.0 && value < 0.01));
}

#[test]
fn gauspuls_cutoff_accepts_documented_empty_default_placeholder() {
    let empty = Tensor::new(Vec::new(), vec![0, 0]).expect("empty double placeholder");
    let output = call(
        Value::CharArray(CharArray::new_row("cutoff")),
        vec![
            Value::Num(50_000.0),
            Value::Num(0.6),
            Value::Tensor(empty),
            Value::Num(-40.0),
        ],
    )
    .expect("documented cutoff form");
    assert!(matches!(output, Value::Num(value) if value.is_finite() && value > 0.0));
}

#[test]
fn gauspuls_cutoff_mode_honors_requested_output_count() {
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let out = call(
        Value::CharArray(CharArray::new_row("cutoff")),
        vec![
            Value::Num(1000.0),
            Value::Num(0.5),
            Value::Num(-6.0),
            Value::Num(-60.0),
        ],
    )
    .expect("cutoff");
    let Value::OutputList(outputs) = out else {
        panic!("expected output list");
    };
    assert_eq!(outputs.len(), 2);
    assert!(matches!(outputs[0], Value::Num(value) if value > 0.0 && value < 0.01));
    assert!(matches!(outputs[1], Value::Num(value) if value == 0.0));
}

#[test]
fn gauspuls_rejects_bad_parameters() {
    let err = call(Value::Num(0.0), vec![Value::Num(0.0)]).expect_err("fc");
    assert_eq!(
        err.identifier(),
        GAUSPULS_ERROR_INVALID_PARAMETER.identifier
    );

    let err = call(Value::Num(0.0), vec![Value::Num(1000.0), Value::Num(0.0)]).expect_err("bw");
    assert_eq!(
        err.identifier(),
        GAUSPULS_ERROR_INVALID_PARAMETER.identifier
    );
}

#[test]
fn gauspuls_is_registered() {
    let entry = runmat_builtins::builtin_catalog_entry_by_name("gauspuls").unwrap();
    assert!(builtin_function_by_name("gauspuls").is_none());
    assert_eq!(entry.bindings.len(), 1);
    let bindings = crate::builtin::runtime_builtin_bindings_by_name("gauspuls");
    assert_eq!(bindings.len(), 1);
    assert_eq!(
        bindings[0].identity,
        entry.binding_identity(&entry.bindings[0])
    );
}
