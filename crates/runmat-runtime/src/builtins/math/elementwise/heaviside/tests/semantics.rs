use runmat_builtins::HEAVISIDE_ERROR_INVALID_INPUT;
use runmat_value::{
    CharArray, IntValue, LogicalArray, NumericStorage, SymbolicExpr, Tensor, Value,
};

use super::execute;

#[test]
fn preserves_single_and_real_step_values() {
    let input = Tensor::from_f32(vec![-1.0, 0.0, 2.0], vec![1, 3]).unwrap();
    let output = super::super::host::apply(input).unwrap();
    assert_eq!(
        output.into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![0.0, 0.5, 1.0])
    );

    for (input, expected) in [(3.5, 1.0), (-2.0, 0.0), (0.0, 0.5), (-0.0, 0.5)] {
        assert_eq!(execute(Value::Num(input)).unwrap(), Value::Num(expected));
    }
    let nan = execute(Value::Num(f64::NAN)).unwrap();
    assert!(matches!(nan, Value::Num(value) if value.is_nan()));
}

#[test]
fn preserves_dense_shape_and_handles_infinities() {
    let input = Tensor::new(
        vec![f64::NEG_INFINITY, -2.0, -0.0, 0.0, 2.0, f64::INFINITY],
        vec![2, 3],
    )
    .unwrap();
    let Value::Tensor(output) = execute(Value::Tensor(input)).unwrap() else {
        panic!("expected tensor")
    };
    assert_eq!(output.shape, vec![2, 3]);
    assert_eq!(output.materialize_f64(), vec![0.0, 0.0, 0.5, 0.5, 1.0, 1.0]);
}

#[test]
fn runmat_extensions_promote_logical_integer_and_character_values() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let logical = LogicalArray::new(vec![0, 1, 0, 1], vec![2, 2]).unwrap();
    let Value::Tensor(output) = execute(Value::LogicalArray(logical)).unwrap() else {
        panic!("expected logical output tensor")
    };
    assert_eq!(output.materialize_f64(), vec![0.5, 1.0, 0.5, 1.0]);

    assert_eq!(
        execute(Value::Int(IntValue::I32(-7))).unwrap(),
        Value::Num(0.0)
    );
    assert_eq!(
        execute(Value::Int(IntValue::I32(0))).unwrap(),
        Value::Num(0.5)
    );
    assert_eq!(
        execute(Value::Int(IntValue::U16(7))).unwrap(),
        Value::Num(1.0)
    );

    let array = CharArray::new("RunMat".chars().collect(), 1, 6).unwrap();
    let Value::Tensor(output) = execute(Value::CharArray(array)).unwrap() else {
        panic!("expected character output tensor")
    };
    assert!(output.materialize_f64().iter().all(|value| *value == 1.0));
}

#[test]
fn rejects_complex_and_string_inputs_with_catalog_error() {
    for input in [Value::Complex(1.0, 1.0), Value::String("runmat".to_owned())] {
        let error = execute(input).expect_err("unsupported input");
        assert_eq!(error.identifier(), HEAVISIDE_ERROR_INVALID_INPUT.identifier);
    }
}

#[test]
fn preserves_symbolic_expression() {
    let output = execute(Value::Symbolic(SymbolicExpr::variable("x"))).unwrap();
    assert!(
        matches!(output, Value::Symbolic(expression) if expression.to_string() == "heaviside(x)")
    );
}

#[test]
fn compatibility_mode_rejects_each_extension_class() {
    for (value, identifier) in [
        (
            Value::Int(IntValue::I32(1)),
            "RunMat:compatibility:HeavisideIntegerInputExtension",
        ),
        (
            Value::Bool(true),
            "RunMat:compatibility:HeavisideLogicalInputExtension",
        ),
        (
            Value::CharArray(CharArray::new(vec!['A'], 1, 1).unwrap()),
            "RunMat:compatibility:HeavisideCharacterInputExtension",
        ),
    ] {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = execute(value).expect_err("strict mode rejects extension");
        assert_eq!(error.identifier(), Some(identifier));
        assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
    }
}
