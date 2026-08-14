#[path = "support/mod.rs"]
mod test_helpers;

use runmat_builtins::{IntValue, Value};
use test_helpers::{compile_source, execute_source};

#[test]
fn callable_vector_and_sparse_validators_accept_documented_integer_empty_forms() {
    let values = execute_source(
        r#"
        value = zeros(0, 3, 'uint16');
        mustBeSparse(value);
        mustBeVector(value, 'allow-all-empties');
        count = numel(value);
        "#,
    )
    .expect("documented empty validator forms should execute");

    assert!(
        values
            .iter()
            .any(|value| matches!(value, Value::Num(number) if *number == 0.0)),
        "expected empty element count, got {values:?}"
    );
}

#[test]
fn arguments_block_vector_option_accepts_all_empty_integer_shapes() {
    let values = execute_source(
        r#"
        result = checked(zeros(0, 3, 'uint16'));
        function out = checked(value)
            arguments
                value {mustBeVector(value, 'allow-all-empties')}
            end
            out = uint64(9007199254740992) + uint64(1);
        end
        "#,
    )
    .expect("arguments-block option should execute");

    assert!(
        values
            .iter()
            .any(|value| matches!(value, Value::Int(IntValue::U64(9_007_199_254_740_993)))),
        "expected exact integer result, got {values:?}"
    );
}

#[test]
fn arguments_block_vector_default_rejects_nonvector_empty_shape() {
    let error = execute_source(
        r#"
        result = checked(zeros(0, 3, 'uint16'));
        function out = checked(value)
            arguments
                value {mustBeVector}
            end
            out = value;
        end
        "#,
    )
    .expect_err("ordinary mustBeVector must reject a 0-by-3 empty array");

    assert_eq!(
        error.identifier(),
        Some("RunMat:ArgumentValidationFunction")
    );
    assert!(error.message().contains("mustBeVector"));
}

#[test]
fn arguments_block_vector_rejects_unknown_literal_option_during_lowering() {
    let error = compile_source(
        r#"
        function out = checked(value)
            arguments
                value {mustBeVector(value, 'unsupported')}
            end
            out = value;
        end
        "#,
    )
    .expect_err("unknown mustBeVector option must not enter bytecode");

    assert!(error.message().contains("allow-all-empties"));
}
