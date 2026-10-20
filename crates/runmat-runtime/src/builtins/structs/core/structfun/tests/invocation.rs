use super::*;
use runmat_value::Tensor;

#[test]
fn maps_fields_in_order_into_a_column() {
    let mut structure = runmat_value::UnboundedStructValueBuilder::new();
    structure.insert(
        "alpha",
        Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap()),
    );
    structure.insert(
        "beta",
        Value::Tensor(Tensor::new(vec![4.0, 5.0], vec![1, 2]).unwrap()),
    );
    let result = call(
        Value::FunctionHandle("length".into()),
        structure.finish(),
        Vec::new(),
    )
    .unwrap();
    let Value::Tensor(output) = result.value() else {
        panic!("expected tensor")
    };
    assert_eq!(output.shape(), &[2, 1]);
    assert_eq!(output.materialize_f64(), vec![3.0, 2.0]);
}

#[test]
fn nonuniform_output_preserves_names_and_values() {
    let mut structure = runmat_value::UnboundedStructValueBuilder::new();
    structure.insert("name", Value::String("runmat".into()));
    structure.insert("flag", Value::Bool(true));
    let result = call(
        Value::FunctionHandle("class".into()),
        structure.finish(),
        vec![Value::String("UniformOutput".into()), Value::Bool(false)],
    )
    .unwrap();
    let Value::Struct(output) = result.value() else {
        panic!("expected structure")
    };
    assert_eq!(
        output.fields.get("name"),
        Some(&Value::String("string".into()))
    );
    assert_eq!(
        output.fields.get("flag"),
        Some(&Value::String("logical".into()))
    );
}

#[test]
fn empty_structure_has_explicit_outputs() {
    let result = call(
        Value::FunctionHandle("length".into()),
        StructValue::new(),
        Vec::new(),
    )
    .unwrap();
    let Value::Tensor(output) = result.value() else {
        panic!("expected tensor")
    };
    assert_eq!(output.shape(), &[0, 1]);
    let result = call(
        Value::FunctionHandle("length".into()),
        StructValue::new(),
        vec![Value::String("UniformOutput".into()), Value::Bool(false)],
    )
    .unwrap();
    let Value::Struct(output) = result.value() else {
        panic!("expected structure")
    };
    assert!(output.fields.is_empty());
}

#[test]
fn rejects_non_struct_input() {
    let error = futures::executor::block_on(super::super::structfun_builtin(
        None,
        Value::FunctionHandle("length".into()),
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        runmat_builtins::STRUCTFUN_ERROR_NOT_SCALAR_STRUCT.identifier
    );
}
