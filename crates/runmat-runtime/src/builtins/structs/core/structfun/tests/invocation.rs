use super::*;
use runmat_value::Tensor;

#[test]
fn maps_fields_in_order_into_a_column() {
    let mut structure = StructValue::new();
    structure.insert(
        "alpha",
        Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap()),
    );
    structure.insert(
        "beta",
        Value::Tensor(Tensor::new(vec![4.0, 5.0], vec![1, 2]).unwrap()),
    );
    let Value::Tensor(output) = call(
        Value::FunctionHandle("length".into()),
        structure,
        Vec::new(),
    )
    .unwrap() else {
        panic!("expected tensor")
    };
    assert_eq!(output.shape, vec![2, 1]);
    assert_eq!(output.materialize_f64(), vec![3.0, 2.0]);
}

#[test]
fn nonuniform_output_preserves_names_and_values() {
    let mut structure = StructValue::new();
    structure.insert("name", Value::String("runmat".into()));
    structure.insert("flag", Value::Bool(true));
    let Value::Struct(output) = call(
        Value::FunctionHandle("class".into()),
        structure,
        vec![Value::String("UniformOutput".into()), Value::Bool(false)],
    )
    .unwrap() else {
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
    let Value::Tensor(output) = call(
        Value::FunctionHandle("length".into()),
        StructValue::new(),
        Vec::new(),
    )
    .unwrap() else {
        panic!("expected tensor")
    };
    assert_eq!(output.shape, vec![0, 1]);
    let Value::Struct(output) = call(
        Value::FunctionHandle("length".into()),
        StructValue::new(),
        vec![Value::String("UniformOutput".into()), Value::Bool(false)],
    )
    .unwrap() else {
        panic!("expected structure")
    };
    assert!(output.fields.is_empty());
}

#[test]
fn rejects_non_struct_input() {
    let error = futures::executor::block_on(super::super::structfun_builtin(
        Value::FunctionHandle("length".into()),
        Value::Num(1.0),
        Vec::new(),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        runmat_builtins::STRUCTFUN_ERROR_NOT_SCALAR_STRUCT.identifier
    );
}
