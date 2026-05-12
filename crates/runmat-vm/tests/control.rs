#[path = "support/mod.rs"]
mod test_helpers;

use runmat_builtins::Value;
use runmat_parser::parse;
use test_helpers::{execute, lower};

#[test]
fn tf_constructs_object_through_vm_dispatch() {
    let program = r#"
        H = tf(20, [1 5]);
        c = class(H);
        n = H.Numerator;
        d = H.Denominator;
    "#;
    let hir = lower(&parse(program).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();

    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Object(object) if object.class_name == "tf")));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::String(class_name) if class_name == "tf")));
    assert!(vars.iter().any(|value| {
        matches!(value, Value::Tensor(tensor) if tensor.shape == vec![1, 1] && tensor.data == vec![20.0])
    }));
    assert!(vars.iter().any(|value| {
        matches!(value, Value::Tensor(tensor) if tensor.shape == vec![1, 2] && tensor.data == vec![1.0, 5.0])
    }));
}
