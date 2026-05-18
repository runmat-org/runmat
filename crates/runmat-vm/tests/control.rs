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

#[test]
fn impulse_returns_siso_response_through_vm_dispatch() {
    let program = r#"
        H = tf(20, [1 5]);
        [y, t] = impulse(H, [0 0.1 0.2]);
        y1 = y(1);
        y2 = y(2);
        t3 = t(3);
    "#;
    let hir = lower(&parse(program).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();

    assert!(vars
        .iter()
        .any(|value| { matches!(value, Value::Tensor(tensor) if tensor.shape == vec![3, 1]) }));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if (*n - 20.0).abs() < 1.0e-8)));
    assert!(vars.iter().any(|value| {
        matches!(value, Value::Num(n) if (*n - (20.0 * (-0.5f64).exp())).abs() < 1.0e-8)
    }));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if (*n - 0.2).abs() < 1.0e-12)));
}

#[test]
fn impulse_docs_examples_execute() {
    let first_order = r#"
        H = tf(20, [1 5]);
        t = 0:0.1:1;
        [y, tout] = impulse(H, t);
        y1 = y(1);
        y11 = y(11);
        tout11 = tout(11);
    "#;
    let hir = lower(&parse(first_order).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars
        .iter()
        .any(|value| { matches!(value, Value::Tensor(tensor) if tensor.shape == vec![11, 1]) }));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if (*n - 20.0).abs() < 1.0e-8)));
    assert!(vars.iter().any(|value| {
        matches!(value, Value::Num(n) if (*n - (20.0 * (-5.0f64).exp())).abs() < 1.0e-8)
    }));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if (*n - 1.0).abs() < 1.0e-12)));

    let plot = r#"
        H = tf(1, [1 3 2]);
        impulse(H);
    "#;
    let hir = lower(&parse(plot).unwrap()).unwrap();
    execute(&hir).unwrap();

    let discrete = r#"
        H = tf(1, [1 -0.5], 0.1);
        [y, t] = impulse(H, 0:0.1:0.5);
        y1 = y(1);
        y2 = y(2);
        y6 = y(6);
        t6 = t(6);
    "#;
    let hir = lower(&parse(discrete).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars
        .iter()
        .any(|value| { matches!(value, Value::Tensor(tensor) if tensor.shape == vec![6, 1]) }));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if n.abs() < 1.0e-12)));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if (*n - 1.0).abs() < 1.0e-12)));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if (*n - 0.0625).abs() < 1.0e-12)));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if (*n - 0.5).abs() < 1.0e-12)));
}
