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
fn step_returns_siso_response_through_vm_dispatch() {
    let program = r#"
        H = tf(1, [1 1]);
        [y, t] = step(H, [0 1 2]);
    "#;
    let hir = lower(&parse(program).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();

    assert!(vars.iter().any(|value| {
        matches!(value, Value::Tensor(tensor)
            if tensor.shape == vec![3, 1]
                && (tensor.data[0] - 0.0).abs() < 1.0e-8
                && (tensor.data[1] - (1.0 - (-1.0_f64).exp())).abs() < 1.0e-5
                && (tensor.data[2] - (1.0 - (-2.0_f64).exp())).abs() < 1.0e-5)
    }));
    assert!(vars.iter().any(|value| {
        matches!(value, Value::Tensor(tensor)
            if tensor.shape == vec![3, 1] && tensor.data == vec![0.0, 1.0, 2.0])
    }));
}

#[test]
fn step_single_output_assignment_returns_response() {
    let program = r#"
        H = tf(1, [1 1]);
        y = step(H, 0:0.5:1);
    "#;
    let hir = lower(&parse(program).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();

    assert!(vars.iter().any(|value| {
        matches!(value, Value::Tensor(tensor)
            if tensor.shape == vec![3, 1]
                && (tensor.data[0] - 0.0).abs() < 1.0e-8
                && (tensor.data[2] - (1.0 - (-1.0_f64).exp())).abs() < 1.0e-5)
    }));
}

#[test]
fn step_statement_form_plots_without_error() {
    let program = r#"
        H = tf(1, [1 1]);
        step(H);
    "#;
    let hir = lower(&parse(program).unwrap()).unwrap();
    execute(&hir).unwrap();
}
