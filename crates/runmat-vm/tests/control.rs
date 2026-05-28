#[path = "support/mod.rs"]
mod test_helpers;

use runmat_builtins::Value;
use test_helpers::execute_source;

#[test]
fn tf_constructs_object_through_vm_dispatch() {
    let program = r#"
        H = tf(20, [1 5]);
        c = class(H);
        n = H.Numerator;
        d = H.Denominator;
    "#;
    let vars = execute_source(program).unwrap();

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
    let vars = execute_source(program).unwrap();

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
    let vars = execute_source(program).unwrap();

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
    execute_source(program).unwrap();
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
    let vars = execute_source(program).unwrap();

    assert!(vars
        .iter()
        .any(|value| { matches!(value, Value::Tensor(tensor) if tensor.shape == vec![3, 1]) }));
    assert!(vars.iter().any(|value| match value {
        Value::Num(n) => (*n - 20.0).abs() < 1.0e-8,
        _ => false,
    }));
    assert!(vars.iter().any(|value| match value {
        Value::Num(n) => (*n - (20.0 * (-0.5f64).exp())).abs() < 1.0e-8,
        _ => false,
    }));
    assert!(vars.iter().any(|value| match value {
        Value::Num(n) => (*n - 0.2).abs() < 1.0e-12,
        _ => false,
    }));
}

#[test]
fn impulse_discrete_response_through_vm_dispatch() {
    let discrete = r#"
        H = tf(1, [1 -0.5], 0.1);
        [y, t] = impulse(H, 0:0.1:0.5);
        y1 = y(1);
        y2 = y(2);
        y6 = y(6);
        t6 = t(6);
    "#;
    let vars = execute_source(discrete).unwrap();
    assert!(vars
        .iter()
        .any(|value| { matches!(value, Value::Tensor(tensor) if tensor.shape == vec![6, 1]) }));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if n.abs() < 1.0e-12)));
    assert!(vars.iter().any(|value| match value {
        Value::Num(n) => (*n - 10.0).abs() < 1.0e-12,
        _ => false,
    }));
    assert!(vars.iter().any(|value| match value {
        Value::Num(n) => (*n - 0.625).abs() < 1.0e-12,
        _ => false,
    }));
    assert!(vars.iter().any(|value| match value {
        Value::Num(n) => (*n - 0.5).abs() < 1.0e-12,
        _ => false,
    }));
}

#[test]
fn impulse_statement_form_plots_without_error() {
    let program = r#"
        H = tf(20, [1 5]);
        impulse(H);
    "#;
    execute_source(program).unwrap();
}

#[test]
fn nyquist_returns_frequency_response_through_vm_dispatch() {
    let program = r#"
        H = tf(1, [1 1]);
        [re, im, w] = nyquist(H, [0 1 2]);
        re1 = re(1);
        re2 = re(2);
        im2 = im(2);
        w3 = w(3);
    "#;
    let vars = execute_source(program).unwrap();

    assert!(vars
        .iter()
        .any(|value| { matches!(value, Value::Tensor(tensor) if tensor.shape == vec![3, 1]) }));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if (n - 1.0).abs() < 1.0e-12)));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if (n - 0.5).abs() < 1.0e-12)));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if (n + 0.5).abs() < 1.0e-12)));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if (n - 2.0).abs() < 1.0e-12)));
}

#[test]
fn nyquist_statement_form_plots_without_error() {
    let program = r#"
        H = tf(1, [1 2 1]);
        nyquist(H);
    "#;
    execute_source(program).unwrap();
}
