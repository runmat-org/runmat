#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_semantic_source;

#[test]
fn closure_simple_no_capture() {
    let vars = execute_semantic_source("f = @(x) x + 1; y = feval(f, 2);").unwrap();
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 3.0).abs() < 1e-9)));
}

#[test]
fn closure_captures_free_variables() {
    let vars = execute_semantic_source("a=1; b=2; f=@(x) x + a + b; y = feval(f, 3);").unwrap();
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 6.0).abs() < 1e-9)));
}

#[test]
fn nested_closures_capture_outer() {
    let vars =
        execute_semantic_source("a=10; f=@(x) @(y) (x + y + a); g = feval(f, 2); r = feval(g, 3);")
            .unwrap();
    // Expect 2 + 3 + 10 = 15
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 15.0).abs() < 1e-9)));
}

#[test]
fn feval_with_string_handle() {
    let vars = execute_semantic_source("r = feval('@max', 2, 5);").unwrap();
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 5.0).abs() < 1e-9)));
}

#[test]
fn fzero_accepts_anonymous_function() {
    let vars = execute_semantic_source("f = @(x) cos(x) - x; r = fzero(f, 0.5);").unwrap();
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 0.7390851332).abs() < 1e-6)));
}

#[test]
fn fzero_accepts_optimset_options() {
    let vars = execute_semantic_source(
        "opts = optimset('TolX', 1e-10, 'Display', 'off'); r = fzero(@sin, [3 4], opts);",
    )
    .unwrap();
    assert!(vars.iter().any(
        |v| matches!(v, runmat_builtins::Value::Num(n) if (*n - std::f64::consts::PI).abs() < 1e-8)
    ));
}

#[test]
fn fsolve_accepts_anonymous_vector_function() {
    let vars = execute_semantic_source(
        "F = @(x) [x(1)^2 + x(2)^2 - 4; x(1)*x(2) - 1]; x = fsolve(F, [1; 1]);",
    )
    .unwrap();
    assert!(vars.iter().any(|v| {
        if let runmat_builtins::Value::Tensor(t) = v {
            t.data.len() == 2
                && (t.data[0] * t.data[0] + t.data[1] * t.data[1] - 4.0).abs() < 1e-5
                && (t.data[0] * t.data[1] - 1.0).abs() < 1e-5
        } else {
            false
        }
    }));
}

#[test]
fn ode45_accepts_anonymous_rhs_function() {
    let vars = execute_semantic_source("y = ode45(@(t, y) -y, [0 1], 1);").unwrap();
    assert!(vars.iter().any(|v| {
        if let runmat_builtins::Value::Tensor(t) = v {
            t.cols() == 1 && (t.data[t.rows() - 1] - (-1.0_f64).exp()).abs() < 1e-2
        } else {
            false
        }
    }));
}

#[test]
fn ode23_accepts_two_output_assignment() {
    let vars =
        execute_semantic_source("[t, y] = ode23(@(t, y) -2*y, [0 0.25 0.5 1.0], 1);").unwrap();
    assert!(vars.iter().any(|v| {
        if let runmat_builtins::Value::Tensor(tensor) = v {
            tensor.cols() == 1
                && tensor.rows() == 4
                && (tensor.data[tensor.rows() - 1] - (-2.0_f64).exp()).abs() < 2e-2
        } else {
            false
        }
    }));
}
