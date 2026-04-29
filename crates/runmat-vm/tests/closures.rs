#[path = "support/mod.rs"]
mod test_helpers;

use runmat_parser::parse;
use test_helpers::execute;
use test_helpers::lower;

#[test]
fn closure_simple_no_capture() {
    let ast = parse("f = @(x) x + 1; y = feval(f, 2);").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 3.0).abs() < 1e-9)));
}

#[test]
fn closure_captures_free_variables() {
    let ast = parse("a=1; b=2; f=@(x) x + a + b; y = feval(f, 3);").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 6.0).abs() < 1e-9)));
}

#[test]
fn nested_closures_capture_outer() {
    let ast = parse("a=10; f=@(x) @(y) (x + y + a); g = feval(f, 2); r = feval(g, 3);").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    // Expect 2 + 3 + 10 = 15
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 15.0).abs() < 1e-9)));
}

#[test]
fn feval_with_string_handle() {
    let ast = parse("r = feval('@max', 2, 5);").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 5.0).abs() < 1e-9)));
}

#[test]
fn fzero_accepts_anonymous_function() {
    let ast = parse("f = @(x) cos(x) - x; r = fzero(f, 0.5);").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 0.7390851332).abs() < 1e-6)));
}

#[test]
fn fzero_accepts_optimset_options() {
    let ast =
        parse("opts = optimset('TolX', 1e-10, 'Display', 'off'); r = fzero(@sin, [3 4], opts);")
            .unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars.iter().any(
        |v| matches!(v, runmat_builtins::Value::Num(n) if (*n - std::f64::consts::PI).abs() < 1e-8)
    ));
}

#[test]
fn fsolve_accepts_anonymous_vector_function() {
    let ast =
        parse("F = @(x) [x(1)^2 + x(2)^2 - 4; x(1)*x(2) - 1]; x = fsolve(F, [1; 1]);").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
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
