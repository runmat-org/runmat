#[path = "support/mod.rs"]
mod test_helpers;

use runmat_builtins::Value;
use test_helpers::execute_source;

#[test]
fn symbolic_vpa_source_workflow() {
    let vars = execute_source(
        "old = digits(20); r = vpa(sym('1/3')); p = vpa(pi, 50); digits('default');",
    )
    .unwrap();

    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Num(n) if (*n - 32.0).abs() < 1.0e-12)));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Symbolic(expr) if expr.to_string() == "0.33333333333333333333")));
    assert!(vars.iter().any(|value| {
        matches!(value, Value::Symbolic(expr) if {
            let text = expr.to_string();
            text.starts_with("3.141592653589793")
                && text.chars().filter(|ch| ch.is_ascii_digit()).count() == 50
        })
    }));
}

#[test]
fn symbolic_int_source_workflow() {
    let vars = execute_source("syms x; F = int(x^2); A = int(sin(x), 0, pi); E = int(exp(x), x);")
        .unwrap();

    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Symbolic(expr) if expr.to_string() == "x^3/3")));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Symbolic(expr) if expr.to_string() == "2")));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Symbolic(expr) if expr.to_string() == "exp(x)")));
}
