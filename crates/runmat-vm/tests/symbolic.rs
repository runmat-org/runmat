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

#[test]
fn symbolic_mixed_numeric_horizontal_concatenation_promotes_to_symbolic_array() {
    let vars = execute_source("syms dA; heightAB = 95; A_pt = [dA, heightAB, 0]; first = A_pt(1);")
        .unwrap();

    assert!(vars.iter().any(|value| {
        matches!(value, Value::SymbolicArray(array) if {
            array.shape == vec![1, 3]
                && array
                    .data
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
            == vec!["dA", "95", "0"]
        })
    }));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Symbolic(expr) if expr.to_string() == "dA")));
}

#[test]
fn symbolic_mixed_numeric_vertical_concatenation_promotes_to_symbolic_array() {
    let vars = execute_source("syms dA; A_col = [dA; 95; 0];").unwrap();

    assert!(vars.iter().any(|value| {
        matches!(value, Value::SymbolicArray(array) if {
            array.shape == vec![3, 1]
                && array
                    .data
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    == vec!["dA", "95", "0"]
        })
    }));
}

#[test]
fn symbolic_array_reports_matlab_compatible_shape_metadata() {
    let vars = execute_source(
        "syms dA; A_pt = [dA, 95, 0]; sz = size(A_pt); n = numel(A_pt); L = length(A_pt);",
    )
    .unwrap();

    assert!(
        vars.iter()
            .any(|value| { matches!(value, Value::Tensor(t) if t.data == vec![1.0, 3.0]) }),
        "size(A_pt) should be the row vector [1 3]"
    );
    assert!(
        vars.iter()
            .any(|value| matches!(value, Value::Num(n) if (*n - 3.0).abs() < 1.0e-12)),
        "numel(A_pt) and length(A_pt) should both be 3"
    );
}
