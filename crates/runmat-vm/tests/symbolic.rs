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
fn symbolic_syms_piecewise_repro_binds_workspace_variables() {
    let vars =
        execute_source("clear; clc; close all; syms t w; f = piecewise(abs(t)<2, 1, abs(t)>2, 0);")
            .unwrap();

    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Symbolic(expr) if expr.to_string() == "t")));
    assert!(vars
        .iter()
        .any(|value| matches!(value, Value::Symbolic(expr) if expr.to_string() == "w")));
    assert!(vars.iter().any(|value| {
        matches!(value, Value::Symbolic(expr) if {
            let text = expr.to_string();
            text.contains("piecewise")
                && text.contains("lt(abs(t), 2)")
                && text.contains("gt(abs(t), 2)")
        })
    }));
}

#[test]
fn symbolic_syms_invalid_declaration_reports_syms_diagnostic() {
    let err = execute_source("syms 1; x = 2;").expect_err("invalid syms declaration should fail");

    assert_eq!(err.identifier.as_deref(), Some("RunMat:syms:InvalidName"));
    assert_eq!(err.context.builtin.as_deref(), Some("syms"));
    assert!(err.message().contains("invalid symbolic variable name"));
    assert!(err.span.is_some(), "expected syms call span on diagnostic");
}
