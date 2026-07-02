#[path = "support/mod.rs"]
mod test_helpers;

use runmat_builtins::Value;

#[test]
fn symbolic_vpa_source_workflow() {
    let vars = test_helpers::execute_source(
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
    let vars = test_helpers::execute_source("syms x; F = int(x^2); A = int(sin(x), 0, pi); E = int(exp(x), x);")
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
    let bytecode = test_helpers::compile_source(
        "syms dA; heightAB = 95; A_pt = [dA, heightAB, 0]; first = A_pt(1);"
    )
    .unwrap();
    let vars = test_helpers::interpret(&bytecode).unwrap();

    // Find the slot index for 'A_pt' and verify it's a SymbolicArray with the correct shape
    let a_pt_slot = bytecode.var_names.iter()
        .find(|(_, name)| name.as_str() == "A_pt")
        .map(|(idx, _)| *idx)
        .expect("Variable 'A_pt' should exist in bytecode");

    assert!(matches!(&vars[a_pt_slot], Value::SymbolicArray(array) if {
        array.shape == vec![1, 3]
            && array
                .data
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
        == vec!["dA", "95", "0"]
    }), "A_pt should be a 1x3 SymbolicArray with elements [dA, 95, 0]");

    // Verify that the indexed result 'first = A_pt(1)' is specifically Value::Symbolic("dA")
    let first_slot = bytecode.var_names.iter()
        .find(|(_, name)| name.as_str() == "first")
        .map(|(idx, _)| *idx)
        .expect("Variable 'first' should exist in bytecode");

    assert!(matches!(&vars[first_slot], Value::Symbolic(expr) if expr.to_string() == "dA"),
        "Expected 'first = A_pt(1)' to be Value::Symbolic(\"dA\")");
}

#[test]
fn symbolic_mixed_numeric_vertical_concatenation_promotes_to_symbolic_array() {
    let vars = test_helpers::execute_source("syms dA; A_col = [dA; 95; 0];").unwrap();

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
fn symbolic_syms_piecewise_repro_binds_workspace_variables() {
    let vars =
        test_helpers::execute_source("clear; clc; close all; syms t w; f = piecewise(abs(t)<2, 1, abs(t)>2, 0);")
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
fn symbolic_array_reports_matlab_compatible_shape_metadata() {
    let bytecode = test_helpers::compile_source(
        "syms dA; A_pt = [dA, 95, 0]; sz = size(A_pt); n = numel(A_pt); L = length(A_pt);",
    )
    .unwrap();
    let vars = test_helpers::interpret(&bytecode).unwrap();

    // Verify size(A_pt) returns [1, 3]
    let sz_slot = bytecode.var_names.iter()
        .find(|(_, name)| name.as_str() == "sz")
        .map(|(idx, _)| *idx)
        .expect("Variable 'sz' should exist in bytecode");

    assert!(
        matches!(&vars[sz_slot], Value::Tensor(t) if t.data == vec![1.0, 3.0]),
        "size(A_pt) should be the row vector [1 3]"
    );

    // Verify numel(A_pt) returns 3
    let n_slot = bytecode.var_names.iter()
        .find(|(_, name)| name.as_str() == "n")
        .map(|(idx, _)| *idx)
        .expect("Variable 'n' should exist in bytecode");

    assert!(
        matches!(&vars[n_slot], Value::Num(n) if (*n - 3.0).abs() < 1.0e-12),
        "numel(A_pt) should be 3"
    );

    // Verify length(A_pt) returns 3
    let l_slot = bytecode.var_names.iter()
        .find(|(_, name)| name.as_str() == "L")
        .map(|(idx, _)| *idx)
        .expect("Variable 'L' should exist in bytecode");

    assert!(
        matches!(&vars[l_slot], Value::Num(l) if (*l - 3.0).abs() < 1.0e-12),
        "length(A_pt) should be 3"
    );
}

#[test]
fn symbolic_syms_invalid_declaration_reports_syms_diagnostic() {
    let err = test_helpers::execute_source("syms 1; x = 2;").expect_err("invalid syms declaration should fail");

    assert_eq!(err.identifier.as_deref(), Some("RunMat:syms:InvalidName"));
    assert_eq!(err.context.builtin.as_deref(), Some("syms"));
    assert!(err.message().contains("invalid symbolic variable name"));
    assert!(err.span.is_some(), "expected syms call span on diagnostic");
}
