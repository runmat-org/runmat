mod test_helpers;

use runmat_builtins::Value;
use runmat_ignition::{compile, Instr};
use runmat_parser::parse;
use std::collections::HashMap;
use test_helpers::{execute, lower};

fn tensor_values(value: &Value) -> (&[f64], &[usize]) {
    match value {
        Value::Tensor(tensor) => (&tensor.data, &tensor.shape),
        other => panic!("expected real tensor, got {other:?}"),
    }
}

fn assert_tensor_close(actual: &Value, expected: &Value) {
    let (actual_data, actual_shape) = tensor_values(actual);
    let (expected_data, expected_shape) = tensor_values(expected);
    assert_eq!(actual_shape, expected_shape, "shape mismatch");
    assert_eq!(actual_data.len(), expected_data.len(), "length mismatch");
    for (idx, (a, b)) in actual_data.iter().zip(expected_data.iter()).enumerate() {
        assert!((a - b).abs() < 1e-9, "tensor mismatch at {idx}: {a} vs {b}");
    }
}

#[test]
fn compiler_emits_dedicated_matrix_and_elementwise_division_instructions() {
    let input = "A = [1 2; 3 4]; B = [5 6; 7 8]; a = A / B; b = A \\ B; c = A ./ B; d = A .\\ B;";
    let ast = parse(input).expect("parse division script");
    let hir = lower(&ast).expect("lower division script");
    let bytecode = compile(&hir, &HashMap::new()).expect("compile division script");

    assert!(
        bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::RightDiv)),
        "expected matrix right division to compile to RightDiv"
    );
    assert!(
        bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::LeftDiv)),
        "expected matrix left division to compile to LeftDiv"
    );
    assert!(
        bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::ElemDiv)),
        "expected elementwise right division to compile to ElemDiv"
    );
    assert!(
        bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::ElemLeftDiv)),
        "expected elementwise left division to compile to ElemLeftDiv"
    );
}

#[test]
fn matrix_left_division_operator_matches_mldivide_builtin() {
    let input = "A = [1 2; 3 4]; b = [5; 6]; x = A \\ b; y = mldivide(A, b);";
    let ast = parse(input).expect("parse left division script");
    let hir = lower(&ast).expect("lower left division script");
    let vars = execute(&hir).expect("execute left division script");

    assert_tensor_close(&vars[2], &vars[3]);
}

#[test]
fn matrix_right_division_operator_matches_mrdivide_builtin() {
    let input = "A = [5 6]; B = [1 2; 3 4]; x = A / B; y = mrdivide(A, B);";
    let ast = parse(input).expect("parse right division script");
    let hir = lower(&ast).expect("lower right division script");
    let vars = execute(&hir).expect("execute right division script");

    assert_tensor_close(&vars[2], &vars[3]);
}

#[test]
fn matrix_left_division_solves_system_instead_of_elementwise_inverse() {
    let input = "A = [1 2; 3 4]; b = [5; 6]; x = A \\ b;";
    let ast = parse(input).expect("parse left division solve script");
    let hir = lower(&ast).expect("lower left division solve script");
    let vars = execute(&hir).expect("execute left division solve script");

    let (data, shape) = tensor_values(&vars[2]);
    assert_eq!(shape, &[2, 1]);
    assert!(
        (data[0] + 4.0).abs() < 1e-9,
        "expected x(1) = -4, got {}",
        data[0]
    );
    assert!(
        (data[1] - 4.5).abs() < 1e-9,
        "expected x(2) = 4.5, got {}",
        data[1]
    );
}
