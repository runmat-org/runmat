mod test_helpers;

use runmat_accelerate::graph::{AccelNodeLabel, PrimitiveOp};
use runmat_builtins::Value;
use runmat_ignition::{compile, Instr};
use runmat_parser::parse;
use std::collections::HashMap;
use test_helpers::{execute, lower};

fn compile_bytecode(source: &str) -> runmat_ignition::Bytecode {
    let ast = parse(source).expect("parse");
    let hir = lower(&ast).expect("lower");
    compile(&hir, &HashMap::new()).expect("compile")
}

fn execute_program(source: &str) -> Vec<Value> {
    let ast = parse(source).expect("parse");
    let hir = lower(&ast).expect("lower");
    execute(&hir).expect("execute")
}

fn assert_same_real_tensor(lhs: &Value, rhs: &Value) {
    match (lhs, rhs) {
        (Value::Tensor(left), Value::Tensor(right)) => {
            assert_eq!(left.shape, right.shape);
            assert_eq!(left.data, right.data);
        }
        (Value::Num(left), Value::Num(right)) => {
            assert!((left - right).abs() < 1e-12, "left={left} right={right}");
        }
        other => panic!("expected matching real results, got {other:?}"),
    }
}

fn assert_same_complex_tensor(lhs: &Value, rhs: &Value) {
    match (lhs, rhs) {
        (Value::ComplexTensor(left), Value::ComplexTensor(right)) => {
            assert_eq!(left.shape, right.shape);
            assert_eq!(left.data, right.data);
        }
        (Value::Complex(lr, li), Value::Complex(rr, ri)) => {
            assert!((lr - rr).abs() < 1e-12, "re left={lr} right={rr}");
            assert!((li - ri).abs() < 1e-12, "im left={li} right={ri}");
        }
        other => panic!("expected matching complex results, got {other:?}"),
    }
}

fn has_builtin(bytecode: &runmat_ignition::Bytecode, name: &str) -> bool {
    let graph = bytecode.accel_graph.as_ref().expect("accel graph");
    graph.nodes.iter().any(|node| match &node.label {
        AccelNodeLabel::Builtin { name: node_name } => node_name.eq_ignore_ascii_case(name),
        _ => false,
    })
}

fn count_primitives(bytecode: &runmat_ignition::Bytecode, op: PrimitiveOp) -> usize {
    let graph = bytecode.accel_graph.as_ref().expect("accel graph");
    graph
        .nodes
        .iter()
        .filter(|node| matches!(node.label, AccelNodeLabel::Primitive(p) if p == op))
        .count()
}

#[test]
fn matrix_and_elementwise_division_lower_to_distinct_instructions() {
    let bytecode = compile_bytecode("a = 6 / 2; b = 6 \\ 2; c = 6 ./ 2; d = 6 .\\ 2;");
    assert!(
        bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::RightDiv)),
        "missing RightDiv in {:?}",
        bytecode.instructions
    );
    assert!(
        bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::LeftDiv)),
        "missing LeftDiv in {:?}",
        bytecode.instructions
    );
    assert!(
        bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::ElemDiv)),
        "missing ElemDiv in {:?}",
        bytecode.instructions
    );
    assert!(
        bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::ElemLeftDiv)),
        "missing ElemLeftDiv in {:?}",
        bytecode.instructions
    );
}

#[test]
fn left_division_operator_matches_mldivide_builtin_for_square_systems() {
    let vars = execute_program(
        "A = [1 2; 3 4]; b = [5; 6]; x = A \\ b; y = mldivide(A, b);",
    );
    assert_same_real_tensor(&vars[2], &vars[3]);
}

#[test]
fn left_division_operator_matches_mldivide_builtin_for_least_squares() {
    let vars = execute_program(
        "A = [1 2; 3 4; 5 6]; b = [7; 8; 9]; x = A \\ b; y = mldivide(A, b);",
    );
    assert_same_real_tensor(&vars[2], &vars[3]);
}

#[test]
fn right_division_operator_matches_mrdivide_builtin_for_square_systems() {
    let vars = execute_program(
        "A = [1 2; 3 4]; B = [2 1; 1 2]; x = A / B; y = mrdivide(A, B);",
    );
    assert_same_real_tensor(&vars[2], &vars[3]);
}

#[test]
fn division_operators_match_complex_builtins() {
    let vars = execute_program(
        "A = [2+1i 1; 0 3-1i]; b = [1-2i; 4]; B = [1+1i 2; 0 2-1i]; \
         x = A \\ b; y = mldivide(A, b); z = A / B; w = mrdivide(A, B);",
    );
    assert_same_complex_tensor(&vars[3], &vars[4]);
    assert_same_complex_tensor(&vars[5], &vars[6]);
}

#[test]
fn matrix_division_scalar_rhs_stays_fusible_in_accel_graph() {
    let bytecode = compile_bytecode("A = rand(4, 4); B = A / 2;");
    assert_eq!(count_primitives(&bytecode, PrimitiveOp::ElemDiv), 1);
    assert!(!has_builtin(&bytecode, "mrdivide"));
}

#[test]
fn true_matrix_division_uses_builtin_accel_nodes() {
    let bytecode = compile_bytecode("A = rand(4, 4); B = rand(4, 4); C = A / B; D = A \\ B;");
    assert!(has_builtin(&bytecode, "mrdivide"));
    assert!(has_builtin(&bytecode, "mldivide"));
    assert_eq!(count_primitives(&bytecode, PrimitiveOp::ElemDiv), 0);
}

#[test]
fn matrix_and_elementwise_object_overloads_dispatch_separately() {
    let vars = execute_program(
        "__register_test_classes(); \
         o = new_object('OverIdx'); \
         o = call_method(o, 'subsasgn', '.', 'k', 5); \
         a = o / 2; \
         b = o \\ 2; \
         c = o ./ 2; \
         d = o .\\ 2;",
    );
    match (&vars[1], &vars[2], &vars[3], &vars[4]) {
        (Value::Num(a), Value::Num(b), Value::Num(c), Value::Num(d)) => {
            assert!((*a - 2.5).abs() < 1e-12, "a={a}");
            assert!((*b - 0.4).abs() < 1e-12, "b={b}");
            assert!((*c - 2.5).abs() < 1e-12, "c={c}");
            assert!((*d - 0.4).abs() < 1e-12, "d={d}");
        }
        other => panic!("expected scalar overload results, got {other:?}"),
    }
}
