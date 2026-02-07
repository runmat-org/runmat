mod test_helpers;

use runmat_accelerate::ShapeInfo;
use runmat_builtins::Value;
use runmat_ignition::{compile, Instr};
use runmat_parser::parse;
use std::collections::HashMap;
use std::convert::TryInto;
use test_helpers::execute;
use test_helpers::lower;

#[test]
fn arithmetic_and_assignment() {
    let input = "x = 1 + 2; y = x * x";
    let ast = parse(input).unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let x: f64 = (&vars[0]).try_into().unwrap();
    let y: f64 = (&vars[1]).try_into().unwrap();
    assert_eq!(x, 3.0);
    assert_eq!(y, 9.0);
}

#[test]
fn call_builtin_multi_output_advances_pc_for_zero_outputs() {
    let input = "disp('hi'); x = 42;";
    let ast = parse(input).expect("parse disp script");
    let hir = lower(&ast).expect("lower disp script");
    let vars = execute(&hir).expect("execute disp script");
    let x: f64 = (&vars[0]).try_into().expect("convert x to f64");
    assert_eq!(x, 42.0);
}

#[test]
fn array_construct_like_and_size_vector_inference() {
    // zeros('like', A)
    let src_like = "A = rand(3,4); B = zeros('like', A);";
    let ast_like = parse(src_like).expect("parse like");
    let hir_like = lower(&ast_like).expect("lower like");
    let bytecode_like =
        runmat_ignition::bytecode::compile(&hir_like, &HashMap::new()).expect("compile like");
    let graph_like = runmat_ignition::accel_graph::build_accel_graph(
        &bytecode_like.instructions,
        &hir_like.var_types,
    );
    let last_like = graph_like.nodes.last().expect("node");
    let out_id = *last_like.outputs.first().unwrap();
    let out_info = graph_like.value(out_id).expect("out value");
    match &out_info.shape {
        ShapeInfo::Tensor(dims) => {
            assert_eq!(dims, &vec![Some(3), Some(4)]);
        }
        other => panic!("unexpected shape: {:?}", other),
    }

    // zeros([5,6]) via size vector
    let src_sz = "sz = [5,6]; B = zeros(sz);";
    let ast_sz = parse(src_sz).expect("parse sz");
    let hir_sz = lower(&ast_sz).expect("lower sz");
    let bytecode_sz =
        runmat_ignition::bytecode::compile(&hir_sz, &HashMap::new()).expect("compile sz");
    let graph_sz = runmat_ignition::accel_graph::build_accel_graph(
        &bytecode_sz.instructions,
        &hir_sz.var_types,
    );
    let last_sz = graph_sz.nodes.last().expect("node");
    let out_id_sz = *last_sz.outputs.first().unwrap();
    let out_info_sz = graph_sz.value(out_id_sz).expect("out value");
    match &out_info_sz.shape {
        ShapeInfo::Tensor(dims) => {
            assert_eq!(dims, &vec![Some(5), Some(6)]);
        }
        other => panic!("unexpected shape: {:?}", other),
    }
}

#[test]
fn complex_literal_matrix_uses_dynamic_path() {
    let input = "A = [1+2i 3-4j];";
    let ast = parse(input).unwrap();
    let hir = lower(&ast).unwrap();
    let bytecode = compile(&hir, &HashMap::new()).unwrap();
    assert!(
        bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::LoadComplex(_, _))),
        "expected complex constant to compile into LoadComplex"
    );
    assert!(
        bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::CreateMatrixDynamic(_))),
        "expected complex literal matrix to use dynamic construction"
    );
}

#[test]
fn complex_literal_matrix_executes() {
    let input = "A = [1+2i 3-4j];";
    let ast = parse(input).unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    match &vars[0] {
        Value::ComplexTensor(tensor) => {
            assert_eq!(tensor.shape, vec![1, 2]);
            assert_eq!(tensor.data, vec![(1.0, 2.0), (3.0, -4.0)]);
        }
        other => panic!("expected complex tensor, got {other:?}"),
    }
}
