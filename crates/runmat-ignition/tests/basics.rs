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

#[test]
fn leading_dot_complex_literals_execute() {
    let input = "A = [.1i .5e-2j];";
    let ast = parse(input).unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    match &vars[0] {
        Value::ComplexTensor(tensor) => {
            assert_eq!(tensor.shape, vec![1, 2]);
            assert_eq!(tensor.data, vec![(0.0, 0.1), (0.0, 0.005)]);
        }
        other => panic!("expected complex tensor, got {other:?}"),
    }
}

#[test]
fn matrix_literal_with_leading_dot_entries_executes() {
    let input = "A = [1 .2 .3];";
    let ast = parse(input).unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    match &vars[0] {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape, vec![1, 3]);
            assert_eq!(tensor.data, vec![1.0, 0.2, 0.3]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn elementwise_division_accepts_leading_dot_rhs() {
    let input = "A = [1 2 3]; B = A./.5;";
    let ast = parse(input).unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    match &vars[1] {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape, vec![1, 3]);
            assert_eq!(tensor.data, vec![2.0, 4.0, 6.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn chol_multiassign_reports_failure() {
    let input = "A = [1 2; 2 1]; [R, p] = chol(A);";
    let ast = parse(input).unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let p: f64 = (&vars[2]).try_into().unwrap();
    assert_eq!(p, 2.0);
    match &vars[1] {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape, vec![2, 2]);
        }
        other => panic!("expected chol factor tensor, got {other:?}"),
    }
}

#[test]
fn uint16_cast_is_callable_in_vm() {
    let input = "A = uint16([3.49 -2 70000]);";
    let ast = parse(input).unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    match &vars[0] {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape, vec![1, 3]);
            assert_eq!(tensor.data, vec![3.0, 0.0, u16::MAX as f64]);
        }
        other => panic!("expected tensor output, got {other:?}"),
    }
}

#[test]
fn atan2_with_rhs_expression_executes_without_stack_underflow() {
    let input = r#"
        Vq_drop = 1.2;
        V_pcc = 2.4;
        Vd_drop = 0.3;
        delta_g0 = atan2(Vq_drop, V_pcc + Vd_drop);
    "#;
    let ast = parse(input).expect("parse atan2 rhs expression script");
    let hir = lower(&ast).expect("lower atan2 rhs expression script");
    let vars = execute(&hir).expect("atan2 rhs expression should execute");
    let delta: f64 = (&vars[3]).try_into().expect("convert delta_g0 to f64");
    assert!((delta - 1.2f64.atan2(2.7)).abs() < 1e-12);
}

#[test]
fn atan2_with_rhs_expression_lowers_to_add_then_builtin_call() {
    let input = "Vq_drop = 1; V_pcc = 2; Vd_drop = 3; delta_g0 = atan2(Vq_drop, V_pcc + Vd_drop);";
    let ast = parse(input).expect("parse atan2 lowering script");
    let hir = lower(&ast).expect("lower atan2 lowering script");
    let bytecode = compile(&hir, &HashMap::new()).expect("compile atan2 lowering script");

    let has_expected_shape = bytecode.instructions.windows(5).any(|window| {
        matches!(window[0], Instr::LoadVar(_))
            && matches!(window[1], Instr::LoadVar(_))
            && matches!(window[2], Instr::LoadVar(_))
            && matches!(window[3], Instr::Add)
            && matches!(window[4], Instr::CallBuiltin(ref name, 2) if name == "atan2")
    });

    assert!(
        has_expected_shape,
        "expected LoadVar,LoadVar,LoadVar,Add,CallBuiltin(atan2,2) sequence; got {:?}",
        bytecode.instructions
    );
}

#[test]
fn atan2_multi_output_argument_path_unpacks_before_call() {
    let input = r#"
        function [a,b] = g()
          a = 1;
          b = 2;
        end
        x = atan2(g());
    "#;
    let ast = parse(input).expect("parse atan2 multi-output script");
    let hir = lower(&ast).expect("lower atan2 multi-output script");
    let vars = execute(&hir).expect("atan2 multi-output script should execute");
    let x: f64 = (&vars[2]).try_into().expect("convert x to f64");
    assert!((x - 1.0f64.atan2(2.0)).abs() < 1e-12);

    let bytecode = compile(&hir, &HashMap::new()).expect("compile atan2 multi-output script");
    let has_unpack_barrier = bytecode.instructions.windows(3).any(|window| {
        matches!(window[0], Instr::CallFunctionMulti(ref name, 0, 2) if name == "g")
            && matches!(window[1], Instr::Unpack(2))
            && matches!(window[2], Instr::CallBuiltin(ref name, 2) if name == "atan2")
    });
    assert!(
        has_unpack_barrier,
        "expected CallFunctionMulti(g,0,2) -> Unpack(2) -> CallBuiltin(atan2,2) in bytecode"
    );
}

#[test]
fn fft_output_supports_scalar_and_range_indexing() {
    let input = r#"
        x = [1 2 3 4 5 6 7 8];
        Y = fft(x);
        a = Y(1);
        h = Y(1:4);
        n = numel(h);
        ra = real(a);
        ia = imag(a);
    "#;
    let ast = parse(input).expect("parse fft indexing script");
    let hir = lower(&ast).expect("lower fft indexing script");
    let vars = execute(&hir).expect("fft output indexing should execute");
    assert!(vars
        .iter()
        .any(|v| matches!(v, Value::Num(n) if (*n - 4.0).abs() < 1e-12)));
    assert!(vars
        .iter()
        .any(|v| matches!(v, Value::Num(n) if (*n - 36.0).abs() < 1e-12)));
    assert!(vars
        .iter()
        .any(|v| matches!(v, Value::Num(n) if n.abs() < 1e-12)));
}

#[test]
fn fft_output_supports_end_arithmetic_range_indexing() {
    let input = r#"
        x = [1 2 3 4 5 6 7 8];
        Y = fft(x);
        h = Y(1:end/2);
        ok = (numel(h) == 4);
    "#;
    let ast = parse(input).expect("parse fft end range script");
    let hir = lower(&ast).expect("lower fft end range script");
    let vars = execute(&hir).expect("fft end-range indexing should execute");
    assert!(
        vars.iter().any(|v| matches!(v, Value::Bool(true))),
        "expected boolean true marker in vars, got {vars:?}"
    );
}

#[test]
fn fft2_output_supports_two_dimensional_indexing() {
    let input = r#"
        A = [1 2; 3 4];
        F = fft2(A);
        c = F(1,2);
        col = F(:,1);
        n = numel(col);
    "#;
    let ast = parse(input).expect("parse fft2 indexing script");
    let hir = lower(&ast).expect("lower fft2 indexing script");
    let vars = execute(&hir).expect("fft2 output indexing should execute");
    assert!(vars
        .iter()
        .any(|v| matches!(v, Value::Num(n) if (*n - 2.0).abs() < 1e-12)));
}
