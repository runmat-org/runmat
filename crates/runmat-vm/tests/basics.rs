#[path = "support/mod.rs"]
mod test_helpers;

use runmat_accelerate::ShapeInfo;
use runmat_builtins::Value;
use runmat_parser::parse;
use runmat_vm::{compile, Instr};
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
fn nextpow2_supports_common_fft_zero_padding_pattern() {
    let input = "x = [1 2 3 4 5 6 7 8 9]; N = 2^nextpow2(length(x));";
    let ast = parse(input).unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let n: f64 = (&vars[1]).try_into().unwrap();
    assert_eq!(n, 16.0);
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
        runmat_vm::bytecode::compile(&hir_like, &HashMap::new()).expect("compile like");
    let graph_like = runmat_vm::accel::graph::build_accel_graph(
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
    let bytecode_sz = runmat_vm::bytecode::compile(&hir_sz, &HashMap::new()).expect("compile sz");
    let graph_sz =
        runmat_vm::accel::graph::build_accel_graph(&bytecode_sz.instructions, &hir_sz.var_types);
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
    let bytecode = compile(&hir, &HashMap::new()).expect("compile fft end range script");
    assert!(
        bytecode
            .instructions
            .iter()
            .any(|ins| matches!(ins, Instr::IndexSliceExpr { .. })),
        "expected IndexSliceExpr in lowered bytecode, got {:?}",
        bytecode.instructions
    );
    let vars = execute(&hir).expect("fft end-range indexing should execute");
    assert!(
        vars.iter().any(|v| {
            matches!(v, Value::Bool(true)) || matches!(v, Value::Num(n) if (*n - 1.0).abs() < 1e-12)
        }),
        "expected true/equivalent marker in vars, got {vars:?}"
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

#[test]
fn fft_output_accepts_gpu_backed_range_selector() {
    let input = r#"
        fs = 1000;
        t = (0:fs-1)'/fs;
        x = 0.8*sin(2*pi*120*t) + 0.35*sin(2*pi*260*t);
        N = 2^nextpow2(length(x));
        X = fft(x, N);
        k = floor(N/2) + 1;
        Y = X(1:k);
        ok = (numel(Y) == 513);
    "#;
    let ast = parse(input).expect("parse fft gpu-backed range script");
    let hir = lower(&ast).expect("lower fft gpu-backed range script");
    let vars = execute(&hir).expect("fft gpu-backed range indexing should execute");
    assert!(
        vars.iter().any(|v| {
            matches!(v, Value::Bool(true)) || matches!(v, Value::Num(n) if (*n - 1.0).abs() < 1e-12)
        }),
        "expected true/equivalent marker in vars, got {vars:?}"
    );
}

#[test]
fn fft_output_supports_scalar_end_div_indexing() {
    let input = r#"
        x = [1 2 3 4 5 6 7 8];
        Y = fft(x);
        a = Y(end/2);
        ok = (abs(real(a) + 4) < 1e-12) && (imag(a) > 1.6) && (imag(a) < 1.7);
    "#;
    let ast = parse(input).expect("parse fft scalar end-div indexing script");
    let hir = lower(&ast).expect("lower fft scalar end-div indexing script");
    let vars = execute(&hir).expect("fft scalar end-div indexing should execute");
    assert!(
        vars.iter().any(|v| {
            matches!(v, Value::Bool(true)) || matches!(v, Value::Num(n) if (*n - 1.0).abs() < 1e-12)
        }),
        "expected true/equivalent marker in vars, got {vars:?}"
    );
}

#[test]
fn fft_output_supports_complex_range_assignment_with_end_div() {
    let input = r#"
        x = [1 2 3 4 5 6 7 8];
        Y = fft(x);
        Y(1:end/2) = 1 + 2i;
        a = Y(1);
        b = Y(4);
        ok = (real(a) == 1) && (imag(a) == 2) && (real(b) == 1) && (imag(b) == 2);
    "#;
    let ast = parse(input).expect("parse fft complex range assign script");
    let hir = lower(&ast).expect("lower fft complex range assign script");
    let vars = execute(&hir).expect("fft complex range assign should execute");
    assert!(
        vars.iter().any(|v| {
            matches!(v, Value::Bool(true)) || matches!(v, Value::Num(n) if (*n - 1.0).abs() < 1e-12)
        }),
        "expected true/equivalent marker in vars, got {vars:?}"
    );
}

#[test]
fn fft2_output_supports_complex_multidim_end_ranges() {
    let input = r#"
        A = [1 2; 3 4];
        F = fft2(A);
        S = F(1:end/2, 1:end);
        ok = (numel(S) == 2);
    "#;
    let ast = parse(input).expect("parse fft2 complex multidim end range script");
    let hir = lower(&ast).expect("lower fft2 complex multidim end range script");
    let vars = execute(&hir).expect("fft2 complex multidim end range should execute");
    assert!(
        vars.iter().any(|v| {
            matches!(v, Value::Bool(true)) || matches!(v, Value::Num(n) if (*n - 1.0).abs() < 1e-12)
        }),
        "expected true/equivalent marker in vars, got {vars:?}"
    );
}

#[test]
fn fft_end_arithmetic_supports_general_scalar_and_range_forms() {
    let input = r#"
        x = [1 2 3 4 5 6 7 8];
        Y = fft(x);
        a = Y(end*1 - 3 + 2/2);
        h = Y(2:(end*1 - 1/2));
        ok = (abs(real(a) + 4) < 1e-12) && (numel(h) == 6);
    "#;
    let ast = parse(input).expect("parse general end arithmetic script");
    let hir = lower(&ast).expect("lower general end arithmetic script");
    let vars = execute(&hir).expect("general end arithmetic should execute");
    assert!(
        vars.iter().any(|v| {
            matches!(v, Value::Bool(true)) || matches!(v, Value::Num(n) if (*n - 1.0).abs() < 1e-12)
        }),
        "expected true/equivalent marker in vars, got {vars:?}"
    );
}

#[test]
fn fft_end_arithmetic_out_of_bounds_raises_error() {
    let input = r#"
        x = [1 2 3 4 5 6 7 8];
        Y = fft(x);
        z = Y(end + 1);
    "#;
    let ast = parse(input).expect("parse end arithmetic oob script");
    let hir = lower(&ast).expect("lower end arithmetic oob script");
    let err = execute(&hir).expect_err("end+1 should be out-of-bounds");
    assert!(
        err.to_string().contains("Index out of bounds")
            || err.to_string().contains("Subscript out of bounds"),
        "unexpected error: {err:?}"
    );
}

#[test]
fn fft_complex_assignment_covers_scalar_slice_and_multidim_broadcast() {
    let input = r#"
        x = [1 2 3 4 5 6 7 8];
        Y = fft(x);
        Y(1) = 3 + 4i;
        Y(1:end/2) = 1 + 2i;
        A = [1 2; 3 4];
        F = fft2(A);
        F(:, 1) = 9 + 10i;
        ok = (real(Y(1)) == 1) && (imag(Y(1)) == 2) && (real(F(2,1)) == 9) && (imag(F(2,1)) == 10);
    "#;
    let ast = parse(input).expect("parse complex assignment coverage script");
    let hir = lower(&ast).expect("lower complex assignment coverage script");
    let vars = execute(&hir).expect("complex assignment coverage should execute");
    assert!(
        vars.iter().any(|v| {
            matches!(v, Value::Bool(true)) || matches!(v, Value::Num(n) if (*n - 1.0).abs() < 1e-12)
        }),
        "expected true/equivalent marker in vars, got {vars:?}"
    );
}

#[test]
fn object_range_end_assignment_accepts_rich_end_expression_payload() {
    let input = r#"
        __register_test_classes();
        o = new_object('OverIdx');
        o(1:(end*1 - 1/2)) = 7;
        r = o(1);
        ok = (r == 99);
    "#;
    let ast = parse(input).expect("parse object range-end payload script");
    let hir = lower(&ast).expect("lower object range-end payload script");
    let vars = execute(&hir).expect("object range-end payload script should execute");
    assert!(
        vars.iter().any(|v| {
            matches!(v, Value::Bool(true)) || matches!(v, Value::Num(n) if (*n - 1.0).abs() < 1e-12)
        }),
        "expected true/equivalent marker in vars, got {vars:?}"
    );
}

#[test]
fn fft_end_arithmetic_supports_pow_round_floor_fix_and_leftdiv() {
    let input = r#"
        x = [1 2 3 4 5 6 7 8];
        Y = fft(x);
        a = Y(round(end^1 / 2));
        b = Y(floor(end ./ 2));
        c = Y(fix(2 \ end));
        ok = (abs(real(a) - real(b)) < 1e-12) && (abs(real(c) - real(Y(2))) < 1e-12);
    "#;
    let ast = parse(input).expect("parse advanced end arithmetic functions script");
    let hir = lower(&ast).expect("lower advanced end arithmetic functions script");
    let vars = execute(&hir).expect("advanced end arithmetic functions should execute");
    assert!(
        vars.iter().any(|v| {
            matches!(v, Value::Bool(true)) || matches!(v, Value::Num(n) if (*n - 1.0).abs() < 1e-12)
        }),
        "expected true/equivalent marker in vars, got {vars:?}"
    );
}

#[test]
fn fft_end_arithmetic_supports_variable_offsets() {
    let input = r#"
        x = [1 2 3 4 5 6 7 8];
        k = 2;
        Y = fft(x);
        a = Y(end - k);
        h = Y(1:(end - k));
        ok = (abs(real(a) + 4) < 1e-12) && (numel(h) == 6);
    "#;
    let ast = parse(input).expect("parse variable end arithmetic script");
    let hir = lower(&ast).expect("lower variable end arithmetic script");
    let vars = execute(&hir).expect("variable end arithmetic should execute");
    assert!(
        vars.iter().any(|v| {
            matches!(v, Value::Bool(true)) || matches!(v, Value::Num(n) if (*n - 1.0).abs() < 1e-12)
        }),
        "expected true/equivalent marker in vars, got {vars:?}"
    );
}

#[test]
fn end_expression_supports_builtin_calls_in_index_context() {
    let input = r#"
        x = [10 20 30 40 50 60 70 80];
        a = x(abs(end-3));
        b = x(max(end-6, 2));
        ok = (a == 50) && (b == 20);
    "#;
    let ast = parse(input).expect("parse builtin-in-end-expression script");
    let hir = lower(&ast).expect("lower builtin-in-end-expression script");
    let vars = execute(&hir).expect("builtin-in-end-expression should execute");
    assert!(
        vars.iter().any(|v| {
            matches!(v, Value::Bool(true)) || matches!(v, Value::Num(n) if (*n - 1.0).abs() < 1e-12)
        }),
        "expected true/equivalent marker in vars, got {vars:?}"
    );
}

#[test]
fn end_expression_supports_user_function_calls_in_index_context() {
    let input = r#"
        function y = pick(n)
            y = n;
        end
        x = [10 20 30 40 50 60 70 80];
        a = x(pick(end-3));
        ok = (a == 50);
    "#;
    let ast = parse(input).expect("parse userfunc-in-end-expression script");
    let hir = lower(&ast).expect("lower userfunc-in-end-expression script");
    let vars = execute(&hir).expect("userfunc-in-end-expression should execute");
    assert!(
        vars.iter().any(|v| {
            matches!(v, Value::Bool(true)) || matches!(v, Value::Num(n) if (*n - 1.0).abs() < 1e-12)
        }),
        "expected true/equivalent marker in vars, got {vars:?}"
    );
}

#[test]
fn fftn_and_ifftn_execute_with_size_vector_and_indexing() {
    let input = r#"
        A = reshape(1:8, [2 2 2]);
        F = fftn(A, [2 2 2]);
        s = F(1:end/2);
        B = ifftn(F, [2 2 2]);
        ok = (numel(s) == 4) && (round(real(B(1))) == 1);
    "#;
    let ast = parse(input).expect("parse fftn/ifftn script");
    let hir = lower(&ast).expect("lower fftn/ifftn script");
    let vars = execute(&hir).expect("fftn/ifftn script should execute");
    assert!(
        vars.iter().any(|v| {
            matches!(v, Value::Bool(true)) || matches!(v, Value::Num(n) if (*n - 1.0).abs() < 1e-12)
        }),
        "expected true/equivalent marker in vars, got {vars:?}"
    );
}
