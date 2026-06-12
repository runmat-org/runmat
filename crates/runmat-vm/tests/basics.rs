#[path = "support/mod.rs"]
mod test_helpers;

use runmat_accelerate::ShapeInfo;
use runmat_builtins::Value;
use runmat_vm::{EndExpr, Instr};
use std::convert::TryInto;
use test_helpers::compile_source;
use test_helpers::interpret;

fn execute_source(source: &str) -> Vec<Value> {
    let bytecode = compile_source(source).expect("compile source");
    interpret(&bytecode).expect("execute bytecode")
}

#[test]
fn arithmetic_and_assignment() {
    let input = "x = 1 + 2; y = x * x";
    let vars = execute_source(input);
    let x: f64 = (&vars[0]).try_into().unwrap();
    let y: f64 = (&vars[1]).try_into().unwrap();
    assert_eq!(x, 3.0);
    assert_eq!(y, 9.0);
}

#[test]
fn bare_random_builtin_identifiers_execute_as_zero_arg_calls() {
    let input = "\
        rng(123);
        a = rand;
        b = rand(2, 3);
        c = randn;
        rand = 7;
        d = rand;
        out = [a, numel(b), numel(c), d];
    ";
    let vars = execute_source(input);
    let out = vars
        .iter()
        .find_map(|value| match value {
            Value::Tensor(tensor) if tensor.shape == vec![1, 4] => Some(tensor),
            _ => None,
        })
        .expect("expected output tensor");
    assert_eq!(out.shape, vec![1, 4]);
    assert!(out.data[0] > 0.0 && out.data[0] < 1.0);
    assert_eq!(out.data[1], 6.0);
    assert_eq!(out.data[2], 1.0);
    assert_eq!(out.data[3], 7.0);
}

#[test]
fn hilbert_builtin_executes_for_fm_demod_shape() {
    let input = r#"
        t = 0:0.001:0.01;
        signal = cos(2*pi*100*t);
        analytic = hilbert(signal);
        phase = unwrap(angle(analytic));
        demod = [diff(phase) 0];
        [b, a] = butter(4, 0.05);
        filtered = filter(b, a, demod);
        out = [numel(analytic), numel(phase), numel(demod), numel(filtered)];
    "#;
    let vars = execute_source(input);
    let out = vars
        .iter()
        .find_map(|value| match value {
            Value::Tensor(tensor) if tensor.shape == vec![1, 4] => Some(tensor),
            _ => None,
        })
        .expect("expected output tensor");
    assert_eq!(out.data, vec![11.0, 11.0, 11.0, 11.0]);
}

#[test]
fn pulstran_rectpuls_builtin_executes_for_impulse_train_shape() {
    let input = r#"
        t = -1:0.5:1;
        d = [-0.5 0.5];
        x = pulstran(t, d, 'rectpuls', 0.25);
        out = [numel(x), x(1), x(2), x(3), x(4), x(5)];
    "#;
    let vars = execute_source(input);
    let out = vars
        .iter()
        .find_map(|value| match value {
            Value::Tensor(tensor) if tensor.shape == vec![1, 6] => Some(tensor),
            _ => None,
        })
        .expect("expected pulse train summary tensor");
    assert_eq!(out.data, vec![5.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
}

#[test]
fn struct_aggregate_literal_uses_typed_instruction_and_overwrites_duplicates() {
    let bytecode = compile_source("s = struct{version = 1, version = 2};").expect("compile source");
    assert!(bytecode.instructions.iter().any(|instr| matches!(
        instr,
        Instr::CreateStructLiteral(fields)
            if fields == &vec!["version".to_string(), "version".to_string()]
    )));

    let vars = interpret(&bytecode).expect("execute bytecode");
    let Value::Struct(st) = &vars[0] else {
        panic!("expected struct value");
    };
    assert_eq!(st.fields.len(), 1);
    assert!(matches!(st.fields.get("version"), Some(Value::Num(v)) if *v == 2.0));
}

#[test]
fn object_aggregate_literal_uses_typed_instruction_and_sets_properties() {
    let bytecode = compile_source("p = ?Point{x = 1, y = 2};").expect("compile source");
    assert!(bytecode.instructions.iter().any(|instr| matches!(
        instr,
        Instr::CreateObjectLiteral { class_name, fields }
            if class_name == "Point"
                && fields == &vec!["x".to_string(), "y".to_string()]
    )));

    let vars = interpret(&bytecode).expect("execute bytecode");
    let Value::Object(obj) = &vars[0] else {
        panic!("expected object value");
    };
    assert_eq!(obj.class_name, "Point");
    assert!(matches!(obj.properties.get("x"), Some(Value::Num(v)) if *v == 1.0));
    assert!(matches!(obj.properties.get("y"), Some(Value::Num(v)) if *v == 2.0));
}

#[test]
fn logical_ops_use_typed_bytecode() {
    let bytecode = compile_source("a = ~0; b = 1 & 0; c = 1 | 0;").unwrap();

    assert!(bytecode
        .instructions
        .iter()
        .any(|instr| matches!(instr, Instr::LogicalNot)));
    assert!(bytecode
        .instructions
        .iter()
        .any(|instr| matches!(instr, Instr::LogicalAnd)));
    assert!(bytecode
        .instructions
        .iter()
        .any(|instr| matches!(instr, Instr::LogicalOr)));
    assert!(!bytecode.instructions.iter().any(|instr| matches!(
        instr,
        Instr::CallBuiltinMulti(name, _, _) if matches!(name.as_str(), "not" | "ne")
    )));
}

#[test]
fn nextpow2_supports_common_fft_zero_padding_pattern() {
    let input = "x = [1 2 3 4 5 6 7 8 9]; N = 2^nextpow2(length(x));";
    let vars = execute_source(input);
    let n: f64 = (&vars[1]).try_into().unwrap();
    assert_eq!(n, 16.0);
}

#[test]
fn call_builtin_multi_output_advances_pc_for_zero_outputs() {
    let input = "disp('hi'); x = 42;";
    let vars = execute_source(input);
    let x: f64 = (&vars[0]).try_into().expect("convert x to f64");
    assert_eq!(x, 42.0);
}

#[test]
fn array_construct_like_and_size_vector_inference() {
    // zeros('like', A)
    let src_like = "A = rand(3,4); B = zeros('like', A);";
    let bytecode_like = compile_source(src_like).expect("compile semantic like");
    if let Some(graph_like) = bytecode_like.accel_graph.as_ref() {
        let last_like = graph_like.nodes.last().expect("node");
        let out_id = *last_like.outputs.first().unwrap();
        let out_info = graph_like.value(out_id).expect("out value");
        match &out_info.shape {
            ShapeInfo::Tensor(dims) => {
                assert_eq!(dims, &vec![Some(3), Some(4)]);
            }
            other => panic!("unexpected shape: {:?}", other),
        }
    } else {
        assert_eq!(
            bytecode_like.fusion_metadata.mir_fusion_signal_count, 0,
            "accel graph should only be omitted for non-fusion-signal programs"
        );
        let vars_like = execute_source(src_like);
        let Value::Tensor(tensor_like) = &vars_like[1] else {
            panic!("expected tensor result for zeros('like', A)");
        };
        assert_eq!(tensor_like.shape, vec![3, 4]);
    }

    // zeros([5,6]) via size vector
    let src_sz = "sz = [5,6]; B = zeros(sz);";
    let bytecode_sz = compile_source(src_sz).expect("compile semantic sz");
    if let Some(graph_sz) = bytecode_sz.accel_graph.as_ref() {
        let last_sz = graph_sz.nodes.last().expect("node");
        let out_id_sz = *last_sz.outputs.first().unwrap();
        let out_info_sz = graph_sz.value(out_id_sz).expect("out value");
        match &out_info_sz.shape {
            ShapeInfo::Tensor(dims) => {
                assert_eq!(dims, &vec![Some(5), Some(6)]);
            }
            other => panic!("unexpected shape: {:?}", other),
        }
    } else {
        assert_eq!(
            bytecode_sz.fusion_metadata.mir_fusion_signal_count, 0,
            "accel graph should only be omitted for non-fusion-signal programs"
        );
        let vars_sz = execute_source(src_sz);
        let Value::Tensor(tensor_sz) = &vars_sz[1] else {
            panic!("expected tensor result for zeros(sz)");
        };
        assert_eq!(tensor_sz.shape, vec![5, 6]);
    }
}

#[test]
fn complex_literal_matrix_uses_fixed_size_construction() {
    let input = "A = [1+2i 3-4j];";
    let bytecode = compile_source(input).unwrap();
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
            .any(|instr| matches!(instr, Instr::CreateMatrix(1, 2))),
        "expected complex literal matrix to use semantic fixed-size construction"
    );
}

#[test]
fn logical_slice_read_and_write_execute() {
    let bytecode =
        compile_source("A = [1 2 3 4]; mask = A > 2; B = A(mask); A(mask) = 9; C = A;").unwrap();
    let vars = test_helpers::interpret(&bytecode).unwrap();

    let Value::Tensor(selected) = &vars[2] else {
        panic!("expected selected tensor, got {:?}", vars[2]);
    };
    assert_eq!(selected.data, vec![3.0, 4.0]);

    let Value::Tensor(updated) = &vars[3] else {
        panic!("expected updated tensor, got {:?}", vars[3]);
    };
    assert_eq!(updated.data, vec![1.0, 2.0, 9.0, 9.0]);
}

#[test]
fn call_result_slice_index_executes() {
    let bytecode =
        compile_source("A = [1 2 3 4]; idx = find(A > 2); B = A(idx); A(idx) = 9; C = A;").unwrap();
    let vars = test_helpers::interpret(&bytecode).unwrap();

    let Value::Tensor(selected) = &vars[2] else {
        panic!("expected selected tensor, got {:?}", vars[2]);
    };
    assert_eq!(selected.data, vec![3.0, 4.0]);

    let Value::Tensor(updated) = &vars[3] else {
        panic!("expected updated tensor, got {:?}", vars[3]);
    };
    assert_eq!(updated.data, vec![1.0, 2.0, 9.0, 9.0]);
}

#[test]
fn scalar_call_result_index_assignment_executes() {
    let bytecode = compile_source("A = [1 2 3]; idx = length(A); A(idx) = 9; B = A;").unwrap();
    let vars = test_helpers::interpret(&bytecode).unwrap();

    let Value::Tensor(updated) = &vars[2] else {
        panic!("expected updated tensor, got {:?}", vars[2]);
    };
    assert_eq!(updated.data, vec![1.0, 2.0, 9.0]);
}

#[test]
fn scalar_value_index_assignment_executes() {
    let bytecode = compile_source("x = 1; x(1) = 2; y = x;").unwrap();
    let vars = test_helpers::interpret(&bytecode).unwrap();
    let updated = vars.last().expect("expected final variable");
    match updated {
        Value::Num(n) => assert!((*n - 2.0).abs() < 1e-9),
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![1, 1]);
            assert_eq!(t.data, vec![2.0]);
        }
        other => panic!("expected scalar numeric assignment result, got {other:?}"),
    }
}

#[test]
fn undefined_root_index_assignment_uses_index_assignment_load_semantics() {
    let bytecode = compile_source("x(1) = 7; y = x;").unwrap();
    assert!(
        bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, runmat_vm::Instr::LoadVarForIndexAssignment(_))),
        "indexed assignment to undefined root should lower through LoadVarForIndexAssignment"
    );
    let vars = test_helpers::interpret(&bytecode).unwrap();
    let updated = vars.last().expect("expected final variable");
    let Value::Tensor(tensor) = updated else {
        panic!("expected tensor assignment result, got {updated:?}");
    };
    assert_eq!(tensor.shape, vec![1, 1]);
    assert_eq!(tensor.data, vec![7.0]);
}

#[test]
fn string_array_scalar_index_assignment_executes() {
    let bytecode = compile_source(r#"S = ["a" "b"]; S(2) = "z"; T = S;"#).unwrap();
    let vars = test_helpers::interpret(&bytecode).unwrap();
    let updated = vars.last().expect("expected final variable");
    let Value::StringArray(sa) = updated else {
        panic!("expected string array assignment result, got {updated:?}");
    };
    assert_eq!(sa.shape, vec![1, 2]);
    assert_eq!(sa.data, vec!["a".to_string(), "z".to_string()]);
}

#[test]
fn complex_literal_matrix_executes() {
    let input = "A = [1+2i 3-4j];";
    let vars = execute_source(input);
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
    let vars = execute_source(input);
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
    let vars = execute_source(input);
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
    let vars = execute_source(input);
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
    let bytecode = compile_source(input).expect("compile chol multi-assign");
    assert!(
        bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::CallBuiltinMulti(name, 1, 2) if name == "chol")),
        "expected semantic multi-output chol call shape in bytecode"
    );
    let vars = execute_source(input);
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
    let vars = execute_source(input);
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
    let vars = execute_source(input);
    let delta: f64 = (&vars[3]).try_into().expect("convert delta_g0 to f64");
    assert!((delta - 1.2f64.atan2(2.7)).abs() < 1e-12);
}

#[test]
fn atan2_with_rhs_expression_lowers_to_add_then_builtin_call() {
    let input = "Vq_drop = 1; V_pcc = 2; Vd_drop = 3; delta_g0 = atan2(Vq_drop, V_pcc + Vd_drop);";
    let bytecode = compile_source(input).expect("compile atan2 lowering script");

    let add_index = bytecode
        .instructions
        .iter()
        .position(|instr| matches!(instr, Instr::Add));
    let atan2_index = bytecode
        .instructions
        .iter()
        .position(|instr| matches!(instr, Instr::CallBuiltinMulti(name, 2, 1) if name == "atan2"));

    assert!(
        matches!((add_index, atan2_index), (Some(add), Some(atan2)) if add < atan2),
        "expected Add before atan2 builtin call; got {:?}",
        bytecode.instructions
    );
}

#[test]
fn atan2_explicit_comma_list_argument_path_unpacks_before_call() {
    let input = "C = {1, 2}; x = atan2(C{:});";
    let vars = execute_source(input);
    let x: f64 = (&vars[1]).try_into().expect("convert x to f64");
    assert!((x - 1.0f64.atan2(2.0)).abs() < 1e-12);

    let bytecode = compile_source(input).expect("compile atan2 comma-list script");
    let has_output_list_expansion = bytecode.instructions.iter().any(|instr| {
        matches!(instr, Instr::CallBuiltinExpandMultiOutput(name, specs, out_count)
            if name == "atan2"
                && *out_count == 1
                && specs.len() == 1
                && specs[0].is_expand
                && specs[0].expand_all
                && specs[0].num_indices == 0)
    });
    assert!(
        has_output_list_expansion,
        "expected explicit CallBuiltinExpandMultiOutput(atan2) expansion in bytecode"
    );
    assert!(
        !bytecode.instructions.iter().any(
            |instr| matches!(instr, Instr::CallBuiltinMulti(name, 2, 1) if name == "atan2")
        ),
        "atan2(C{{:}}) should lower through expand-multi-output call shape, not fixed-arity builtin call"
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
    let vars = execute_source(input);
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
    let bytecode = compile_source(input).expect("compile semantic fft end range script");
    assert!(
        bytecode
            .instructions
            .iter()
            .any(|ins| matches!(ins, Instr::IndexSliceExpr { .. })),
        "expected IndexSliceExpr in semantic bytecode, got {:?}",
        bytecode.instructions
    );
    let vars = interpret(&bytecode).expect("fft end-range indexing should execute");
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
    let vars = execute_source(input);
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
    let vars = execute_source(input);
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
    let vars = execute_source(input);
    assert!(
        vars.iter().any(|v| {
            matches!(v, Value::Bool(true)) || matches!(v, Value::Num(n) if (*n - 1.0).abs() < 1e-12)
        }),
        "expected true/equivalent marker in vars, got {vars:?}"
    );
}

#[test]
fn scalar_end_div_indexing_rejects_fractional_result() {
    let input = r#"
        x = [10 20 30 40 50];
        y = x(end/2);
    "#;
    let bytecode = compile_source(input).expect("compile semantic end-div script");
    let err = interpret(&bytecode).expect_err("fractional end/2 scalar index must fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UnsupportedIndexType"),
        "unexpected identifier: {:?} ({err:?})",
        err.identifier()
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
    let vars = execute_source(input);
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
    let vars = execute_source(input);
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
    let vars = execute_source(input);
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
    let bytecode = compile_source(input).expect("compile semantic end arithmetic oob script");
    let err = interpret(&bytecode).expect_err("end+1 should be out-of-bounds");
    assert_eq!(
        err.identifier(),
        Some("RunMat:IndexOutOfBounds"),
        "unexpected identifier: {:?} ({err:?})",
        err.identifier()
    );
}

#[test]
fn scalar_slice_with_nonnumeric_selector_errors() {
    let input = r#"
        x = 42;
        idx = "a";
        y = x(idx);
    "#;
    let bytecode = compile_source(input).expect("compile semantic scalar slice script");
    let err = interpret(&bytecode).expect_err("scalar slice with nonnumeric selector must error");
    assert_eq!(
        err.identifier(),
        Some("RunMat:UnsupportedIndexType"),
        "unexpected identifier: {:?} ({err:?})",
        err.identifier()
    );
}

#[test]
fn string_slice_assignment_on_scalar_string_reports_slice_non_tensor() {
    let input = r#"
        x = "abc";
        x(1:1) = "z";
    "#;
    let bytecode = compile_source(input).expect("compile semantic string slice assign");
    let err = interpret(&bytecode).expect_err("string scalar slice assignment must error");
    assert_eq!(
        err.identifier(),
        Some("RunMat:SliceNonTensor"),
        "unexpected identifier: {:?} ({err:?})",
        err.identifier()
    );
}

#[test]
fn numeric_linear_slice_assignment_with_string_rhs_reports_invalid_rhs_identifier() {
    let input = r#"
        x = [1 2];
        x([1 2]) = "z";
    "#;
    let bytecode = compile_source(input).expect("compile semantic numeric linear slice assign");
    let err =
        interpret(&bytecode).expect_err("numeric linear slice assignment must reject string rhs");
    assert_eq!(
        err.identifier(),
        Some("RunMat:InvalidSliceAssignmentRhs"),
        "unexpected identifier: {:?} ({err:?})",
        err.identifier()
    );
}

#[test]
fn numeric_nd_slice_assignment_with_string_rhs_reports_invalid_rhs_identifier() {
    let input = r#"
        x = [1 2; 3 4];
        x(:, 1) = "z";
    "#;
    let bytecode = compile_source(input).expect("compile semantic numeric nd slice assign");
    let err = interpret(&bytecode).expect_err("numeric nd slice assignment must reject string rhs");
    assert_eq!(
        err.identifier(),
        Some("RunMat:InvalidSliceAssignmentRhs"),
        "unexpected identifier: {:?} ({err:?})",
        err.identifier()
    );
}

#[test]
fn complex_linear_slice_assignment_with_string_rhs_reports_invalid_rhs_identifier() {
    let input = r#"
        x = [1 + 2i, 3 + 4i];
        x([1 2]) = "z";
    "#;
    let bytecode = compile_source(input).expect("compile semantic complex linear slice assign");
    let err =
        interpret(&bytecode).expect_err("complex linear slice assignment must reject string rhs");
    assert_eq!(
        err.identifier(),
        Some("RunMat:InvalidSliceAssignmentRhs"),
        "unexpected identifier: {:?} ({err:?})",
        err.identifier()
    );
}

#[test]
fn complex_nd_slice_assignment_with_string_rhs_reports_invalid_rhs_identifier() {
    let input = r#"
        x = [1 + 2i, 3 + 4i; 5 + 6i, 7 + 8i];
        x(:, 1) = "z";
    "#;
    let bytecode = compile_source(input).expect("compile semantic complex nd slice assign");
    let err = interpret(&bytecode).expect_err("complex nd slice assignment must reject string rhs");
    assert_eq!(
        err.identifier(),
        Some("RunMat:InvalidSliceAssignmentRhs"),
        "unexpected identifier: {:?} ({err:?})",
        err.identifier()
    );
}

#[test]
fn string_linear_slice_assignment_with_numeric_rhs_reports_invalid_rhs_identifier() {
    let input = r#"
        x = ["a", "b"];
        x(1) = 1;
    "#;
    let bytecode = compile_source(input).expect("compile semantic string linear slice assign");
    let err =
        interpret(&bytecode).expect_err("string linear slice assignment must reject numeric rhs");
    assert_eq!(
        err.identifier(),
        Some("RunMat:InvalidSliceAssignmentRhs"),
        "unexpected identifier: {:?} ({err:?})",
        err.identifier()
    );
}

#[test]
fn string_nd_slice_assignment_with_numeric_rhs_reports_invalid_rhs_identifier() {
    let input = r#"
        x = ["a", "b"; "c", "d"];
        x(:, 1) = 1;
    "#;
    let bytecode = compile_source(input).expect("compile semantic string nd slice assign");
    let err = interpret(&bytecode).expect_err("string nd slice assignment must reject numeric rhs");
    assert_eq!(
        err.identifier(),
        Some("RunMat:InvalidSliceAssignmentRhs"),
        "unexpected identifier: {:?} ({err:?})",
        err.identifier()
    );
}

#[test]
fn logical_linear_slice_assignment_with_string_rhs_reports_invalid_rhs_identifier() {
    let input = r#"
        x = [1 0] > 0;
        x([1 2]) = "z";
    "#;
    let bytecode = compile_source(input).expect("compile semantic logical linear slice assign");
    let err =
        interpret(&bytecode).expect_err("logical linear slice assignment must reject string rhs");
    assert_eq!(
        err.identifier(),
        Some("RunMat:InvalidSliceAssignmentRhs"),
        "unexpected identifier: {:?} ({err:?})",
        err.identifier()
    );
}

#[test]
fn logical_nd_slice_assignment_with_string_rhs_reports_invalid_rhs_identifier() {
    let input = r#"
        x = [1 0; 0 1] > 0;
        x(:, 1) = "z";
    "#;
    let bytecode = compile_source(input).expect("compile semantic logical nd slice assign");
    let err = interpret(&bytecode).expect_err("logical nd slice assignment must reject string rhs");
    assert_eq!(
        err.identifier(),
        Some("RunMat:InvalidSliceAssignmentRhs"),
        "unexpected identifier: {:?} ({err:?})",
        err.identifier()
    );
}

#[test]
fn logical_slice_assignment_executes_and_coerces_numeric_rhs() {
    let input = r#"
        x = [1 0] > 0;
        x([2]) = 2;
        s = sum(x);
    "#;
    let vars = execute_source(input);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 2.0).abs() < 1e-9)));
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
    let vars = execute_source(input);
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
    let bytecode = compile_source(input).expect("compile object end-range assignment");
    assert!(
        bytecode.instructions.iter().any(|instr| {
            matches!(
                instr,
                Instr::StoreSliceExpr {
                    range_end_exprs,
                    ..
                } if matches!(
                    range_end_exprs.as_slice(),
                    [EndExpr::Sub(lhs, rhs)]
                        if matches!(&**lhs, EndExpr::Mul(mul_lhs, mul_rhs)
                            if matches!(&**mul_lhs, EndExpr::End)
                                && matches!(&**mul_rhs, EndExpr::Const(v) if (*v - 1.0).abs() < 1e-12))
                            && matches!(&**rhs, EndExpr::Div(div_lhs, div_rhs)
                                if matches!(&**div_lhs, EndExpr::Const(v) if (*v - 1.0).abs() < 1e-12)
                                    && matches!(&**div_rhs, EndExpr::Const(v) if (*v - 2.0).abs() < 1e-12))
                )
            )
        }),
        "expected StoreSliceExpr to preserve rich end arithmetic payload for object indexing"
    );
    let vars = execute_source(input);
    assert!(
        vars.iter().any(|v| {
            matches!(v, Value::Bool(true)) || matches!(v, Value::Num(n) if (*n - 1.0).abs() < 1e-12)
        }),
        "expected true/equivalent marker in vars, got {vars:?}"
    );
}

#[test]
fn object_range_end_indexing_accepts_mixed_string_selector_payload() {
    let input = r#"
        __register_test_classes();
        o = new_object('OverIdx');
        r = o(1:(end*1 - 1/2), "key");
        ok = (r == 99);
    "#;
    let vars = execute_source(input);
    assert!(
        vars.iter().any(|v| {
            matches!(v, Value::Bool(true)) || matches!(v, Value::Num(n) if (*n - 1.0).abs() < 1e-12)
        }),
        "expected true/equivalent marker in vars, got {vars:?}"
    );
}

#[test]
fn object_range_end_assignment_accepts_mixed_string_selector_payload() {
    let input = r#"
        __register_test_classes();
        o = new_object('OverIdx');
        o(1:(end*1 - 1/2), "key") = 7;
        r = o.last;
        ok = (r == 7);
    "#;
    let vars = execute_source(input);
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
    let vars = execute_source(input);
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
    let vars = execute_source(input);
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
    let vars = execute_source(input);
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
    let vars = execute_source(input);
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
    let vars = execute_source(input);
    assert!(
        vars.iter().any(|v| {
            matches!(v, Value::Bool(true)) || matches!(v, Value::Num(n) if (*n - 1.0).abs() < 1e-12)
        }),
        "expected true/equivalent marker in vars, got {vars:?}"
    );
}
