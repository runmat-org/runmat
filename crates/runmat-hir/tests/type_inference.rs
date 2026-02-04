use runmat_hir::*;

use runmat_runtime as _;

fn lower(code: &str) -> runmat_hir::HirProgram {
    let ast = runmat_parser::parse(code).unwrap();
    runmat_hir::lower(&ast, &LoweringContext::empty())
        .map(|result| result.hir)
        .unwrap()
}

fn lower_result(code: &str) -> runmat_hir::LoweringResult {
    let ast = runmat_parser::parse(code).unwrap();
    runmat_hir::lower(&ast, &LoweringContext::empty()).unwrap()
}

#[test]
fn infer_simple_function_return_types() {
    let prog = lower("function y = f(x); if x>0; y=1; else; y=2.0; end; end");
    let m = infer_function_output_types(&prog);
    let tys = m.get("f").unwrap();
    assert_eq!(tys.len(), 1);
    // 1 and 2.0 unify to Num
    assert_eq!(tys[0], runmat_builtins::Type::Num);
}

#[test]
fn infer_function_with_fallthrough() {
    let prog = lower("function y = g(x); y = x; if x<0; y = 3; end; end");
    let m = infer_function_output_types(&prog);
    let tys = m.get("g").unwrap();
    // y starts as Unknown (from x), then may be assigned Num in branch; join yields Num
    assert_eq!(tys[0], runmat_builtins::Type::Num);
}

#[test]
fn infer_loop_assignment() {
    let prog = lower("function y = h(n); y = 0; while n>0; y = y + 1; n = n - 1; end; end");
    let m = infer_function_output_types(&prog);
    let tys = m.get("h").unwrap();
    assert_eq!(tys[0], runmat_builtins::Type::Num);
}

#[test]
fn infer_variable_types_env() {
    let prog = lower("function y = q(x); y = x; if x>0; y = 1; else; y = 2.0; end; end");
    let envs = infer_function_variable_types(&prog);
    let env = envs.get("q").unwrap();
    // y should be Num after joins; x remains Unknown
    let y_id = if let runmat_hir::HirStmt::Function { outputs, .. } = &prog.body[0] {
        outputs[0]
    } else {
        panic!("expected function")
    };
    assert_eq!(
        env.get(&y_id)
            .cloned()
            .unwrap_or(runmat_builtins::Type::Unknown),
        runmat_builtins::Type::Num
    );
}

#[test]
fn interprocedural_recursive_summary_terminates() {
    // Simple recursion; bound iteration must terminate; outputs remain Unknown or Num conservatively
    let prog = lower("function y = f(n); if n<=0; y = 0; else; y = f(n-1); end; end");
    let m = infer_function_output_types(&prog);
    let tys = m.get("f").unwrap();
    // Conservatively Num due to base case assignment
    assert_eq!(tys[0], runmat_builtins::Type::Num);
}

#[test]
fn interprocedural_mutual_recursion_summary() {
    let prog = lower(
        "function y = a(n); if n<=0; y=0; else; y = b(n-1); end; end\nfunction y = b(n); if n<=0; y=1; else; y = a(n-1); end; end",
    );
    let m = infer_function_output_types(&prog);
    assert_eq!(m.get("a").unwrap()[0], runmat_builtins::Type::Num);
    assert_eq!(m.get("b").unwrap()[0], runmat_builtins::Type::Num);
}

#[test]
fn struct_flow_from_string_array_and_cell() {
    // s is refined when guarded by ismember/strcmpi with string-array/cell inputs
    let prog = lower(
        "function y = g(s)\n if ismember('x', fieldnames(s)) && any(strcmpi(fieldnames(s), 'y'))\n  s.x = 1; s.y = 2;\n end; y = 0; end",
    );
    let envs = infer_function_variable_types(&prog);
    let env = envs.get("g").unwrap();
    // We cannot get VarId of s directly; ensure types map includes at least one Struct
    let has_struct = env
        .values()
        .any(|t| matches!(t, runmat_builtins::Type::Struct { .. }));
    assert!(has_struct);
}

#[test]
fn multi_lhs_resolution_from_summary() {
    // f returns two nums; [u,v]=f(..) should map per position
    let prog = lower(
        "function [x,y] = f(a); x=a; y=a+1; end; function z = caller()\n [u,v] = f(3); z = u+v; end",
    );
    let envs = infer_function_variable_types(&prog);
    let env = envs.get("caller").unwrap();
    // Find VarIds for u and v by scanning Hir
    let mut u_id = None;
    let mut v_id = None;
    if let runmat_hir::HirStmt::Function { body, .. } = &prog.body[1] {
        for s in body {
            if let runmat_hir::HirStmt::MultiAssign(vars, _, _, _) = s {
                u_id = vars[0];
                v_id = vars[1];
            }
        }
    }
    let u_ty = env
        .get(&u_id.unwrap())
        .cloned()
        .unwrap_or(runmat_builtins::Type::Unknown);
    let v_ty = env
        .get(&v_id.unwrap())
        .cloned()
        .unwrap_or(runmat_builtins::Type::Unknown);
    assert_eq!(u_ty, runmat_builtins::Type::Num);
    assert_eq!(v_ty, runmat_builtins::Type::Num);
}

#[test]
fn infer_range_shape_in_globals() {
    let result = lower_result("x = 0:1:100; y = sin(x);");
    match &result.hir.body[0] {
        runmat_hir::HirStmt::Assign(_, expr, _, _) => match &expr.kind {
            runmat_hir::HirExprKind::Range(start, step, end) => {
                let start_text = match &start.kind {
                    runmat_hir::HirExprKind::Number(text) => text.as_str(),
                    other => panic!("unexpected range start: {other:?}"),
                };
                let step_text = match step {
                    Some(step) => match &step.kind {
                        runmat_hir::HirExprKind::Number(text) => Some(text.as_str()),
                        other => panic!("unexpected range step: {other:?}"),
                    },
                    None => None,
                };
                let end_text = match &end.kind {
                    runmat_hir::HirExprKind::Number(text) => text.as_str(),
                    other => panic!("unexpected range end: {other:?}"),
                };
                assert_eq!(start_text, "0");
                assert_eq!(step_text, Some("1"));
                assert_eq!(end_text, "100");
            }
            runmat_hir::HirExprKind::Binary(_, runmat_parser::BinOp::Colon, _) => {}
            other => panic!("unexpected range expression: {other:?}"),
        },
        other => panic!("unexpected statement {other:?}"),
    }
    let x_id = runmat_hir::VarId(*result.variables.get("x").unwrap());
    let y_id = runmat_hir::VarId(*result.variables.get("y").unwrap());
    if !result.inferred_globals.contains_key(&x_id) {
        panic!(
            "missing inferred global for x: {:?}",
            result.inferred_globals
        );
    }
    let recomputed =
        runmat_hir::infer_global_variable_types(&result.hir, &result.inferred_function_returns);
    if !recomputed.contains_key(&x_id) {
        panic!("missing recomputed global for x: {:?}", recomputed);
    }
    let x_ty = recomputed
        .get(&x_id)
        .cloned()
        .unwrap_or(runmat_builtins::Type::Unknown);
    if matches!(x_ty, runmat_builtins::Type::Unknown) {
        panic!("x inferred as Unknown: {:?}", recomputed);
    }
    let y_ty = recomputed
        .get(&y_id)
        .cloned()
        .unwrap_or(runmat_builtins::Type::Unknown);
    assert_eq!(
        x_ty,
        runmat_builtins::Type::Tensor {
            shape: Some(vec![Some(1), Some(101)])
        }
    );
    if runmat_builtins::builtin_functions().is_empty() {
        assert_eq!(y_ty, runmat_builtins::Type::Unknown);
    } else {
        assert_eq!(
            y_ty,
            runmat_builtins::Type::Tensor {
                shape: Some(vec![Some(1), Some(101)])
            }
        );
    }
}

#[test]
fn infer_range_shape_with_constants() {
    if !runmat_builtins::constants()
        .iter()
        .any(|c| c.name.eq_ignore_ascii_case("pi"))
    {
        return;
    }
    let result = lower_result("a = 0:pi/100:2*pi;");
    let a_id = runmat_hir::VarId(*result.variables.get("a").unwrap());
    let globals =
        runmat_hir::infer_global_variable_types(&result.hir, &result.inferred_function_returns);
    let a_ty = globals
        .get(&a_id)
        .cloned()
        .unwrap_or(runmat_builtins::Type::Unknown);
    assert_eq!(
        a_ty,
        runmat_builtins::Type::Tensor {
            shape: Some(vec![Some(1), Some(201)])
        }
    );
}

#[test]
fn infer_index_shapes_for_scalar_and_range() {
    if !runmat_builtins::constants()
        .iter()
        .any(|c| c.name.eq_ignore_ascii_case("pi"))
    {
        return;
    }
    let result = lower_result("a = 0:pi/100:2*pi; b = sin(a); c = a[5]; d = a[1:10];");
    let globals =
        runmat_hir::infer_global_variable_types(&result.hir, &result.inferred_function_returns);
    let c_id = runmat_hir::VarId(*result.variables.get("c").unwrap());
    let d_id = runmat_hir::VarId(*result.variables.get("d").unwrap());
    let c_ty = globals
        .get(&c_id)
        .cloned()
        .unwrap_or(runmat_builtins::Type::Unknown);
    let d_ty = globals
        .get(&d_id)
        .cloned()
        .unwrap_or(runmat_builtins::Type::Unknown);
    assert_eq!(c_ty, runmat_builtins::Type::Num);
    assert_eq!(
        d_ty,
        runmat_builtins::Type::Tensor {
            shape: Some(vec![Some(1), Some(10)])
        }
    );
}

#[test]
fn infer_matmul_shape_with_known_dims() {
    let result = lower_result("a = ones(2,3); b = ones(3,4); c = a * b;");
    let globals =
        runmat_hir::infer_global_variable_types(&result.hir, &result.inferred_function_returns);
    let c_id = runmat_hir::VarId(*result.variables.get("c").unwrap());
    let c_ty = globals
        .get(&c_id)
        .cloned()
        .unwrap_or(runmat_builtins::Type::Unknown);
    assert_eq!(
        c_ty,
        runmat_builtins::Type::Tensor {
            shape: Some(vec![Some(2), Some(4)])
        }
    );
}

#[test]
fn lint_shape_mismatches() {
    let text = "a = ones(2,3); b = ones(4,2); c = a * b; d = a + b;";
    let result = lower_result(text);
    let diags = runmat_hir::lint_shapes(&result);
    assert!(diags.iter().any(|d| d.code == "lint.shape.matmul"));
    assert!(diags.iter().any(|d| d.code == "lint.shape.broadcast"));
}

#[test]
fn lint_dot_and_reshape() {
    let text = "a = ones(1,3); b = ones(1,4); c = dot(a, b); d = reshape(a, 2, 2); e = reshape(a, -1, -1);";
    let result = lower_result(text);
    let diags = runmat_hir::lint_shapes(&result);
    assert!(diags.iter().any(|d| d.code == "lint.shape.dot"));
    assert!(diags.iter().any(|d| d.code == "lint.shape.reshape"));
}

#[test]
fn lint_logical_index_mismatch() {
    let text = "a = ones(2,2); m = ones(1,2) > 0; b = a[m];";
    let result = lower_result(text);
    let diags = runmat_hir::lint_shapes(&result);
    assert!(diags.iter().any(|d| d.code == "lint.shape.logical_index"));
}

#[test]
fn lint_repmat_and_permute() {
    let bad_text =
        "a = ones(2,2); b = repmat(a, 1.5, 2); c = permute(a, [1 2 3]); d = permute(a, [1 1]);";
    let bad_result = lower_result(bad_text);
    let bad_diags = runmat_hir::lint_shapes(&bad_result);
    assert!(bad_diags.iter().any(|d| d.code == "lint.shape.repmat"));
    assert!(bad_diags.iter().any(|d| d.code == "lint.shape.permute"));

    let good_text = "a = ones(2,2); b = repmat(a, 2, 3); c = permute(a, [2 1]);";
    let good_result = lower_result(good_text);
    let good_diags = runmat_hir::lint_shapes(&good_result);
    assert!(!good_diags.iter().any(|d| d.code == "lint.shape.repmat"));
    assert!(!good_diags.iter().any(|d| d.code == "lint.shape.permute"));
}

#[test]
fn lint_concat_mismatches() {
    let bad_text = "B = ones(2,3); C = ones(4,3); D = ones(2,4); A = [B, C]; E = [B; D];";
    let bad_result = lower_result(bad_text);
    let bad_diags = runmat_hir::lint_shapes(&bad_result);
    assert!(bad_diags.iter().any(|d| d.code == "lint.shape.horzcat"));
    assert!(bad_diags.iter().any(|d| d.code == "lint.shape.vertcat"));

    let good_text = "B = ones(2,3); C = ones(2,4); D = ones(4,3); A = [B, C]; E = [B; D];";
    let good_result = lower_result(good_text);
    let good_diags = runmat_hir::lint_shapes(&good_result);
    assert!(!good_diags.iter().any(|d| d.code == "lint.shape.horzcat"));
    assert!(!good_diags.iter().any(|d| d.code == "lint.shape.vertcat"));
}

#[test]
fn lint_reduction_dim_out_of_range() {
    let bad_text = "a = ones(2,2); b = sum(a, 3);";
    let bad_result = lower_result(bad_text);
    let bad_diags = runmat_hir::lint_shapes(&bad_result);
    assert!(bad_diags.iter().any(|d| d.code == "lint.shape.reduction"));

    let good_text = "a = ones(2,2); b = sum(a, 2);";
    let good_result = lower_result(good_text);
    let good_diags = runmat_hir::lint_shapes(&good_result);
    assert!(!good_diags.iter().any(|d| d.code == "lint.shape.reduction"));
}
