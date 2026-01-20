use runmat_hir::*;

fn lower(code: &str) -> runmat_hir::HirProgram {
    let ast = runmat_parser::parse(code).unwrap();
    runmat_hir::lower(&ast, &LoweringContext::empty())
        .map(|result| result.hir)
        .unwrap()
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
