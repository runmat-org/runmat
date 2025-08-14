use runmat_hir::*;

fn lower(code: &str) -> runmat_hir::HirProgram {
    let ast = runmat_parser::parse(code).unwrap();
    runmat_hir::lower(&ast).unwrap()
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
    let y_id = if let runmat_hir::HirStmt::Function { outputs, .. } = &prog.body[0] { outputs[0] } else { panic!("expected function") };
    assert_eq!(env.get(&y_id).cloned().unwrap_or(runmat_builtins::Type::Unknown), runmat_builtins::Type::Num);
}


