use runmat_hir::{
    lower, HirAssembly, HirCallableRef, HirExprKind, HirPlace, HirStmtKind, IndexKind,
    LoweringContext,
};
use runmat_parser::parse;

fn entry_body(src: &str) -> (HirAssembly, Vec<runmat_hir::HirStmt>) {
    let ast = parse(src).expect("parse");
    let assembly = lower(&ast, &LoweringContext::empty())
        .expect("lower")
        .assembly;
    let entry = assembly.entrypoints[0].target;
    let body = assembly.functions[entry.0].body.statements.clone();
    (assembly, body)
}

#[test]
fn ident_call_with_range_is_builtin_call_when_no_shadowing() {
    let (_, body) = entry_body("x = sin(0:5);");
    assert_eq!(body.len(), 1);
    match &body[0].kind {
        HirStmtKind::Assign(_, expr, _) => match &expr.kind {
            HirExprKind::Call(call) => {
                assert!(
                    matches!(
                        &call.callee,
                        HirCallableRef::Builtin(id) if id.0 == "sin"
                    ) || matches!(
                        &call.callee,
                        HirCallableRef::Unresolved(name) if name.0.len() == 1 && name.0[0].0 == "sin"
                    )
                );
                assert_eq!(call.args.len(), 1);
                assert!(matches!(call.args[0].kind, HirExprKind::Range(_, _, _)));
            }
            other => panic!("expected builtin call, got {other:?}"),
        },
        other => panic!("expected Assign, got {other:?}"),
    }
}

#[test]
fn variable_shadowing_turns_ident_call_into_index() {
    let (assembly, body) = entry_body("single = 3; y = single(1);");
    let single = assembly
        .bindings
        .iter()
        .find(|binding| binding.name.0 == "single")
        .unwrap()
        .id;
    assert_eq!(body.len(), 2);
    match &body[1].kind {
        HirStmtKind::Assign(_, expr, _) => match &expr.kind {
            HirExprKind::Index(base, indexing) => {
                assert!(matches!(base.kind, HirExprKind::Binding(id) if id == single));
                assert_eq!(indexing.kind, IndexKind::Paren);
                assert_eq!(indexing.components.len(), 1);
            }
            other => panic!("expected Index due to shadowing, got {other:?}"),
        },
        other => panic!("expected Assign, got {other:?}"),
    }
}

#[test]
fn array_indexing_remains_index() {
    let (assembly, body) = entry_body("A = rand(2,2); z = A(1,2);");
    let a = assembly
        .bindings
        .iter()
        .find(|binding| binding.name.0 == "A")
        .unwrap()
        .id;
    assert_eq!(body.len(), 2);
    match &body[1].kind {
        HirStmtKind::Assign(HirPlace::Binding(_), expr, _) => match &expr.kind {
            HirExprKind::Index(base, indexing) => {
                assert!(matches!(base.kind, HirExprKind::Binding(id) if id == a));
                assert_eq!(indexing.components.len(), 2);
            }
            other => panic!("expected Index for A(1,2), got {other:?}"),
        },
        other => panic!("expected Assign, got {other:?}"),
    }
}
