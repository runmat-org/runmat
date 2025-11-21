use runmat_hir::{lower, HirExprKind, HirStmt};
use runmat_parser::parse;

#[test]
fn ident_call_with_range_is_func_call_when_no_shadowing() {
    let ast = parse("x = single(0:5);").expect("parse");
    let hir = lower(&ast).expect("lower");
    assert_eq!(hir.body.len(), 1);
    match &hir.body[0] {
        HirStmt::Assign(_, expr, _) => match &expr.kind {
            HirExprKind::FuncCall(name, args) => {
                assert_eq!(name, "single");
                assert_eq!(args.len(), 1);
                assert!(matches!(args[0].kind, HirExprKind::Range(_, _, _)));
            }
            other => panic!("expected FuncCall(single,_), got {other:?}"),
        },
        other => panic!("expected Assign, got {other:?}"),
    }
}

#[test]
fn variable_shadowing_turns_ident_call_into_index() {
    // Shadow builtin 'single' with a variable, then 'single(1)' must be indexing
    let ast = parse("single = 3; y = single(1);").expect("parse");
    let hir = lower(&ast).expect("lower");
    assert_eq!(hir.body.len(), 2);
    match &hir.body[1] {
        HirStmt::Assign(_, expr, _) => match &expr.kind {
            HirExprKind::Index(base, args) => {
                // Base should be the Var for 'single'
                assert!(matches!(base.kind, HirExprKind::Var(_)));
                assert_eq!(args.len(), 1);
            }
            other => panic!("expected Index due to shadowing, got {other:?}"),
        },
        other => panic!("expected Assign, got {other:?}"),
    }
}

#[test]
fn array_indexing_remains_index() {
    let ast = parse("A = rand(2,2); z = A(1,2);").expect("parse");
    let hir = lower(&ast).expect("lower");
    assert_eq!(hir.body.len(), 2);
    match &hir.body[1] {
        HirStmt::Assign(_, expr, _) => match &expr.kind {
            HirExprKind::Index(base, args) => {
                assert!(matches!(base.kind, HirExprKind::Var(_)));
                assert_eq!(args.len(), 2);
            }
            other => panic!("expected Index for A(1,2), got {other:?}"),
        },
        other => panic!("expected Assign, got {other:?}"),
    }
}
