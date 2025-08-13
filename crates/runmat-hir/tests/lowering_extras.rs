use runmat_hir::{lower, HirExprKind, HirProgram, HirStmt};
use runmat_parser::parse_simple as parse;

fn lower_src(src: &str) -> HirProgram {
    let ast = parse(src).unwrap();
    lower(&ast).unwrap()
}

#[test]
fn import_and_metaclass_lowering() {
    let prog = lower_src("import pkg.*; ?pkg.Class");
    assert_eq!(prog.body.len(), 2);
    // Import lowers to Import node
    match &prog.body[0] {
        HirStmt::Import { path, wildcard } => {
            assert_eq!(path, &vec!["pkg".to_string()]);
            assert!(*wildcard);
        }
        _ => panic!("expected import node"),
    }
    // MetaClass lowers to a MetaClass expr carrying the qualified name
    match &prog.body[1] {
        HirStmt::ExprStmt(expr, false) => match &expr.kind {
            HirExprKind::MetaClass(s) => assert_eq!(s, "pkg.Class"),
            _ => panic!("expected metaclass expr"),
        },
        _ => panic!("expected expr stmt for metaclass"),
    }
}

#[test]
fn lvalue_assignment_lowering_total() {
    let prog = lower_src("A=1; A(1)=2; A{1}=3; s.f = 4");
    // Ensure lowering doesn't panic and produces statements for complex lvalues
    assert_eq!(prog.body.len(), 4);
    // second: paren-index assignment
    match &prog.body[1] {
        HirStmt::AssignLValue(_, _, _) => {}
        other => panic!("expected AssignLValue for second stmt, got {:?}", other),
    }
    // third: brace-index assignment
    match &prog.body[2] {
        HirStmt::AssignLValue(_, _, _) => {}
        other => panic!("expected AssignLValue for third stmt, got {:?}", other),
    }
    // fourth: member assignment
    match &prog.body[3] {
        HirStmt::AssignLValue(_, _, _) => {}
        other => panic!("expected AssignLValue for fourth stmt, got {:?}", other),
    }
}


