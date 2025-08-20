use runmat_parser::{parse_simple as parse, Expr, Stmt};

#[test]
fn command_form_basic() {
    let program = parse("plot 1 2 3").unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::FuncCall(name, args), _) => {
            assert_eq!(name, "plot");
            assert_eq!(args.len(), 3);
        }
        _ => panic!("expected func call"),
    }
}

#[test]
fn command_form_not_triggered_by_assignment_or_index() {
    // Should parse as assignment, not command form
    let program = parse("x = 1").unwrap();
    match &program.body[0] {
        Stmt::Assign(name, _, _) => assert_eq!(name, "x"),
        _ => panic!("expected assign"),
    }

    // Not command-form when followed by '('
    let program = parse("foo(1,2)").unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::Index(_, _), _) | Stmt::ExprStmt(Expr::FuncCall(_, _), _) => {}
        _ => panic!("expected call/index"),
    }
}

#[test]
fn metaclass_qualified() {
    let program = parse("?pkg.sub.Class").unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::MetaClass(s), _) => assert_eq!(s, "pkg.sub.Class"),
        _ => panic!("expected metaclass"),
    }
}
