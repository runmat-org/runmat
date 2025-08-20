use runmat_parser::{parse_simple as parse, BinOp, Expr, Stmt};

#[test]
fn cell_literal_and_indexing() {
    let program = parse("C = {1,2;3,4}; C{1,2}").unwrap();
    assert_eq!(program.body.len(), 2);
    match &program.body[0] {
        Stmt::Assign(name, Expr::Cell(rows), true) => {
            assert_eq!(name, "C");
            assert_eq!(rows.len(), 2);
            assert_eq!(rows[0].len(), 2);
        }
        _ => panic!("expected cell assignment"),
    }
    match &program.body[1] {
        Stmt::ExprStmt(Expr::IndexCell(base, idxs), _) => {
            assert!(matches!(**base, Expr::Ident(ref n) if n == "C"));
            assert_eq!(idxs.len(), 2);
        }
        _ => panic!("expected cell indexing"),
    }
}

#[test]
fn indexing_with_end_and_member_method() {
    let program = parse("A(5:end); obj.method(a,b)").unwrap();
    assert_eq!(program.body.len(), 2);
    match &program.body[0] {
        Stmt::ExprStmt(Expr::Index(base, idxs), true) => {
            assert!(matches!(**base, Expr::Ident(ref n) if n == "A"));
            assert!(matches!(
                idxs.as_slice(),
                [Expr::Binary(_, BinOp::Colon, _)]
                    | [Expr::Range(_, _, _)]
                    | [Expr::Number(_), Expr::EndKeyword]
                    | [Expr::Number(_), Expr::Colon, Expr::EndKeyword]
            ));
        }
        _ => panic!("expected indexing with end"),
    }
    match &program.body[1] {
        Stmt::ExprStmt(Expr::MethodCall(obj, name, args), false) => {
            assert!(matches!(**obj, Expr::Ident(ref n) if n == "obj"));
            assert_eq!(name, "method");
            assert_eq!(args.len(), 2);
        }
        _ => panic!("expected method call expression"),
    }
}
