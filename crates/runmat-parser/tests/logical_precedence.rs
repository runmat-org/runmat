use runmat_parser::{parse_simple as parse, BinOp, Expr, Stmt};

#[test]
fn logical_and_short_circuit_precedence() {
    let program = parse("a && b || c").unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::Binary(lhs, BinOp::OrOr, rhs, _), false, _) => {
            assert!(matches!(**lhs, Expr::Binary(_, BinOp::AndAnd, _, _)));
            assert!(matches!(**rhs, Expr::Ident(ref n, _) if n == "c"));
        }
        _ => panic!("expected 'a && b || c' shape"),
    }

    let program = parse("a & b | c").unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::Binary(lhs, BinOp::BitOr, rhs, _), false, _) => {
            assert!(matches!(**lhs, Expr::Binary(_, BinOp::BitAnd, _, _)));
            assert!(matches!(**rhs, Expr::Ident(ref n, _) if n == "c"));
        }
        _ => panic!("expected 'a & b | c' shape"),
    }
}
