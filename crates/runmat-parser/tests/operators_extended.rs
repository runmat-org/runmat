use runmat_parser::{parse_simple as parse, BinOp, Expr, Stmt, UnOp};

#[test]
fn non_conjugate_transpose_and_dot_plus() {
    let program = parse("A.'; A .+ B").unwrap();
    assert_eq!(program.body.len(), 2);
    match &program.body[0] {
        Stmt::ExprStmt(Expr::Unary(UnOp::NonConjugateTranspose, _), true) => {}
        _ => panic!("expected non-conjugate transpose"),
    }
    match &program.body[1] {
        Stmt::ExprStmt(Expr::Binary(_, BinOp::Add, _), false) => {}
        _ => panic!("expected dot-plus to parse as add"),
    }
}


