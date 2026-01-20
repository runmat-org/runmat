use runmat_parser::{BinOp, Expr, Stmt, UnOp};

mod parse;
use parse::parse;

#[test]
fn non_conjugate_transpose_and_dot_plus() {
    let program = parse("A.'; A .+ B").unwrap();
    assert_eq!(program.body.len(), 2);
    match &program.body[0] {
        Stmt::ExprStmt(Expr::Unary(UnOp::NonConjugateTranspose, _, _), true, _) => {}
        _ => panic!("expected non-conjugate transpose"),
    }
    match &program.body[1] {
        Stmt::ExprStmt(Expr::Binary(_, BinOp::Add, _, _), false, _) => {}
        _ => panic!("expected dot-plus to parse as add"),
    }
}
