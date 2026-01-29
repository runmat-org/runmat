use runmat_parser::{BinOp, Expr, Span, Stmt};

fn span() -> Span {
    Span::default()
}

pub fn num(value: String) -> Expr {
    Expr::Number(value, span())
}

pub fn ident(value: String) -> Expr {
    Expr::Ident(value, span())
}

pub fn expr_stmt(expr: Expr, suppressed: bool) -> Stmt {
    Stmt::ExprStmt(expr, suppressed, span())
}

pub fn binary_boxed(lhs: Box<Expr>, op: BinOp, rhs: Box<Expr>) -> Expr {
    Expr::Binary(lhs, op, rhs, span())
}

pub fn func_call(name: String, args: Vec<Expr>) -> Expr {
    Expr::FuncCall(name, args, span())
}

pub fn tensor(rows: Vec<Vec<Expr>>) -> Expr {
    Expr::Tensor(rows, span())
}
