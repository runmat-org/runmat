use runmat_parser::{Expr, Span, Stmt, UnOp};

fn span() -> Span {
    Span::default()
}

pub fn assign(name: String, expr: Expr, suppressed: bool) -> Stmt {
    Stmt::Assign(name, expr, suppressed, span())
}

pub fn range(start: Box<Expr>, step: Option<Box<Expr>>, end: Box<Expr>) -> Expr {
    Expr::Range(start, step, end, span())
}

pub fn string(value: String) -> Expr {
    Expr::String(value, span())
}

pub fn unary_boxed(op: UnOp, expr: Box<Expr>) -> Expr {
    Expr::Unary(op, expr, span())
}

pub fn span_value() -> Span {
    span()
}
