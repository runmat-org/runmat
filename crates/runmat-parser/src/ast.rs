use serde::{Deserialize, Serialize};

use crate::Span;

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum Expr {
    Number(String, Span),
    String(String, Span),
    Ident(String, Span),
    EndKeyword(Span), // 'end' used in indexing contexts
    Unary(UnOp, Box<Expr>, Span),
    Binary(Box<Expr>, BinOp, Box<Expr>, Span),
    Tensor(Vec<Vec<Expr>>, Span),
    Cell(Vec<Vec<Expr>>, Span),
    Index(Box<Expr>, Vec<Expr>, Span),
    IndexCell(Box<Expr>, Vec<Expr>, Span),
    Range(Box<Expr>, Option<Box<Expr>>, Box<Expr>, Span),
    Colon(Span),
    FuncCall(String, Vec<Expr>, Span),
    Member(Box<Expr>, String, Span),
    // Dynamic field: s.(expr)
    MemberDynamic(Box<Expr>, Box<Expr>, Span),
    DottedInvoke(Box<Expr>, String, Vec<Expr>, Span),
    MethodCall(Box<Expr>, String, Vec<Expr>, Span),
    AnonFunc {
        params: Vec<String>,
        body: Box<Expr>,
        span: Span,
    },
    FuncHandle(String, Span),
    MetaClass(String, Span),
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::Number(_, span)
            | Expr::String(_, span)
            | Expr::Ident(_, span)
            | Expr::EndKeyword(span)
            | Expr::Unary(_, _, span)
            | Expr::Binary(_, _, _, span)
            | Expr::Tensor(_, span)
            | Expr::Cell(_, span)
            | Expr::Index(_, _, span)
            | Expr::IndexCell(_, _, span)
            | Expr::Range(_, _, _, span)
            | Expr::Colon(span)
            | Expr::FuncCall(_, _, span)
            | Expr::Member(_, _, span)
            | Expr::MemberDynamic(_, _, span)
            | Expr::DottedInvoke(_, _, _, span)
            | Expr::MethodCall(_, _, _, span)
            | Expr::FuncHandle(_, span)
            | Expr::MetaClass(_, span) => *span,
            Expr::AnonFunc { span, .. } => *span,
        }
    }

    pub fn with_span(self, span: Span) -> Expr {
        match self {
            Expr::Number(value, _) => Expr::Number(value, span),
            Expr::String(value, _) => Expr::String(value, span),
            Expr::Ident(value, _) => Expr::Ident(value, span),
            Expr::EndKeyword(_) => Expr::EndKeyword(span),
            Expr::Unary(op, expr, _) => Expr::Unary(op, expr, span),
            Expr::Binary(lhs, op, rhs, _) => Expr::Binary(lhs, op, rhs, span),
            Expr::Tensor(rows, _) => Expr::Tensor(rows, span),
            Expr::Cell(rows, _) => Expr::Cell(rows, span),
            Expr::Index(base, indices, _) => Expr::Index(base, indices, span),
            Expr::IndexCell(base, indices, _) => Expr::IndexCell(base, indices, span),
            Expr::Range(start, step, end, _) => Expr::Range(start, step, end, span),
            Expr::Colon(_) => Expr::Colon(span),
            Expr::FuncCall(name, args, _) => Expr::FuncCall(name, args, span),
            Expr::Member(base, name, _) => Expr::Member(base, name, span),
            Expr::MemberDynamic(base, name, _) => Expr::MemberDynamic(base, name, span),
            Expr::DottedInvoke(base, name, args, _) => Expr::DottedInvoke(base, name, args, span),
            Expr::MethodCall(base, name, args, _) => Expr::MethodCall(base, name, args, span),
            Expr::AnonFunc { params, body, .. } => Expr::AnonFunc { params, body, span },
            Expr::FuncHandle(name, _) => Expr::FuncHandle(name, span),
            Expr::MetaClass(name, _) => Expr::MetaClass(name, span),
        }
    }
}

#[derive(Debug, PartialEq, Copy, Clone, Serialize, Deserialize)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    RightDiv,
    Pow,
    LeftDiv,
    Colon,
    // Element-wise operations
    ElemMul,     // .*
    ElemDiv,     // ./
    ElemPow,     // .^
    ElemLeftDiv, // .\
    // Logical operations
    AndAnd, // && (short-circuit)
    OrOr,   // || (short-circuit)
    BitAnd, // &
    BitOr,  // |
    // Comparison operations
    Equal,        // ==
    NotEqual,     // ~=
    Less,         // <
    LessEqual,    // <=
    Greater,      // >
    GreaterEqual, // >=
}

#[derive(Debug, PartialEq, Copy, Clone, Serialize, Deserialize)]
pub enum UnOp {
    Plus,
    Minus,
    Transpose,
    NonConjugateTranspose,
    Not, // ~
}

#[derive(Debug, PartialEq)]
pub enum Stmt {
    ExprStmt(Expr, bool, Span), // Expression and whether it's semicolon-terminated (suppressed)
    Assign(String, Expr, bool, Span), // Variable, Expression, and whether it's semicolon-terminated (suppressed)
    MultiAssign(Vec<String>, Expr, bool, Span),
    AssignLValue(LValue, Expr, bool, Span),
    If {
        cond: Expr,
        then_body: Vec<Stmt>,
        elseif_blocks: Vec<(Expr, Vec<Stmt>)>,
        else_body: Option<Vec<Stmt>>,
        span: Span,
    },
    While {
        cond: Expr,
        body: Vec<Stmt>,
        span: Span,
    },
    For {
        var: String,
        expr: Expr,
        body: Vec<Stmt>,
        span: Span,
    },
    Switch {
        expr: Expr,
        cases: Vec<(Expr, Vec<Stmt>)>,
        otherwise: Option<Vec<Stmt>>,
        span: Span,
    },
    TryCatch {
        try_body: Vec<Stmt>,
        catch_var: Option<String>,
        catch_body: Vec<Stmt>,
        span: Span,
    },
    Global(Vec<String>, Span),
    Persistent(Vec<String>, Span),
    Break(Span),
    Continue(Span),
    Return(Span),
    Function {
        name: String,
        params: Vec<String>,
        outputs: Vec<String>,
        body: Vec<Stmt>,
        span: Span,
    },
    Import {
        path: Vec<String>,
        wildcard: bool,
        span: Span,
    },
    ClassDef {
        name: String,
        super_class: Option<String>,
        members: Vec<ClassMember>,
        span: Span,
    },
}

impl Stmt {
    pub fn span(&self) -> Span {
        match self {
            Stmt::ExprStmt(_, _, span)
            | Stmt::Assign(_, _, _, span)
            | Stmt::MultiAssign(_, _, _, span)
            | Stmt::AssignLValue(_, _, _, span)
            | Stmt::Global(_, span)
            | Stmt::Persistent(_, span)
            | Stmt::Break(span)
            | Stmt::Continue(span)
            | Stmt::Return(span) => *span,
            Stmt::If { span, .. }
            | Stmt::While { span, .. }
            | Stmt::For { span, .. }
            | Stmt::Switch { span, .. }
            | Stmt::TryCatch { span, .. }
            | Stmt::Function { span, .. }
            | Stmt::Import { span, .. }
            | Stmt::ClassDef { span, .. } => *span,
        }
    }
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum LValue {
    Var(String),
    Member(Box<Expr>, String),
    MemberDynamic(Box<Expr>, Box<Expr>),
    Index(Box<Expr>, Vec<Expr>),
    IndexCell(Box<Expr>, Vec<Expr>),
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct Attr {
    pub name: String,
    pub value: Option<String>,
}

#[derive(Debug, PartialEq)]
pub enum ClassMember {
    Properties {
        attributes: Vec<Attr>,
        names: Vec<String>,
    },
    Methods {
        attributes: Vec<Attr>,
        body: Vec<Stmt>,
    },
    Events {
        attributes: Vec<Attr>,
        names: Vec<String>,
    },
    Enumeration {
        attributes: Vec<Attr>,
        names: Vec<String>,
    },
    Arguments {
        attributes: Vec<Attr>,
        names: Vec<String>,
    },
}

#[derive(Debug, PartialEq)]
pub struct Program {
    pub body: Vec<Stmt>,
}
