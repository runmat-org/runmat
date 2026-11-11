use super::{FeaName, FeaTokenRange};
use runmat_types::Span;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeaExpression {
    pub kind: FeaExpressionKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeaExpressionKind {
    Number(String),
    String(String),
    Reference(FeaReference),
    List(Vec<FeaExpression>),
    Record(Vec<FeaRecordEntry>),
    Group(Box<FeaExpression>),
    Unary {
        operator: FeaUnaryOperator,
        operand: Box<FeaExpression>,
    },
    Binary {
        left: Box<FeaExpression>,
        operator: FeaBinaryOperator,
        right: Box<FeaExpression>,
    },
    Call {
        callee: Box<FeaExpression>,
        arguments: Vec<FeaCallArgument>,
    },
    PrefixApplication {
        callee: Box<FeaExpression>,
        argument: Box<FeaExpression>,
    },
    Quantity {
        value: Box<FeaExpression>,
        unit: FeaTokenRange,
    },
    Missing,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeaReference {
    pub segments: Vec<FeaName>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeaCallArgument {
    Positional(FeaExpression),
    Named {
        name: FeaName,
        value: FeaExpression,
        span: Span,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeaRecordEntry {
    pub name: FeaName,
    pub value: FeaExpression,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeaUnaryOperator {
    Positive,
    Negative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeaBinaryOperator {
    Range,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
}
