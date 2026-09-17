use super::{FeaExpression, FeaName, FeaReference, FeaTokenRange};
use runmat_types::Span;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeaBlock {
    pub statements: Vec<FeaStatement>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeaStatement {
    pub kind: FeaStatementKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeaStatementKind {
    Block {
        header: FeaTokenRange,
        body: FeaBlock,
    },
    Input {
        name: FeaName,
        type_syntax: FeaTokenRange,
        default: Option<FeaExpression>,
    },
    Let {
        name: FeaName,
        value: FeaExpression,
    },
    Assignment {
        target: FeaReference,
        relation: FeaAssignmentRelation,
        value: FeaExpression,
    },
    MaterialAssignment {
        material: FeaExpression,
        target: FeaExpression,
    },
    Output {
        name: FeaName,
        value: FeaExpression,
    },
    Include(FeaExpression),
    Retain(FeaExpression),
    Command {
        syntax: FeaTokenRange,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeaAssignmentRelation {
    Equal,
    LessEqual,
    GreaterEqual,
}
