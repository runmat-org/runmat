use runmat_parser::fea::{FeaAssignmentRelation, FeaModelDefinitionKind};
use runmat_types::Span;

use crate::CheckedExpression;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckedFeaStatement {
    pub kind: CheckedFeaStatementKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckedFeaStatementKind {
    GeneratedByFunction {
        function: Vec<String>,
        statements: Vec<CheckedFeaStatement>,
    },
    UniformSample {
        start: CheckedExpression,
        end: CheckedExpression,
        rate: CheckedExpression,
    },
    AppliedLoad {
        kind: String,
        target: CheckedExpression,
        value: CheckedExpression,
    },
    SignalDefinition {
        name: String,
        quantity: String,
        domain: String,
        statements: Vec<CheckedFeaStatement>,
    },
    ModelDefinition {
        kind: FeaModelDefinitionKind,
        name: String,
        statements: Vec<CheckedFeaStatement>,
    },
    Block {
        header: String,
        header_span: Span,
        statements: Vec<CheckedFeaStatement>,
    },
    Input(String),
    Let {
        name: String,
        value: CheckedExpression,
    },
    Feature {
        name: String,
        value: CheckedExpression,
    },
    SourceParameterUpdate {
        target: Vec<String>,
        value: CheckedExpression,
    },
    Use(Vec<String>),
    Requirement {
        target: Vec<String>,
        value: CheckedExpression,
    },
    Assignment {
        target: Vec<String>,
        relation: FeaAssignmentRelation,
        value: CheckedExpression,
    },
    PropertyAssignment {
        property: CheckedExpression,
        target: CheckedExpression,
    },
    Output {
        name: String,
        value: CheckedExpression,
    },
    OutputDefinition {
        name: String,
        statements: Vec<CheckedFeaStatement>,
    },
    Include(CheckedExpression),
    Retain(CheckedExpression),
    Command(String),
}
