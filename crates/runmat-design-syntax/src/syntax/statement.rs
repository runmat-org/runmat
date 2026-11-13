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

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeaModelDefinitionKind {
    Section,
    Layup,
    Orientation,
    Constraint,
    Load,
    LoadCase,
    Interface,
    Connector,
    Excitation,
    ResponseSpectrum,
    RandomExcitation,
    Initial,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeaStatementKind {
    GeneratedByFunction {
        function: FeaReference,
        body: FeaBlock,
    },
    UniformSample {
        start: FeaExpression,
        end: FeaExpression,
        rate: FeaExpression,
    },
    AppliedLoad {
        kind: FeaName,
        target: FeaExpression,
        value: FeaExpression,
    },
    SignalDefinition {
        name: FeaName,
        quantity: FeaName,
        domain: FeaName,
        body: FeaBlock,
    },
    ModelDefinition {
        kind: FeaModelDefinitionKind,
        name: FeaName,
        body: FeaBlock,
    },
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
    Feature {
        name: FeaName,
        value: FeaExpression,
    },
    SourceParameterUpdate {
        target: FeaReference,
        value: FeaExpression,
    },
    Use(FeaReference),
    Requirement {
        target: FeaReference,
        value: FeaExpression,
    },
    Assignment {
        target: FeaReference,
        relation: FeaAssignmentRelation,
        value: FeaExpression,
    },
    PropertyAssignment {
        property: FeaExpression,
        target: FeaExpression,
    },
    Output {
        name: FeaName,
        value: FeaExpression,
    },
    OutputDefinition {
        name: FeaName,
        body: FeaBlock,
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
