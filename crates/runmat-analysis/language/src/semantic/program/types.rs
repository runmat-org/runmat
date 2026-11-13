use std::collections::BTreeMap;

use runmat_design_syntax::{FeaAssignmentRelation, FeaModelDefinitionKind};
use runmat_types::Span;
use runmat_types::{OperationIdentity, OperationOverloadIdentity, OperationSignature};

use crate::{
    BoundFeaProgram, CheckedCallableSignature, CheckedExpression, CheckedFeaDeclaration,
    FeaDiagnostic, FeaSymbolKey, LogicalSourcePath,
};

#[derive(Debug)]
pub struct CheckedFeaProgram {
    pub bound: BoundFeaProgram,
    pub declarations: BTreeMap<(LogicalSourcePath, FeaSymbolKey), CheckedFeaDeclaration>,
    pub call_sites: Vec<CheckedFeaCallSite>,
    pub diagnostics: Vec<FeaDiagnostic>,
    pub geometry_inspections:
        BTreeMap<(LogicalSourcePath, String), super::super::CheckedGeometryInspection>,
    pub studies: BTreeMap<(LogicalSourcePath, FeaSymbolKey), runmat_analysis_core::AuthoredStudy>,
    pub sweeps: BTreeMap<(LogicalSourcePath, FeaSymbolKey), runmat_analysis_core::AuthoredSweep>,
    pub execution_policies:
        BTreeMap<(LogicalSourcePath, FeaSymbolKey), runmat_analysis_core::AuthoredExecutionPolicy>,
    /// Launch-policy bindings are intentionally separate from study and sweep definitions.
    pub execution_bindings: BTreeMap<
        (LogicalSourcePath, FeaSymbolKey),
        runmat_analysis_core::AuthoredExecutionPolicyRef,
    >,
    pub materials: BTreeMap<(LogicalSourcePath, FeaSymbolKey), runmat_analysis_core::MaterialModel>,
    pub geometries:
        BTreeMap<(LogicalSourcePath, FeaSymbolKey), runmat_geometry_core::GeometryDefinition>,
    pub meshes: BTreeMap<(LogicalSourcePath, FeaSymbolKey), runmat_meshing_core::MeshDefinition>,
    pub models: BTreeMap<(LogicalSourcePath, FeaSymbolKey), runmat_analysis_core::AuthoredModel>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckedFeaCallSite {
    pub module: LogicalSourcePath,
    pub span: Span,
    pub callable: CheckedFeaCallable,
    pub named_arguments: Vec<(String, Span)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckedFeaCallable {
    Operation {
        operation: OperationIdentity,
        overload: OperationOverloadIdentity,
        signature: OperationSignature,
    },
    Named {
        path: Vec<String>,
        signature: CheckedCallableSignature,
    },
}

impl CheckedFeaProgram {
    pub fn is_valid(&self) -> bool {
        self.bound.is_valid()
            && !self.diagnostics.iter().any(|diagnostic| {
                diagnostic.severity == crate::FeaDiagnosticSeverity::Error
                    && diagnostic.availability == crate::SourceAvailability::Available
            })
    }
}

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
        rate: Box<CheckedExpression>,
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
    AssessmentDefinition {
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
