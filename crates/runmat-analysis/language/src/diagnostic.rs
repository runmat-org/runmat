use runmat_types::Span;
use serde::{Deserialize, Serialize};

use crate::LogicalSourcePath;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceAvailability {
    Available,
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
#[error("{code}: {message}")]
#[serde(deny_unknown_fields)]
pub struct SourceDiagnostic {
    pub code: String,
    pub message: String,
    pub availability: SourceAvailability,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
#[error("{code}: {message}")]
#[serde(deny_unknown_fields)]
pub struct OperationDiagnostic {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeaDiagnosticSeverity {
    Error,
    Warning,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeaDiagnosticPhase {
    Syntax,
    ModuleGraph,
    Binding,
    Semantic,
    Inspection,
    Lowering,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaDiagnosticLocation {
    pub path: LogicalSourcePath,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaRelatedDiagnostic {
    pub message: String,
    pub location: FeaDiagnosticLocation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaDiagnostic {
    pub code: String,
    pub message: String,
    pub severity: FeaDiagnosticSeverity,
    pub phase: FeaDiagnosticPhase,
    pub location: Option<FeaDiagnosticLocation>,
    /// Short label printed under the located span, such as the expected unit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub related: Vec<FeaRelatedDiagnostic>,
    pub availability: SourceAvailability,
}

impl FeaDiagnostic {
    pub fn error(
        code: impl Into<String>,
        message: impl Into<String>,
        phase: FeaDiagnosticPhase,
        location: Option<FeaDiagnosticLocation>,
    ) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            severity: FeaDiagnosticSeverity::Error,
            phase,
            location,
            label: None,
            related: Vec::new(),
            availability: SourceAvailability::Available,
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn unavailable(mut self) -> Self {
        self.availability = SourceAvailability::Unavailable;
        self
    }

    pub fn with_related(mut self, related: FeaRelatedDiagnostic) -> Self {
        self.related.push(related);
        self
    }
}
