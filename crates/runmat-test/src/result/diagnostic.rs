use serde::{Deserialize, Serialize};

use crate::descriptor::SourceDescriptor;
use crate::lifecycle::ExecutionPhase;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Diagnostic {
    pub identifier: String,
    pub message: String,
    pub severity: DiagnosticSeverity,
    pub phase: ExecutionPhase,
    pub source: Option<SourceDescriptor>,
    #[serde(default)]
    pub details: Vec<DiagnosticDetail>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticSeverity {
    Information,
    Warning,
    Error,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticDetail {
    pub label: String,
    pub value: String,
}
