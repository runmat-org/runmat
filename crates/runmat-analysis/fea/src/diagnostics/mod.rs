use serde::{Deserialize, Serialize};

pub(crate) mod builders;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeaDiagnosticSeverity {
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeaDiagnostic {
    pub code: String,
    pub severity: FeaDiagnosticSeverity,
    pub message: String,
}
