use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InferenceSeverity {
    Error,
    Warning,
    Note,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferenceDiagnostic {
    pub code: String,
    pub severity: InferenceSeverity,
    pub message: String,
    pub argument: Option<usize>,
    pub dimension: Option<usize>,
}

impl InferenceDiagnostic {
    pub fn error(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            severity: InferenceSeverity::Error,
            message: message.into(),
            argument: None,
            dimension: None,
        }
    }

    pub fn at_dimension(mut self, dimension: usize) -> Self {
        self.dimension = Some(dimension);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FactInference {
    pub fact: crate::ValueFact,
    pub diagnostics: Vec<InferenceDiagnostic>,
}

impl FactInference {
    pub fn exact(fact: crate::ValueFact) -> Self {
        Self {
            fact,
            diagnostics: Vec::new(),
        }
    }
}
