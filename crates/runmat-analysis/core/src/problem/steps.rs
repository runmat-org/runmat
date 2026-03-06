use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisStepKind {
    Static,
    Modal,
    Transient,
    Nonlinear,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisStep {
    pub step_id: String,
    pub kind: AnalysisStepKind,
}
