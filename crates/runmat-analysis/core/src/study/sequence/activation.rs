//! Orthogonal procedure-step object activation and excitation-case selection.
use serde::{Deserialize, Serialize};

use super::ProcedureStepId;

/// Selects load cases independently of support, material and other object activations.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExcitationCaseSelection {
    #[default]
    All,
    /// An empty selection intentionally suppresses all excitation actions.
    Only(Vec<crate::ExcitationCaseId>),
}

impl ExcitationCaseSelection {
    pub fn includes(&self, case_id: &crate::ExcitationCaseId) -> bool {
        match self {
            Self::All => true,
            Self::Only(ids) => ids.contains(case_id),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepObjectRef {
    Constraint(String),
    Boundary(String),
    Connector(String),
    Interface(String),
    Excitation(String),
    InitialCondition(String),
    MaterialParameter(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StepObjectActivation {
    pub object: StepObjectRef,
    pub lifecycle: ActivationLifecycle,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivationLifecycle {
    Activate,
    Continue {
        from_step: ProcedureStepId,
    },
    Modify {
        prior_step: ProcedureStepId,
        prior_object: StepObjectRef,
    },
    Deactivate {
        from_step: ProcedureStepId,
    },
}
