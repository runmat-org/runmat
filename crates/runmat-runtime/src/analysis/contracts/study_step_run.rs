//! One published step of a study run and whether it saved render topology.
use super::{
    AnalysisExecutionStep, AnalysisStudyStepNumericalResultData, QualityGate, QualityReason,
    RunProvenance, RunStatus,
};
use runmat_analysis_core::ProcedureStep;
use runmat_analysis_fea::AnalysisRunIdentity;
use serde::{Deserialize, Serialize};

/// Whether publication saved the render tiles a field scene is drawn from. `absent` runs have
/// no field scene; a host shows their saved figures instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RenderTopologyPresence {
    Persisted,
    Absent,
}

impl RenderTopologyPresence {
    /// The presence of the render topology publication hands to the artifact store, which
    /// saves it as the run's render tiles exactly when it is present.
    pub fn of<T>(staged_topology: Option<&T>) -> Self {
        if staged_topology.is_some() {
            Self::Persisted
        } else {
            Self::Absent
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisStudyStepRunData {
    pub ordinal: usize,
    pub procedure_step: ProcedureStep,
    pub execution: AnalysisExecutionStep,
    pub identity: AnalysisRunIdentity,
    pub run_operation: String,
    pub run_op_version: String,
    pub run_id: String,
    pub run_status: RunStatus,
    pub numerically_acceptable: bool,
    pub solver_convergence: QualityGate,
    pub result_quality: QualityGate,
    #[serde(default)]
    pub quality_reasons: Vec<QualityReason>,
    pub provenance: RunProvenance,
    pub numerical_result: AnalysisStudyStepNumericalResultData,
    pub evidence_artifact_path: String,
    /// Written at publication from the render topology the run saved. A record published
    /// before this field reads as `None`: unknown, never `persisted`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub render_topology: Option<RenderTopologyPresence>,
}
