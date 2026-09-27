use runmat_analysis_core::ProcedureStep;
use runmat_analysis_fea::{
    execution::ThermalWorkload,
    study::{plan_thermal_step, plan_thermal_study, ThermalStudyPlanningPolicy},
};
use serde::{Deserialize, Serialize};

use super::{invalid, ThermalStudyPreparation};
use crate::AnalysisBridgeResult;

/// Completion scope is explicit and retained by the durable root. Completing a selected
/// thermal step does not complete the surrounding authored multi-physics study.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ThermalStudyScope {
    Standalone,
    SelectedStep,
}

impl ThermalStudyPreparation {
    pub(super) fn planned_workload(&self) -> AnalysisBridgeResult<ThermalWorkload> {
        let policy = ThermalStudyPlanningPolicy {
            backend: self.product.workload.backend,
            formulation_requirement: self.product.workload.formulation_requirement,
        };
        match self.scope {
            ThermalStudyScope::Standalone => plan_thermal_study(&self.product.compiled, policy),
            ThermalStudyScope::SelectedStep => plan_thermal_step(
                &self.product.compiled,
                &self.product.workload.step_id.0,
                policy,
            ),
        }
        .map_err(|error| invalid(error.to_string()))
    }

    /// Only used after exact selected-workload validation by the coordinator constructor.
    pub(super) fn selected_step(&self) -> &ProcedureStep {
        self.product
            .workload
            .model
            .sequence
            .steps
            .iter()
            .find(|step| step.step_id == self.product.workload.step_id)
            .expect("validated selected thermal step")
    }
}
