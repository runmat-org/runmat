use runmat_analysis_core::ProcedureStep;
use runmat_analysis_fea::{execution::ThermalWorkload, study::ThermalStudyPlanningPolicy};
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
    pub(super) fn validate_workload(&self) -> AnalysisBridgeResult<()> {
        if self.field_source.is_some() {
            if self.planned_workload()? != self.product.workload {
                return Err(invalid(
                    "thermal workload differs from its retained source binding",
                ));
            }
            return Ok(());
        }
        let policy = ThermalStudyPlanningPolicy {
            backend: self.product.workload.backend,
            element_heat_source: None,
        };
        let workload = &self.product.workload;
        match self.scope {
            ThermalStudyScope::Standalone => {
                workload.validate_authored_standalone(&self.product.compiled, policy)
            }
            ThermalStudyScope::SelectedStep => {
                workload.validate_authored_selected_step(&self.product.compiled, policy)
            }
        }
        .map_err(|e| invalid(e.to_string()))
    }
    pub(super) fn planned_workload(&self) -> AnalysisBridgeResult<ThermalWorkload> {
        let source = self
            .field_source
            .as_ref()
            .ok_or_else(|| invalid("retained thermal source is absent"))?;
        if self.scope != ThermalStudyScope::SelectedStep {
            return Err(invalid(
                "retained thermal source requires selected-step scope",
            ));
        }
        source
            .thermal_intent()
            .validate_authored_selected_step(
                &self.product.compiled,
                ThermalStudyPlanningPolicy {
                    backend: self.product.workload.backend,
                    element_heat_source: None,
                },
            )
            .map_err(|e| invalid(e.to_string()))?;
        source.bind(&self.mesh).map_err(|e| invalid(e.to_string()))
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
