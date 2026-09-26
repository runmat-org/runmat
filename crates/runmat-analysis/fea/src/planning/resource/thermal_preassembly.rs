//! Composition of thermal preassembly component estimates; not whole-process admission.
use super::{
    ByteCount, ResourcePlanningError, ThermalHistoryResourceFacts, ThermalObservationResourceFacts,
    ThermalOperatorResourceFacts, ThermalRealizationResourceFacts,
    ThermalSourceEvidenceResourceFacts,
};
use crate::execution::ThermalSolveSchedule;
use runmat_analysis_core::{AnalysisModel, ProcedureStep};
use runmat_meshing_core::SolverMeshArtifact;

pub struct ThermalPreassemblyResourceFacts {
    pub history: ThermalHistoryResourceFacts,
    pub operators: ThermalOperatorResourceFacts,
    pub sources: ThermalSourceEvidenceResourceFacts,
    pub observations: ThermalObservationResourceFacts,
    pub realization: ThermalRealizationResourceFacts,
    /// Shared codec estimate for authored model materialization plus a possible filtered copy.
    /// The usual unfiltered numerical path borrows the model and uses less than this envelope.
    pub model_materialization_bytes: ByteCount,
}

impl ThermalPreassemblyResourceFacts {
    pub fn derive(
        model: &AnalysisModel,
        step: &ProcedureStep,
        mesh: &SolverMeshArtifact,
        schedule: &ThermalSolveSchedule,
    ) -> Result<Self, ResourcePlanningError> {
        let history = ThermalHistoryResourceFacts::derive(mesh, schedule)?;
        history.require_fits(step.accuracy.resources.maximum_resident_bytes)?;
        // Reuse the workload's bounded input envelope; do not allocate canonical bytes or a
        // serde value tree to decide whether the model fits the authored resident ceiling.
        let model_bytes =
            runmat_canonical_codec::serialized_materialization_resident_bytes_estimate(
                model,
                crate::execution::THERMAL_WORKLOAD_LIMITS,
            )
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        let model_materialization_bytes = ByteCount(model_bytes.checked_mul(2).ok_or(
            ResourcePlanningError::Overflow("thermal model materialization"),
        )?);
        require_ceiling(
            model_materialization_bytes.0,
            step.accuracy.resources.maximum_resident_bytes,
        )?;
        let operators = ThermalOperatorResourceFacts::derive(mesh)?;
        operators.require_fits(&history, step.accuracy.resources.maximum_resident_bytes)?;
        Ok(Self {
            history,
            operators,
            sources: ThermalSourceEvidenceResourceFacts::derive(model, step, schedule)?,
            observations: ThermalObservationResourceFacts::derive(mesh, step)?,
            realization: ThermalRealizationResourceFacts::derive(model, step)?,
            model_materialization_bytes,
        })
    }

    pub fn reservation_bytes(&self) -> Result<ByteCount, ResourcePlanningError> {
        let observations = self.observations.reservation_bytes()?.0;
        let realization = self.realization.reservation_bytes()?.0;
        self.operators
            .numerical_reservation_bytes(&self.history)?
            .0
            .checked_add(self.sources.reservation_bytes()?.0)
            .and_then(|bytes| bytes.checked_add(self.model_materialization_bytes.0))
            .and_then(|bytes| bytes.checked_add(observations))
            .and_then(|bytes| bytes.checked_add(realization))
            .map(ByteCount)
            .ok_or(ResourcePlanningError::Overflow(
                "thermal preassembly reservation",
            ))
    }

    pub(crate) fn require_fits(&self, maximum: u64) -> Result<(), ResourcePlanningError> {
        require_ceiling(self.reservation_bytes()?.0, maximum)
    }
}

fn require_ceiling(required: u64, maximum: u64) -> Result<(), ResourcePlanningError> {
    if required > maximum {
        Err(ResourcePlanningError::AccuracyCeiling {
            resource: "thermal preassembly components",
            required,
            maximum,
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests;
