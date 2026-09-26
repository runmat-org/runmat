//! Composition of thermal preassembly component estimates; not whole-process admission.
use super::{
    ByteCount, ResourcePlanningError, ThermalFieldConstructionResourceFacts,
    ThermalHistoryResourceFacts, ThermalMetadataResourceFacts, ThermalObservationResourceFacts,
    ThermalOperatorResourceFacts, ThermalRealizationResourceFacts,
    ThermalSourceEvidenceResourceFacts,
};
use crate::execution::ThermalSolveSchedule;
use runmat_analysis_core::{AnalysisModel, ProcedureStep};
use runmat_meshing_core::SolverMeshArtifact;

pub struct ThermalPreassemblyResourceFacts {
    pub(crate) properties: crate::properties::CompiledPropertyMap,
    /// Compiler selection, owned properties and concurrent canonical identity scratch estimate.
    pub property_compilation_bytes: ByteCount,
    pub history: ThermalHistoryResourceFacts,
    pub operators: ThermalOperatorResourceFacts,
    pub sources: ThermalSourceEvidenceResourceFacts,
    pub observations: ThermalObservationResourceFacts,
    pub realization: ThermalRealizationResourceFacts,
    pub metadata: ThermalMetadataResourceFacts,
    pub field_construction: ThermalFieldConstructionResourceFacts,
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
        // Admit projection, canonical validation and support construction before deriving them.
        // Operator facts retain this estimate in the final composed numerical envelope.
        let planning = ThermalOperatorResourceFacts::derivation_resident_bytes_estimate(mesh)?;
        let before_projection = history
            .reservation_bytes()?
            .0
            .checked_add(model_materialization_bytes.0)
            .and_then(|bytes| bytes.checked_add(planning.0))
            .ok_or(ResourcePlanningError::Overflow("thermal operator planning"))?;
        require_ceiling(
            before_projection,
            step.accuracy.resources.maximum_resident_bytes,
        )?;
        let operators = ThermalOperatorResourceFacts::derive(mesh)?;
        operators.require_fits(&history, step.accuracy.resources.maximum_resident_bytes)?;
        let sources = ThermalSourceEvidenceResourceFacts::derive(model, step, schedule)?;
        let observations = ThermalObservationResourceFacts::derive(mesh, step)?;
        let realization = ThermalRealizationResourceFacts::derive(model, step)?;
        let metadata = ThermalMetadataResourceFacts::derive(model, step, schedule)?;
        let field_construction = ThermalFieldConstructionResourceFacts::derive(mesh, schedule)?;
        let base = component_bytes(
            &history,
            &operators,
            &sources,
            &observations,
            &realization,
            model_materialization_bytes,
            &metadata,
        )?
        .checked_add(field_construction.reservation_bytes().0)
        .ok_or(ResourcePlanningError::Overflow(
            "thermal field construction reservation",
        ))?;
        let maximum = step.accuracy.resources.maximum_resident_bytes;
        require_ceiling(base, maximum)?;
        // Admit actual selected metadata before owned constitutive/section/audit cloning.
        // The compiler owns selection for both the profile and materialization passes.
        let prepared =
            crate::properties::PreparedProperties::prepare(model, Some(mesh), maximum - base)
                .map_err(|error| match error {
                    crate::properties::PropertyCompilationError::ResourceCeiling {
                        required,
                        ..
                    } => match base.checked_add(required) {
                        Some(required) => ResourcePlanningError::AccuracyCeiling {
                            resource: "thermal preassembly components",
                            required,
                            maximum,
                        },
                        None => ResourcePlanningError::Overflow("thermal property compilation"),
                    },
                    error => ResourcePlanningError::Invalid(error.to_string()),
                })?;
        let property_compilation_bytes = ByteCount(prepared.estimated_resident_bytes);
        let properties = prepared
            .materialize()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        properties
            .validate_thermal_solid_capabilities()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        Ok(Self {
            properties,
            property_compilation_bytes,
            history,
            operators,
            sources,
            observations,
            realization,
            metadata,
            field_construction,
            model_materialization_bytes,
        })
    }

    pub fn reservation_bytes(&self) -> Result<ByteCount, ResourcePlanningError> {
        component_bytes(
            &self.history,
            &self.operators,
            &self.sources,
            &self.observations,
            &self.realization,
            self.model_materialization_bytes,
            &self.metadata,
        )?
        .checked_add(self.property_compilation_bytes.0)
        .and_then(|bytes| bytes.checked_add(self.field_construction.reservation_bytes().0))
        .map(ByteCount)
        .ok_or(ResourcePlanningError::Overflow(
            "thermal preassembly reservation",
        ))
    }

    pub(crate) fn require_fits(&self, maximum: u64) -> Result<(), ResourcePlanningError> {
        require_ceiling(self.reservation_bytes()?.0, maximum)
    }
}

fn component_bytes(
    history: &ThermalHistoryResourceFacts,
    operators: &ThermalOperatorResourceFacts,
    sources: &ThermalSourceEvidenceResourceFacts,
    observations: &ThermalObservationResourceFacts,
    realization: &ThermalRealizationResourceFacts,
    model: ByteCount,
    metadata: &ThermalMetadataResourceFacts,
) -> Result<u64, ResourcePlanningError> {
    let observations = observations.reservation_bytes()?.0;
    let realization = realization.reservation_bytes()?.0;
    let metadata = metadata.reservation_bytes()?.0;
    operators
        .numerical_reservation_bytes(history)?
        .0
        .checked_add(sources.reservation_bytes()?.0)
        .and_then(|bytes| bytes.checked_add(model.0))
        .and_then(|bytes| bytes.checked_add(observations))
        .and_then(|bytes| bytes.checked_add(realization))
        .and_then(|bytes| bytes.checked_add(metadata))
        .ok_or(ResourcePlanningError::Overflow(
            "thermal preassembly components",
        ))
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
