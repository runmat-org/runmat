//! Composition of thermal preassembly component estimates; not whole-process admission.
use super::{
    ByteCount, ResourcePlanningError, ThermalFieldConstructionResourceFacts,
    ThermalHistoryResourceFacts, ThermalManifestCatalogResourceFacts,
    ThermalManifestMetadataResourceFacts, ThermalManifestRetainedResourceFacts,
    ThermalMetadataResourceFacts, ThermalObservationResourceFacts, ThermalOperatorResourceFacts,
    ThermalRealizationResourceFacts, ThermalSourceEvidenceResourceFacts,
};
use crate::execution::ThermalSolveSchedule;
use runmat_analysis_core::{AnalysisModel, ProcedureStep};
use runmat_meshing_core::SolverMeshArtifact;

pub struct ThermalPreassemblyResourceFacts {
    pub(crate) admitted_identity: crate::admission::AnalysisRunIdentity,
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
    pub manifest_catalog: ThermalManifestCatalogResourceFacts,
    pub manifest_metadata: ThermalManifestMetadataResourceFacts,
    pub manifest_retained: ThermalManifestRetainedResourceFacts,
    /// Shared codec estimate for authored model materialization plus a possible filtered copy.
    /// The usual unfiltered numerical path borrows the model and uses less than this envelope.
    pub model_materialization_bytes: ByteCount,
}

impl ThermalPreassemblyResourceFacts {
    /// Borrowed input preflight before an owned preparation clones the field closure.
    pub(crate) fn require_input_fits(
        model: &AnalysisModel,
        fields: &[crate::execution::InitialFieldBinding],
        maximum: u64,
    ) -> Result<(), ResourcePlanningError> {
        require_ceiling(Self::input_materialization_bytes(model, fields)?.0, maximum)
    }

    fn input_materialization_bytes(
        model: &AnalysisModel,
        fields: &[crate::execution::InitialFieldBinding],
    ) -> Result<ByteCount, ResourcePlanningError> {
        let model_bytes = runmat_canonical_codec::profile_serialized_shape(
            model,
            crate::execution::THERMAL_WORKLOAD_LIMITS,
        )
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?
        .materialization_resident_bytes_estimate;
        let bytes = model_bytes
            .checked_mul(2)
            .ok_or(ResourcePlanningError::Overflow(
                "thermal model materialization",
            ))?
            .checked_add(
                super::initial_field_reservation_bytes(
                    fields,
                    crate::execution::THERMAL_WORKLOAD_LIMITS,
                )?
                .0,
            )
            .ok_or(ResourcePlanningError::Overflow(
                "thermal model and initial-field materialization",
            ))?;
        Ok(ByteCount(bytes))
    }

    /// Estimate the selected mesh after exact formulation admission. `identity` must be the
    /// registry receipt for this mesh; the numerical pipeline obtains it immediately before
    /// this call. These estimates do not replace formulation or mesh admission.
    pub fn derive(
        initial_fields: &[crate::execution::InitialFieldBinding],
        model: &AnalysisModel,
        step: &ProcedureStep,
        mesh: &SolverMeshArtifact,
        schedule: &ThermalSolveSchedule,
        identity: &crate::admission::AnalysisRunIdentity,
    ) -> Result<Self, ResourcePlanningError> {
        Self::derive_with_observations(
            initial_fields,
            model,
            step,
            mesh,
            schedule,
            identity,
            crate::pipeline::thermal::ThermalObservationExecution::Inline,
        )
    }

    pub(crate) fn derive_with_observations(
        initial_fields: &[crate::execution::InitialFieldBinding],
        model: &AnalysisModel,
        step: &ProcedureStep,
        mesh: &SolverMeshArtifact,
        schedule: &ThermalSolveSchedule,
        identity: &crate::admission::AnalysisRunIdentity,
        observations: crate::pipeline::thermal::ThermalObservationExecution,
    ) -> Result<Self, ResourcePlanningError> {
        if identity.family() != runmat_analysis_core::PhysicsFamily::Thermal
            || identity.procedure() != step.procedure
        {
            return Err(ResourcePlanningError::Invalid(
                "thermal resource identity differs from the selected procedure".into(),
            ));
        }
        let history = ThermalHistoryResourceFacts::derive(mesh, schedule, &model.interfaces)?;
        history.require_fits(step.accuracy.resources.resident_byte_budget())?;
        // Reuse the workload's bounded input envelope; do not allocate canonical bytes or a
        // serde value tree to decide whether the model fits the authored resident ceiling.
        let model_materialization_bytes = Self::input_materialization_bytes(model, initial_fields)?;
        require_ceiling(
            model_materialization_bytes.0,
            step.accuracy.resources.resident_byte_budget(),
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
            step.accuracy.resources.resident_byte_budget(),
        )?;
        let operators = select_thermal_solve_path(
            ThermalOperatorResourceFacts::derive(mesh, &model.interfaces)?,
            &history,
            model,
            step,
        )?;
        let sources = ThermalSourceEvidenceResourceFacts::derive(model, step, schedule)?;
        let observations = match observations {
            crate::pipeline::thermal::ThermalObservationExecution::Inline => {
                ThermalObservationResourceFacts::derive(mesh, step)?
            }
            crate::pipeline::thermal::ThermalObservationExecution::Deferred => {
                ThermalObservationResourceFacts::derive_deferred(mesh, step)?
            }
        };
        let realization = ThermalRealizationResourceFacts::derive(model, step)?;
        let metadata = ThermalMetadataResourceFacts::derive(model, step, schedule)?;
        let field_construction =
            ThermalFieldConstructionResourceFacts::derive(mesh, schedule, history.interface_nodes)?;
        let manifest_catalog =
            ThermalManifestCatalogResourceFacts::derive(mesh, schedule, history.interface_nodes)?;
        let manifest_metadata = ThermalManifestMetadataResourceFacts::derive(
            identity,
            mesh,
            schedule,
            &metadata,
            &sources,
            &manifest_catalog,
            history.interface_nodes,
        )?;
        let catalog_bytes = manifest_catalog.reservation_bytes()?.0;
        let manifest_retained = ThermalManifestRetainedResourceFacts::derive(
            model,
            step,
            mesh,
            schedule,
            identity,
            &metadata,
            &sources,
            history.interface_nodes,
        )?;
        let retained_bytes = manifest_retained.reservation_bytes()?.0;
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
        .and_then(|bytes| bytes.checked_add(catalog_bytes))
        .and_then(|bytes| bytes.checked_add(manifest_metadata.reservation_bytes().0))
        .and_then(|bytes| bytes.checked_add(retained_bytes))
        .ok_or(ResourcePlanningError::Overflow(
            "thermal field construction reservation",
        ))?;
        let maximum = step.accuracy.resources.resident_byte_budget();
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
            admitted_identity: identity.clone(),
            properties,
            property_compilation_bytes,
            history,
            operators,
            sources,
            observations,
            realization,
            metadata,
            field_construction,
            manifest_catalog,
            manifest_metadata,
            manifest_retained,
            model_materialization_bytes,
        })
    }

    pub fn reservation_bytes(&self) -> Result<ByteCount, ResourcePlanningError> {
        let catalog_bytes = self.manifest_catalog.reservation_bytes()?.0;
        let retained_bytes = self.manifest_retained.reservation_bytes()?.0;
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
        .and_then(|bytes| bytes.checked_add(catalog_bytes))
        .and_then(|bytes| bytes.checked_add(self.manifest_metadata.reservation_bytes().0))
        .and_then(|bytes| bytes.checked_add(retained_bytes))
        .map(ByteCount)
        .ok_or(ResourcePlanningError::Overflow(
            "thermal preassembly reservation",
        ))
    }

    pub(crate) fn require_fits(&self, maximum: u64) -> Result<(), ResourcePlanningError> {
        require_ceiling(self.reservation_bytes()?.0, maximum)
    }
}

/// The proving factorization when its reservation fits the run's resident budget; otherwise the
/// conduction multigrid path for steady linear conduction (no radiation), whose reservation must
/// fit the same budget. The budget is the step's resident ceiling, which a study run sets to
/// min(authored, the memory the run is admitted to use), so the plan and every re-derivation
/// select the same path.
pub(crate) fn select_thermal_solve_path(
    operators: ThermalOperatorResourceFacts,
    history: &ThermalHistoryResourceFacts,
    model: &AnalysisModel,
    step: &ProcedureStep,
) -> Result<ThermalOperatorResourceFacts, ResourcePlanningError> {
    let maximum = step.accuracy.resources.resident_byte_budget();
    let factorization = operators.numerical_reservation_bytes(history)?.0;
    if factorization <= maximum {
        return Ok(operators);
    }
    // A steady thermal step is static; transient conduction keeps the factor its time steps reuse.
    let multigrid_admitted = step.procedure == runmat_analysis_core::AnalysisProcedure::Static
        && !model.boundaries.iter().any(|boundary| {
            matches!(
                boundary.operator,
                runmat_analysis_core::BoundaryOperator::ThermalRadiation { .. }
            )
        });
    if !multigrid_admitted {
        return Err(ResourcePlanningError::ThermalSolveCeiling {
            factorization_resident_bytes: factorization,
            multigrid_resident_bytes: None,
            maximum,
        });
    }
    let multigrid = operators.on_path(super::ThermalSolvePath::ConductionMultigrid);
    let multigrid_bytes = multigrid.numerical_reservation_bytes(history)?.0;
    if multigrid_bytes > maximum {
        return Err(ResourcePlanningError::ThermalSolveCeiling {
            factorization_resident_bytes: factorization,
            multigrid_resident_bytes: Some(multigrid_bytes),
            maximum,
        });
    }
    Ok(multigrid)
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
