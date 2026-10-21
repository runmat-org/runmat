//! Authenticated native cell/all-face catalog authority, without numerical arrays.
use super::*;
use crate::{
    execution::FluidSteadyWorkload, results::field::FiniteVolumeFieldTopology,
    topology::AnalysisProjection,
};

impl ObservationFieldCatalog {
    pub fn finite_volume_recovery(
        request: FieldRecoveryRequest,
        manifest: RecoveredFieldArtifactManifest,
        layout: FiniteVolumeFieldTopology,
    ) -> AnalysisExecutionResult<Self> {
        Self::new(ObservationFieldCatalogSource::FiniteVolumeRecovery {
            recovery_request: Box::new(request),
            manifest: Box::new(manifest),
            layout: Box::new(layout),
        })
    }
    pub fn validate_finite_volume_selection(
        &self,
        selection: &ObservationFieldSelection,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        let ObservationFieldCatalogSource::FiniteVolumeRecovery {
            recovery_request,
            manifest,
            layout,
        } = &self.source
        else {
            return Err(invalid("expected native finite-volume catalog"));
        };
        selection.validate_finite_volume(recovery_request, manifest, layout)
    }
    /// Independently rederive exact original geometry and authored native observation targets.
    pub fn validate_finite_volume_plan(
        &self,
        plan: &ResolvedObservationPlan,
        workload: &FluidSteadyWorkload,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        workload.validate()?;
        let ObservationFieldCatalogSource::FiniteVolumeRecovery {
            recovery_request,
            manifest,
            layout,
        } = &self.source
        else {
            return Err(invalid("expected native finite-volume catalog"));
        };
        if recovery_request.producer != workload.producer_identity()? {
            return Err(invalid(
                "finite-volume catalog differs from original authored producer",
            ));
        }
        let definitions = crate::execution::fluid_field_definition_inventory(workload.coupling())?;
        if manifest
            .fields
            .iter()
            .map(|entry| &entry.definition)
            .ne(definitions.iter())
        {
            return Err(invalid(
                "finite-volume catalog differs from original physical U/p/q inventory",
            ));
        }
        let projection = AnalysisProjection::derive(mesh).map_err(invalid)?;
        let geometry = workload.fluid_cell_complex(mesh, &projection)?;
        layout
            .validate_for(&projection, &geometry)
            .map_err(invalid)?;
        let [step] = workload.model.sequence.steps.as_slice() else {
            return Err(invalid("finite-volume authored step absent"));
        };
        super::super::finite_volume::verify_finite_volume_observation_plan(
            step,
            layout,
            &projection,
            &geometry,
            &definitions,
            plan,
            super::super::ObservationLimits::default(),
        )
        .map_err(invalid)
    }
}
