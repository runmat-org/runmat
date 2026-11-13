//! Original sampled request/layout authority; numerical arrays remain in field partitions.
use super::*;
use crate::execution::StepIdentity;
use crate::results::field::SampledVolumeFieldTopology;

#[cfg(test)]
mod tests;

impl ObservationFieldCatalog {
    pub fn sampled_recovery(
        recovery_request: FieldRecoveryRequest,
        manifest: RecoveredFieldArtifactManifest,
        sampling: SampledVolumeFieldTopology,
    ) -> AnalysisExecutionResult<Self> {
        Self::new(ObservationFieldCatalogSource::SampledRecovery {
            recovery_request: Box::new(recovery_request),
            manifest: Box::new(manifest),
            sampling: Box::new(sampling),
        })
    }

    pub fn validate_sampled_selection(
        &self,
        selection: &ObservationFieldSelection,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        let ObservationFieldCatalogSource::SampledRecovery {
            recovery_request,
            manifest,
            sampling,
        } = &self.source
        else {
            return Err(invalid("expected sampled recovery catalog"));
        };
        selection.validate_sampled(recovery_request, manifest, sampling)
    }

    /// Producer must be independently derived from the original numerical workload.
    /// This generic layer authenticates native storage layout and authored selections;
    /// physics adapters additionally authenticate original intent and retained solve lineage.
    pub fn validate_sampled_plan(
        &self,
        plan: &ResolvedObservationPlan,
        step: &runmat_analysis_core::ProcedureStep,
        producer: StepIdentity,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        let ObservationFieldCatalogSource::SampledRecovery {
            recovery_request,
            manifest,
            sampling,
        } = &self.source
        else {
            return Err(invalid("expected sampled recovery catalog"));
        };
        if recovery_request.producer != producer {
            return Err(invalid(
                "sampled recovery differs from original authored producer",
            ));
        }
        let definitions = manifest
            .fields
            .iter()
            .map(|entry| entry.definition.clone())
            .collect::<Vec<_>>();
        super::super::resolve::verify_sampled_observation_plan(
            step,
            sampling,
            mesh,
            &definitions,
            plan,
        )
        .map_err(invalid)
    }
}
