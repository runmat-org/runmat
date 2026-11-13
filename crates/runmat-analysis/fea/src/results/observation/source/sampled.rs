//! Physics-neutral selection over an authenticated native sampled-field artifact.
use super::*;
use crate::results::field::{
    RecoveredFieldDerivation, RecoveredFieldEntityDomain, RecoveredFieldSample,
    SampledVolumeFieldTopology,
};
use runmat_analysis_core::{ObservationQuantity, RecoveryPolicy};

pub(crate) fn validate_sampled_authority(
    request: &FieldRecoveryRequest,
    manifest: &RecoveredFieldArtifactManifest,
    sampling: &SampledVolumeFieldTopology,
) -> AnalysisExecutionResult<()> {
    request.validate()?;
    manifest.validate().map_err(invalid)?;
    sampling.validate().map_err(invalid)?;
    if request.source.solver_result().is_none()
        || request.topology != sampling.identity().map_err(invalid)?
        || manifest.recovery_request != request.identity()
        || manifest.source != request.source
        || manifest.topology_digest != request.topology.discretization_digest()
        || manifest.entity_counts != sampling.entity_counts()
        || manifest.fields.len() != request.plan.requirements.len()
        || manifest
            .fields
            .iter()
            .any(|field| field.sample != RecoveredFieldSample::Final)
    {
        return Err(invalid("sampled manifest differs from original recovery request, source, topology or final coordinate"));
    }
    request
        .plan
        .validate_definitions(manifest.fields.iter().map(|field| field.definition.clone()))?;
    for requirement in &request.plan.requirements {
        let count = match requirement.entity_domain {
            RecoveredFieldEntityDomain::VolumeIntegrationPoints
                if !matches!(
                    requirement.quantity,
                    ObservationQuantity::MagneticEdgeCirculation
                        | ObservationQuantity::ElectricEdgeCirculation
                ) =>
            {
                sampling.integration_sample_count()
            }
            RecoveredFieldEntityDomain::OrientedEdges
                if matches!(
                    requirement.quantity,
                    ObservationQuantity::MagneticEdgeCirculation
                        | ObservationQuantity::ElectricEdgeCirculation
                ) =>
            {
                sampling.oriented_edges().len()
            }
            // Element averages of the sampled field over each element's own samples.
            RecoveredFieldEntityDomain::VolumeElements => sampling.element_identities().len(),
            _ => {
                return Err(invalid(
                    "sampled native quantity differs from its exact entity domain",
                ))
            }
        };
        let count = u64::try_from(count).map_err(invalid)?;
        if requirement.recovery != RecoveryPolicy::Native
            || requirement.cost.source_entity_count != count
            || requirement.cost.output_entity_count != count
            || requirement.cost.topology_incidence_count != 0
        {
            return Err(invalid(
                "sampled native recovery cost differs from retained sample cardinality",
            ));
        }
    }
    if manifest
        .fields
        .iter()
        .any(|entry| entry.definition.derivation() != RecoveredFieldDerivation::Native)
    {
        return Err(invalid(
            "sampled recovery cannot masquerade as transferred native values",
        ));
    }
    Ok(())
}

impl ObservationFieldSelection {
    pub fn sampled_recovery(
        request: &FieldRecoveryRequest,
        manifest: &RecoveredFieldArtifactManifest,
        sampling: &SampledVolumeFieldTopology,
        selected: Vec<RecoveredFieldIdentity>,
    ) -> AnalysisExecutionResult<Self> {
        validate_sampled_authority(request, manifest, sampling)?;
        let selection =
            RecoveredFieldSelection::new(manifest.clone(), selected).map_err(invalid)?;
        let fields = manifest
            .fields
            .iter()
            .filter(|entry| selection.fields.contains(&entry.field))
            .cloned()
            .collect();
        Self::new(
            request.producer,
            ObservationFieldSource::SampledRecovery {
                recovery_request: request.identity(),
                artifact: manifest.identity,
                numerical_source: request.source,
                topology: request.topology,
            },
            fields,
        )
    }

    pub fn validate_sampled(
        &self,
        request: &FieldRecoveryRequest,
        manifest: &RecoveredFieldArtifactManifest,
        sampling: &SampledVolumeFieldTopology,
    ) -> AnalysisExecutionResult<()> {
        let expected = Self::sampled_recovery(
            request,
            manifest,
            sampling,
            self.fields.iter().map(|entry| entry.field).collect(),
        )?;
        if self != &expected {
            return Err(invalid(
                "selection differs from original sampled recovery authority",
            ));
        }
        Ok(())
    }
}
