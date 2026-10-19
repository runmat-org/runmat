//! Physics-neutral selection over an authenticated native finite-volume-field artifact.
use super::*;
use crate::results::field::{
    FiniteVolumeFieldTopology, RecoveredFieldDerivation, RecoveredFieldEntityDomain,
    RecoveredFieldSample,
};
use runmat_analysis_core::RecoveryPolicy;

pub(crate) fn validate_finite_volume_authority(
    request: &FieldRecoveryRequest,
    manifest: &RecoveredFieldArtifactManifest,
    layout: &FiniteVolumeFieldTopology,
) -> AnalysisExecutionResult<()> {
    request.validate()?;
    manifest.validate().map_err(invalid)?;
    layout.validate().map_err(invalid)?;
    if request.source.solver_result().is_none()
        || request.topology != layout.identity()
        || manifest.recovery_request != request.identity()
        || manifest.source != request.source
        || manifest.topology_digest != request.topology.discretization_digest()
        || manifest.entity_counts != layout.entity_counts()
        || manifest.fields.len() != request.plan.requirements.len()
        || manifest
            .fields
            .iter()
            .any(|field| field.sample != RecoveredFieldSample::Final)
    {
        return Err(invalid("finite-volume manifest differs from original recovery request, source, topology or final coordinate"));
    }
    request
        .plan
        .validate_definitions(manifest.fields.iter().map(|field| field.definition.clone()))?;
    for requirement in &request.plan.requirements {
        let count = match requirement.entity_domain {
            RecoveredFieldEntityDomain::VolumeElements => layout.cell_identities().len(),
            RecoveredFieldEntityDomain::OrientedFaces => layout.oriented_faces().len(),
            _ => {
                return Err(invalid(
                    "finite-volume native quantity requires cells or all oriented faces",
                ))
            }
        };
        let count = u64::try_from(count).map_err(invalid)?;
        let components = requirement
            .value_shape
            .component_count()
            .ok_or_else(|| invalid("finite-volume native component shape absent"))?
            as u64;
        if requirement.recovery != RecoveryPolicy::Native
            || requirement.cost.source_entity_count != count
            || requirement.cost.output_entity_count != count
            || requirement.cost.topology_incidence_count != 0
            || requirement.cost.arithmetic_operation_count
                != count
                    .checked_mul(components)
                    .ok_or_else(|| invalid("finite-volume native copy count overflow"))?
        {
            return Err(invalid(
                "finite-volume native recovery cost differs from retained sample cardinality",
            ));
        }
    }
    if manifest
        .fields
        .iter()
        .any(|entry| entry.definition.derivation() != RecoveredFieldDerivation::Native)
    {
        return Err(invalid(
            "finite-volume recovery cannot masquerade as transferred native values",
        ));
    }
    Ok(())
}

impl ObservationFieldSelection {
    pub fn finite_volume_recovery(
        request: &FieldRecoveryRequest,
        manifest: &RecoveredFieldArtifactManifest,
        layout: &FiniteVolumeFieldTopology,
        selected: Vec<RecoveredFieldIdentity>,
    ) -> AnalysisExecutionResult<Self> {
        validate_finite_volume_authority(request, manifest, layout)?;
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
            ObservationFieldSource::FiniteVolumeRecovery {
                recovery_request: request.identity(),
                artifact: manifest.identity,
                numerical_source: request.source,
                topology: request.topology,
            },
            fields,
        )
    }

    pub fn validate_finite_volume(
        &self,
        request: &FieldRecoveryRequest,
        manifest: &RecoveredFieldArtifactManifest,
        layout: &FiniteVolumeFieldTopology,
    ) -> AnalysisExecutionResult<()> {
        let expected = Self::finite_volume_recovery(
            request,
            manifest,
            layout,
            self.fields.iter().map(|entry| entry.field).collect(),
        )?;
        if self != &expected {
            return Err(invalid(
                "selection differs from original finite-volume recovery authority",
            ));
        }
        Ok(())
    }
}
