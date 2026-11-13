//! Physics-neutral selection over native nodal and element fields retained on the canonical
//! solver mesh (for example acoustic pressure, fluid cell velocity or face volume flux). Every sample of a
//! required definition is selected; fan-out chooses the evaluated sample.
use super::*;
use crate::results::field::{RecoveredFieldDerivation, RecoveredFieldEntityDomain};
use runmat_analysis_core::RecoveryPolicy;

pub(crate) fn validate_volume_mesh_authority(
    request: &FieldRecoveryRequest,
    manifest: &RecoveredFieldArtifactManifest,
) -> AnalysisExecutionResult<()> {
    request.validate()?;
    manifest.validate().map_err(invalid)?;
    if request.source.solver_result().is_none()
        || !matches!(request.topology, FieldTopologyIdentity::VolumeMesh { .. })
        || manifest.recovery_request != request.identity()
        || manifest.source != request.source
        || manifest.topology_digest != request.topology.discretization_digest()
        || manifest.fields.is_empty()
    {
        return Err(invalid(
            "volume-mesh manifest differs from its original recovery request, source or topology",
        ));
    }
    for requirement in &request.plan.requirements {
        if requirement.recovery != RecoveryPolicy::Native
            || !matches!(
                requirement.entity_domain,
                RecoveredFieldEntityDomain::Nodes
                    | RecoveredFieldEntityDomain::VolumeElements
                    | RecoveredFieldEntityDomain::OrientedFaces
            )
        {
            return Err(invalid(
                "volume-mesh recovery admits native nodal, element or oriented-face quantities",
            ));
        }
    }
    if manifest
        .fields
        .iter()
        .any(|entry| entry.definition.derivation() != RecoveredFieldDerivation::Native)
    {
        return Err(invalid(
            "volume-mesh recovery cannot masquerade as transferred native values",
        ));
    }
    Ok(())
}

impl ObservationFieldSelection {
    /// `projection_digest` must be derived by the calling boundary from the authenticated mesh.
    pub fn volume_mesh_recovery(
        request: &FieldRecoveryRequest,
        manifest: &RecoveredFieldArtifactManifest,
        projection_digest: Digest,
        selected: Vec<RecoveredFieldIdentity>,
    ) -> AnalysisExecutionResult<Self> {
        validate_volume_mesh_authority(request, manifest)?;
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
            ObservationFieldSource::VolumeMeshRecovery {
                recovery_request: request.identity(),
                artifact: manifest.identity,
                numerical_source: request.source,
                topology: request.topology,
                projection_digest,
            },
            fields,
        )
    }

    pub fn validate_volume_mesh(
        &self,
        request: &FieldRecoveryRequest,
        manifest: &RecoveredFieldArtifactManifest,
        projection_digest: Digest,
    ) -> AnalysisExecutionResult<()> {
        let expected = Self::volume_mesh_recovery(
            request,
            manifest,
            projection_digest,
            self.fields.iter().map(|entry| entry.field).collect(),
        )?;
        if self != &expected {
            return Err(invalid(
                "selection differs from original volume-mesh recovery authority",
            ));
        }
        Ok(())
    }
}
