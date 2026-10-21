//! Native nodal/element field catalog on the canonical solver mesh, without numerical arrays.
use super::*;
use crate::topology::AnalysisProjection;

impl ObservationFieldCatalog {
    pub fn volume_mesh_recovery(
        request: FieldRecoveryRequest,
        manifest: RecoveredFieldArtifactManifest,
    ) -> AnalysisExecutionResult<Self> {
        Self::new(ObservationFieldCatalogSource::VolumeMeshRecovery {
            recovery_request: Box::new(request),
            manifest: Box::new(manifest),
        })
    }

    pub fn validate_volume_mesh_selection(
        &self,
        selection: &ObservationFieldSelection,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        let ObservationFieldCatalogSource::VolumeMeshRecovery {
            recovery_request,
            manifest,
        } = &self.source
        else {
            return Err(invalid("expected native volume-mesh catalog"));
        };
        let projection = AnalysisProjection::derive(mesh).map_err(invalid)?;
        selection.validate_volume_mesh(recovery_request, manifest, projection.projection_digest)
    }

    /// Selection metadata against a plan whose projection digest was already authenticated.
    pub fn validate_volume_mesh_selection_metadata(
        &self,
        selection: &ObservationFieldSelection,
        projection_digest: Digest,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        let ObservationFieldCatalogSource::VolumeMeshRecovery {
            recovery_request,
            manifest,
        } = &self.source
        else {
            return Err(invalid("expected native volume-mesh catalog"));
        };
        selection.validate_volume_mesh(recovery_request, manifest, projection_digest)
    }

    /// Independently re-resolve the authored plan on the exact mesh projection.
    pub fn validate_volume_mesh_plan(
        &self,
        plan: &ResolvedObservationPlan,
        step: &runmat_analysis_core::ProcedureStep,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        let ObservationFieldCatalogSource::VolumeMeshRecovery {
            recovery_request,
            manifest,
        } = &self.source
        else {
            return Err(invalid("expected native volume-mesh catalog"));
        };
        if !matches!(
            recovery_request.topology,
            crate::execution::FieldTopologyIdentity::VolumeMesh { mesh_digest }
                if mesh_digest == mesh.canonical_digest
        ) {
            return Err(invalid("volume-mesh catalog differs from the canonical mesh"));
        }
        verify_authored_observations(step, plan).map_err(invalid)?;
        let projection = AnalysisProjection::derive(mesh).map_err(invalid)?;
        let mut seen = std::collections::BTreeSet::new();
        let definitions = manifest
            .fields
            .iter()
            .filter(|entry| seen.insert(entry.definition.identity()))
            .map(|entry| &entry.definition)
            .collect::<Vec<_>>();
        super::super::resolve::verify_observation_plan(
            step,
            &projection,
            definitions.into_iter(),
            plan,
        )
            .map_err(invalid)
    }
}
