//! Native nodal/element field catalog on the canonical solver mesh, without numerical arrays.
use super::*;
use crate::topology::AnalysisProjection;

impl ObservationFieldCatalog {
    /// `step` is the producing authored step (with its observations); `sequence` names its
    /// authored sequence. Both are bound to the recovery request's producer identity.
    pub fn volume_mesh_recovery(
        request: FieldRecoveryRequest,
        manifest: RecoveredFieldArtifactManifest,
        sequence: crate::execution::SequenceIdentity,
        step: runmat_analysis_core::ProcedureStep,
    ) -> AnalysisExecutionResult<Self> {
        Self::new(ObservationFieldCatalogSource::VolumeMeshRecovery {
            recovery_request: Box::new(request),
            manifest: Box::new(manifest),
            sequence,
            step: Box::new(step),
        })
    }

    /// The producing authored step of a volume-mesh catalog.
    pub fn volume_mesh_step(
        &self,
    ) -> AnalysisExecutionResult<&runmat_analysis_core::ProcedureStep> {
        match &self.source {
            ObservationFieldCatalogSource::VolumeMeshRecovery { step, .. } => Ok(step),
            _ => Err(invalid("expected native volume-mesh catalog")),
        }
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
            ..
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
            ..
        } = &self.source
        else {
            return Err(invalid("expected native volume-mesh catalog"));
        };
        selection.validate_volume_mesh(recovery_request, manifest, projection_digest)
    }

    /// Resolve this catalog's producing step on the exact mesh projection from the catalog's
    /// own field manifest. The plan is canonical for this catalog by construction (it is the
    /// resolution `validate_volume_mesh_plan` repeats for a plan from elsewhere), so the pair
    /// is sealed and no holder verifies it again. Returns the projection it was resolved on.
    pub fn resolve_volume_mesh_plan(
        self,
        mesh: &SolverMeshArtifact,
        limits: super::super::ObservationLimits,
    ) -> AnalysisExecutionResult<(AnalysisProjection, VolumeMeshObservationAuthority)> {
        self.validate()?;
        let ObservationFieldCatalogSource::VolumeMeshRecovery {
            recovery_request,
            manifest,
            step,
            ..
        } = &self.source
        else {
            return Err(invalid("expected native volume-mesh catalog"));
        };
        if !matches!(
            recovery_request.topology,
            crate::execution::FieldTopologyIdentity::VolumeMesh { mesh_digest }
                if mesh_digest == mesh.canonical_digest
        ) {
            return Err(invalid(
                "volume-mesh catalog differs from the canonical mesh",
            ));
        }
        let projection = AnalysisProjection::derive(mesh).map_err(invalid)?;
        let mut seen = std::collections::BTreeSet::new();
        let definitions = manifest
            .fields
            .iter()
            .filter(|entry| seen.insert(entry.definition.identity()))
            .map(|entry| entry.definition.clone())
            .collect::<Vec<_>>();
        let plan = super::super::resolve_observation_plan(step, &projection, &definitions, limits)
            .map_err(invalid)?;
        let mesh_digest = mesh.canonical_digest;
        Ok((
            projection,
            VolumeMeshObservationAuthority {
                catalog: self,
                plan,
                mesh_digest,
            },
        ))
    }

    /// Independently re-resolve the authored plan on the exact mesh projection.
    pub fn validate_volume_mesh_plan(
        &self,
        plan: &ResolvedObservationPlan,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        let ObservationFieldCatalogSource::VolumeMeshRecovery {
            recovery_request,
            manifest,
            step,
            ..
        } = &self.source
        else {
            return Err(invalid("expected native volume-mesh catalog"));
        };
        if !matches!(
            recovery_request.topology,
            crate::execution::FieldTopologyIdentity::VolumeMesh { mesh_digest }
                if mesh_digest == mesh.canonical_digest
        ) {
            return Err(invalid(
                "volume-mesh catalog differs from the canonical mesh",
            ));
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

/// A volume-mesh field catalog and the plan resolved from its own step and manifest on one
/// canonical mesh. Only [`ObservationFieldCatalog::resolve_volume_mesh_plan`] pairs them, so
/// holders carry the pair instead of re-resolving the plan to check it against the catalog.
#[derive(Clone, Debug, PartialEq)]
pub struct VolumeMeshObservationAuthority {
    catalog: ObservationFieldCatalog,
    plan: ResolvedObservationPlan,
    mesh_digest: runmat_meshing_core::StableDigest,
}

impl VolumeMeshObservationAuthority {
    pub fn catalog(&self) -> &ObservationFieldCatalog {
        &self.catalog
    }

    pub fn plan(&self) -> &ResolvedObservationPlan {
        &self.plan
    }

    /// Whether the plan was resolved on exactly this mesh.
    pub(in crate::results::observation) fn resolved_on(&self, mesh: &SolverMeshArtifact) -> bool {
        self.mesh_digest == mesh.canonical_digest
    }

    pub(in crate::results::observation) fn into_parts(
        self,
    ) -> (ObservationFieldCatalog, ResolvedObservationPlan) {
        (self.catalog, self.plan)
    }
}

#[cfg(test)]
mod tests;
