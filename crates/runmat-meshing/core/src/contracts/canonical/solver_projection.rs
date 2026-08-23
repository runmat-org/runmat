use serde::{Deserialize, Serialize};

use super::{
    CanonicalMeshingContract, GeometryRevisionRef, MeshingContractError, MeshingRequest,
    SolverMeshArtifact, SolverMeshTopology, StableDigest, ANALYSIS_MESH_ARTIFACT_SCHEMA_VERSION,
};

pub const SOLVER_MESH_PROJECTION_SCHEMA_VERSION: u16 = 1;
pub const SOLVER_MESH_VALIDATION_SCHEMA_VERSION: u16 = 1;

/// Canonical solver topology before the final validation manifest is known.
///
/// Keeping this projection separate avoids a content-addressing cycle: the final artifact may
/// point at the validation-stage manifest, while that manifest closes over this projection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverMeshProjection {
    pub schema_version: u16,
    pub geometry: GeometryRevisionRef,
    pub resolved_request: MeshingRequest,
    pub topology: SolverMeshTopology,
}

impl SolverMeshProjection {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        if self.schema_version != SOLVER_MESH_PROJECTION_SCHEMA_VERSION {
            return Err(MeshingContractError::invalid(
                "solver mesh projection schema version",
                format!("expected {SOLVER_MESH_PROJECTION_SCHEMA_VERSION}"),
            ));
        }
        self.geometry.validate()?;
        self.resolved_request.validate()?;
        self.topology.validate(&self.resolved_request)
    }

    pub fn into_artifact(
        self,
        validation_manifest_digest: StableDigest,
    ) -> Result<SolverMeshArtifact, MeshingContractError> {
        validation_manifest_digest.validate_nonzero("solver mesh validation manifest digest")?;
        let mut artifact = SolverMeshArtifact {
            schema_version: ANALYSIS_MESH_ARTIFACT_SCHEMA_VERSION,
            canonical_digest: StableDigest::ZERO,
            root_stage_manifest_digest: validation_manifest_digest,
            geometry: self.geometry,
            resolved_request: self.resolved_request,
            topology: self.topology,
        };
        artifact.seal_canonical_digest()?;
        Ok(artifact)
    }
}

/// Independently checked receipt that binds the validated topology and its factual inventory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverMeshValidation {
    pub schema_version: u16,
    pub projection_digest: StableDigest,
    pub geometry: GeometryRevisionRef,
    pub resolved_request_digest: StableDigest,
    pub node_count: u64,
    pub volume_element_count: u64,
    pub boundary_face_count: u64,
    pub boundary_edge_count: u64,
}

impl SolverMeshValidation {
    pub fn from_validated_projection(
        projection: &SolverMeshProjection,
    ) -> Result<Self, MeshingContractError> {
        projection.validate()?;
        Ok(Self {
            schema_version: SOLVER_MESH_VALIDATION_SCHEMA_VERSION,
            projection_digest: projection.canonical_digest()?,
            geometry: projection.geometry.clone(),
            resolved_request_digest: projection.resolved_request.canonical_digest()?,
            node_count: projection.topology.nodes.len() as u64,
            volume_element_count: projection.topology.volume_elements.len() as u64,
            boundary_face_count: projection.topology.boundary_faces.len() as u64,
            boundary_edge_count: projection.topology.boundary_edges.len() as u64,
        })
    }

    pub fn validate_against(
        &self,
        projection: &SolverMeshProjection,
    ) -> Result<(), MeshingContractError> {
        self.validate()?;
        projection.validate()?;
        if *self != Self::from_validated_projection(projection)? {
            return Err(MeshingContractError::invalid(
                "solver mesh validation",
                "receipt does not match the independently validated projection",
            ));
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<(), MeshingContractError> {
        if self.schema_version != SOLVER_MESH_VALIDATION_SCHEMA_VERSION {
            return Err(MeshingContractError::invalid(
                "solver mesh validation schema version",
                format!("expected {SOLVER_MESH_VALIDATION_SCHEMA_VERSION}"),
            ));
        }
        self.projection_digest
            .validate_nonzero("validated solver projection digest")?;
        self.geometry.validate()?;
        self.resolved_request_digest
            .validate_nonzero("validated solver request digest")?;
        if self.node_count == 0 || self.volume_element_count == 0 {
            return Err(MeshingContractError::invalid(
                "solver mesh validation inventory",
                "validated solver topology must contain nodes and volume elements",
            ));
        }
        Ok(())
    }
}
