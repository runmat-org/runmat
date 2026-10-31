//! Exact native cell/all-face storage authority, independent of the fluid equations.
use super::{RecoveredFieldEntityCounts, RecoveredFieldError};
use crate::{
    execution::FieldTopologyIdentity,
    topology::{AnalysisProjection, FiniteVolumeTopology},
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::StableDigest;
use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests;
mod validation;

const PREFIX: &[u8] = b"runmat-finite-volume-field-layout/v1\0";
/// Per-entity topology of any admitted mesh: collections are bounded by the byte ceiling.
/// Layout size follows the admitted mesh (its plan and the object inventory admit it), so the
/// codec bounds it only by the wire range.
const LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeOrientedFace {
    pub identity: StableDigest,
    pub corner_identities: [StableDigest; 3],
    /// Positive flux leaves this exact original cell.
    pub owner: StableDigest,
    pub neighbor: Option<StableDigest>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FiniteVolumeFieldTopology {
    mesh_digest: StableDigest,
    projection_digest: Digest,
    geometry_digest: Digest,
    node_identities: Vec<StableDigest>,
    cell_identities: Vec<StableDigest>,
    oriented_faces: Vec<NativeOrientedFace>,
}

fn invalid(e: impl std::fmt::Display) -> RecoveredFieldError {
    RecoveredFieldError::InvalidArtifact(e.to_string())
}

fn face_identity(
    corners: [StableDigest; 3],
    owner: StableDigest,
    neighbor: Option<StableDigest>,
) -> Result<StableDigest, RecoveredFieldError> {
    let digest = crate::contracts::canonical_digest::canonical_contract_digest(
        PREFIX,
        "owner-oriented-face",
        &(corners, owner, neighbor),
        LIMITS,
    )
    .map_err(invalid)?;
    Ok(StableDigest::from_bytes(*digest.bytes()))
}

impl FiniteVolumeFieldTopology {
    pub fn derive(
        projection: &AnalysisProjection,
        topology: &FiniteVolumeTopology,
    ) -> Result<Self, RecoveredFieldError> {
        topology.validate_for(projection).map_err(invalid)?;
        Self::from_derived(projection, topology)
    }

    /// The layout of `topology`, which the caller derived from `projection` in this process
    /// (`FiniteVolumeTopology::derive*`), so it is not re-derived to check it. Face identities
    /// are computed here once; validation checks the inventory without hashing them again.
    pub(crate) fn from_derived(
        projection: &AnalysisProjection,
        topology: &FiniteVolumeTopology,
    ) -> Result<Self, RecoveredFieldError> {
        let mut faces = Vec::with_capacity(topology.faces.len());
        for face in &topology.faces {
            let owner = topology.cells[face.owner].element_identity;
            let neighbor = face.neighbor.map(|i| topology.cells[i].element_identity);
            faces.push(NativeOrientedFace {
                identity: face_identity(face.corner_identities, owner, neighbor)?,
                corner_identities: face.corner_identities,
                owner,
                neighbor,
            });
        }
        let value = Self {
            mesh_digest: topology.mesh_digest,
            projection_digest: topology.projection_digest,
            geometry_digest: topology.geometry_digest,
            node_identities: projection.nodes.iter().map(|n| n.stable_identity).collect(),
            cell_identities: topology.cells.iter().map(|c| c.element_identity).collect(),
            oriented_faces: faces,
        };
        validation::validate(&value, validation::FaceIdentities::Computed)?;
        value.entity_counts().validate()?;
        Ok(value)
    }

    pub fn identity(&self) -> FieldTopologyIdentity {
        FieldTopologyIdentity::FiniteVolumeMesh {
            mesh_digest: self.mesh_digest,
            geometry_digest: self.geometry_digest,
        }
    }
    pub fn entity_counts(&self) -> RecoveredFieldEntityCounts {
        RecoveredFieldEntityCounts {
            nodes: self.node_identities.len(),
            volume_elements: self.cell_identities.len(),
            oriented_faces: self.oriented_faces.len(),
            volume_integration_points: 0,
            oriented_edges: 0,
            beam_elements: 0,
            shell_elements: 0,
            connectors: 0,
            constraint_groups: 0,
        }
    }
    pub fn cell_identities(&self) -> &[StableDigest] {
        &self.cell_identities
    }
    pub fn oriented_faces(&self) -> &[NativeOrientedFace] {
        &self.oriented_faces
    }

    /// Internal wire integrity is not original-source admission.
    pub fn validate(&self) -> Result<(), RecoveredFieldError> {
        validation::validate(self, validation::FaceIdentities::Recompute)?;
        self.entity_counts().validate()
    }
    pub fn validate_for(
        &self,
        projection: &AnalysisProjection,
        topology: &FiniteVolumeTopology,
    ) -> Result<(), RecoveredFieldError> {
        // Equal to the freshly derived (valid) layout implies valid; no separate validate.
        if self != &Self::derive(projection, topology)? {
            return Err(RecoveredFieldError::ArtifactTopologyMismatch);
        }
        Ok(())
    }
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, RecoveredFieldError> {
        self.validate()?;
        encode_contract(PREFIX, "finite-volume-field-layout", self, LIMITS).map_err(invalid)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, RecoveredFieldError> {
        let value: Self = decode_contract(PREFIX, "finite-volume-field-layout", bytes, LIMITS)
            .map_err(invalid)?;
        value.validate()?;
        Ok(value)
    }
}
