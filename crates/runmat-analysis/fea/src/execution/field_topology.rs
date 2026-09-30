//! Storage topology authority for recovered fields, independent of consuming physics.
use runmat_execution::Digest;
use runmat_meshing_core::StableDigest;
use serde::{Deserialize, Serialize};

use super::{AnalysisExecutionError, AnalysisExecutionResult};
use crate::physics::structural::topology::CanonicalStructuralTopologyIdentity;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
pub enum FieldTopologyIdentity {
    VolumeMesh {
        mesh_digest: StableDigest,
    },
    Structural {
        topology: CanonicalStructuralTopologyIdentity,
    },
}

impl FieldTopologyIdentity {
    pub fn discretization_digest(self) -> Digest {
        match self {
            Self::VolumeMesh { mesh_digest } => Digest::from_bytes(*mesh_digest.bytes()),
            Self::Structural { topology } => topology.discretization_digest(),
        }
    }

    pub fn has_valid_identity(self) -> bool {
        match self {
            Self::VolumeMesh { mesh_digest } => {
                mesh_digest != StableDigest::ZERO
                    && self.discretization_digest() != Digest::sha256([])
            }
            Self::Structural { topology } => topology.has_valid_identity(),
        }
    }

    pub fn require_structural(
        self,
    ) -> AnalysisExecutionResult<CanonicalStructuralTopologyIdentity> {
        match self {
            Self::Structural { topology } => Ok(topology),
            Self::VolumeMesh { .. } => Err(AnalysisExecutionError::Invalid(
                "structural recovery requires structural topology authority".into(),
            )),
        }
    }
}

#[cfg(test)]
mod tests;
