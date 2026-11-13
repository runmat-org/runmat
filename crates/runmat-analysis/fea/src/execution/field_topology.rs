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
    SampledVolumeMesh {
        mesh_digest: StableDigest,
        sampling_digest: Digest,
    },
    FiniteVolumeMesh {
        mesh_digest: StableDigest,
        geometry_digest: Digest,
    },
    Structural {
        topology: CanonicalStructuralTopologyIdentity,
    },
}

impl FieldTopologyIdentity {
    pub fn discretization_digest(self) -> Digest {
        match self {
            Self::VolumeMesh { mesh_digest } => Digest::from_bytes(*mesh_digest.bytes()),
            Self::SampledVolumeMesh {
                mesh_digest,
                sampling_digest,
            } => {
                let mut bytes = b"runmat-fea-sampled-volume-field-topology/v1\0".to_vec();
                bytes.extend_from_slice(mesh_digest.bytes());
                bytes.extend_from_slice(sampling_digest.bytes());
                Digest::sha256(bytes)
            }
            Self::Structural { topology } => topology.discretization_digest(),
            Self::FiniteVolumeMesh {
                mesh_digest,
                geometry_digest,
            } => {
                let mut bytes = b"runmat-fea-finite-volume-field-topology/v1\0".to_vec();
                bytes.extend_from_slice(mesh_digest.bytes());
                bytes.extend_from_slice(geometry_digest.bytes());
                Digest::sha256(bytes)
            }
        }
    }

    pub fn has_valid_identity(self) -> bool {
        match self {
            Self::VolumeMesh { mesh_digest } => {
                mesh_digest != StableDigest::ZERO
                    && self.discretization_digest() != Digest::sha256([])
            }
            Self::SampledVolumeMesh {
                mesh_digest,
                sampling_digest,
            } => {
                mesh_digest != StableDigest::ZERO
                    && Digest::from_bytes(*mesh_digest.bytes()) != Digest::sha256([])
                    && sampling_digest != Digest::from_bytes([0; 32])
                    && sampling_digest != Digest::sha256([])
            }
            Self::Structural { topology } => topology.has_valid_identity(),
            Self::FiniteVolumeMesh {
                mesh_digest,
                geometry_digest,
            } => {
                mesh_digest != StableDigest::ZERO
                    && Digest::from_bytes(*mesh_digest.bytes()) != Digest::sha256([])
                    && geometry_digest != Digest::from_bytes([0; 32])
                    && geometry_digest != Digest::sha256([])
            }
        }
    }

    pub fn require_structural(
        self,
    ) -> AnalysisExecutionResult<CanonicalStructuralTopologyIdentity> {
        match self {
            Self::Structural { topology } => Ok(topology),
            Self::VolumeMesh { .. }
            | Self::SampledVolumeMesh { .. }
            | Self::FiniteVolumeMesh { .. } => Err(AnalysisExecutionError::Invalid(
                "structural recovery requires structural topology authority".into(),
            )),
        }
    }
}

#[cfg(test)]
mod tests;
