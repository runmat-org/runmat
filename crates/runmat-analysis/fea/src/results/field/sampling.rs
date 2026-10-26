//! Physics-independent authority for canonical volume field storage layout.
//! Sample rules name geometry/quadrature only; they do not select a solver.
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::{SolverMeshArtifact, StableDigest};
use serde::{Deserialize, Serialize};

use super::{RecoveredFieldEntityCounts, RecoveredFieldError};
use crate::execution::FieldTopologyIdentity;

mod derive;
#[cfg(test)]
mod tests;
mod validation;

const PREFIX: &[u8] = b"runmat-sampled-volume-field-topology/v1\0";
/// Per-entity topology of any admitted mesh: collections are bounded by the byte ceiling.
const LIMITS: CanonicalLimits = CanonicalLimits::content_bounded(1024 * 1024 * 1024, 65_536, 64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VolumeFieldSamplingRule {
    TetrahedronFourPoint,
    TetrahedronKeastFifteen,
}
impl VolumeFieldSamplingRule {
    pub const fn points_per_element(self) -> usize {
        match self {
            Self::TetrahedronFourPoint => 4,
            Self::TetrahedronKeastFifteen => 15,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampledOrientedEdge {
    pub identity: StableDigest,
    /// Canonical direction: lesser persistent node identity to greater identity.
    pub node_identities: [StableDigest; 2],
}
impl SampledOrientedEdge {
    pub fn orientation_sign(&self, endpoints: [StableDigest; 2]) -> Option<i8> {
        if endpoints == self.node_identities {
            Some(1)
        } else if endpoints == [self.node_identities[1], self.node_identities[0]] {
            Some(-1)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VolumeFieldIntegrationSample {
    pub identity: StableDigest,
    pub element_identity: StableDigest,
    /// Zero-based ordinal in the named shared tetrahedral rule.
    pub ordinal: usize,
    pub position_m: [f64; 3],
    pub volume_weight_m3: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SamplingData {
    schema_version: u16,
    mesh_digest: StableDigest,
    projection_digest: Digest,
    rule: VolumeFieldSamplingRule,
    node_identities: Vec<StableDigest>,
    element_identities: Vec<StableDigest>,
    element_corner_identities: Vec<[StableDigest; 4]>,
    oriented_edges: Vec<SampledOrientedEdge>,
    integration_samples: Vec<VolumeFieldIntegrationSample>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "SamplingWire")]
pub struct SampledVolumeFieldTopology {
    data: SamplingData,
    sampling_digest: Digest,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SamplingWire {
    data: SamplingData,
    sampling_digest: Digest,
}
impl TryFrom<SamplingWire> for SampledVolumeFieldTopology {
    type Error = RecoveredFieldError;
    fn try_from(wire: SamplingWire) -> Result<Self, Self::Error> {
        let value = Self {
            data: wire.data,
            sampling_digest: wire.sampling_digest,
        };
        value.validate()?;
        Ok(value)
    }
}

impl SampledVolumeFieldTopology {
    pub fn identity(&self) -> Result<FieldTopologyIdentity, RecoveredFieldError> {
        self.validate()?;
        Ok(FieldTopologyIdentity::SampledVolumeMesh {
            mesh_digest: self.data.mesh_digest,
            sampling_digest: self.sampling_digest,
        })
    }
    /// Original canonical mesh digest, not the composite sampled layout digest.
    pub fn mesh_digest(&self) -> Digest {
        Digest::from_bytes(*self.data.mesh_digest.bytes())
    }
    pub fn projection_digest(&self) -> Digest {
        self.data.projection_digest
    }
    pub fn sampling_digest(&self) -> Digest {
        self.sampling_digest
    }
    pub fn rule(&self) -> VolumeFieldSamplingRule {
        self.data.rule
    }
    pub fn node_identities(&self) -> &[StableDigest] {
        &self.data.node_identities
    }
    pub fn element_identities(&self) -> &[StableDigest] {
        &self.data.element_identities
    }
    pub fn oriented_edges(&self) -> &[SampledOrientedEdge] {
        &self.data.oriented_edges
    }
    pub fn integration_samples(&self) -> &[VolumeFieldIntegrationSample] {
        &self.data.integration_samples
    }
    pub fn entity_counts(&self) -> RecoveredFieldEntityCounts {
        RecoveredFieldEntityCounts {
            oriented_faces: 0,
            nodes: self.data.node_identities.len(),
            volume_elements: self.data.element_identities.len(),
            volume_integration_points: self.data.integration_samples.len(),
            oriented_edges: self.data.oriented_edges.len(),
            beam_elements: 0,
            shell_elements: 0,
            connectors: 0,
            constraint_groups: 0,
        }
    }
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, RecoveredFieldError> {
        self.validate()?;
        encode("sampled-volume-field-topology", self)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, RecoveredFieldError> {
        decode_contract(PREFIX, "sampled-volume-field-topology", bytes, LIMITS)
            .map_err(|error| RecoveredFieldError::Encoding(error.to_string()))
    }
    /// Decode validation proves internal layout integrity. Source admission must
    /// additionally compare against the original mesh; no numerical field is recovered.
    pub fn validate_for_mesh(&self, mesh: &SolverMeshArtifact) -> Result<(), RecoveredFieldError> {
        self.validate()?;
        if self != &Self::derive(mesh, self.rule())? {
            return Err(invalid(
                "sampled field topology differs from canonical mesh",
            ));
        }
        Ok(())
    }
}

fn encode<T: Serialize>(label: &str, value: &T) -> Result<Vec<u8>, RecoveredFieldError> {
    encode_contract(PREFIX, label, value, LIMITS)
        .map_err(|error| RecoveredFieldError::Encoding(error.to_string()))
}
fn stable_identity<T: Serialize>(
    label: &str,
    value: &T,
) -> Result<StableDigest, RecoveredFieldError> {
    Ok(StableDigest::from_bytes(
        *Digest::sha256(encode(label, value)?).bytes(),
    ))
}
fn invalid(message: impl Into<String>) -> RecoveredFieldError {
    RecoveredFieldError::InvalidArtifact(message.into())
}
