//! Physics-independent authority for canonical volume field storage layout.
//! Sample rules name geometry/quadrature only; they do not select a solver.
use runmat_canonical_codec::CanonicalLimits;
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
/// Format bounds of the layout identity. Its size grows with the admitted mesh, so it is only
/// ever digested from the streamed encoding, bounded by the wire range (see `digest`).
const FORMAT: CanonicalLimits = CanonicalLimits::new(0, 0, 65_536, 64);

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

/// One integration sample of the layout, materialized on demand. The stored layout names a
/// sample only by (element, rule point); its identity is derived from that pair and its
/// geometry from the canonical mesh and the named quadrature rule.
#[derive(Debug, Clone, PartialEq)]
pub struct VolumeFieldIntegrationSample {
    pub identity: StableDigest,
    pub element_identity: StableDigest,
    /// Zero-based ordinal in the named shared tetrahedral rule.
    pub ordinal: usize,
    pub position_m: [f64; 3],
    pub volume_weight_m3: f64,
}

/// Geometry of one integration sample, derived from the canonical mesh and the rule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VolumeFieldSampleGeometry {
    /// Canonical sample index: `element * points_per_element + ordinal`.
    pub index: usize,
    pub element: usize,
    pub ordinal: usize,
    pub position_m: [f64; 3],
    pub volume_weight_m3: f64,
}

/// The stored layout: identities once per node and element, element corners as indices into
/// the node inventory. Oriented edges follow from the corners, sample identities from
/// (rule, element, ordinal) and sample geometry from the mesh, so none is repeated.
/// Fields are declared in canonical (UTF-8 key) order so the streamed encoding holds none of
/// the mesh-sized collections back for sorting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SamplingData {
    element_corner_nodes: Vec<[u32; 4]>,
    element_identities: Vec<StableDigest>,
    mesh_digest: StableDigest,
    node_identities: Vec<StableDigest>,
    projection_digest: Digest,
    rule: VolumeFieldSamplingRule,
    schema_version: u16,
}
const SCHEMA_VERSION: u16 = 2;

/// `count` copies of one value, serialized as a sequence without allocating it: the wire
/// shape of a mesh-sized collection for encoded-size planning.
pub(crate) struct RepeatedShape<T> {
    value: T,
    count: usize,
}
impl<T> RepeatedShape<T> {
    pub(crate) fn new(value: T, count: usize) -> Self {
        Self { value, count }
    }
}
impl<T: Serialize> Serialize for RepeatedShape<T> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_seq(std::iter::repeat_n(&self.value, self.count))
    }
}
impl<T: runmat_types::physical::QuantityStorage> runmat_types::physical::QuantityStorage
    for RepeatedShape<T>
{
    fn all_finite(&self) -> bool {
        self.value.all_finite()
    }
}

/// Wire shape of a layout with a given inventory, for encoded-size planning. It serializes
/// exactly like `SampledVolumeFieldTopology`, with maximal index and digest spellings.
#[derive(Serialize)]
pub(crate) struct SampledTopologyShape {
    data: SampledTopologyShapeData,
    sampling_digest: Digest,
}
#[derive(Serialize)]
struct SampledTopologyShapeData {
    element_corner_nodes: RepeatedShape<[u32; 4]>,
    element_identities: RepeatedShape<StableDigest>,
    mesh_digest: StableDigest,
    node_identities: RepeatedShape<StableDigest>,
    projection_digest: Digest,
    rule: VolumeFieldSamplingRule,
    schema_version: u16,
}
impl SampledTopologyShape {
    pub(crate) fn new(nodes: usize, elements: usize, rule: VolumeFieldSamplingRule) -> Self {
        let stable = StableDigest::from_bytes([0xff; 32]);
        Self {
            data: SampledTopologyShapeData {
                element_corner_nodes: RepeatedShape::new([u32::MAX; 4], elements),
                element_identities: RepeatedShape::new(stable, elements),
                mesh_digest: stable,
                node_identities: RepeatedShape::new(stable, nodes),
                projection_digest: Digest::from_bytes([0xff; 32]),
                rule,
                schema_version: SCHEMA_VERSION,
            },
            sampling_digest: Digest::from_bytes([0xff; 32]),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SampledVolumeFieldTopology {
    data: SamplingData,
    sampling_digest: Digest,
    /// Derived from the element corners when the layout is admitted; never stored.
    #[serde(skip)]
    oriented_edges: Vec<SampledOrientedEdge>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SamplingWire {
    data: SamplingData,
    sampling_digest: Digest,
}
impl<'de> Deserialize<'de> for SampledVolumeFieldTopology {
    fn deserialize<D: serde::Deserializer<'de>>(decoder: D) -> Result<Self, D::Error> {
        let wire = SamplingWire::deserialize(decoder)?;
        Self::admit(wire.data, Some(wire.sampling_digest)).map_err(serde::de::Error::custom)
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
        &self.oriented_edges
    }
    /// Element corners as indices into `node_identities`.
    pub fn element_corner_nodes(&self) -> &[[u32; 4]] {
        &self.data.element_corner_nodes
    }
    pub fn integration_sample_count(&self) -> usize {
        self.data.element_identities.len() * self.data.rule.points_per_element()
    }
    /// Identity of the sample at canonical index `element * points_per_element + ordinal`,
    /// derived from (rule, element identity, ordinal).
    pub fn integration_sample_identity(
        &self,
        index: usize,
    ) -> Result<StableDigest, RecoveredFieldError> {
        let points = self.data.rule.points_per_element();
        let element = self
            .data
            .element_identities
            .get(index / points)
            .ok_or_else(|| invalid("integration sample index exceeds sampled topology"))?;
        sample_identity(self.data.rule, *element, index % points)
    }
    /// Visit every sample's geometry in canonical order, derived from `projection` and the
    /// named rule. The projection must be the one this layout was derived from.
    pub fn visit_integration_geometry(
        &self,
        projection: &crate::topology::AnalysisProjection,
        visit: impl FnMut(VolumeFieldSampleGeometry),
    ) -> Result<(), RecoveredFieldError> {
        if projection.projection_digest != self.data.projection_digest
            || projection.elements.len() != self.data.element_identities.len()
            || projection
                .elements
                .iter()
                .zip(&self.data.element_identities)
                .any(|(element, identity)| element.stable_identity != *identity)
        {
            return Err(invalid(
                "sample geometry projection differs from the sampled topology",
            ));
        }
        derive::visit_geometry(projection, self.data.rule, visit)
    }
    /// Every sample with identity and geometry, materialized. Mesh-sized: for small layouts and
    /// tests; production paths visit geometry and derive identities on demand.
    pub fn integration_samples_for(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> Result<Vec<VolumeFieldIntegrationSample>, RecoveredFieldError> {
        self.validate_for_mesh(mesh)?;
        let projection = crate::topology::AnalysisProjection::derive(mesh)
            .map_err(|error| invalid(error.to_string()))?;
        let mut samples = Vec::with_capacity(self.integration_sample_count());
        let mut failure = None;
        self.visit_integration_geometry(&projection, |point| {
            match self.integration_sample_identity(point.index) {
                Ok(identity) => samples.push(VolumeFieldIntegrationSample {
                    identity,
                    element_identity: self.data.element_identities[point.element],
                    ordinal: point.ordinal,
                    position_m: point.position_m,
                    volume_weight_m3: point.volume_weight_m3,
                }),
                Err(error) => failure = failure.take().or(Some(error)),
            }
        })?;
        failure.map_or(Ok(samples), Err)
    }
    pub fn entity_counts(&self) -> RecoveredFieldEntityCounts {
        RecoveredFieldEntityCounts {
            oriented_faces: 0,
            nodes: self.data.node_identities.len(),
            volume_elements: self.data.element_identities.len(),
            volume_integration_points: self.integration_sample_count(),
            oriented_edges: self.oriented_edges.len(),
            beam_elements: 0,
            shell_elements: 0,
            connectors: 0,
            constraint_groups: 0,
        }
    }
    /// Decode validation proves internal layout integrity. Source admission must
    /// additionally compare against the original mesh; no numerical field is recovered.
    /// As [`Self::validate_for_mesh`] against the caller's canonical projection of `mesh`.
    pub(crate) fn validate_for_projection(
        &self,
        mesh: &SolverMeshArtifact,
        projection: &crate::topology::AnalysisProjection,
    ) -> Result<(), RecoveredFieldError> {
        self.validate()?;
        if self != &Self::derive_from_projection(mesh, projection, self.rule())? {
            return Err(invalid(
                "sampled field topology differs from canonical mesh",
            ));
        }
        Ok(())
    }
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

/// SHA-256 of the canonical encoding, streamed: a layout identity never materializes the
/// mesh-sized encoding it names.
fn digest<T: Serialize>(label: &str, value: &T) -> Result<Digest, RecoveredFieldError> {
    crate::contracts::canonical_digest::canonical_contract_digest(PREFIX, label, value, FORMAT)
        .map_err(|error| RecoveredFieldError::Encoding(error.to_string()))
}
fn stable_identity<T: Serialize>(
    label: &str,
    value: &T,
) -> Result<StableDigest, RecoveredFieldError> {
    Ok(StableDigest::from_bytes(*digest(label, value)?.bytes()))
}
fn sample_identity(
    rule: VolumeFieldSamplingRule,
    element: StableDigest,
    ordinal: usize,
) -> Result<StableDigest, RecoveredFieldError> {
    stable_identity("integration-sample", &(rule, element, ordinal))
}
fn invalid(message: impl Into<String>) -> RecoveredFieldError {
    RecoveredFieldError::InvalidArtifact(message.into())
}
