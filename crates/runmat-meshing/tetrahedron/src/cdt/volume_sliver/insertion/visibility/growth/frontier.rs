//! Canonical scheduling order is retained; cached later failures remain deferred.
use super::*;
use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::quality::predicate::SpatialPredicateError;
type Key = (u32, usize);
type QualityKey = (u64, [StableDigest; 3], u32, usize);
struct Face {
    vertices: [u32; 3],
    identities: [StableDigest; 3],
    visible: Result<bool, SpatialPredicateError>,
    expandable: Option<u32>,
    quality: Option<QualityKey>,
}
pub(super) enum Decision {
    Done,
    Expand(u32),
    Reject(&'static str),
}
pub(super) struct Frontier<'a, T: VolumeTopologyRead + ?Sized> {
    topology: &'a T,
    node: DelaunayVolumeNode,
    protected: &'a BTreeSet<[StableDigest; 3]>,
    region: Option<PersistentEntityId>,
    metrics: Option<(InsertionMetric<'a>, f64)>,
    faces: BTreeMap<Key, Face>,
    invisible: BTreeSet<Key>,
    pending: BTreeSet<Key>,
    quality: BTreeSet<QualityKey>,
}
impl<'a, T: VolumeTopologyRead + ?Sized> Frontier<'a, T> {
    pub(super) fn new(
        topology: &'a T,
        node: DelaunayVolumeNode,
        protected: &'a BTreeSet<[StableDigest; 3]>,
        region: Option<PersistentEntityId>,
        metrics: Option<(InsertionMetric<'a>, f64)>,
    ) -> Self {
        Self {
            topology,
            node,
            protected,
            region,
            metrics,
            faces: BTreeMap::new(),
            invisible: BTreeSet::new(),
            pending: BTreeSet::new(),
            quality: BTreeSet::new(),
        }
    }
    pub(super) fn insert(&mut self, cell: u32, opposite: usize, vertices: [u32; 3]) {
        let key = (cell, opposite);
        let tetrahedron = self.topology.cell(cell);
        let nodes = vertices.map(|v| self.topology.node(v));
        let identities = nodes.map(|node| node.identity);
        let interior = self.topology.node(tetrahedron.vertex_indices[opposite]);
        let visible =
            orient_volume_nodes([nodes[0], nodes[1], nodes[2], interior]).and_then(|expected| {
                orient_volume_nodes([nodes[0], nodes[1], nodes[2], self.node])
                    .map(|observed| observed != PredicateSign::Zero && observed == expected)
            });
        if !matches!(visible, Ok(true)) {
            self.invisible.insert(key);
        }
        let expandable = tetrahedron.neighbors[opposite].filter(|index| {
            !self.protected.contains(&identities)
                && self.topology.cell(*index).region_id == &self.region
        });
        if expandable.is_some() && self.metrics.is_some() {
            self.pending.insert(key);
        }
        self.faces.insert(
            key,
            Face {
                vertices,
                identities,
                visible,
                expandable,
                quality: None,
            },
        );
    }
    pub(super) fn remove(&mut self, cell: u32, opposite: usize) {
        let key = (cell, opposite);
        if let Some(face) = self.faces.remove(&key) {
            self.invisible.remove(&key);
            self.pending.remove(&key);
            if let Some(quality) = face.quality {
                self.quality.remove(&quality);
            }
        }
    }
    pub(super) fn next(&mut self) -> Result<Decision, DelaunayVolumeSliverError> {
        if let Some(key) = self.invisible.first() {
            let face = &self.faces[key];
            face.visible.map_err(visibility_predicate_error)?;
            if self.protected.contains(&face.identities) {
                return Ok(Decision::Reject(
                    "sliver cavity visibility growth reached a protected facet",
                ));
            }
            let Some(adjacent) = self.topology.cell(key.0).neighbors[key.1] else {
                return Ok(Decision::Reject(
                    "sliver insertion point lies outside the volume boundary",
                ));
            };
            if self.topology.cell(adjacent).region_id != &self.region {
                return Ok(Decision::Reject(
                    "sliver cavity visibility growth reached a region boundary",
                ));
            }
            return Ok(Decision::Expand(adjacent));
        }
        let Some((metrics, target)) = self.metrics else {
            return Ok(Decision::Done);
        };
        while let Some(key) = self.pending.pop_first() {
            let face = &self.faces[&key];
            let [a, b, c] = face
                .vertices
                .map(|v| self.topology.node(v).coordinates_m.canonical_value());
            let value = crate::cdt::volume_quality::evaluate_tetrahedron_jacobian(
                [a, b, c, self.node.coordinates_m.canonical_value()],
                metrics(face.vertices)?,
            )
            .map_err(|cause| {
                error(
                    DelaunayVolumeSliverErrorKind::Accounting(cause),
                    "frontier score numerical accounting failed",
                )
            })?
            .unwrap_or(0.);
            let quality = (value.to_bits(), face.identities, key.0, key.1);
            self.faces
                .get_mut(&key)
                .expect("pending boundary face")
                .quality = Some(quality);
            self.quality.insert(quality);
        }
        let Some(&(bits, _, cell, opposite)) = self.quality.first() else {
            return Ok(Decision::Done);
        };
        if f64::from_bits(bits) > target {
            return Ok(Decision::Done);
        }
        Ok(Decision::Expand(
            self.faces[&(cell, opposite)]
                .expandable
                .expect("quality face has admitted neighbor"),
        ))
    }
}
