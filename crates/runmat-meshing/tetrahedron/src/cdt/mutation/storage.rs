use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::quality::predicate::PredicateSign;
use runmat_meshing_core::StableDigest;

use crate::cdt::{
    topology::orient_materialized_volume_nodes, DelaunayVolumeNode, DelaunayVolumeTopology,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::cdt) struct TopologyMutationError {
    pub(in crate::cdt) kind: TopologyMutationErrorKind,
    pub(in crate::cdt) reason: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::cdt) enum TopologyMutationErrorKind {
    InvalidTopology,
    ResourceLimit,
    Cancelled,
}

impl std::fmt::Display for TopologyMutationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.reason)
    }
}

impl std::error::Error for TopologyMutationError {}

#[derive(Clone, Debug, PartialEq)]
pub(in crate::cdt) struct MutableTetrahedron {
    pub(in crate::cdt) vertex_indices: [u32; 4],
    pub(in crate::cdt) neighbors: [Option<u32>; 4],
    pub(in crate::cdt) region_id: Option<PersistentEntityId>,
}

#[derive(Debug)]
#[cfg_attr(test, derive(Clone))]
pub(in crate::cdt) struct TopologyMutation {
    pub(super) trace_enabled: bool,
    pub(super) edit_trace: super::trace::TopologyEdits,
    pub(super) trial_journals: Vec<super::trial::TrialJournal>,
    pub(super) edge_observers: Vec<BTreeMap<u32, Option<[u32; 4]>>>,
    pub(in crate::cdt) nodes: Vec<DelaunayVolumeNode>,
    pub(in crate::cdt) node_indices: BTreeMap<StableDigest, u32>,
    pub(super) positions: BTreeSet<[u64; 3]>,
    pub(super) spatial_index:
        RefCell<Option<std::sync::Arc<runmat_meshing_core::PointSpatialIndex<StableDigest>>>>,
    pub(super) coordinate_scale_m: f64,
    pub(super) cell_spatial_index: RefCell<Option<super::spatial_cells::CellSpatialCache>>,
    pub(super) connectivity_state: RefCell<Option<crate::cdt::topology::ConnectivitySearchState>>,
    pub(super) physical_dirty_faces: Option<BTreeSet<[u32; 3]>>,
    pub(in crate::cdt) tetrahedra: Vec<Option<MutableTetrahedron>>,
    pub(super) tetrahedron_indices: BTreeMap<[u32; 4], u32>,
    pub(super) free_tetrahedra: BTreeSet<u32>,
    pub(in crate::cdt) vertex_stars: Vec<BTreeSet<u32>>,
    pub(super) face_uses: BTreeMap<[u32; 3], Vec<(u32, usize)>>,
    pub(in crate::cdt) boundary_faces: BTreeMap<[u32; 3], (u32, usize)>,
}

impl TopologyMutation {
    /// Cavity commits maintain canonical corner keys in this ordered index.
    /// Seed once from maintained keys, then update the cycle hint with each
    /// atomic cavity. Fixed nodes/regions are a search precondition, not proof.
    pub(in crate::cdt) fn connectivity_search_fingerprint<E>(
        &self,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<[u8; 32], E> {
        let mut retained = self.connectivity_state.borrow_mut();
        if let Some(state) = *retained {
            checkpoint()?;
            return Ok(state.fingerprint());
        }
        let mut state = crate::cdt::topology::ConnectivitySearchState::default();
        for vertices in self.tetrahedron_indices.keys() {
            checkpoint()?;
            state.insert(*vertices);
        }
        *retained = Some(state);
        Ok(state.fingerprint())
    }

    pub(in crate::cdt) fn contains_position(&self, coordinates: [f64; 3]) -> bool {
        self.positions.contains(&coordinates.map(coordinate_bits))
    }
    pub(in crate::cdt) fn from_canonical(topology: DelaunayVolumeTopology) -> Self {
        let mut incidence = topology.incidence;
        let original_stars = std::mem::take(&mut incidence.vertex_stars);
        // The working owner reconstructs faces and region support. Release the
        // obsolete canonical incidence before allocating its mutation indexes.
        drop(incidence);
        let vertex_stars = original_stars
            .into_iter()
            .map(|star| star.into_iter().collect())
            .collect();
        let tetrahedra = topology
            .tetrahedra
            .into_iter()
            .map(|tetrahedron| {
                Some(MutableTetrahedron {
                    vertex_indices: tetrahedron.vertex_indices,
                    neighbors: tetrahedron.neighbors,
                    region_id: tetrahedron.region_id,
                })
            })
            .collect::<Vec<_>>();
        let tetrahedron_indices = tetrahedra
            .iter()
            .enumerate()
            .filter_map(|(index, tetrahedron)| {
                tetrahedron.as_ref().map(|tetrahedron| {
                    let mut key = tetrahedron.vertex_indices;
                    key.sort_unstable();
                    (key, index as u32)
                })
            })
            .collect();
        let nodes = topology.nodes;
        let node_indices = nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.identity, index as u32))
            .collect();
        let positions = nodes
            .iter()
            .map(|node| node.coordinates_m.canonical_value().map(coordinate_bits))
            .collect();
        let coordinate_scale_m = nodes
            .iter()
            .flat_map(|node| node.coordinates_m.canonical_value())
            .map(f64::abs)
            .fold(1.0, f64::max);
        let mut result = Self {
            trace_enabled: false,
            edit_trace: super::trace::TopologyEdits::default(),
            trial_journals: Vec::new(),
            edge_observers: Vec::new(),
            nodes,
            node_indices,
            coordinate_scale_m,
            spatial_index: RefCell::new(None),
            cell_spatial_index: RefCell::new(None),
            connectivity_state: RefCell::new(None),
            physical_dirty_faces: None,
            positions,
            tetrahedra,
            tetrahedron_indices,
            free_tetrahedra: BTreeSet::new(),
            vertex_stars,
            face_uses: BTreeMap::new(),
            boundary_faces: BTreeMap::new(),
        };
        result.rebuild_face_uses();
        result
    }

    pub(in crate::cdt) fn append_node(
        &mut self,
        node: DelaunayVolumeNode,
    ) -> Result<u32, TopologyMutationError> {
        if node.identity == runmat_meshing_core::StableDigest::ZERO
            || node
                .coordinates_m
                .canonical_value()
                .iter()
                .any(|coordinate| !coordinate.is_finite())
            || self.node_indices.contains_key(&node.identity)
            || self
                .positions
                .contains(&node.coordinates_m.canonical_value().map(coordinate_bits))
        {
            return Err(error(
                "mutable topology rejected an invalid or duplicate node",
            ));
        }
        let index = u32::try_from(self.nodes.len())
            .map_err(|_| error("mutable topology exhausted its node index space"))?;
        self.coordinate_scale_m = node
            .coordinates_m
            .canonical_value()
            .into_iter()
            .map(f64::abs)
            .fold(self.coordinate_scale_m, f64::max);
        self.nodes.push(node);
        self.node_indices.insert(node.identity, index);
        self.positions
            .insert(node.coordinates_m.canonical_value().map(coordinate_bits));
        self.vertex_stars.push(BTreeSet::new());
        if self.trace_enabled && !self.trial_journals.is_empty() {
            self.edit_trace
                .push(super::trace::TopologyEdit::AppendNode(node));
        }
        Ok(index)
    }

    pub(in crate::cdt) fn tetrahedron(&self, index: u32) -> Option<&MutableTetrahedron> {
        self.tetrahedra.get(index as usize)?.as_ref()
    }

    pub(in crate::cdt) fn tetrahedron_index(&self, mut vertices: [u32; 4]) -> Option<u32> {
        vertices.sort_unstable();
        self.tetrahedron_indices.get(&vertices).copied()
    }

    pub(super) fn oriented_vertices(
        &self,
        mut vertices: [u32; 4],
    ) -> Result<[u32; 4], TopologyMutationError> {
        if vertices
            .iter()
            .any(|vertex| *vertex as usize >= self.nodes.len())
        {
            return Err(error("mutable tetrahedron references a missing node"));
        }
        let mut unique = vertices;
        unique.sort_unstable();
        if unique.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(error("mutable tetrahedron repeats a node"));
        }
        let nodes = vertices.map(|vertex| self.nodes[vertex as usize]);
        let sign = orient_materialized_volume_nodes(nodes).map_err(|failure| {
            error(format!(
                "mutable materialized orientation predicate failed: {failure:?}"
            ))
        })?;
        match sign {
            PredicateSign::Positive => Ok(vertices),
            PredicateSign::Negative => {
                vertices.swap(0, 1);
                Ok(vertices)
            }
            PredicateSign::Zero => Err(error(format!(
                "mutable tetrahedron {vertices:?} is coplanar at materialized solver coordinates: nodes={nodes:?}",
            ))),
        }
    }

    pub(super) fn face(vertices: [u32; 4], opposite: usize) -> [u32; 3] {
        let mut face = [0; 3];
        let mut cursor = 0;
        for (slot, vertex) in vertices.into_iter().enumerate() {
            if slot != opposite {
                face[cursor] = vertex;
                cursor += 1;
            }
        }
        face.sort_unstable();
        face
    }

    pub(super) fn rebuild_face_uses(&mut self) {
        self.face_uses.clear();
        for (tetrahedron_index, tetrahedron) in self.tetrahedra.iter().enumerate() {
            let Some(tetrahedron) = tetrahedron else {
                continue;
            };
            for opposite in 0..4 {
                self.face_uses
                    .entry(Self::face(tetrahedron.vertex_indices, opposite))
                    .or_default()
                    .push((tetrahedron_index as u32, opposite));
            }
        }
        self.boundary_faces = self
            .face_uses
            .iter()
            .filter(|(_, uses)| uses.len() == 1)
            .map(|(face, uses)| (*face, uses[0]))
            .collect();
    }

    pub(super) fn remove_face_uses(&mut self, tetrahedron_index: u32, vertices: [u32; 4]) {
        for opposite in 0..4 {
            let face = Self::face(vertices, opposite);
            let remove_face = if let Some(uses) = self.face_uses.get_mut(&face) {
                uses.retain(|(index, _)| *index != tetrahedron_index);
                uses.is_empty()
            } else {
                false
            };
            if remove_face {
                self.face_uses.remove(&face);
            }
            self.sync_boundary_face(face);
        }
    }

    pub(super) fn insert_face_uses(&mut self, tetrahedron_index: u32, vertices: [u32; 4]) {
        for opposite in 0..4 {
            let uses = self
                .face_uses
                .entry(Self::face(vertices, opposite))
                .or_default();
            uses.push((tetrahedron_index, opposite));
            uses.sort_unstable();
            self.sync_boundary_face(Self::face(vertices, opposite));
        }
    }
}

pub(super) fn coordinate_bits(value: f64) -> u64 {
    if value == 0.0 {
        0
    } else {
        value.to_bits()
    }
}

pub(super) fn error(reason: impl Into<String>) -> TopologyMutationError {
    TopologyMutationError {
        kind: TopologyMutationErrorKind::InvalidTopology,
        reason: reason.into(),
    }
}

pub(super) fn execution_error(
    kind: TopologyMutationErrorKind,
    reason: impl Into<String>,
) -> TopologyMutationError {
    TopologyMutationError {
        kind,
        reason: reason.into(),
    }
}
