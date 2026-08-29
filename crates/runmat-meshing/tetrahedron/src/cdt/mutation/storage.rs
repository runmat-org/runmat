use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::quality::predicate::{orient3d, PredicateSign};

use crate::cdt::{DelaunayVolumeNode, DelaunayVolumeTopology};

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

#[derive(Clone, Debug)]
pub(super) struct MutableTetrahedron {
    pub(super) vertex_indices: [u32; 4],
    pub(super) neighbors: [Option<u32>; 4],
    pub(super) region_id: Option<PersistentEntityId>,
}

#[derive(Clone, Debug)]
pub(in crate::cdt) struct TopologyMutation {
    pub(super) nodes: Vec<DelaunayVolumeNode>,
    pub(super) tetrahedra: Vec<Option<MutableTetrahedron>>,
    pub(super) free_tetrahedra: BTreeSet<u32>,
    pub(super) vertex_stars: Vec<BTreeSet<u32>>,
    pub(super) face_uses: BTreeMap<[u32; 3], Vec<(u32, usize)>>,
}

impl TopologyMutation {
    pub(in crate::cdt) fn from_canonical(topology: DelaunayVolumeTopology) -> Self {
        let vertex_stars = topology
            .incidence
            .vertex_stars
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
        let mut result = Self {
            nodes: topology.nodes,
            tetrahedra,
            free_tetrahedra: BTreeSet::new(),
            vertex_stars,
            face_uses: BTreeMap::new(),
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
                .iter()
                .any(|coordinate| !coordinate.is_finite())
            || self.nodes.iter().any(|existing| {
                existing.identity == node.identity || existing.coordinates_m == node.coordinates_m
            })
        {
            return Err(error(
                "mutable topology rejected an invalid or duplicate node",
            ));
        }
        let index = u32::try_from(self.nodes.len())
            .map_err(|_| error("mutable topology exhausted its node index space"))?;
        self.nodes.push(node);
        self.vertex_stars.push(BTreeSet::new());
        Ok(index)
    }

    pub(super) fn tetrahedron(&self, index: u32) -> Option<&MutableTetrahedron> {
        self.tetrahedra.get(index as usize)?.as_ref()
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
        match orient3d(vertices.map(|vertex| self.nodes[vertex as usize].coordinates_m)).map_err(
            |failure| error(format!("mutable orientation predicate failed: {failure:?}")),
        )? {
            PredicateSign::Positive => Ok(vertices),
            PredicateSign::Negative => {
                vertices.swap(0, 1);
                Ok(vertices)
            }
            PredicateSign::Zero => Err(error("mutable tetrahedron is exactly coplanar")),
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
        }
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
