use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::{
    quality::predicate::{orient3d, PredicateSign, SpatialPredicateError},
    MeshingCancellationSignal, StableDigest,
};

use super::incidence::{build_volume_incidence, DelaunayVolumeIncidence};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DelaunayTopologyOptions {
    pub maximum_nodes: u64,
    pub maximum_tetrahedra: u64,
    pub cancellation_check_interval: u64,
}

impl Default for DelaunayTopologyOptions {
    fn default() -> Self {
        Self {
            maximum_nodes: 1_000_000_000,
            maximum_tetrahedra: 2_000_000_000,
            cancellation_check_interval: 1_024,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DelaunayVolumeNode {
    pub identity: StableDigest,
    pub coordinates_m: [f64; 3],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelaunayVolumeTetrahedron {
    pub vertex_indices: [u32; 4],
    /// Neighbor opposite each correspondingly indexed vertex.
    pub neighbors: [Option<u32>; 4],
    pub region_id: Option<PersistentEntityId>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DelaunayVolumeTopology {
    pub nodes: Vec<DelaunayVolumeNode>,
    pub tetrahedra: Vec<DelaunayVolumeTetrahedron>,
    pub incidence: DelaunayVolumeIncidence,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelaunayTopologyErrorKind {
    InvalidOptions,
    InvalidNode,
    InvalidTetrahedron,
    InvalidRegion,
    DegenerateTetrahedron,
    NonManifoldFace,
    ResourceLimit,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelaunayTopologyError {
    pub kind: DelaunayTopologyErrorKind,
    pub reason: String,
}

impl std::fmt::Display for DelaunayTopologyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "3D Delaunay topology {:?}: {}",
            self.kind, self.reason
        )
    }
}

impl std::error::Error for DelaunayTopologyError {}

pub fn build_delaunay_volume_topology(
    nodes: Vec<DelaunayVolumeNode>,
    tetrahedra: Vec<[u32; 4]>,
    options: DelaunayTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeTopology, DelaunayTopologyError> {
    build_delaunay_volume_topology_with_regions(
        nodes,
        tetrahedra
            .into_iter()
            .map(|vertices| (vertices, None))
            .collect(),
        options,
        cancellation,
    )
}

pub(super) fn build_delaunay_volume_topology_with_regions(
    nodes: Vec<DelaunayVolumeNode>,
    tetrahedra: Vec<([u32; 4], Option<PersistentEntityId>)>,
    options: DelaunayTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeTopology, DelaunayTopologyError> {
    validate_options(options)?;
    if nodes.is_empty() || nodes.len() as u64 > options.maximum_nodes {
        return Err(error(
            DelaunayTopologyErrorKind::ResourceLimit,
            "node inventory is empty or exceeds its hard limit",
        ));
    }
    if tetrahedra.is_empty() || tetrahedra.len() as u64 > options.maximum_tetrahedra {
        return Err(error(
            DelaunayTopologyErrorKind::ResourceLimit,
            "tetrahedron inventory is empty or exceeds its hard limit",
        ));
    }
    for (index, node) in nodes.iter().enumerate() {
        checkpoint(index as u64, options, cancellation)?;
        if node.identity == StableDigest::ZERO
            || node.coordinates_m.iter().any(|value| !value.is_finite())
            || index > 0 && nodes[index - 1].identity >= node.identity
        {
            return Err(error(
                DelaunayTopologyErrorKind::InvalidNode,
                "nodes must have finite coordinates and strictly ordered nonzero identities",
            ));
        }
    }

    let mut oriented = Vec::with_capacity(tetrahedra.len());
    let mut tetrahedron_keys = BTreeSet::new();
    for (index, (vertices, region_id)) in tetrahedra.into_iter().enumerate() {
        checkpoint(index as u64, options, cancellation)?;
        if vertices
            .iter()
            .any(|vertex| *vertex as usize >= nodes.len())
        {
            return Err(error(
                DelaunayTopologyErrorKind::InvalidTetrahedron,
                "tetrahedron references a node outside the admitted inventory",
            ));
        }
        let mut key = vertices;
        key.sort_unstable();
        if key.windows(2).any(|pair| pair[0] == pair[1]) || !tetrahedron_keys.insert(key) {
            return Err(error(
                DelaunayTopologyErrorKind::InvalidTetrahedron,
                "tetrahedron vertices must be distinct and tetrahedra unique",
            ));
        }
        let mut canonical_vertices = key;
        match orient3d(canonical_vertices.map(|vertex| nodes[vertex as usize].coordinates_m))
            .map_err(predicate_error)?
        {
            PredicateSign::Positive => {}
            PredicateSign::Negative => canonical_vertices.swap(0, 1),
            PredicateSign::Zero => {
                return Err(error(
                    DelaunayTopologyErrorKind::DegenerateTetrahedron,
                    "tetrahedron vertices are exactly coplanar",
                ));
            }
        }
        oriented.push((key, canonical_vertices, region_id));
    }
    oriented.sort_by_key(|(key, _, _)| *key);

    let mut face_uses = BTreeMap::<[u32; 3], Vec<(usize, usize)>>::new();
    for (tetrahedron_index, (_, vertices, _)) in oriented.iter().enumerate() {
        for opposite in 0..4 {
            let mut face = [0u32; 3];
            let mut cursor = 0;
            for (vertex_index, vertex) in vertices.iter().enumerate() {
                if vertex_index != opposite {
                    face[cursor] = *vertex;
                    cursor += 1;
                }
            }
            face.sort_unstable();
            face_uses
                .entry(face)
                .or_default()
                .push((tetrahedron_index, opposite));
        }
    }
    let mut result = oriented
        .into_iter()
        .map(|(_, vertex_indices, region_id)| DelaunayVolumeTetrahedron {
            vertex_indices,
            neighbors: [None; 4],
            region_id,
        })
        .collect::<Vec<_>>();
    for uses in face_uses.values() {
        match uses.as_slice() {
            [_] => {}
            [(left, left_opposite), (right, right_opposite)] => {
                result[*left].neighbors[*left_opposite] = Some(*right as u32);
                result[*right].neighbors[*right_opposite] = Some(*left as u32);
            }
            _ => {
                return Err(error(
                    DelaunayTopologyErrorKind::NonManifoldFace,
                    "more than two tetrahedra share one face",
                ));
            }
        }
    }
    let incidence = build_volume_incidence(&nodes, &result, cancellation, options)?;
    Ok(DelaunayVolumeTopology {
        nodes,
        tetrahedra: result,
        incidence,
    })
}

fn validate_options(options: DelaunayTopologyOptions) -> Result<(), DelaunayTopologyError> {
    if options.maximum_nodes == 0
        || options.maximum_tetrahedra == 0
        || options.cancellation_check_interval == 0
    {
        return Err(error(
            DelaunayTopologyErrorKind::InvalidOptions,
            "topology limits and cancellation interval must be nonzero",
        ));
    }
    Ok(())
}

fn checkpoint(
    index: u64,
    options: DelaunayTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayTopologyError> {
    if index.is_multiple_of(options.cancellation_check_interval) && cancellation.is_cancelled() {
        return Err(error(DelaunayTopologyErrorKind::Cancelled, "cancelled"));
    }
    Ok(())
}

fn predicate_error(error: SpatialPredicateError) -> DelaunayTopologyError {
    self::error(
        DelaunayTopologyErrorKind::InvalidNode,
        format!("spatial predicate rejected node coordinates: {error:?}"),
    )
}

pub(super) fn error(
    kind: DelaunayTopologyErrorKind,
    reason: impl Into<String>,
) -> DelaunayTopologyError {
    DelaunayTopologyError {
        kind,
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use runmat_meshing_core::NeverCancelled;

    use super::*;

    fn node(identity: u8, coordinates_m: [f64; 3]) -> DelaunayVolumeNode {
        DelaunayVolumeNode {
            identity: StableDigest::from_bytes([identity; 32]),
            coordinates_m,
        }
    }

    #[test]
    fn topology_orients_tetrahedra_and_builds_reciprocal_neighbors() {
        let nodes = vec![
            node(1, [0.0, 0.0, 0.0]),
            node(2, [1.0, 0.0, 0.0]),
            node(3, [0.0, 1.0, 0.0]),
            node(4, [0.0, 0.0, 1.0]),
            node(5, [0.0, 0.0, -1.0]),
        ];
        let topology = build_delaunay_volume_topology(
            nodes.clone(),
            vec![[0, 1, 2, 3], [2, 1, 0, 4]],
            DelaunayTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap();
        assert_eq!(topology.nodes, nodes);
        assert_eq!(
            topology.tetrahedra[0]
                .neighbors
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>(),
            vec![1]
        );
        assert_eq!(
            topology.tetrahedra[1]
                .neighbors
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>(),
            vec![0]
        );
        for tetrahedron in &topology.tetrahedra {
            assert_eq!(
                orient3d(
                    tetrahedron
                        .vertex_indices
                        .map(|index| topology.nodes[index as usize].coordinates_m)
                )
                .unwrap(),
                PredicateSign::Positive
            );
        }
    }

    #[test]
    fn topology_rejects_degenerate_duplicate_and_nonmanifold_inputs() {
        let nodes = vec![
            node(1, [0.0, 0.0, 0.0]),
            node(2, [1.0, 0.0, 0.0]),
            node(3, [0.0, 1.0, 0.0]),
            node(4, [0.0, 0.0, 1.0]),
            node(5, [0.0, 0.0, -1.0]),
            node(6, [0.0, 0.0, 2.0]),
        ];
        assert_eq!(
            build_delaunay_volume_topology(
                nodes.clone(),
                vec![[0, 1, 2, 2]],
                DelaunayTopologyOptions::default(),
                &NeverCancelled,
            )
            .unwrap_err()
            .kind,
            DelaunayTopologyErrorKind::InvalidTetrahedron
        );
        let mut coplanar = nodes.clone();
        coplanar.push(node(7, [1.0, 1.0, 0.0]));
        assert_eq!(
            build_delaunay_volume_topology(
                coplanar,
                vec![[0, 1, 2, 6]],
                DelaunayTopologyOptions::default(),
                &NeverCancelled,
            )
            .unwrap_err()
            .kind,
            DelaunayTopologyErrorKind::DegenerateTetrahedron
        );
        assert_eq!(
            build_delaunay_volume_topology(
                nodes,
                vec![[0, 1, 2, 3], [0, 2, 1, 4], [0, 1, 2, 5]],
                DelaunayTopologyOptions::default(),
                &NeverCancelled,
            )
            .unwrap_err()
            .kind,
            DelaunayTopologyErrorKind::NonManifoldFace
        );
    }

    struct Cancelled;

    impl MeshingCancellationSignal for Cancelled {
        fn is_cancelled(&self) -> bool {
            true
        }
    }

    #[test]
    fn topology_enforces_hard_limits_and_cancellation() {
        let nodes = vec![
            node(1, [0.0, 0.0, 0.0]),
            node(2, [1.0, 0.0, 0.0]),
            node(3, [0.0, 1.0, 0.0]),
            node(4, [0.0, 0.0, 1.0]),
        ];
        assert_eq!(
            build_delaunay_volume_topology(
                nodes.clone(),
                vec![[0, 1, 2, 3]],
                DelaunayTopologyOptions {
                    maximum_nodes: 3,
                    ..DelaunayTopologyOptions::default()
                },
                &NeverCancelled,
            )
            .unwrap_err()
            .kind,
            DelaunayTopologyErrorKind::ResourceLimit
        );
        assert_eq!(
            build_delaunay_volume_topology(
                nodes,
                vec![[0, 1, 2, 3]],
                DelaunayTopologyOptions::default(),
                &Cancelled,
            )
            .unwrap_err()
            .kind,
            DelaunayTopologyErrorKind::Cancelled
        );
    }
}
