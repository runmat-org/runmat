mod indexed;
pub(in crate::cdt) use indexed::IndexedCanonicalTopology;
pub(in crate::cdt) mod queries;
mod read;
pub(in crate::cdt) use read::VolumeTopologyRead;
mod validation;
pub(in crate::cdt) use validation::validate_canonical_volume_topology;
mod search_state;
#[cfg(test)]
pub(in crate::cdt) use search_state::connectivity_search_fingerprint;
pub(in crate::cdt) use search_state::ConnectivitySearchState;
mod canonical;
mod neighbors;
pub(in crate::cdt) use canonical::CanonicalVolumeTopology;
pub(in crate::cdt) use neighbors::connect as connect_volume_neighbors;

use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::{
    quality::predicate::{
        insphere3d_constructed, orient3d_constructed, Point3Construction, PredicateSign,
        SpatialPredicateError, SpatialPredicatePoint,
    },
    MeshingCancellationSignal, StableDigest,
};
use serde::{Deserialize, Serialize};

use super::incidence::{build_volume_incidence, DelaunayVolumeIncidence};

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
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

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelaunayVolumeNode {
    pub identity: StableDigest,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub coordinates_m:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Position, [f64; 3]>,
    pub construction: Point3Construction,
}

impl DelaunayVolumeNode {
    pub fn explicit(identity: StableDigest, coordinates_m: [f64; 3]) -> Self {
        Self {
            identity,
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(coordinates_m),
            construction: Point3Construction::Explicit,
        }
    }

    pub(crate) fn predicate_point(self) -> SpatialPredicatePoint {
        SpatialPredicatePoint {
            identity: self.identity,
            coordinates: self.coordinates_m.canonical_value(),
            construction: self.construction,
        }
    }

    /// Predicates for connectivity consumed by the solver use its stored
    /// coordinates. Exact construction provenance remains available separately.
    pub(crate) fn materialized_predicate_point(self) -> SpatialPredicatePoint {
        Self::explicit(self.identity, self.coordinates_m.canonical_value()).predicate_point()
    }
}

pub(super) fn orient_volume_nodes(
    nodes: [DelaunayVolumeNode; 4],
) -> Result<PredicateSign, SpatialPredicateError> {
    orient3d_constructed(nodes.map(DelaunayVolumeNode::predicate_point))
}

pub(super) fn orient_materialized_volume_nodes(
    nodes: [DelaunayVolumeNode; 4],
) -> Result<PredicateSign, SpatialPredicateError> {
    orient3d_constructed(nodes.map(DelaunayVolumeNode::materialized_predicate_point))
}

pub(super) fn insphere_materialized_volume_nodes(
    nodes: [DelaunayVolumeNode; 5],
) -> Result<PredicateSign, SpatialPredicateError> {
    insphere3d_constructed(nodes.map(DelaunayVolumeNode::materialized_predicate_point))
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelaunayVolumeTetrahedron {
    pub vertex_indices: [u32; 4],
    /// Neighbor opposite each correspondingly indexed vertex.
    pub neighbors: [Option<u32>; 4],
    pub region_id: Option<PersistentEntityId>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
    build_canonical_volume_topology(nodes, tetrahedra, options, cancellation)
        .map(CanonicalVolumeTopology::into_inner)
}

pub(in crate::cdt) fn build_canonical_volume_topology(
    nodes: Vec<DelaunayVolumeNode>,
    tetrahedra: Vec<([u32; 4], Option<PersistentEntityId>)>,
    options: DelaunayTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<CanonicalVolumeTopology, DelaunayTopologyError> {
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
            || node
                .coordinates_m
                .canonical_value()
                .iter()
                .any(|value| !value.is_finite())
            || index > 0 && nodes[index - 1].identity >= node.identity
        {
            return Err(error(
                DelaunayTopologyErrorKind::InvalidNode,
                "nodes must have finite coordinates and strictly ordered nonzero identities",
            ));
        }
    }

    let mut oriented = Vec::with_capacity(tetrahedra.len());
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
        if key.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(error(
                DelaunayTopologyErrorKind::InvalidTetrahedron,
                "tetrahedron vertices must be distinct and tetrahedra unique",
            ));
        }
        let mut canonical_vertices = key;
        let tetrahedron_nodes = canonical_vertices.map(|vertex| nodes[vertex as usize]);
        let sign = orient_materialized_volume_nodes(tetrahedron_nodes).map_err(predicate_error)?;
        match sign {
            PredicateSign::Positive => {}
            PredicateSign::Negative => canonical_vertices.swap(0, 1),
            PredicateSign::Zero => {
                return Err(error(
                    DelaunayTopologyErrorKind::DegenerateTetrahedron,
                    format!(
                        "tetrahedron {index} with vertices {key:?} is coplanar at materialized solver coordinates"
                    ),
                ));
            }
        }
        oriented.push((key, canonical_vertices, region_id));
    }
    oriented.sort_by_key(|(key, _, _)| *key);
    if oriented.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err(error(
            DelaunayTopologyErrorKind::InvalidTetrahedron,
            "tetrahedra must be unique",
        ));
    }

    let mut result = oriented
        .into_iter()
        .map(|(_, vertex_indices, region_id)| DelaunayVolumeTetrahedron {
            vertex_indices,
            neighbors: [None; 4],
            region_id,
        })
        .collect::<Vec<_>>();
    neighbors::connect(&nodes, &mut result, options, cancellation)?;
    let incidence = build_volume_incidence(&nodes, &result, cancellation, options)?;
    Ok(CanonicalVolumeTopology::from_construction(
        DelaunayVolumeTopology {
            nodes,
            tetrahedra: result,
            incidence,
        },
    ))
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
        DelaunayVolumeNode::explicit(StableDigest::from_bytes([identity; 32]), coordinates_m)
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
                runmat_meshing_core::quality::predicate::orient3d(tetrahedron.vertex_indices.map(
                    |index| {
                        topology.nodes[index as usize]
                            .coordinates_m
                            .canonical_value()
                    }
                ))
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

    #[test]
    fn topology_rejects_constructed_volume_lost_by_solver_coordinate_rounding() {
        let plane = 1.0e16;
        let constructed = DelaunayVolumeNode {
            identity: StableDigest::from_bytes([4; 32]),
            // The exact dyadic midpoint is one unit above the plane, but that offset is below one
            // binary64 ULP at this magnitude and therefore is not present in solver coordinates.
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new([0.25, 0.25, plane]),
            construction: runmat_meshing_core::quality::predicate::Point3Construction::Segment {
                endpoints: [[0.25, 0.25, plane], [0.25, 0.25, plane + 2.0]],
                parameter_numerator: 1,
                parameter_exponent: 1,
            },
        };
        let nodes = vec![
            node(1, [0.0, 0.0, plane]),
            node(2, [1.0, 0.0, plane]),
            node(3, [0.0, 1.0, plane]),
            constructed,
        ];
        assert_ne!(
            orient_volume_nodes(nodes.clone().try_into().unwrap()).unwrap(),
            PredicateSign::Zero
        );
        let failure = build_delaunay_volume_topology(
            nodes,
            vec![[0, 1, 2, 3]],
            DelaunayTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err();
        assert_eq!(
            failure.kind,
            DelaunayTopologyErrorKind::DegenerateTetrahedron
        );
        assert!(failure.reason.contains("materialized solver coordinates"));
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
