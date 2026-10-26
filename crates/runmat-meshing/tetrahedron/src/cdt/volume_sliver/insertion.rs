use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    quality::predicate::{
        compose_dyadic_point, Point3Construction, PredicateSign, SpatialPredicatePoint,
    },
    MeshingCancellationSignal, StableDigest,
};
use sha2::{Digest, Sha256};

use super::admission::QualityObjective;
#[cfg(test)]
use super::admission::{
    candidate_quality_is_admissible, compare_spectra, evaluate_canonical_candidate_unranked,
    quality_spectrum,
};
#[cfg(test)]
use super::reconnection::reconnection_candidates;
use super::work::OptimizationWork;
use super::{
    error, DelaunayVolumeSliverError, DelaunayVolumeSliverErrorKind, DelaunayVolumeSliverInsertion,
    DelaunayVolumeSliverMutation, DelaunayVolumeSliverOptions, SliverSearchPolicy,
    CAVITY_SMOOTHING_STEPS, MAXIMUM_CAVITY_SEARCH_DIRECTIONS, MAXIMUM_CAVITY_SMOOTHING_ITERATIONS,
};
use crate::cdt::{
    topology::{orient_materialized_volume_nodes as orient_volume_nodes, VolumeTopologyRead},
    DelaunayTetrahedronQuality, DelaunayVolumeNode,
};
#[cfg(test)]
use crate::reconnect::local_tetrahedron_boundary_faces;

#[cfg(test)]
use crate::cdt::{
    insertion::insert_delaunay_volume_node_locally_with_barriers,
    mutation::TopologyMutation,
    topology::{build_canonical_volume_topology, CanonicalVolumeTopology},
    DelaunayInsertionErrorKind, DelaunayTopologyErrorKind, DelaunayVolumeQuality,
    DelaunayVolumeRefinementInput, DelaunayVolumeTopology,
};

const INSERTION_IDENTITY_DOMAIN: &[u8] = b"runmat/meshing/cdt/volume-optimization-insertion/1\0";
const KERNEL_PROBE_IDENTITY_DOMAIN: &[u8] = b"runmat/meshing/cdt/volume-kernel-probe/1\0";
const EDGE_STAR_DIRECTION_WEIGHTS: [(u64, u8); 3] = [(61, 6), (13, 4), (1, 2)];

mod choices;
mod geometry;
mod offcenter;
pub(super) mod refill;
#[cfg(test)]
mod retained;
#[cfg(test)]
mod search;
pub(super) mod visibility;
pub(super) mod working;
#[cfg(test)]
use geometry::split_complete_edge_star;
use geometry::{
    edge_star_boundary_faces, kernel_probe_identity, pattern_search_edge_star,
    point_is_in_edge_star_kernel, smooth_edge_star_point, PreparedEdgeStar,
};
#[cfg(test)]
pub(super) use search::insertion_candidates;
use visibility::smooth_visibility_cavity_point;
#[cfg(test)]
use visibility::split_two_tetrahedron_cavity;

#[cfg(test)]
enum EdgeStarSplit {
    Topology(CanonicalVolumeTopology),
    Rejected(String),
}

type VisibilityCavity = (
    BTreeSet<u32>,
    Option<runmat_geometry_core::PersistentEntityId>,
);
pub(super) type VisibilityCavityResult = Result<VisibilityCavity, String>;

type InsertionMetric<'a> =
    &'a dyn Fn(
        [u32; 3],
    )
        -> Result<runmat_meshing_size::metric::MetricTensor3, DelaunayVolumeSliverError>;

pub(super) struct VisibilityCavityRequest<'a> {
    pub(super) node: DelaunayVolumeNode,
    pub(super) source_index: u32,
    pub(super) relocated_vertices: &'a [u32],
    pub(super) neighbor_index: u32,
    pub(super) protected_faces: &'a BTreeSet<[StableDigest; 3]>,
    pub(super) quality_growth: Option<(InsertionMetric<'a>, f64)>,
    pub(super) options: DelaunayVolumeSliverOptions,
    pub(super) cancellation: &'a dyn MeshingCancellationSignal,
}

struct VisibilityContext<'a> {
    source_vertices: [u32; 4],
    neighbor_index: u32,
    protected_faces: &'a BTreeSet<[StableDigest; 3]>,
    metrics: InsertionMetric<'a>,
    target_scaled_jacobian: f64,
    options: DelaunayVolumeSliverOptions,
    cancellation: &'a dyn MeshingCancellationSignal,
}

#[derive(Clone, Copy)]
struct EdgeStarChoice {
    edge_identities: [StableDigest; 2],
    index: u16,
    coordinates_m:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Position, [f64; 3]>,
    construction: Point3Construction,
    proposal_score: f64,
    strategy: InsertionStrategy,
}

#[derive(Clone, Copy)]
enum InsertionStrategy {
    EdgeStar,
    Cavity { neighbor_index: u32 },
    BoundaryOffcenter { neighbor_index: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EdgeCandidatePolicy {
    ExcludedSegment,
    OffEdgeOnly,
    IncludeMidpoint,
}

#[cfg(test)]
fn insertion_error(failure: crate::cdt::DelaunayInsertionError) -> DelaunayVolumeSliverError {
    let kind = match failure.kind {
        DelaunayInsertionErrorKind::ResourceLimit => DelaunayVolumeSliverErrorKind::ResourceLimit,
        DelaunayInsertionErrorKind::Cancelled => DelaunayVolumeSliverErrorKind::Cancelled,
        _ => DelaunayVolumeSliverErrorKind::InvalidTopology,
    };
    error(kind, failure.to_string())
}

#[cfg(test)]
fn edge_candidate_policy(
    edge: [StableDigest; 2],
    protected_faces: &[[StableDigest; 3]],
    protected_segments: &BTreeSet<[StableDigest; 2]>,
) -> EdgeCandidatePolicy {
    if protected_segments.contains(&edge) {
        EdgeCandidatePolicy::ExcludedSegment
    } else if protected_faces
        .iter()
        .any(|facet| facet.contains(&edge[0]) && facet.contains(&edge[1]))
    {
        EdgeCandidatePolicy::OffEdgeOnly
    } else {
        EdgeCandidatePolicy::IncludeMidpoint
    }
}

/// Finds a point in the kernel of an interior edge-star cavity while maximizing the minimum
/// created scaled Jacobian. This is one deterministic scale of TetGen-style link-face smoothing:
/// start at the edge midpoint, move toward each cavity boundary-face centroid, and retain the
/// best strict improvement until convergence. The caller searches several bounded step scales.
fn average(points: impl Iterator<Item = [f64; 3]>) -> [f64; 3] {
    let (sum, count) = points.fold(([0.0; 3], 0_u64), |(mut sum, count), point| {
        for axis in 0..3 {
            sum[axis] += point[axis];
        }
        (sum, count + 1)
    });
    sum.map(|value| value / count as f64)
}

fn insertion_identity(
    source: [StableDigest; 4],
    edge: [StableDigest; 2],
    index: u16,
) -> StableDigest {
    let mut hasher = Sha256::new();
    hasher.update(INSERTION_IDENTITY_DOMAIN);
    for identity in source.into_iter().chain(edge) {
        hasher.update(identity.bytes());
    }
    hasher.update(index.to_be_bytes());
    StableDigest::from_bytes(hasher.finalize().into())
}

pub(in crate::cdt) fn insertion_identity_is_valid(
    insertion: &DelaunayVolumeSliverInsertion,
) -> bool {
    insertion.inserted_node.identity
        == insertion_identity(
            insertion.source_tetrahedron_node_identities,
            insertion.source_edge_node_identities,
            insertion.construction_point_index,
        )
}

fn reject(counts: &mut BTreeMap<&'static str, u64>, reason: &'static str) {
    *counts.entry(reason).or_default() += 1;
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
    use runmat_meshing_core::{NeverCancelled, StableDigest};

    use super::{
        edge_candidate_policy, split_complete_edge_star, EdgeCandidatePolicy, EdgeStarSplit,
    };
    use crate::cdt::{
        topology::build_delaunay_volume_topology_with_regions, DelaunayTopologyOptions,
        DelaunayVolumeNode, DelaunayVolumeSliverOptions,
    };

    #[test]
    fn protected_facets_allow_off_edge_sliver_candidates_but_segments_do_not() {
        let identity = |value| StableDigest::from_bytes([value; 32]);
        let edge = [identity(1), identity(2)];
        let facets = [[identity(1), identity(2), identity(3)]];

        assert_eq!(
            edge_candidate_policy(edge, &facets, &BTreeSet::new()),
            EdgeCandidatePolicy::OffEdgeOnly
        );
        assert_eq!(
            edge_candidate_policy(edge, &[], &BTreeSet::new()),
            EdgeCandidatePolicy::IncludeMidpoint
        );
        assert_eq!(
            edge_candidate_policy(edge, &facets, &BTreeSet::from([edge])),
            EdgeCandidatePolicy::ExcludedSegment
        );
    }

    #[test]
    fn complete_edge_star_split_preserves_the_cavity_boundary() {
        let identity = |value| StableDigest::from_bytes([value; 32]);
        let node = |value, coordinates| DelaunayVolumeNode::explicit(identity(value), coordinates);
        let nodes = vec![
            node(1, [0.0, 0.0, -1.0]),
            node(2, [0.0, 0.0, 1.0]),
            node(3, [1.0, 0.0, 0.0]),
            node(4, [-0.5, 0.866_025_403_784, 0.0]),
            node(5, [-0.5, -0.866_025_403_784, 0.0]),
        ];
        let region =
            PersistentEntityId::new(PersistentEntityKind::Region, "solid".to_owned(), Vec::new());
        let topology = build_delaunay_volume_topology_with_regions(
            nodes,
            vec![
                ([0, 1, 2, 3], Some(region.clone())),
                ([0, 1, 3, 4], Some(region.clone())),
                ([0, 1, 4, 2], Some(region)),
            ],
            DelaunayTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap();
        let mut inserted_identity = [3; 32];
        inserted_identity[31] = 4;
        let inserted = DelaunayVolumeNode::explicit(
            StableDigest::from_bytes(inserted_identity),
            [0.05, 0.03, 0.02],
        );

        let EdgeStarSplit::Topology(split) = split_complete_edge_star(
            &topology,
            inserted,
            [identity(1), identity(2)],
            DelaunayVolumeSliverOptions::default(),
            &NeverCancelled,
        )
        .unwrap() else {
            panic!("strict interior edge-star point should be admissible");
        };

        assert_eq!(split.nodes.len(), 6);
        assert_eq!(split.tetrahedra.len(), 6);
        assert_eq!(split.incidence.boundary_facets.len(), 6);
    }
}
