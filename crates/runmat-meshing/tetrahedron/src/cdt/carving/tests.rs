use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
use runmat_meshing_core::{
    quality::predicate::{orient3d, PredicateSign},
    MeshingCancellationSignal, NeverCancelled, StableDigest,
};

use super::*;
use crate::cdt::{
    build_delaunay_volume_point_set, build_delaunay_volume_topology, recover_delaunay_facets,
    recover_delaunay_segments, DelaunayConstraintFacet, DelaunayConstraintFacetSide,
    DelaunayConstraintNode, DelaunayConstraintSegment, DelaunayPointSetOptions,
    DelaunaySegmentRecoveryOptions, DelaunayTopologyOptions,
};

#[path = "tests/close_parallel.rs"]
mod close_parallel;
#[path = "tests/small_void.rs"]
mod small_void;

fn entity(kind: PersistentEntityKind, value: &str) -> PersistentEntityId {
    PersistentEntityId {
        kind,
        source_topology_id: value.to_owned(),
        assembly_path: Vec::new(),
    }
}

fn fixture(interface: bool) -> (DelaunayConstraints, DelaunayFacetRecovery) {
    let coordinates = [
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [-3.0, -4.0, 0.0],
        [0.0, 0.0, 5.0],
        [0.0, 0.0, -5.0],
    ];
    let all_facets = [
        [0, 1, 2],
        [0, 1, 3],
        [0, 1, 4],
        [0, 2, 3],
        [0, 2, 4],
        [1, 2, 3],
        [1, 2, 4],
    ];
    let facet_vertices = if interface {
        all_facets.to_vec()
    } else {
        vec![all_facets[0], all_facets[1], all_facets[3], all_facets[5]]
    };
    let mut segment_vertices = facet_vertices
        .iter()
        .flat_map(|facet| {
            let mut edges = [
                [facet[0], facet[1]],
                [facet[1], facet[2]],
                [facet[2], facet[0]],
            ];
            for edge in &mut edges {
                edge.sort_unstable();
            }
            edges
        })
        .collect::<Vec<_>>();
    segment_vertices.sort_unstable();
    segment_vertices.dedup();
    let constraints = DelaunayConstraints {
        nodes: coordinates
            .into_iter()
            .enumerate()
            .map(|(index, coordinates_m)| DelaunayConstraintNode {
                identity: StableDigest::from_bytes([(index + 70) as u8; 32]),
                source_vertex_id: None,
                coordinates_m,
            })
            .collect(),
        segments: segment_vertices
            .into_iter()
            .map(|vertex_indices| DelaunayConstraintSegment {
                vertex_indices,
                source_edge_id: None,
                source_edge_parameters: None,
            })
            .collect(),
        facets: facet_vertices
            .iter()
            .copied()
            .enumerate()
            .map(|(index, vertex_indices)| DelaunayConstraintFacet {
                facet_id: StableDigest::from_bytes([(index + 90) as u8; 32]),
                chart_id: StableDigest::from_bytes([(index + 120) as u8; 32]),
                vertex_indices,
                source_face_id: entity(PersistentEntityKind::Face, &format!("face:{index}")),
                positive_side: if vertex_indices == [0, 1, 2] {
                    DelaunayConstraintFacetSide::Region(region("upper"))
                } else {
                    boundary_sides(
                        coordinates,
                        vertex_indices,
                        opposite_vertex(vertex_indices),
                        if vertex_indices.contains(&3) {
                            "upper"
                        } else {
                            "lower"
                        },
                    )
                    .0
                },
                negative_side: if vertex_indices == [0, 1, 2] {
                    if interface {
                        DelaunayConstraintFacetSide::Region(region("lower"))
                    } else {
                        DelaunayConstraintFacetSide::Exterior
                    }
                } else {
                    boundary_sides(
                        coordinates,
                        vertex_indices,
                        opposite_vertex(vertex_indices),
                        if vertex_indices.contains(&3) {
                            "upper"
                        } else {
                            "lower"
                        },
                    )
                    .1
                },
                contact_ids: Vec::new(),
            })
            .collect(),
    };
    let topology = build_delaunay_volume_topology(
        constraints.volume_nodes(),
        vec![[0, 1, 2, 3], [0, 2, 1, 4]],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let segments = recover_delaunay_segments(
        topology,
        &constraints,
        DelaunaySegmentRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let facets = recover_delaunay_facets(
        segments,
        &constraints,
        DelaunayFacetRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    (constraints, facets)
}

#[test]
fn carving_floods_only_across_unconstrained_faces() {
    let (constraints, recovery) = fixture(false);
    let carving = carve_delaunay_volume(
        &recovery,
        &constraints,
        DelaunayCarvingOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    assert_eq!(carving.topology.tetrahedra.len(), 1);
    assert_eq!(carving.topology.nodes.len(), 4);
    assert_eq!(carving.removed_tetrahedra.len(), 1);
    assert_eq!(carving.topology.incidence.regions.len(), 1);
    assert_eq!(
        carving.topology.incidence.regions[0].region_id,
        region("upper")
    );
    assert!(carving
        .topology
        .incidence
        .unassigned_tetrahedron_indices
        .is_empty());
    assert_eq!(carving.facets[0].region_ids, vec![region("upper")]);
    assert!(!carving.facets[0].borders_void);
    assert!(carving.facets[0].borders_exterior);
    validate_delaunay_carving(
        &recovery,
        &constraints,
        &carving,
        DelaunayCarvingOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
}

#[test]
fn carving_classifies_conformal_region_interfaces() {
    let (constraints, recovery) = fixture(true);

    let carving = carve_delaunay_volume(
        &recovery,
        &constraints,
        DelaunayCarvingOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    assert_eq!(carving.topology.tetrahedra.len(), 2);
    assert!(carving.removed_tetrahedra.is_empty());
    assert_eq!(carving.topology.incidence.regions.len(), 2);
    assert_eq!(
        carving.facets[0].region_ids,
        vec![region("lower"), region("upper")]
    );
    assert!(!carving.facets[0].borders_exterior);
    assert!(!carving.facets[0].borders_void);
    validate_delaunay_carving(
        &recovery,
        &constraints,
        &carving,
        DelaunayCarvingOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
}

#[test]
fn carving_rejects_contradictory_facet_sides_and_tampering() {
    let (mut constraints, recovery) = fixture(false);
    constraints.facets[0].negative_side = DelaunayConstraintFacetSide::Void;
    assert_eq!(
        carve_delaunay_volume(
            &recovery,
            &constraints,
            DelaunayCarvingOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayCarvingErrorKind::AmbiguousClassification
    );

    let (constraints, recovery) = fixture(false);
    let mut carving = carve_delaunay_volume(
        &recovery,
        &constraints,
        DelaunayCarvingOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    carving.removed_tetrahedra.clear();
    assert_eq!(
        validate_delaunay_carving(
            &recovery,
            &constraints,
            &carving,
            DelaunayCarvingOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayCarvingErrorKind::InvalidTopology
    );
}

struct Cancelled;

impl MeshingCancellationSignal for Cancelled {
    fn is_cancelled(&self) -> bool {
        true
    }
}

#[test]
fn carving_enforces_work_limits_and_cancellation() {
    let (constraints, recovery) = fixture(false);
    assert_eq!(
        carve_delaunay_volume(
            &recovery,
            &constraints,
            DelaunayCarvingOptions {
                maximum_flood_steps: 0,
                ..DelaunayCarvingOptions::default()
            },
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayCarvingErrorKind::InvalidOptions
    );
    let bounded = DelaunayCarvingOptions {
        maximum_flood_steps: 1,
        ..DelaunayCarvingOptions::default()
    };
    assert_eq!(
        carve_delaunay_volume(&recovery, &constraints, bounded, &NeverCancelled)
            .unwrap_err()
            .kind,
        DelaunayCarvingErrorKind::ResourceLimit
    );
    let cancelled = DelaunayCarvingOptions {
        facet_recovery: DelaunayFacetRecoveryOptions {
            segment_recovery: DelaunaySegmentRecoveryOptions {
                constraints: crate::cdt::DelaunayConstraintOptions {
                    cancellation_check_interval: 1,
                    ..crate::cdt::DelaunayConstraintOptions::default()
                },
                ..DelaunaySegmentRecoveryOptions::default()
            },
            ..DelaunayFacetRecoveryOptions::default()
        },
        ..DelaunayCarvingOptions::default()
    };
    assert_eq!(
        carve_delaunay_volume(&recovery, &constraints, cancelled, &Cancelled)
            .unwrap_err()
            .kind,
        DelaunayCarvingErrorKind::Cancelled
    );
}

fn boundary_sides(
    coordinates: [[f64; 3]; 5],
    facet: [u32; 3],
    opposite: u32,
    region_id: &str,
) -> (DelaunayConstraintFacetSide, DelaunayConstraintFacetSide) {
    sides_containing_point(
        coordinates,
        facet,
        coordinates[opposite as usize],
        region_id,
    )
}

fn sides_containing_point<const N: usize>(
    coordinates: [[f64; 3]; N],
    facet: [u32; 3],
    contained_point: [f64; 3],
    region_id: &str,
) -> (DelaunayConstraintFacetSide, DelaunayConstraintFacetSide) {
    let region = DelaunayConstraintFacetSide::Region(region(region_id));
    sides_for_point(
        coordinates,
        facet,
        contained_point,
        region,
        DelaunayConstraintFacetSide::Exterior,
    )
}

fn sides_for_point<const N: usize>(
    coordinates: [[f64; 3]; N],
    facet: [u32; 3],
    contained_point: [f64; 3],
    contained_side: DelaunayConstraintFacetSide,
    other_side: DelaunayConstraintFacetSide,
) -> (DelaunayConstraintFacetSide, DelaunayConstraintFacetSide) {
    let points = facet.map(|vertex| coordinates[vertex as usize]);
    match orient3d([points[0], points[1], points[2], contained_point]).unwrap() {
        PredicateSign::Negative => (contained_side, other_side),
        PredicateSign::Positive => (other_side, contained_side),
        PredicateSign::Zero => panic!("fixture facet must be nondegenerate"),
    }
}

fn opposite_vertex(facet: [u32; 3]) -> u32 {
    let tetrahedron = if facet.contains(&3) {
        [0, 1, 2, 3]
    } else {
        [0, 1, 2, 4]
    };
    tetrahedron
        .into_iter()
        .find(|vertex| !facet.contains(vertex))
        .unwrap()
}

fn region(value: &str) -> PersistentEntityId {
    PersistentEntityId {
        kind: PersistentEntityKind::Region,
        source_topology_id: value.to_owned(),
        assembly_path: Vec::new(),
    }
}
