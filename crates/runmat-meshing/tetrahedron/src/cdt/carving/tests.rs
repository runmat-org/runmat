use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
use runmat_meshing_core::{
    contracts::{MeshingStage, TopologyEntityId},
    MeshingCancellationSignal, NeverCancelled, StableDigest,
};

use super::*;
use crate::cdt::{
    build_delaunay_volume_topology, recover_delaunay_facets, recover_delaunay_segments,
    DelaunayConstraintFacet, DelaunayConstraintNode, DelaunayConstraintSegment,
    DelaunaySegmentRecoveryOptions, DelaunayTopologyOptions,
};

fn fixture() -> (DelaunayConstraints, DelaunayFacetRecovery) {
    let coordinates = [
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [-3.0, -4.0, 0.0],
        [0.0, 0.0, 5.0],
        [0.0, 0.0, -5.0],
    ];
    let facet_vertices = [
        [0, 1, 2],
        [0, 1, 3],
        [0, 1, 4],
        [0, 2, 3],
        [0, 2, 4],
        [1, 2, 3],
        [1, 2, 4],
    ];
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
                source_node_id: id(
                    MeshingStage::ProtectedBoundaryComplex,
                    &format!("node:{index}"),
                ),
                coordinates_m,
            })
            .collect(),
        segments: segment_vertices
            .into_iter()
            .map(|vertex_indices| DelaunayConstraintSegment {
                vertex_indices,
                protected_edge_id: None,
                source_edge_id: None,
            })
            .collect(),
        facets: facet_vertices
            .into_iter()
            .enumerate()
            .map(|(index, vertex_indices)| DelaunayConstraintFacet {
                facet_id: id(
                    MeshingStage::ProtectedBoundaryComplex,
                    &format!("facet:{index}"),
                ),
                vertex_indices,
                source_face_id: id(MeshingStage::SurfaceMesh, &format!("face:{index}")),
                material_interface_ids: Vec::new(),
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

fn seeds() -> DelaunayCarvingSeeds {
    DelaunayCarvingSeeds {
        regions: vec![DelaunayRegionSeed {
            region_id: region("upper"),
            coordinates_m: [0.5, 0.25, 1.0],
        }],
        voids: vec![DelaunayVoidSeed {
            coordinates_m: [0.5, 0.25, -1.0],
        }],
    }
}

#[test]
fn carving_floods_only_across_unconstrained_faces() {
    let (constraints, recovery) = fixture();
    let carving = carve_delaunay_volume(
        &recovery,
        &constraints,
        &seeds(),
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
    validate_delaunay_carving(
        &recovery,
        &constraints,
        &seeds(),
        &carving,
        DelaunayCarvingOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
}

#[test]
fn carving_rejects_missing_component_seeds_and_tampering() {
    let (constraints, recovery) = fixture();
    let missing_void = DelaunayCarvingSeeds {
        regions: seeds().regions,
        voids: Vec::new(),
    };
    assert_eq!(
        carve_delaunay_volume(
            &recovery,
            &constraints,
            &missing_void,
            DelaunayCarvingOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayCarvingErrorKind::AmbiguousClassification
    );

    let mut carving = carve_delaunay_volume(
        &recovery,
        &constraints,
        &seeds(),
        DelaunayCarvingOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    carving.removed_tetrahedra.clear();
    assert_eq!(
        validate_delaunay_carving(
            &recovery,
            &constraints,
            &seeds(),
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
    let (constraints, recovery) = fixture();
    let bounded = DelaunayCarvingOptions {
        maximum_flood_steps: 1,
        ..DelaunayCarvingOptions::default()
    };
    assert_eq!(
        carve_delaunay_volume(&recovery, &constraints, &seeds(), bounded, &NeverCancelled)
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
        carve_delaunay_volume(&recovery, &constraints, &seeds(), cancelled, &Cancelled)
            .unwrap_err()
            .kind,
        DelaunayCarvingErrorKind::Cancelled
    );
}

fn region(value: &str) -> PersistentEntityId {
    PersistentEntityId {
        kind: PersistentEntityKind::Region,
        source_topology_id: value.to_owned(),
        assembly_path: Vec::new(),
    }
}

fn id(stage: MeshingStage, value: &str) -> TopologyEntityId {
    TopologyEntityId {
        stage,
        id: value.to_owned(),
    }
}
