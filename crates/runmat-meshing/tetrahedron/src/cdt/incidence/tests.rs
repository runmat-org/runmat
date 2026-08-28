use runmat_meshing_core::{
    quality::predicate::{orient3d, PredicateSign},
    NeverCancelled, StableDigest,
};

use super::*;
use crate::cdt::{
    build_delaunay_volume_topology, insert_delaunay_volume_node, validate_delaunay_volume_topology,
    DelaunayInsertionErrorKind, DelaunayInsertionOptions,
};

fn node(identity: u8, coordinates_m: [f64; 3]) -> DelaunayVolumeNode {
    DelaunayVolumeNode {
        identity: StableDigest::from_bytes([identity; 32]),
        coordinates_m,
    }
}

fn region(id: &str) -> PersistentEntityId {
    PersistentEntityId {
        kind: PersistentEntityKind::Region,
        source_topology_id: id.to_owned(),
        assembly_path: Vec::new(),
    }
}

fn bipyramid() -> DelaunayVolumeTopology {
    build_delaunay_volume_topology(
        vec![
            node(10, [0.0, 0.0, 0.0]),
            node(20, [2.0, 0.0, 0.0]),
            node(30, [0.0, 2.0, 0.0]),
            node(40, [0.0, 0.0, 2.0]),
            node(50, [0.0, 0.0, -2.0]),
        ],
        vec![[0, 1, 2, 3], [0, 2, 1, 4]],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
}

#[test]
fn incidence_derives_vertex_stars_and_outward_boundary_facets() {
    let topology = bipyramid();
    assert_eq!(topology.incidence.vertex_stars[0], vec![0, 1]);
    assert_eq!(topology.incidence.vertex_stars[3], vec![0]);
    assert_eq!(topology.incidence.boundary_facets.len(), 6);
    assert_eq!(
        topology.incidence.unassigned_tetrahedron_indices,
        vec![0, 1]
    );
    assert!(topology.incidence.regions.is_empty());

    for facet in &topology.incidence.boundary_facets {
        let tetrahedron = &topology.tetrahedra[facet.tetrahedron_index as usize];
        let opposite = tetrahedron.vertex_indices[facet.opposite_vertex_slot as usize];
        let points = [
            topology.nodes[facet.oriented_vertex_indices[0] as usize].coordinates_m,
            topology.nodes[facet.oriented_vertex_indices[1] as usize].coordinates_m,
            topology.nodes[facet.oriented_vertex_indices[2] as usize].coordinates_m,
            topology.nodes[opposite as usize].coordinates_m,
        ];
        assert_eq!(orient3d(points).unwrap(), PredicateSign::Positive);
    }
}

#[test]
fn checked_region_assignment_builds_canonical_incidence_and_rejects_bad_input() {
    let topology = assign_delaunay_volume_regions(
        bipyramid(),
        vec![region("solid:a"), region("solid:b")],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    assert!(topology.incidence.unassigned_tetrahedron_indices.is_empty());
    assert_eq!(topology.incidence.regions.len(), 2);
    assert_eq!(topology.incidence.regions[0].tetrahedron_indices, vec![0]);
    assert_eq!(topology.incidence.regions[1].tetrahedron_indices, vec![1]);
    validate_delaunay_volume_topology(
        &topology,
        DelaunayInsertionOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    assert_eq!(
        assign_delaunay_volume_regions(
            bipyramid(),
            vec![region("solid:a")],
            DelaunayTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayTopologyErrorKind::InvalidRegion
    );
    let mut wrong_kind = region("face:a");
    wrong_kind.kind = PersistentEntityKind::Face;
    assert_eq!(
        assign_delaunay_volume_regions(
            bipyramid(),
            vec![wrong_kind.clone(), wrong_kind],
            DelaunayTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayTopologyErrorKind::InvalidRegion
    );
}

#[test]
fn insertion_preserves_one_region_and_rejects_cross_region_cavities() {
    let single = build_delaunay_volume_topology(
        bipyramid().nodes[..4].to_vec(),
        vec![[0, 1, 2, 3]],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let assigned = assign_delaunay_volume_regions(
        single,
        vec![region("solid:a")],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let refined = insert_delaunay_volume_node(
        assigned,
        node(25, [0.25, 0.25, 0.25]),
        DelaunayInsertionOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(refined.incidence.regions.len(), 1);
    assert_eq!(refined.incidence.regions[0].tetrahedron_indices.len(), 4);

    let split = assign_delaunay_volume_regions(
        bipyramid(),
        vec![region("solid:a"), region("solid:b")],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(
        insert_delaunay_volume_node(
            split,
            node(25, [0.5, 0.5, 0.0]),
            DelaunayInsertionOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayInsertionErrorKind::InvalidTopology
    );
}

#[test]
fn independent_validation_rejects_tampered_incidence() {
    let mut topology = bipyramid();
    topology.incidence.vertex_stars[0].clear();
    assert_eq!(
        validate_delaunay_volume_topology(
            &topology,
            DelaunayInsertionOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayInsertionErrorKind::InvalidTopology
    );
}

struct Cancelled;

impl MeshingCancellationSignal for Cancelled {
    fn is_cancelled(&self) -> bool {
        true
    }
}

#[test]
fn region_assignment_honors_cancellation_before_mutation() {
    let options = DelaunayTopologyOptions {
        cancellation_check_interval: 1,
        ..DelaunayTopologyOptions::default()
    };
    assert_eq!(
        assign_delaunay_volume_regions(
            bipyramid(),
            vec![region("solid:a"), region("solid:b")],
            options,
            &Cancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayTopologyErrorKind::Cancelled
    );
}
