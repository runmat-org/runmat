use std::collections::BTreeSet;

use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind, TopologicalOrientation};
use runmat_meshing_core::{NeverCancelled, StableDigest};

use super::*;
use crate::{
    build_exact_face_pslg, triangulate_exact_face_pslg, ExactFaceBoundary, ExactFaceBoundaryLoop,
    ExactFaceBoundarySegment, ExactFaceDelaunayOptions,
};

#[test]
fn concave_face_recovers_every_missing_protected_segment() {
    let (boundary, pslg) = concave_fixture();
    let options = ExactFaceDelaunayOptions::default();
    let delaunay = triangulate_exact_face_pslg(&pslg, &boundary, &NeverCancelled, options).unwrap();
    let initial_edges = triangle_edges(&delaunay.triangles);
    assert!(pslg
        .segments
        .iter()
        .any(|segment| !initial_edges.contains(&sorted(segment.vertex_indices))));

    let constrained =
        recover_exact_face_segments(&delaunay, &pslg, &boundary, &NeverCancelled, options).unwrap();
    let recovered_edges = triangle_edges(&constrained.triangles);
    assert!(pslg
        .segments
        .iter()
        .all(|segment| recovered_edges.contains(&sorted(segment.vertex_indices))));
    validate_exact_face_constrained_delaunay(
        &constrained,
        &pslg,
        &boundary,
        &NeverCancelled,
        options,
    )
    .unwrap();
    assert!(constrained.recovery_edge_flip_count > 0);
}

#[test]
fn deterministic_cavity_retriangulates_an_intersected_triangle_strip() {
    let (boundary, pslg) = concave_fixture();
    let options = ExactFaceDelaunayOptions {
        maximum_cavity_retriangulations: 1,
        ..ExactFaceDelaunayOptions::default()
    };
    let delaunay = triangulate_exact_face_pslg(&pslg, &boundary, &NeverCancelled, options).unwrap();
    let initial_edges = triangle_edges(&delaunay.triangles);
    let target = pslg
        .segments
        .iter()
        .map(|segment| sorted(segment.vertex_indices))
        .find(|edge| !initial_edges.contains(edge))
        .expect("fixture must have a missing protected segment");
    let mut control = super::recover::RecoveryControl::new(&pslg, &NeverCancelled, options);
    let mut triangles = delaunay.triangles.clone();

    assert!(super::cavity::recover_segment_cavity(
        &mut triangles,
        &pslg,
        target,
        &BTreeSet::new(),
        &mut control,
    )
    .unwrap());
    assert_eq!(triangles.len(), delaunay.triangles.len());
    assert!(triangle_edges(&triangles).contains(&target));
    super::planarity::validate_planar_edges(
        &super::super::topology::edge_uses(&triangles),
        &pslg,
        &NeverCancelled,
        options,
    )
    .unwrap();

    let mut repeated_control =
        super::recover::RecoveryControl::new(&pslg, &NeverCancelled, options);
    let mut repeated = delaunay.triangles.clone();
    assert!(super::cavity::recover_segment_cavity(
        &mut repeated,
        &pslg,
        target,
        &BTreeSet::new(),
        &mut repeated_control,
    )
    .unwrap());
    assert_eq!(repeated, triangles);

    let mut second_attempt = delaunay.triangles.clone();
    let error = super::cavity::recover_segment_cavity(
        &mut second_attempt,
        &pslg,
        target,
        &BTreeSet::new(),
        &mut control,
    )
    .unwrap_err();
    assert_eq!(
        error.kind,
        crate::ExactFaceDelaunayErrorKind::IterationLimit
    );
    assert_eq!(second_attempt, delaunay.triangles);
}

#[test]
fn independent_validation_rejects_missing_protected_provenance() {
    let (boundary, pslg) = concave_fixture();
    let options = ExactFaceDelaunayOptions::default();
    let delaunay = triangulate_exact_face_pslg(&pslg, &boundary, &NeverCancelled, options).unwrap();
    let mut constrained =
        recover_exact_face_segments(&delaunay, &pslg, &boundary, &NeverCancelled, options).unwrap();
    constrained.protected_segments.pop();

    let error = validate_exact_face_constrained_delaunay(
        &constrained,
        &pslg,
        &boundary,
        &NeverCancelled,
        options,
    )
    .unwrap_err();
    assert_eq!(
        error.kind,
        crate::ExactFaceDelaunayErrorKind::InvalidTopology
    );
}

#[test]
fn independent_validation_rejects_excess_cavity_evidence() {
    let (boundary, pslg) = concave_fixture();
    let options = ExactFaceDelaunayOptions::default();
    let delaunay = triangulate_exact_face_pslg(&pslg, &boundary, &NeverCancelled, options).unwrap();
    let mut constrained =
        recover_exact_face_segments(&delaunay, &pslg, &boundary, &NeverCancelled, options).unwrap();
    constrained.cavity_retriangulation_count = options.maximum_cavity_retriangulations + 1;

    let error = validate_exact_face_constrained_delaunay(
        &constrained,
        &pslg,
        &boundary,
        &NeverCancelled,
        options,
    )
    .unwrap_err();
    assert_eq!(
        error.kind,
        crate::ExactFaceDelaunayErrorKind::IterationLimit
    );
}

fn concave_fixture() -> (ExactFaceBoundary, crate::ExactFacePslg) {
    let outer_coordinates = [[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]];
    let inner_coordinates = [[-4.0, 0.0], [4.0, 0.0], [0.0, 0.1]];
    let outer_nodes = (1u8..=4)
        .map(|value| StableDigest::from_bytes([value; 32]))
        .collect::<Vec<_>>();
    let inner_nodes = (5u8..=7)
        .map(|value| StableDigest::from_bytes([value; 32]))
        .collect::<Vec<_>>();
    let boundary = ExactFaceBoundary {
        source_face_id: id(PersistentEntityKind::Face, "face"),
        outer_loop: ExactFaceBoundaryLoop {
            source_wire_id: id(PersistentEntityKind::Wire, "outer"),
            orientation: TopologicalOrientation::Forward,
            segments: loop_segments("outer", &outer_coordinates, &outer_nodes),
        },
        inner_loops: vec![ExactFaceBoundaryLoop {
            source_wire_id: id(PersistentEntityKind::Wire, "inner"),
            orientation: TopologicalOrientation::Reversed,
            segments: loop_segments("inner", &inner_coordinates, &inner_nodes),
        }],
    };
    let pslg = build_exact_face_pslg(&boundary).unwrap();
    (boundary, pslg)
}

fn loop_segments(
    scope: &str,
    coordinates: &[[f64; 2]],
    nodes: &[StableDigest],
) -> Vec<ExactFaceBoundarySegment> {
    (0..coordinates.len())
        .map(|index| ExactFaceBoundarySegment {
            source_coedge_id: id(
                PersistentEntityKind::Coedge,
                &format!("{scope}:coedge:{index}"),
            ),
            source_edge_id: id(PersistentEntityKind::Edge, &format!("{scope}:edge:{index}")),
            seam_image: None,
            node_ids: [nodes[index], nodes[(index + 1) % nodes.len()]],
            edge_parameters: [0.0, 1.0],
            node_uv: [
                coordinates[index],
                coordinates[(index + 1) % coordinates.len()],
            ],
        })
        .collect()
}

fn triangle_edges(triangles: &[crate::ExactFaceDelaunayTriangle]) -> BTreeSet<[u32; 2]> {
    triangles
        .iter()
        .flat_map(|triangle| {
            let vertices = triangle.vertex_indices;
            [
                sorted([vertices[0], vertices[1]]),
                sorted([vertices[1], vertices[2]]),
                sorted([vertices[2], vertices[0]]),
            ]
        })
        .collect()
}

fn sorted(mut edge: [u32; 2]) -> [u32; 2] {
    edge.sort_unstable();
    edge
}

fn id(kind: PersistentEntityKind, name: &str) -> PersistentEntityId {
    PersistentEntityId {
        kind,
        source_topology_id: name.into(),
        assembly_path: vec!["root".into()],
    }
}
