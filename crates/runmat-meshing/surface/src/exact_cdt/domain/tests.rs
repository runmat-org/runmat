use std::collections::BTreeSet;

use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind, TopologicalOrientation};
use runmat_meshing_core::{NeverCancelled, StableDigest};

use crate::{
    build_exact_face_pslg, recover_exact_face_segments, triangulate_exact_face_pslg,
    ExactFaceBoundary, ExactFaceBoundaryLoop, ExactFaceBoundarySegment, ExactFaceDelaunayOptions,
};

use super::*;

#[test]
fn oriented_hole_is_removed_by_protected_component_traversal() {
    let boundary = boundary(
        &[[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]],
        &[vec![[-4.0, -2.0], [4.0, -2.0], [0.0, 3.0]]],
    );
    let (pslg, constrained, options) = constrained(&boundary);
    let first =
        carve_exact_face_domain(&constrained, &pslg, &boundary, &NeverCancelled, options).unwrap();
    let second =
        carve_exact_face_domain(&constrained, &pslg, &boundary, &NeverCancelled, options).unwrap();

    assert_eq!(first, second);
    assert_eq!(first.removed_exterior_triangle_count, 0);
    assert!(first.removed_hole_triangle_count > 0);
    assert_eq!(boundary_edges(&first.triangles), pslg_edges(&pslg));
    validate_exact_face_trimmed_delaunay(
        &first,
        &constrained,
        &pslg,
        &boundary,
        &NeverCancelled,
        options,
    )
    .unwrap();
}

#[test]
fn hole_carving_is_independent_of_parametric_loop_winding() {
    let boundary = boundary(
        &[[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]],
        &[vec![[0.0, 3.0], [4.0, -2.0], [-4.0, -2.0]]],
    );
    let (pslg, constrained, options) = constrained(&boundary);
    let trimmed =
        carve_exact_face_domain(&constrained, &pslg, &boundary, &NeverCancelled, options).unwrap();

    assert!(trimmed.removed_hole_triangle_count > 0);
    assert_eq!(boundary_edges(&trimmed.triangles), pslg_edges(&pslg));
}

#[test]
fn concave_outer_loop_removes_the_unprotected_exterior_component() {
    let boundary = boundary(
        &[[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [2.0, 2.0], [0.0, 4.0]],
        &[],
    );
    let (pslg, constrained, options) = constrained(&boundary);
    let trimmed =
        carve_exact_face_domain(&constrained, &pslg, &boundary, &NeverCancelled, options).unwrap();

    assert!(trimmed.removed_exterior_triangle_count > 0);
    assert_eq!(trimmed.removed_hole_triangle_count, 0);
    assert_eq!(boundary_edges(&trimmed.triangles), pslg_edges(&pslg));
}

#[test]
fn independent_validation_rejects_trim_leakage_and_false_evidence() {
    let boundary = boundary(
        &[[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]],
        &[vec![[-4.0, -2.0], [4.0, -2.0], [0.0, 3.0]]],
    );
    let (pslg, constrained, options) = constrained(&boundary);
    let trimmed =
        carve_exact_face_domain(&constrained, &pslg, &boundary, &NeverCancelled, options).unwrap();

    let retained = trimmed.triangles.iter().copied().collect::<BTreeSet<_>>();
    let leaked = constrained
        .triangles
        .iter()
        .copied()
        .find(|triangle| !retained.contains(triangle))
        .unwrap();
    let mut with_leak = trimmed.clone();
    with_leak.triangles.push(leaked);
    with_leak.triangles.sort();
    let error = validate_exact_face_trimmed_delaunay(
        &with_leak,
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

    let mut false_evidence = trimmed;
    false_evidence.removed_hole_triangle_count += 1;
    let error = validate_exact_face_trimmed_delaunay(
        &false_evidence,
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

fn constrained(
    boundary: &ExactFaceBoundary,
) -> (
    crate::ExactFacePslg,
    crate::ExactFaceConstrainedDelaunay,
    ExactFaceDelaunayOptions,
) {
    let pslg = build_exact_face_pslg(boundary).unwrap();
    let options = ExactFaceDelaunayOptions::default();
    let delaunay = triangulate_exact_face_pslg(&pslg, boundary, &NeverCancelled, options).unwrap();
    let constrained =
        recover_exact_face_segments(&delaunay, &pslg, boundary, &NeverCancelled, options).unwrap();
    (pslg, constrained, options)
}

fn boundary(outer: &[[f64; 2]], holes: &[Vec<[f64; 2]>]) -> ExactFaceBoundary {
    let mut next_node = 1u8;
    let mut make_loop = |scope: &str, coordinates: &[[f64; 2]], orientation| {
        let nodes = (0..coordinates.len())
            .map(|_| {
                let node = StableDigest::from_bytes([next_node; 32]);
                next_node += 1;
                node
            })
            .collect::<Vec<_>>();
        ExactFaceBoundaryLoop {
            source_wire_id: id(PersistentEntityKind::Wire, scope),
            orientation,
            segments: (0..coordinates.len())
                .map(|index| ExactFaceBoundarySegment {
                    source_coedge_id: id(
                        PersistentEntityKind::Coedge,
                        &format!("{scope}:coedge:{index}"),
                    ),
                    source_edge_id: id(
                        PersistentEntityKind::Edge,
                        &format!("{scope}:edge:{index}"),
                    ),
                    seam_image: None,
                    node_ids: [nodes[index], nodes[(index + 1) % nodes.len()]],
                    node_uv: [
                        coordinates[index],
                        coordinates[(index + 1) % coordinates.len()],
                    ],
                })
                .collect(),
        }
    };
    ExactFaceBoundary {
        source_face_id: id(PersistentEntityKind::Face, "face"),
        outer_loop: make_loop("outer", outer, TopologicalOrientation::Forward),
        inner_loops: holes
            .iter()
            .enumerate()
            .map(|(index, hole)| {
                make_loop(
                    &format!("hole:{index}"),
                    hole,
                    TopologicalOrientation::Reversed,
                )
            })
            .collect(),
    }
}

fn boundary_edges(triangles: &[crate::ExactFaceDelaunayTriangle]) -> BTreeSet<[u32; 2]> {
    let mut counts = std::collections::BTreeMap::<[u32; 2], usize>::new();
    for triangle in triangles {
        let vertices = triangle.vertex_indices;
        for edge in [
            sorted([vertices[0], vertices[1]]),
            sorted([vertices[1], vertices[2]]),
            sorted([vertices[2], vertices[0]]),
        ] {
            *counts.entry(edge).or_default() += 1;
        }
    }
    counts
        .into_iter()
        .filter_map(|(edge, count)| (count == 1).then_some(edge))
        .collect()
}

fn pslg_edges(pslg: &crate::ExactFacePslg) -> BTreeSet<[u32; 2]> {
    pslg.segments
        .iter()
        .map(|segment| sorted(segment.vertex_indices))
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
