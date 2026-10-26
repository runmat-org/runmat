use super::*;
use crate::cdt::{build_delaunay_volume_topology, DelaunayVolumeNode};
use runmat_meshing_core::{NeverCancelled, StableDigest};

fn topology(extra: &[[f64; 3]], scale: f64, permutation: [usize; 3]) -> DelaunayVolumeTopology {
    let corners = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let nodes = corners
        .iter()
        .chain(extra)
        .enumerate()
        .map(|(index, point)| {
            let mut identity = [0u8; 32];
            identity[..8].copy_from_slice(&(index as u64 + 1).to_be_bytes());
            DelaunayVolumeNode::explicit(
                StableDigest::from_bytes(identity),
                permutation.map(|axis| point[axis] * scale),
            )
        })
        .collect();
    build_delaunay_volume_topology(
        nodes,
        vec![[0, 1, 2, 3]],
        Default::default(),
        &NeverCancelled,
    )
    .unwrap()
}
fn exhaustive(topology: &DelaunayVolumeTopology) -> bool {
    topology.tetrahedra.iter().any(|cell| {
        let points = cell.vertex_indices.map(|vertex| {
            topology.nodes[vertex as usize]
                .coordinates_m
                .canonical_value()
        });
        topology.nodes.iter().enumerate().any(|(index, node)| {
            !cell.vertex_indices.contains(&(index as u32))
                && (0..4).all(|replace| {
                    let mut simplex = points;
                    simplex[replace] = node.coordinates_m.canonical_value();
                    orient3d(simplex).unwrap() == PredicateSign::Positive
                })
        })
    })
}
// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Strict containment agrees with exhaustive adaptive predicates at faces, edges and
//   adjacent floating-point positions across three scales and three coordinate permutations. An
//   exact open-box broad phase is not a replacement for canonical topology, Delaunay, physical
//   quality or native throughput admission.
#[test]
fn indexed_containment_matches_exhaustive_predicates_at_faces_edges_and_nearby_points() {
    let below = f64::from_bits(0.5f64.to_bits() - 1);
    let above = f64::from_bits(0.5f64.to_bits() + 1);
    let cases = [
        vec![],
        vec![[0.0, 0.25, 0.25]],
        vec![[0.0, 0.0, 0.25]],
        vec![[0.25, 0.25, 0.5]],
        vec![[0.25, 0.25, below]],
        vec![[0.25, 0.25, above]],
        vec![[0.125, 0.125, 0.125]],
        vec![[-0.125, 0.25, 0.25]],
    ];
    for scale in [1e-6, 1.0, 1e6] {
        for permutation in [[0, 1, 2], [2, 0, 1], [1, 0, 2]] {
            for points in &cases {
                let mesh = topology(points, scale, permutation);
                assert_eq!(
                    contains_embedded_node(&mesh, Default::default(), &NeverCancelled).unwrap(),
                    exhaustive(&mesh),
                    "scale={scale} permutation={permutation:?} points={points:?}"
                );
            }
        }
    }
}
// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A 1,028-node coplanar inventory produces zero strict-interior candidates with fewer
//   than 128 spatial-query checkpoint visits and matches exhaustive containment. This bounds the
//   declared aligned-plane query, not arbitrary occupied geometry, index construction cost,
//   measured allocator storage or whole-study scaling.
#[test]
fn aligned_plane_inventory_is_rejected_before_adaptive_predicates() {
    let points = (0..32)
        .flat_map(|y| {
            (0..32).map(move |z| {
                [
                    0.0,
                    (f64::from(y) + 0.25) / 32.0,
                    (f64::from(z) + 0.25) / 32.0,
                ]
            })
        })
        .collect::<Vec<_>>();
    let mesh = topology(&points, 1.0, [0, 1, 2]);
    let bounds = bounds(
        mesh.tetrahedra[0]
            .vertex_indices
            .map(|vertex| mesh.nodes[vertex as usize].coordinates_m.canonical_value()),
    );
    assert_eq!(mesh.nodes.len(), 1028);
    let entries = mesh
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| {
            SpatialPoint::new(node.coordinates_m.canonical_value(), index).unwrap()
        })
        .collect();
    let index = PointSpatialIndex::from_entries_with_checkpoint(entries, &mut || {
        Ok::<_, DelaunayVolumeSliverError>(())
    })
    .unwrap();
    let mut visited = 0;
    let candidates = candidate_nodes(&index, bounds, &mut || {
        visited += 1;
        Ok(())
    })
    .unwrap();
    assert!(candidates.is_empty());
    assert!(
        visited < 128,
        "coplanar branches must be pruned rather than visited point by point: {visited}"
    );
    assert!(!contains_embedded_node(&mesh, Default::default(), &NeverCancelled).unwrap());
    assert!(!exhaustive(&mesh));
}
#[test]
fn indexed_containment_checks_cancellation_during_index_construction() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct CancelAfter(AtomicUsize);
    impl MeshingCancellationSignal for CancelAfter {
        fn is_cancelled(&self) -> bool {
            self.0.fetch_add(1, Ordering::Relaxed) >= 2
        }
    }
    let mesh = topology(&[[0.2, 0.2, 0.2]], 1.0, [0, 1, 2]);
    let options = DelaunayVolumeSliverOptions {
        cancellation_check_interval: 1,
        ..Default::default()
    };
    assert!(
        matches!(contains_embedded_node(&mesh,options,&CancelAfter(AtomicUsize::new(0))),Err(failure) if failure.kind==DelaunayVolumeSliverErrorKind::Cancelled)
    );
}
