use super::*;
use runmat_meshing_core::NeverCancelled;

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A local cone refill reinserts enclosed original points through the constrained
//   insertion kernel, retains their coordinates and identities, leaves no orphaned vertex, and
//   preserves the exact cavity boundary. Only cavity data is materialized. This fixture is not a
//   whole-task memory measurement or convergence guarantee.
#[test]
fn local_refill_preserves_enclosed_points_and_the_original_cavity_boundary() {
    check_refill(false);
}
// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Local refill moves only explicitly named original enclosed points, preserves the
//   other points and exact cavity boundary, and passes independent canonical construction with
//   every point incident. Duplicate names and protected boundary point names reject. This checks
//   refill authority, not convergence of an original CAD model.
#[test]
fn local_refill_moves_only_explicitly_named_enclosed_points_and_preserves_the_boundary() {
    check_refill(true);
}
fn check_refill(relocate: bool) {
    let (topology, _) = crate::cdt::volume_sliver::tests::interior_sliver_fixture();
    let removed = (0..4).collect();
    let boundary = super::super::visibility::cavity_boundary_faces(&topology, &removed)
        .into_iter()
        .map(|(_, _, face)| face)
        .collect::<Vec<_>>();
    let node = DelaunayVolumeNode::explicit(StableDigest::from_bytes([240; 32]), [0.2, 0.2, 0.2]);
    let moved = DelaunayVolumeNode::explicit(StableDigest::from_bytes([241; 32]), [0.3, 0.1, 0.1]);
    let relocations = if relocate {
        vec![(topology.nodes[4].identity, moved)]
    } else {
        Vec::new()
    };
    let mut work = OptimizationWork::new(1_000_000, 1, &NeverCancelled);
    let created = preserve_vertices(
        &topology,
        &removed,
        &boundary,
        RefillPoint {
            node,
            replaces: None,
            relocations: &relocations,
        },
        Default::default(),
        &mut work,
        &NeverCancelled,
    )
    .unwrap();
    let mut nodes = topology.nodes.clone();
    if relocate {
        nodes[4] = moved;
    }
    nodes.push(node);
    nodes.sort_unstable_by_key(|node| node.identity);
    let index = |identity| {
        nodes
            .binary_search_by_key(&identity, |node| node.identity)
            .unwrap() as u32
    };
    let region = topology.tetrahedra[0].region_id.clone();
    let cells = created
        .into_iter()
        .map(|identities| (identities.map(index), region.clone()))
        .collect();
    let result = crate::cdt::topology::build_canonical_volume_topology(
        nodes,
        cells,
        Default::default(),
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(result.nodes.len(), topology.nodes.len() + 1);
    assert!(result
        .incidence
        .vertex_stars
        .iter()
        .all(|star| !star.is_empty()));
    for (index, original) in topology.nodes.iter().enumerate() {
        if relocate && index == 4 {
            assert!(result.nodes.contains(&moved));
            assert!(!result.nodes.contains(original));
        } else {
            assert!(result.nodes.contains(original));
        }
    }
    if relocate {
        for invalid in [
            vec![(topology.nodes[0].identity, moved)],
            vec![relocations[0], relocations[0]],
        ] {
            assert!(preserve_vertices(
                &topology,
                &removed,
                &boundary,
                RefillPoint {
                    node,
                    replaces: None,
                    relocations: &invalid
                },
                Default::default(),
                &mut work,
                &NeverCancelled
            )
            .is_err());
        }
    }
    let expected = boundary
        .iter()
        .map(|face| face.map(|vertex| topology.nodes[vertex as usize].identity))
        .collect::<BTreeSet<_>>();
    let actual = result
        .incidence
        .boundary_facets
        .iter()
        .map(|facet| {
            let mut identities = facet
                .vertex_indices
                .map(|vertex| result.nodes[vertex as usize].identity);
            identities.sort_unstable();
            identities
        })
        .collect();
    assert_eq!(expected, actual);
}
