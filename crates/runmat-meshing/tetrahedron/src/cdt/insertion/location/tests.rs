use super::*;
use crate::cdt::{
    build_delaunay_volume_topology, mutation::TopologyMutation, DelaunayTopologyOptions,
};
use runmat_meshing_core::{NeverCancelled, StableDigest};

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: After explicitly separate index construction, a bad walking hint still locates the
//   independently specified final disconnected cell and its exact boundary, and rejects a one-step
//   outside point across 16/256 cells. Query predicate and checkpoint bounds exclude repeated
//   whole-inventory traversal in this family; mutation cost, arbitrary cell density, complete
//   native study work and partial CAD loading remain separate evidence.
#[test]
fn maintained_location_finds_disconnected_cells_and_exact_boundary_without_global_scan() {
    for count in [16, 256] {
        let mut nodes = Vec::new();
        let mut cells = Vec::new();
        for cluster in 0..count {
            let first = nodes.len() as u32;
            for point in [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]] {
                let mut identity = [1; 32];
                identity[..8].copy_from_slice(&((nodes.len() + 1) as u64).to_be_bytes());
                nodes.push(DelaunayVolumeNode::explicit(
                    StableDigest::from_bytes(identity),
                    [point[0] + cluster as f64 * 10., point[1], point[2]],
                ));
            }
            cells.push([first, first + 1, first + 2, first + 3]);
        }
        let topology = TopologyMutation::from_canonical(
            build_delaunay_volume_topology(
                nodes,
                cells,
                DelaunayTopologyOptions::default(),
                &NeverCancelled,
            )
            .unwrap(),
        );
        topology
            .prepare_tetrahedron_candidates(&mut || Ok::<_, ()>(()))
            .unwrap();
        let locator = TetrahedronPointLocator::walking(&topology);
        for (height, contained) in [(0.1, true), (0., true), (0.0_f64.next_down(), false)] {
            let query = DelaunayVolumeNode::explicit(
                StableDigest::from_bytes([255; 32]),
                [(count - 1) as f64 * 10. + 0.1, 0.1, height],
            );
            let mut work = InsertionWork::new(Default::default(), &NeverCancelled);
            let found = locator
                .first_containing_from(query, Some(0), &mut work)
                .unwrap();
            assert_eq!(found.is_some(), contained);
            if let Some(found) = found {
                assert_eq!(
                    topology
                        .tetrahedron(found)
                        .unwrap()
                        .vertex_indices
                        .iter()
                        .map(|index| index / 4)
                        .collect::<BTreeSet<_>>(),
                    BTreeSet::from([(count - 1) as u32])
                );
            }
            assert!(
                work.inventory().0 <= 12,
                "only local exact containment predicates"
            );
            assert!(
                work.inventory().1 <= 64,
                "indexed fallback work must not scan all cells"
            );
        }
    }
}
