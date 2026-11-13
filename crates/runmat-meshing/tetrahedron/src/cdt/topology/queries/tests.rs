use super::*;
use crate::cdt::topology::VolumeTopologyRead;
use crate::cdt::{build_delaunay_volume_topology, DelaunayTopologyOptions, DelaunayVolumeNode};
use runmat_meshing_core::NeverCancelled;

pub(in crate::cdt) fn topology(unrelated_cells: usize) -> DelaunayVolumeTopology {
    let mut points = vec![
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.25, 0.25, 0.25],
    ];
    let mut cells = vec![[0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4]];
    for index in 0..unrelated_cells {
        let base = points.len() as u32;
        let x = 10.0 + index as f64 * 4.0;
        points.extend([
            [x, 0.0, 0.0],
            [x + 1.0, 0.0, 0.0],
            [x, 1.0, 0.0],
            [x, 0.0, 1.0],
        ]);
        cells.push([base, base + 1, base + 2, base + 3]);
    }
    let nodes = points
        .into_iter()
        .enumerate()
        .map(|(index, point)| {
            let mut identity = [0; 32];
            identity[24..].copy_from_slice(&((index + 1) as u64).to_be_bytes());
            DelaunayVolumeNode::explicit(StableDigest::from_bytes(identity), point)
        })
        .collect();
    build_delaunay_volume_topology(
        nodes,
        cells,
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Canonical source lookup matches exhaustive oriented-cell search with zero,16 and4096
//   unrelated cells, under128 inventory comparisons per query. Local-star boundary roles match the
//   complete boundary, including an interior vertex. This is a query bound, not complete refinement
//   or native throughput qualification.
#[test]
fn canonical_source_and_local_boundary_queries_match_exhaustive_inventories() {
    for count in [0, 16, 4096] {
        let topology = topology(count);
        for index in [
            0,
            topology.tetrahedra.len() / 2,
            topology.tetrahedra.len() - 1,
        ] {
            let identities = topology.tetrahedra[index]
                .vertex_indices
                .map(|vertex| topology.nodes[vertex as usize].identity);
            for query in [
                identities,
                [identities[1], identities[0], identities[2], identities[3]],
                [StableDigest::ZERO; 4],
            ] {
                let reference = topology.tetrahedra.iter().position(|cell| {
                    cell.vertex_indices
                        .map(|vertex| topology.nodes[vertex as usize].identity)
                        == query
                });
                let mut probes = 0;
                let actual =
                    find_with_probe(&topology, query, &mut || probes += 1).map(|(index, _)| index);
                assert_eq!(actual, reference);
                assert!(
                    probes < 128,
                    "canonical lookup must not scan unrelated cells: {probes}"
                );
            }
        }
        for vertex in [0, 1, 2, 3, 4, topology.nodes.len() - 1] {
            let reference = topology
                .incidence
                .boundary_facets
                .iter()
                .any(|face| face.vertex_indices.contains(&(vertex as u32)));
            assert_eq!(topology.vertex_is_boundary(vertex as u32), reference);
        }
        assert!(
            !topology.vertex_is_boundary(4),
            "the subdivision center is interior"
        );
    }
}
