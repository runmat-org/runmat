use super::*;
use crate::cdt::insertion::topology_access::TetrahedralTopology;
use runmat_meshing_core::NeverCancelled;

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A closed replacement that preserves exterior facets but removes an enclosed vertex is
//   rejected before commit. The original point star remains unchanged. Vertex deletion requires
//   separate explicit authority; exterior-boundary equality alone is insufficient.
#[test]
fn closed_replacement_preserves_enclosed_vertices_before_commit() {
    let nodes = [
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [0.25, 0.25, 0.25],
    ]
    .into_iter()
    .enumerate()
    .map(|(index, point)| {
        crate::cdt::DelaunayVolumeNode::explicit(
            StableDigest::from_bytes([index as u8 + 1; 32]),
            point,
        )
    })
    .collect();
    let topology = crate::cdt::build_delaunay_volume_topology(
        nodes,
        vec![[4, 0, 1, 2], [4, 0, 3, 1], [4, 0, 2, 3], [4, 1, 3, 2]],
        Default::default(),
        &NeverCancelled,
    )
    .unwrap();
    let protected = topology
        .incidence
        .boundary_facets
        .iter()
        .map(|facet| {
            let mut identities = facet
                .vertex_indices
                .map(|vertex| topology.nodes[vertex as usize].identity);
            identities.sort_unstable();
            identities
        })
        .collect();
    let region = None;
    let mut owner = TopologyMutation::from_canonical(topology);
    let removed = (0..4).collect();
    let failure = owner
        .replace_closed_cavity(
            &removed,
            vec![([0, 1, 2, 3], region)],
            &protected,
            Default::default(),
            &NeverCancelled,
        )
        .unwrap_err();
    assert!(failure.reason.contains("existing vertex"));
    assert_eq!(owner.active_tetrahedron_count(), 4);
    assert_eq!(owner.vertex_star(4).len(), 4);
}
