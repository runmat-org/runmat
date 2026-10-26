use super::*;
use crate::cdt::test_fixtures::coupled_quality::fixture;
use runmat_meshing_core::NeverCancelled;

#[test]
fn deleting_a_free_vertex_can_retain_an_unchanged_boundary_target() {
    let nodes = [
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [0.02; 3],
    ]
    .into_iter()
    .enumerate()
    .map(|(index, point)| {
        DelaunayVolumeNode::explicit(StableDigest::from_bytes([index as u8 + 1; 32]), point)
    })
    .collect();
    let topology = crate::cdt::build_delaunay_volume_topology(
        nodes,
        vec![[4, 0, 1, 2], [4, 0, 3, 1], [4, 0, 2, 3], [4, 1, 3, 2]],
        crate::cdt::DelaunayTopologyOptions::default(),
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
                .map(|index| topology.nodes[index as usize].identity);
            identities.sort_unstable();
            identities
        })
        .collect::<Vec<_>>();
    for target in &topology.nodes[..4] {
        let result = collapse_interior_edge(
            &topology,
            4,
            target.identity,
            DelaunayVolumeSliverOptions::default(),
            &NeverCancelled,
        )
        .unwrap()
        .unwrap();
        assert_eq!(result.nodes, topology.nodes[..4]);
        assert_eq!(result.tetrahedra.len(), 1);
        crate::cdt::insertion::validate_conforming_volume_topology(
            &result,
            &protected,
            DelaunayVolumeSliverOptions::default().insertion,
            &NeverCancelled,
        )
        .unwrap();
    }
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Changed-star collapse preparation agrees exactly with the former complete-map oracle,
//   including accepted conforming candidates, identity permutations and0/64/4096 unrelated cells.
//   Complete candidate reconstruction and independent final validation remain global; no
//   whole-relocation locality or native runtime claim is made.
#[test]
fn local_collapse_plan_matches_complete_reference_across_unrelated_inventory() {
    let mut admitted = 0;
    for permuted in [false, true] {
        let mut baseline = None;
        for background in [0, 64, 4096] {
            let (topology, provenance) = fixture(permuted, background);
            let free = topology
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, node)| node.coordinates_m.canonical_value()[0].abs() == 0.95)
                .map(|(index, _)| index)
                .collect::<Vec<_>>();
            let options = DelaunayVolumeSliverOptions::default();
            let plan = prepare(&topology, free[0], free[1], options, &NeverCancelled)
                .unwrap()
                .unwrap();
            let inventory = (plan.removed.len(), plan.changed.len());
            if let Some(expected) = baseline {
                assert_eq!(inventory, expected);
            } else {
                baseline = Some(inventory);
            }
            assert!(inventory.0 < 128 && inventory.1 < 128);
            let target = topology.nodes[free[1]].identity;
            let actual =
                collapse_interior_edge(&topology, free[0], target, options, &NeverCancelled)
                    .unwrap();
            let expected =
                legacy_collapse(&topology, free[0], target, options, &NeverCancelled).unwrap();
            assert_eq!(actual, expected);
            if let Some(result) = actual {
                admitted += 1;
                let faces = provenance
                    .facets
                    .iter()
                    .map(|facet| facet.node_identities)
                    .collect::<Vec<_>>();
                crate::cdt::insertion::validate_conforming_volume_topology(
                    &result,
                    &faces,
                    options.insertion,
                    &NeverCancelled,
                )
                .unwrap();
            }
        }
    }
    assert_eq!(
        admitted, 6,
        "reference agreement must include accepted physical candidates"
    );
}

fn legacy_collapse(
    topology: &DelaunayVolumeTopology,
    source_index: usize,
    target_identity: StableDigest,
    options: DelaunayVolumeSliverOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<DelaunayVolumeTopology>, DelaunayVolumeSliverError> {
    let target_index = topology
        .nodes
        .binary_search_by_key(&target_identity, |node| node.identity)
        .map_err(|_| {
            error(
                DelaunayVolumeSliverErrorKind::InvalidTopology,
                "interior edge-collapse target is absent from canonical topology",
            )
        })?;
    if source_index == target_index {
        return Ok(None);
    }
    let nodes = topology
        .nodes
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != source_index)
        .map(|(_, node)| *node)
        .collect::<Vec<_>>();
    let target_after_removal = target_index - usize::from(target_index > source_index);
    let remap = |vertex: u32| {
        let vertex = vertex as usize;
        if vertex == source_index {
            target_after_removal as u32
        } else {
            (vertex - usize::from(vertex > source_index)) as u32
        }
    };
    let mut unique = BTreeMap::new();
    for tetrahedron in &topology.tetrahedra {
        let vertices = tetrahedron.vertex_indices.map(remap);
        let mut key = vertices;
        key.sort_unstable();
        if key.windows(2).any(|pair| pair[0] == pair[1]) {
            continue;
        }
        match unique.entry(key) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert((vertices, tetrahedron.region_id.clone()));
            }
            std::collections::btree_map::Entry::Occupied(entry) => {
                if entry.get().1 != tetrahedron.region_id {
                    return Ok(None);
                }
            }
        }
    }
    let tetrahedra = unique.into_values().collect::<Vec<_>>();
    match build_delaunay_volume_topology_with_regions(
        nodes,
        tetrahedra,
        options.insertion.topology,
        cancellation,
    ) {
        Ok(topology) => Ok(Some(topology)),
        Err(failure) => match failure.kind {
            DelaunayTopologyErrorKind::ResourceLimit => Err(error(
                DelaunayVolumeSliverErrorKind::ResourceLimit,
                failure.to_string(),
            )),
            DelaunayTopologyErrorKind::Cancelled => Err(error(
                DelaunayVolumeSliverErrorKind::Cancelled,
                failure.to_string(),
            )),
            _ => Ok(None),
        },
    }
}
