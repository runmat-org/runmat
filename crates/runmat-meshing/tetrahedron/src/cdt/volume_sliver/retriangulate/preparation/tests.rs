use super::*;
use runmat_meshing_core::{NeverCancelled, StableDigest};

fn topology() -> DelaunayVolumeTopology {
    let (original, _) = crate::cdt::volume_sliver::tests::interior_sliver_fixture();
    let region = original.tetrahedra[0].region_id.clone();
    let mut nodes = original.nodes;
    nodes.push(DelaunayVolumeNode::explicit(
        StableDigest::from_bytes([6; 32]),
        [0.0, 0.0, -1.0],
    ));
    let mut cells = original
        .tetrahedra
        .into_iter()
        .map(|cell| (cell.vertex_indices, cell.region_id))
        .collect::<Vec<_>>();
    cells.push(([0, 1, 2, 5], region.clone()));
    for index in 0..24u8 {
        let start = nodes.len() as u32;
        let x = 10.0 + 2.0 * f64::from(index);
        for (corner, point) in [
            [x, 0.0, 0.0],
            [x + 1.0, 0.0, 0.0],
            [x, 1.0, 0.0],
            [x, 0.0, 1.0],
        ]
        .into_iter()
        .enumerate()
        {
            nodes.push(DelaunayVolumeNode::explicit(
                StableDigest::from_bytes([10 + 4 * index + corner as u8; 32]),
                point,
            ));
        }
        cells.push(([start, start + 1, start + 2, start + 3], region.clone()));
    }
    build_delaunay_volume_topology_with_regions(nodes, cells, Default::default(), &NeverCancelled)
        .unwrap()
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A vertex-star neighborhood retains five cells against a 29-cell canonical mesh and
//   reproduces the complete cavity, original outside-face owners, regions and exact target volume.
//   This excludes full canonical reconstruction, legalization, allocator measurements and
//   whole-study throughput.
#[test]
fn vertex_star_inventory_preserves_complete_cavity_and_outside_owners() {
    let topology = topology();
    let source = topology
        .nodes
        .iter()
        .position(|node| node.identity == StableDigest::from_bytes([5; 32]))
        .unwrap();
    let selected = topology.incidence.vertex_stars[source]
        .iter()
        .map(|index| *index as usize)
        .collect::<Vec<_>>();
    let all = topology
        .tetrahedra
        .iter()
        .enumerate()
        .map(|(index, cell)| {
            let points = cell.vertex_indices.map(|vertex| {
                topology.nodes[vertex as usize]
                    .coordinates_m
                    .canonical_value()
            });
            CavityTetrahedron {
                tetrahedron_id: index as u32,
                component_id: 0,
                node_ids: cell.vertex_indices,
                source_surface_element_id: 0,
                region_ids: cell
                    .region_id
                    .iter()
                    .map(|region| region.source_topology_id.clone())
                    .collect(),
                volume_m3: tetrahedron_volume(points),
                aspect_ratio: tetrahedron_edge_aspect_ratio(points),
                exact_scaled_jacobian: tetrahedron_scaled_jacobian(points).unwrap(),
            }
        })
        .collect::<Vec<_>>();
    let expected =
        constrained_cavity_from_selected_tetrahedra(&all, &selected, Vec::new()).unwrap();
    let (local, local_selected) = local_cavity_inventory(&topology, &selected).unwrap();
    let actual =
        constrained_cavity_from_selected_tetrahedra(&local, &local_selected, Vec::new()).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(local.len(), 5);
    assert_eq!(all.len(), 29);
    assert!(actual
        .boundary_faces
        .iter()
        .any(|face| !face.outside_tetrahedron_ids.is_empty()));
}

#[test]
fn prepared_vertex_removal_reuses_the_same_refill_and_rejects_collisions() {
    let (topology, _) = crate::cdt::volume_sliver::tests::interior_sliver_fixture();
    let options = crate::cdt::DelaunayInsertionOptions::default();
    let source = topology
        .nodes
        .iter()
        .position(|node| node.identity == StableDigest::from_bytes([5; 32]))
        .unwrap();
    let prepared = PreparedVertexRemoval::prepare(&topology, source, options, &NeverCancelled)
        .unwrap()
        .unwrap();
    let first = prepared.materialize(&[], &NeverCancelled).unwrap().unwrap();
    let second = prepared.materialize(&[], &NeverCancelled).unwrap().unwrap();
    assert_eq!(first, second);
    assert_eq!(first.nodes.len(), 4);
    assert_eq!(first.tetrahedra.len(), 1);
    assert!(prepared
        .relocate(topology.nodes[0], &[], &NeverCancelled)
        .unwrap()
        .is_none());
    let replacement =
        DelaunayVolumeNode::explicit(StableDigest::from_bytes([200; 32]), [0.2, 0.2, 0.2]);
    let relocated = prepared
        .relocate(replacement, &[], &NeverCancelled)
        .unwrap()
        .unwrap();
    assert_eq!(relocated.nodes.len(), 5);
    assert_eq!(relocated.tetrahedra.len(), 4);
    assert!(relocated.nodes.contains(&replacement));
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Every removal position in a canonical node inventory matches identity-map remapping
//   for all unchanged cells, and cells referencing the removed node reject. This verifies remapping
//   semantics without establishing independent candidate admissibility or native runtime.
#[test]
fn removal_index_shift_matches_canonical_identity_remapping() {
    let topology = topology();
    for source in 0..topology.nodes.len() {
        let indices = topology
            .nodes
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != source)
            .enumerate()
            .map(|(new_index, (_, node))| (node.identity, new_index as u32))
            .collect::<BTreeMap<_, _>>();
        for cell in &topology.tetrahedra {
            if cell.vertex_indices.contains(&(source as u32)) {
                assert!(super::super::remap_without_vertex(cell.vertex_indices, source).is_err());
            } else {
                let expected = cell
                    .vertex_indices
                    .map(|vertex| indices[&topology.nodes[vertex as usize].identity]);
                assert_eq!(
                    super::super::remap_without_vertex(cell.vertex_indices, source).unwrap(),
                    expected
                );
            }
        }
    }
}

#[test]
fn vertex_star_preparation_observes_cancellation_before_inventory_work() {
    struct Cancelled;
    impl MeshingCancellationSignal for Cancelled {
        fn is_cancelled(&self) -> bool {
            true
        }
    }
    let topology = topology();
    assert!(
        matches!(PreparedVertexRemoval::prepare(&topology,4,Default::default(),&Cancelled), Err(failure) if failure.kind == DelaunayVolumeSliverErrorKind::Cancelled)
    );
}
