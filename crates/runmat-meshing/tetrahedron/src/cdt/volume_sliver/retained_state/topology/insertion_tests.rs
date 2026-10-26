//! Inserted stable identities can precede or interleave the entire old inventory.
//! Exact independent topology and quality restoration must include those remaps.
use super::super::test_fixtures::{assigned, pair, quality};
use super::super::QualityPatch;
use super::*;
use runmat_meshing_core::NeverCancelled;

struct Cancelled;
impl MeshingCancellationSignal for Cancelled {
    fn is_cancelled(&self) -> bool {
        true
    }
}

fn identity(value: u64) -> StableDigest {
    let mut bytes = [0; 32];
    bytes[24..].copy_from_slice(&value.to_be_bytes());
    StableDigest::from_bytes(bytes)
}

fn split(topology: &DelaunayVolumeTopology, identity: StableDigest) -> DelaunayVolumeTopology {
    let cell = &topology.tetrahedra[0];
    let points = cell.vertex_indices.map(|index| {
        topology.nodes[index as usize]
            .coordinates_m
            .canonical_value()
    });
    let point =
        std::array::from_fn(|axis| points.iter().map(|point| point[axis]).sum::<f64>() / 4.);
    let mut nodes = topology.nodes.clone();
    let inserted = nodes.len() as u32;
    nodes.push(DelaunayVolumeNode::explicit(identity, point));
    let mut cells = topology
        .tetrahedra
        .iter()
        .skip(1)
        .map(|cell| (cell.vertex_indices, cell.region_id.clone()))
        .collect::<Vec<_>>();
    for opposite in 0..4 {
        let mut corners = cell.vertex_indices;
        corners[opposite] = inserted;
        cells.push((corners, cell.region_id.clone()));
    }
    let original_identities = nodes.iter().map(|node| node.identity).collect::<Vec<_>>();
    nodes.sort_unstable_by_key(|node| node.identity);
    for (corners, _) in &mut cells {
        *corners = corners.map(|index| {
            nodes
                .binary_search_by_key(&original_identities[index as usize], |node| node.identity)
                .unwrap() as u32
        });
    }
    build_delaunay_volume_topology_with_regions(
        nodes,
        cells,
        DelaunayVolumeSliverOptions::default().insertion.topology,
        &NeverCancelled,
    )
    .unwrap()
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Physical one-cell insertion and compound insertion with zero or 64 unrelated cells
//   retain only added nodes and exact changed cell records. Inserted identities precede, interleave
//   or follow the baseline node order; materialization restores complete coordinates, orientation,
//   neighbors, assigned regions and incidence. Independently evaluated quality and both priority
//   spectra reconstruct exactly. Changed or removed baseline nodes, cancellation and the unchanged
//   node ceiling reject. This component witness does not establish arbitrary mutation locality,
//   sublinear work, whole-task native memory or quality convergence.
#[test]
fn inserted_node_patches_restore_physical_topology_quality_and_rank_across_index_shifts() {
    for unrelated in [0, 64] {
        let original = assigned(pair(unrelated).0);
        let mut nodes = original.nodes.clone();
        for (index, node) in nodes.iter_mut().enumerate() {
            node.identity = identity((index as u64 + 1) * 4);
        }
        let base = build_delaunay_volume_topology_with_regions(
            nodes,
            original
                .tetrahedra
                .iter()
                .map(|cell| (cell.vertex_indices, cell.region_id.clone()))
                .collect(),
            DelaunayVolumeSliverOptions::default().insertion.topology,
            &NeverCancelled,
        )
        .unwrap();
        for inserted in [2, 6, (base.nodes.len() as u64 + 1) * 4] {
            let candidate = split(&base, identity(inserted));
            let patch = TopologyPatch::between(&base, &candidate).unwrap();
            assert_eq!(patch.inserted_nodes.len(), 1);
            assert_eq!(patch.removed.len(), 1);
            assert_eq!(patch.created.len(), 4);
            assert_eq!(
                patch.keys(&base).collect::<Vec<_>>(),
                connectivity::keys(&candidate).collect::<Vec<_>>()
            );
            let restored = patch
                .materialize(
                    &base,
                    DelaunayVolumeSliverOptions::default(),
                    &NeverCancelled,
                )
                .unwrap();
            assert_eq!(restored, candidate);
            let base_quality = quality(&base);
            let candidate_quality = quality(&candidate);
            let quality_patch = QualityPatch::between(&base_quality, &candidate_quality).unwrap();
            assert_eq!(quality_patch.materialize(&base_quality), candidate_quality);
            for objective in [
                crate::cdt::volume_sliver::admission::QualityObjective::Shape,
                crate::cdt::volume_sliver::admission::QualityObjective::Refinement,
            ] {
                assert_eq!(
                    quality_patch.spectrum(&base_quality, objective),
                    crate::cdt::volume_sliver::admission::quality_spectrum(
                        &candidate_quality,
                        objective
                    )
                );
            }
            // A compound insertion also retains only newly introduced nodes.
            let compound = split(&candidate, identity((base.nodes.len() as u64 + 3) * 4));
            let compound_patch = TopologyPatch::between(&base, &compound).unwrap();
            assert_eq!(compound_patch.inserted_nodes.len(), 2);
            assert_eq!(
                compound_patch
                    .materialize(
                        &base,
                        DelaunayVolumeSliverOptions::default(),
                        &NeverCancelled
                    )
                    .unwrap(),
                compound
            );
            let mut moved = candidate.clone();
            let index = moved
                .nodes
                .binary_search_by_key(&base.nodes[0].identity, |node| node.identity)
                .unwrap();
            {
                let quantity_change = 0.001;
                let quantity_index = 0;
                let quantity_slot = &mut moved.nodes[index].coordinates_m;
                let mut quantity_coordinates = quantity_slot.canonical_value();
                quantity_coordinates[quantity_index] += quantity_change;
                *quantity_slot = runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(quantity_coordinates);
            };
            assert!(TopologyPatch::between(&base, &moved).is_err());
            let mut missing = candidate.clone();
            missing.nodes.remove(index);
            assert!(TopologyPatch::between(&base, &missing).is_err());
            assert_eq!(
                patch
                    .materialize(&base, DelaunayVolumeSliverOptions::default(), &Cancelled)
                    .unwrap_err()
                    .kind,
                DelaunayVolumeSliverErrorKind::Cancelled
            );
            let mut limited = DelaunayVolumeSliverOptions::default();
            limited.insertion.topology.maximum_nodes = base.nodes.len() as u64;
            assert_eq!(
                patch
                    .materialize(&base, limited, &NeverCancelled)
                    .unwrap_err()
                    .kind,
                DelaunayVolumeSliverErrorKind::ResourceLimit
            );
        }
    }
}
