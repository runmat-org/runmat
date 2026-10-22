//! Differential complete operators, including cospherical policy, focus and barriers.
use super::*;
use crate::cdt::{
    build_delaunay_volume_topology, DelaunayInsertionOptions, DelaunayTopologyOptions,
    DelaunayVolumeNode,
};
use runmat_meshing_core::NeverCancelled;

fn neighborhoods(count: usize, height: f64) -> DelaunayVolumeTopology {
    let mut nodes = Vec::new();
    let mut cells = Vec::new();
    for copy in 0..count {
        let offset = nodes.len() as u32;
        for point in [
            [5., 0., 0.],
            [0., 5., 0.],
            [-3., -4., 0.],
            [0., 0., height],
            [0., 0., -height],
        ] {
            let mut identity = [0; 32];
            identity[..8].copy_from_slice(&((nodes.len() + 1) as u64).to_be_bytes());
            nodes.push(DelaunayVolumeNode::explicit(
                StableDigest::from_bytes(identity),
                [point[0] + copy as f64 * 20., point[1], point[2]],
            ));
        }
        cells.extend([
            [offset, offset + 1, offset + 2, offset + 3],
            [offset, offset + 2, offset + 1, offset + 4],
        ]);
    }
    build_delaunay_volume_topology(
        nodes,
        cells,
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
}

// witness: FEA-SHARED-RESOURCES-001/machine_cpu_execution @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Maintained canonical legalization matches the complete legacy operator and flip count
//   on one, four and 32 separated neighborhoods, with sphere-inside, sphere-outside and cospherical
//   cases, focused or unfocused queries and protected faces. It preserves symbolic policy and
//   independently admits final topology. This checks numerical kernel equivalence, not parallel
//   throughput, allocator demand, arbitrary relocation locality or whole-study scaling.
#[test]
fn maintained_canonical_legalization_matches_full_rebuilds_with_focus_and_barriers() {
    let mut accepted = 0;
    for count in [1, 4, 32] {
        for height in [4., 5., 6.] {
            let original = neighborhoods(count, height);
            for protected in [false, true] {
                let barriers = if protected {
                    BTreeSet::from([original.nodes[..3]
                        .iter()
                        .map(|node| node.identity)
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap()])
                } else {
                    BTreeSet::new()
                };
                for focus in [None, Some(original.nodes[3].identity)] {
                    let options = DelaunayInsertionOptions::default();
                    let mut old_work = InsertionWork::new(options, &NeverCancelled);
                    let expected = super::super::reference_canonical_legalization(
                        original.clone(),
                        &barriers,
                        true,
                        focus,
                        &mut old_work,
                    )
                    .unwrap();
                    let mut work = InsertionWork::new(options, &NeverCancelled);
                    let actual =
                        restore(original.clone(), &barriers, true, focus, &mut work).unwrap();
                    assert_eq!(
                        actual, expected,
                        "count={count}, height={height}, protected={protected}, focus={focus:?}"
                    );
                    assert_eq!(work.legalization_flips, old_work.legalization_flips);
                    accepted += work.legalization_flips;
                    assert!(work.inventory().0 <= old_work.inventory().0);
                }
            }
        }
    }
    assert!(accepted > 0, "matrix must perform real flips");
}

#[test]
fn maintained_legalization_preserves_region_barriers_and_rejects_exhausted_work() {
    let mut original = neighborhoods(1, 4.);
    use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
    for (index, cell) in original.tetrahedra.iter_mut().enumerate() {
        cell.region_id = Some(PersistentEntityId::new(
            PersistentEntityKind::Region,
            format!("region-{index}"),
            Vec::new(),
        ));
    }
    original.incidence = crate::cdt::incidence::build_volume_incidence(
        &original.nodes,
        &original.tetrahedra,
        &NeverCancelled,
        DelaunayTopologyOptions::default(),
    )
    .unwrap();
    let options = DelaunayInsertionOptions::default();
    assert_eq!(
        restore(
            original.clone(),
            &BTreeSet::new(),
            true,
            None,
            &mut InsertionWork::new(options, &NeverCancelled)
        )
        .unwrap(),
        original
    );
    let original = neighborhoods(4, 4.);
    let options = DelaunayInsertionOptions {
        maximum_predicate_evaluations: 1,
        ..Default::default()
    };
    let failure = restore(
        original,
        &BTreeSet::new(),
        true,
        None,
        &mut InsertionWork::new(options, &NeverCancelled),
    )
    .unwrap_err();
    assert_eq!(failure.kind, DelaunayInsertionErrorKind::ResourceLimit);
}
