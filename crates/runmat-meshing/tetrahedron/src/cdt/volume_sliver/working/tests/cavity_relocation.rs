//! A point crosses its old star while a bounded joint transaction preserves the
//! complete outer surface. Canonical point-set construction is the test oracle.
use super::*;
use crate::cdt::mutation::CavityRelocation;

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: An interior point crosses its old star in a joint point/cavity transaction.
//   Independent canonical construction checks the complete result; failed replacement and dropped
//   trials restore point/quality state, and a single grouped seed replays through current
//   admission. All retained points participate. This local transaction witness is not whole-model
//   convergence or measured memory evidence.
#[test]
fn joint_cavity_relocation_crosses_the_old_star_and_replays_atomically() {
    check_joint_relocation(false);
}
// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Neighboring interior points share one point/cavity transaction and one replay record.
//   A canonical construction independently checks resulting connectivity and complete quality;
//   failed and dropped candidates restore the original state. This checks transaction authority,
//   not whole-model convergence or throughput.
#[test]
fn neighboring_points_move_and_replay_in_one_atomic_cavity_transaction() {
    check_joint_relocation(true);
}
fn check_joint_relocation(replace_second: bool) {
    let (base, provenance) = interior_sliver_fixture();
    let mut nodes = base.nodes;
    nodes[4].coordinates_m = runmat_types::physical::Quantity::<
        runmat_types::physical::kind::Position,
        _,
    >::new([0.1, 0.1, 0.1]);
    nodes.push(DelaunayVolumeNode::explicit(
        StableDigest::from_bytes([6; 32]),
        [0.6, 0.1, 0.1],
    ));
    let build = |nodes| {
        let topology =
            crate::cdt::build_delaunay_volume_point_set(nodes, Default::default(), &NeverCancelled)
                .unwrap();
        let regions = vec![provenance.facets[0].region_ids[0].clone(); topology.tetrahedra.len()];
        crate::cdt::assign_delaunay_volume_regions(
            topology,
            regions,
            Default::default(),
            &NeverCancelled,
        )
        .unwrap()
    };
    let original = build(nodes.clone());
    let replacement =
        DelaunayVolumeNode::explicit(StableDigest::from_bytes([7; 32]), [0.7, 0.1, 0.1]);
    nodes[4] = replacement;
    let mut replacements = vec![(4, replacement)];
    if replace_second {
        let second =
            DelaunayVolumeNode::explicit(StableDigest::from_bytes([8; 32]), [0.2, 0.2, 0.1]);
        nodes[5] = second;
        replacements.push((5, second));
    }
    let expected = build(nodes);
    let request = metric();
    let prepared = PreparedLocalQuality::prepare(&request, &provenance, quality_options()).unwrap();
    let reference = quality(&original, &provenance);
    let mut state = WorkingState::new(
        original.clone(),
        reference.clone(),
        quality_options(),
        &prepared,
        &[],
        &NeverCancelled,
    )
    .unwrap();
    let options = DelaunayVolumeSliverOptions::default();
    {
        let mut trial = state.trial();
        assert!(
            trial
                .replace_node(4, replacement, options, &NeverCancelled)
                .is_err(),
            "the oracle must cross the fixed star"
        );
    }
    let change = || CavityRelocation {
        replacements: replacements.clone(),
        removed: (0..original.tetrahedra.len() as u32).collect(),
        created: expected
            .tetrahedra
            .iter()
            .map(|cell| {
                (
                    cell.vertex_indices.map(|vertex| {
                        let id = expected.nodes[vertex as usize].identity;
                        if let Some((index, _)) =
                            replacements.iter().find(|(_, node)| node.identity == id)
                        {
                            *index
                        } else {
                            original
                                .nodes
                                .binary_search_by_key(&id, |node| node.identity)
                                .unwrap() as u32
                        }
                    }),
                    cell.region_id.clone(),
                )
            })
            .collect(),
    };
    let seed = {
        let mut trial = state.trial();
        trial
            .relocate_cavity(change(), options, &NeverCancelled)
            .unwrap();
        let seed = trial.capture(Vec::new());
        assert_eq!(seed.edits.len(), 1);
        assert!(matches!(
            seed.edits.iter().next(),
            Some(TopologyEdit::RelocateCavity { .. })
        ));
        seed
    };
    assert_eq!(state.quality().cell_count(), reference.tetrahedra.len());
    assert_eq!(state.topology().node(4), original.nodes[4]);
    {
        let mut trial = state.trial();
        let mut invalid = change();
        invalid.created.clear();
        assert!(trial
            .relocate_cavity(invalid, options, &NeverCancelled)
            .is_err());
    }
    assert_eq!(state.topology().node(4), original.nodes[4]);
    {
        let mut trial = state.trial();
        seed.replay(&mut trial, options, &NeverCancelled).unwrap();
        trial.commit();
    }
    let (actual, actual_quality) = state.finish(options, &NeverCancelled).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(actual_quality, quality(&expected, &provenance));
    assert!(actual
        .incidence
        .vertex_stars
        .iter()
        .all(|star| !star.is_empty()));
}
