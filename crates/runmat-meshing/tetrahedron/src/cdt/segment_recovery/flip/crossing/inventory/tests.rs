use super::*;
use crate::cdt::{build_delaunay_volume_topology, DelaunayTopologyOptions, DelaunayVolumeNode};
use runmat_meshing_core::NeverCancelled;

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Analytic convex-bipyramid crossing, obstruction and edge-contact assessments reuse
//   one conservative inventory without additional spatial search. Checked mutation rederives
//   candidates; rollback restores the original assessment. No cache is retained across an accepted
//   edit, and no full recovery or native resource claim follows.
#[test]
fn read_only_crossing_assessments_share_candidates_and_mutations_rederive_them() {
    let points = [
        [-1.5, 0., 0.],
        [1.4, 0., 0.],
        [0., -0.5, 0.],
        [0., 2., 2.5],
        [0., 2., -2.5],
    ];
    let nodes = points
        .into_iter()
        .enumerate()
        .map(|(index, point)| {
            DelaunayVolumeNode::explicit(StableDigest::from_bytes([(index + 1) as u8; 32]), point)
        })
        .collect();
    let mut topology = TopologyMutation::from_canonical(
        build_delaunay_volume_topology(
            nodes,
            vec![[0, 2, 3, 4], [1, 2, 4, 3]],
            DelaunayTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap(),
    );
    let mut work = RecoveryWork::new(Default::default(), &NeverCancelled);
    let cells = candidate_cells(&topology, 0, 1, 0, &mut work).unwrap();
    let query_work = work.search_inventory()[6..].to_vec();
    // The line joining the two apexes crosses one interior face strictly,
    // inside a convex bipyramid. There is no edge-contact obstruction.
    assert_eq!(
        crossing_face_count(&topology, &cells, 0, 1, 0, &mut work).unwrap(),
        1
    );
    assert_eq!(
        crossing_face_obstruction_count(&topology, &cells, 0, 1, 0, &mut work).unwrap(),
        0
    );
    assert_eq!(
        first_edge_crossing_face(&topology, &cells, 0, 1, 0, &mut work).unwrap(),
        None
    );
    assert_eq!(&work.search_inventory()[6..], query_work);
    {
        let mut trial = topology.cavity_trial();
        trial
            .replace_cavity(
                &BTreeSet::from([0, 1]),
                vec![
                    ([0, 1, 2, 3], None),
                    ([0, 1, 3, 4], None),
                    ([0, 1, 4, 2], None),
                ],
            )
            .unwrap();
        let fresh_cells = candidate_cells(&*trial, 0, 1, 0, &mut work).unwrap();
        assert_eq!(fresh_cells.len(), 3);
        assert_eq!(
            crossing_face_count(&*trial, &fresh_cells, 0, 1, 0, &mut work).unwrap(),
            0
        );
        assert!(work.search_inventory()[7] > query_work[1]);
    }
    assert_eq!(
        crossing_face_count(&topology, &cells, 0, 1, 0, &mut work).unwrap(),
        1
    );
}
