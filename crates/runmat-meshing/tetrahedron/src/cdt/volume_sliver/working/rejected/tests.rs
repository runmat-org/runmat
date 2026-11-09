use super::*;
use crate::cdt::{
    topology::build_delaunay_volume_topology_with_regions, DelaunayTopologyOptions,
    DelaunayVolumeTopology,
};
use runmat_meshing_core::NeverCancelled;

fn identity(index: usize) -> StableDigest {
    StableDigest::from_bytes([index as u8 + 1; 32])
}

/// Five cells in a row, each touching the next at one node: cell `i` has nodes `3i` to `3i + 3`.
/// A node listed in `moved` is shifted a little along y.
fn strip(moved: &[usize]) -> DelaunayVolumeTopology {
    let nodes = (0..16)
        .map(|index| {
            let (cell, slot) = (index / 3, index % 3);
            let x = 2.0 * cell as f64;
            let mut point = match slot {
                0 => [x, 0.0, 0.0],
                1 => [x + 1.0, 1.0, 0.0],
                _ => [x + 1.0, 0.0, 1.0],
            };
            if moved.contains(&index) {
                point[1] += 0.25;
            }
            DelaunayVolumeNode::explicit(identity(index), point)
        })
        .collect();
    let cells = (0..5)
        .map(|cell| {
            let first = 3 * cell;
            ([first, first + 1, first + 2, first + 3], None)
        })
        .collect();
    build_delaunay_volume_topology_with_regions(
        nodes,
        cells,
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
}

fn key() -> CompleteSearchKey {
    CompleteSearchKey::new([3, 1, 2, 0].map(identity), QualityObjective::Shape)
}

fn coupled(floor_ratio: f64) -> Ranking {
    Ranking {
        coupled: true,
        objective: QualityObjective::Shape,
        floor: crate::cdt::volume_quality::violation_priority_key(floor_ratio),
    }
}

fn recording(changed: &[usize], ranked_out: Vec<RankedOutCandidate>) -> SearchRecording {
    SearchRecording {
        changed: changed.iter().map(|index| identity(*index)).collect(),
        ranked_out,
    }
}

#[test]
fn a_rejected_search_stays_rejected_until_its_neighborhood_changes() {
    let mut rejected = RejectedCompleteRelocations::default();
    rejected.record(&strip(&[]), key(), recording(&[], Vec::new()));
    assert!(rejected.still_rejected(&strip(&[]), &key(), coupled(2.0)));
    // Node 12 shares no cell with the source's one-ring (nodes 0 to 6).
    assert!(rejected.still_rejected(&strip(&[12]), &key(), coupled(2.0)));
    // A different floor alone does not rerun a search without ranked-out candidates.
    assert!(rejected.still_rejected(&strip(&[]), &key(), coupled(1.5)));
    // Node 8 lies in the cell around node 6, so the search reads it.
    assert!(!rejected.still_rejected(&strip(&[8]), &key(), coupled(2.0)));
    // The stale record is gone.
    assert!(!rejected.still_rejected(&strip(&[]), &key(), coupled(2.0)));
}

#[test]
fn nodes_a_candidate_changed_extend_the_neighborhood() {
    let mut rejected = RejectedCompleteRelocations::default();
    rejected.record(&strip(&[]), key(), recording(&[11], Vec::new()));
    assert!(!rejected.still_rejected(&strip(&[12]), &key(), coupled(2.0)));
}

/// A candidate that replaces a 1.3 violation by a 1.25 one loses while a worse violation
/// elsewhere (2.0) sets the coupled floor, and wins once the floor drops to its own level.
#[test]
fn a_ranked_out_candidate_reruns_the_search_when_the_floor_admits_it() {
    let candidate = RankedOutCandidate {
        removed: vec![[1.3, 0.5], [0.4, 0.9]],
        added: vec![[1.25, 0.5], [0.4, 0.9], [0.2, 0.3]],
    };
    assert!(!candidate.admissible(coupled(2.0)));
    assert!(candidate.admissible(coupled(1.3)));
    let mut rejected = RejectedCompleteRelocations::default();
    rejected.record(&strip(&[]), key(), recording(&[], vec![candidate]));
    assert!(rejected.still_rejected(&strip(&[]), &key(), coupled(2.0)));
    assert!(!rejected.still_rejected(&strip(&[]), &key(), coupled(1.3)));
}

/// The ranking from changed cells agrees with the spectrum comparison of whole states.
#[test]
fn changed_cell_ranking_matches_whole_spectrum_comparison() {
    let shared = [[1.7, 1.1], [1.05, 0.2], [0.3, 1.4]];
    let cases = [
        (vec![[1.3, 0.5]], vec![[1.25, 0.5]]),
        (vec![[1.3, 0.5]], vec![[1.3, 1.01]]),
        (vec![[1.8, 0.5]], vec![[1.7, 1.7]]),
        (vec![[1.1, 1.2]], vec![[0.4, 0.4], [1.0, 1.0]]),
        (vec![[0.4, 0.4]], vec![[1.2, 0.4]]),
    ];
    for (removed, added) in cases {
        let candidate = RankedOutCandidate {
            removed: removed.clone(),
            added: added.clone(),
        };
        let spectrum = |cells: &[[f64; 2]], both: bool| {
            let mut values = cells
                .iter()
                .flat_map(|cell| if both { cell.to_vec() } else { vec![cell[0]] })
                .filter(|value| *value > 1.0)
                .collect::<Vec<_>>();
            values.sort_by(|a, b| b.total_cmp(a));
            values
        };
        let current = shared.iter().chain(&removed).copied().collect::<Vec<_>>();
        let next = shared.iter().chain(&added).copied().collect::<Vec<_>>();
        for floor in [0.0, 1.2, 1.3, 1.75] {
            let floor_key = if floor > 0.0 {
                crate::cdt::volume_quality::violation_priority_key(floor)
            } else {
                0
            };
            let prefix = |values: Vec<f64>| {
                values
                    .into_iter()
                    .filter(|value| {
                        crate::cdt::volume_quality::violation_priority_key(*value) >= floor_key
                    })
                    .collect::<Vec<_>>()
            };
            let expected = crate::cdt::volume_quality::compare_spectra(
                &prefix(spectrum(&next, true)),
                &prefix(spectrum(&current, true)),
            ) == Ordering::Less;
            let ranking = Ranking {
                coupled: true,
                objective: QualityObjective::Shape,
                floor: floor_key,
            };
            assert_eq!(
                candidate.admissible(ranking),
                expected,
                "{removed:?} -> {added:?} at {floor}"
            );
        }
        let expected = crate::cdt::volume_quality::compare_spectra(
            &spectrum(&next, false),
            &spectrum(&current, false),
        ) == Ordering::Less
            && crate::cdt::volume_quality::compare_spectra(
                &spectrum(
                    &next.iter().map(|cell| [cell[1], 0.0]).collect::<Vec<_>>(),
                    false,
                ),
                &spectrum(
                    &current
                        .iter()
                        .map(|cell| [cell[1], 0.0])
                        .collect::<Vec<_>>(),
                    false,
                ),
            ) != Ordering::Greater;
        let ranking = Ranking {
            coupled: false,
            objective: QualityObjective::Shape,
            floor: 0,
        };
        assert_eq!(
            candidate.admissible(ranking),
            expected,
            "{removed:?} -> {added:?} preserve"
        );
    }
}
