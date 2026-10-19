use super::*;
use crate::reconnect::local_tetrahedron_boundary_faces;
use std::sync::atomic::{AtomicUsize, Ordering};

fn star(count: u32) -> Vec<[u32; 4]> {
    (0..count)
        .map(|index| [0, 1, index + 2, (index + 1) % count + 2])
        .collect()
}

// Independent original exhaustive definition, retained only as an oracle.
fn exhaustive(boundary: &BTreeSet<[u32; 3]>, target: [u32; 2]) -> Vec<[u32; 4]> {
    let other = boundary
        .iter()
        .flatten()
        .copied()
        .filter(|node| !target.contains(node))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let mut created = Vec::new();
    for first in 0..other.len() {
        for second in first + 1..other.len() {
            let mut left = [target[0], other[first], other[second]];
            let mut right = [target[1], other[first], other[second]];
            left.sort_unstable();
            right.sort_unstable();
            if boundary.contains(&left) && boundary.contains(&right) {
                created.push([target[0], target[1], other[first], other[second]]);
            }
        }
    }
    created
}

#[test]
fn boundary_links_match_every_exhaustive_target_and_original_cell_order() {
    for count in [3, 4, 7] {
        let boundary = local_tetrahedron_boundary_faces(&star(count));
        let candidates =
            BoundaryEdgeStarCandidates::new(&boundary, &mut || Ok::<_, ()>(())).unwrap();
        let nodes = boundary
            .iter()
            .flatten()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let mut expected_targets = Vec::new();
        for left in 0..nodes.len() {
            for right in left + 1..nodes.len() {
                let target = [nodes[left], nodes[right]];
                let expected = exhaustive(&boundary, target);
                assert_eq!(
                    candidates
                        .tetrahedra(target, &mut || Ok::<_, ()>(()))
                        .unwrap(),
                    expected
                );
                if expected.len() >= 2 {
                    expected_targets.push(target);
                }
                let reverse = [target[1], target[0]];
                assert_eq!(
                    candidates
                        .tetrahedra(reverse, &mut || Ok::<_, ()>(()))
                        .unwrap(),
                    exhaustive(&boundary, reverse)
                );
            }
        }
        assert_eq!(candidates.targets().collect::<Vec<_>>(), expected_targets);
    }
}

#[test]
fn high_valence_boundary_links_have_linear_candidate_work_and_preserve_coverage() {
    for count in [8, 128] {
        let original = star(count);
        let boundary = local_tetrahedron_boundary_faces(&original);
        let calls = AtomicUsize::new(0);
        let mut checkpoint = || {
            calls.fetch_add(1, Ordering::Relaxed);
            Ok::<_, ()>(())
        };
        let candidates = BoundaryEdgeStarCandidates::new(&boundary, &mut checkpoint).unwrap();
        let mut total_cells = 0;
        assert_eq!(candidates.targets().count(), count as usize + 1);
        for target in candidates.targets() {
            let cells = candidates.tetrahedra(target, &mut checkpoint).unwrap();
            total_cells += cells.len();
            if target == [0, 1] {
                // The fixture is generated around the ring; proposals retain
                // the original exhaustive lexical edge order instead.
                let mut expected = original.clone();
                for cell in &mut expected {
                    cell.sort_unstable();
                }
                expected.sort_unstable();
                assert_eq!(cells, expected);
                assert_eq!(local_tetrahedron_boundary_faces(&cells), boundary);
            }
        }
        assert_eq!(total_cells, count as usize * 3);
        assert_eq!(calls.load(Ordering::Relaxed), count as usize * 8);
    }
}

#[test]
fn every_boundary_link_checkpoint_rejects_without_changing_source_or_queries() {
    let boundary = local_tetrahedron_boundary_faces(&star(8));
    let original = boundary.clone();
    for allowance in 0..40 {
        let mut calls = 0;
        let mut checkpoint = || {
            calls += 1;
            if calls > allowance {
                Err(())
            } else {
                Ok(())
            }
        };
        assert!(BoundaryEdgeStarCandidates::new(&boundary, &mut checkpoint).is_err());
        assert_eq!(boundary, original);
    }
    let candidates = BoundaryEdgeStarCandidates::new(&boundary, &mut || Ok::<_, ()>(())).unwrap();
    for allowance in 0..8 {
        let mut calls = 0;
        assert!(candidates
            .tetrahedra([0, 1], &mut || {
                calls += 1;
                if calls > allowance {
                    Err(())
                } else {
                    Ok(())
                }
            })
            .is_err());
        assert_eq!(
            candidates
                .tetrahedra([0, 1], &mut || Ok::<_, ()>(()))
                .unwrap(),
            exhaustive(&boundary, [0, 1])
        );
        assert_eq!(boundary, original);
    }
}
