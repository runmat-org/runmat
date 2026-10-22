use super::*;
use crate::cdt::{build_delaunay_volume_topology, DelaunayTopologyOptions, DelaunayVolumeNode};
use runmat_meshing_core::{NeverCancelled, StableDigest};
use std::collections::BTreeSet;

fn fan(count: u32) -> TopologyMutation {
    let mut points = vec![[0., 0., -1.], [0., 0., 1.]];
    for vertex in 0..count {
        let angle = std::f64::consts::TAU * f64::from(vertex) / f64::from(count);
        points.push([angle.cos(), angle.sin(), 0.]);
    }
    let nodes = points
        .into_iter()
        .enumerate()
        .map(|(index, coordinates)| {
            let mut identity = [0; 32];
            identity[..4].copy_from_slice(&(index as u32 + 1).to_be_bytes());
            DelaunayVolumeNode::explicit(StableDigest::from_bytes(identity), coordinates)
        })
        .collect();
    let cells = (0..count)
        .map(|index| [0, 1, index + 2, (index + 1) % count + 2])
        .collect();
    TopologyMutation::from_canonical(
        build_delaunay_volume_topology(
            nodes,
            cells,
            DelaunayTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap(),
    )
}

fn check_every_face(topology: &TopologyMutation) {
    let canonical = topology
        .snapshot(Default::default(), &NeverCancelled)
        .unwrap();
    for first in 0..topology.nodes().len() as u32 {
        for second in first + 1..topology.nodes().len() as u32 {
            for third in second + 1..topology.nodes().len() as u32 {
                let face = [first, second, third];
                let expected = (0..topology.tetrahedron_slot_count())
                    .filter_map(|index| {
                        topology
                            .tetrahedron(index as u32)
                            .filter(|cell| {
                                face.iter()
                                    .all(|vertex| cell.vertex_indices.contains(vertex))
                            })
                            .map(|_| index as u32)
                    })
                    .collect::<Vec<_>>();
                for ordering in [face, [third, first, second]] {
                    assert_eq!(
                        topology
                            .face_tetrahedra_with_checkpoint(ordering, &mut || Ok::<_, ()>(()))
                            .unwrap(),
                        expected
                    );
                }
                let canonical_expected = canonical
                    .tetrahedra
                    .iter()
                    .enumerate()
                    .filter_map(|(index, cell)| {
                        face.iter()
                            .all(|vertex| cell.vertex_indices.contains(vertex))
                            .then_some(index as u32)
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    canonical
                        .face_tetrahedra_with_checkpoint(face, &mut || Ok::<_, ()>(()))
                        .unwrap(),
                    canonical_expected
                );
            }
        }
    }
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Maintained mutable face owners and immutable smallest-star queries match exhaustive
//   cell membership, including absent faces, vertex orderings, checked cavity commit and rollback.
//   Does not prove geometric acceptance by owner lookup alone or whole native recovery termination.
#[test]
fn exact_face_owners_match_exhaustive_inventory_through_commit_and_rollback() {
    let mut topology = fan(8);
    check_every_face(&topology);
    let removed = (0..topology.tetrahedron_slot_count())
        .filter_map(|index| {
            let cell = topology.tetrahedron(index as u32).unwrap();
            [0, 1, 3]
                .iter()
                .all(|vertex| cell.vertex_indices.contains(vertex))
                .then_some(index as u32)
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(removed.len(), 2);
    let original = topology
        .snapshot(Default::default(), &NeverCancelled)
        .unwrap();
    {
        let mut trial = topology.cavity_trial();
        trial
            .replace_cavity(
                &removed,
                vec![
                    ([2, 4, 0, 1], None),
                    ([2, 4, 1, 3], None),
                    ([2, 4, 3, 0], None),
                ],
            )
            .unwrap();
        check_every_face(&trial);
    }
    assert_eq!(
        topology
            .snapshot(Default::default(), &NeverCancelled)
            .unwrap(),
        original
    );
    check_every_face(&topology);
    let mut trial = topology.cavity_trial();
    trial
        .replace_cavity(
            &removed,
            vec![
                ([2, 4, 0, 1], None),
                ([2, 4, 1, 3], None),
                ([2, 4, 3, 0], None),
            ],
        )
        .unwrap();
    trial.commit();
    check_every_face(&topology);
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A maintained interior-face query uses three checkpoints across 4/64-cell endpoint
//   stars, returns both original owners and rejects/retries at every query checkpoint. Excludes
//   face-index construction/update work, arbitrary mesh admission and whole-task timing/memory.
#[test]
fn maintained_face_query_work_is_independent_of_endpoint_valence_and_cancellable() {
    let mut costs = Vec::new();
    for count in [4, 64] {
        let topology = fan(count);
        assert_eq!(topology.vertex_star_len(0), count as usize);
        let mut work = 0;
        let expected = topology
            .face_tetrahedra_with_checkpoint([0, 1, 2], &mut || {
                work += 1;
                Ok::<_, ()>(())
            })
            .unwrap();
        assert_eq!(expected.len(), 2);
        costs.push(work);
        for stop in 1..=work {
            let mut calls = 0;
            assert_eq!(
                topology.face_tetrahedra_with_checkpoint([0, 1, 2], &mut || {
                    calls += 1;
                    if calls == stop {
                        Err("cancelled")
                    } else {
                        Ok(())
                    }
                }),
                Err("cancelled")
            );
            assert_eq!(
                topology
                    .face_tetrahedra_with_checkpoint([0, 1, 2], &mut || Ok::<_, ()>(()))
                    .unwrap(),
                expected
            );
        }
    }
    assert_eq!(costs, vec![3, 3]);
}
