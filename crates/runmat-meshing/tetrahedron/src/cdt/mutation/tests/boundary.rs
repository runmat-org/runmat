use super::*;
use std::collections::BTreeMap;

fn check_boundary_index(topology: &TopologyMutation) {
    let expected = topology
        .tetrahedra
        .iter()
        .enumerate()
        .filter_map(|(index, tet)| tet.as_ref().map(|tet| (index, tet)))
        .flat_map(|(index, tet)| {
            (0..4).filter_map(move |opposite| {
                if tet.neighbors[opposite].is_some() {
                    return None;
                }
                let mut face = tet
                    .vertex_indices
                    .iter()
                    .enumerate()
                    .filter_map(|(slot, vertex)| (slot != opposite).then_some(*vertex))
                    .collect::<Vec<_>>();
                face.sort_unstable();
                Some(([face[0], face[1], face[2]], (index as u32, opposite)))
            })
        })
        .collect::<BTreeMap<_, _>>();
    assert_eq!(topology.boundary_faces, expected);
}

#[test]
fn local_mutation_keeps_boundary_index_and_neighbors_in_agreement() {
    let mut topology = triangular_bipyramid();
    check_boundary_index(&topology);
    topology
        .replace_cavity(
            &BTreeSet::from([0, 1]),
            vec![
                ([0, 1, 3, 4], None),
                ([1, 2, 3, 4], None),
                ([2, 0, 3, 4], None),
            ],
        )
        .unwrap();
    check_boundary_index(&topology);
    assert_eq!(topology.active_tetrahedron_count(), 3);
}

fn single_tetrahedron() -> TopologyMutation {
    TopologyMutation::from_canonical(
        build_delaunay_volume_topology(
            vec![
                node(10, [0., 0., 0.]),
                node(20, [1., 0., 0.]),
                node(30, [0., 1., 0.]),
                node(40, [0., 0., 1.]),
            ],
            vec![[0, 1, 2, 3]],
            DelaunayTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap(),
    )
}

#[test]
fn hull_growth_attaches_to_the_opposite_side_of_an_existing_boundary_face() {
    let mut topology = single_tetrahedron();
    let inserted = topology.append_node(node(50, [3., 3., 3.])).unwrap();
    topology
        .replace_hull_cavity(
            &BTreeSet::new(),
            vec![([1, 2, 3, inserted], None)],
            &BTreeSet::from([[1, 2, 3]]),
        )
        .unwrap();
    check_boundary_index(&topology);
    assert!(!topology.boundary_faces.contains_key(&[1, 2, 3]));
    topology
        .freeze(DelaunayTopologyOptions::default(), &NeverCancelled)
        .unwrap();
}

#[test]
fn hull_attachment_rejects_overlap_without_changing_tetrahedra_or_boundary_index() {
    let mut topology = single_tetrahedron();
    let before = topology.boundary_faces.clone();
    let inserted = topology.append_node(node(50, [0.2, 0.2, 0.2])).unwrap();
    assert!(topology
        .replace_hull_cavity(
            &BTreeSet::new(),
            vec![([1, 2, 3, inserted], None)],
            &BTreeSet::from([[1, 2, 3]])
        )
        .is_err());
    assert_eq!(topology.active_tetrahedron_count(), 1);
    assert_eq!(topology.boundary_faces, before);
    check_boundary_index(&topology);
}
