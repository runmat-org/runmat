//! Complete independent checks after every insertion in an admitted prefix.
use super::*;

#[test]
fn admitted_insertions_discharge_each_changed_neighborhood_before_the_next_point() {
    let options = DelaunayInsertionOptions::default();
    let admitted =
        AdmittedPhysicalTopology::verify_global(enclosing_tetrahedron(), options, &NeverCancelled)
            .unwrap();
    let mut topology = TopologyMutation::from_admitted(admitted);
    for (index, coordinates) in [
        [0.0, 0.0, 0.0],
        [0.2, 0.1, 0.1],
        [-0.2, 0.15, 0.2],
        [0.1, -0.4, -0.3],
        [-0.3, -0.2, -0.1],
        [0.15, 0.2, -0.15],
    ]
    .into_iter()
    .enumerate()
    {
        insert_delaunay_volume_node_into_mutation(
            &mut topology,
            node(50 + index as u8, coordinates),
            options,
            &NeverCancelled,
        )
        .unwrap();
        assert!(topology.physical_dirty_faces().unwrap().is_empty());
        let snapshot = topology
            .snapshot(options.topology, &NeverCancelled)
            .unwrap();
        assert_eq!(snapshot.nodes.len(), 5 + index);
        assert_globally_delaunay(&snapshot);
        validate_delaunay_volume_topology(&snapshot, options, &NeverCancelled).unwrap();
    }
}
