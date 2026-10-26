use super::*;
use crate::cdt::topology::VolumeTopologyRead;

fn cone(count: u32) -> TopologyMutation {
    use crate::cdt::{build_delaunay_volume_topology, DelaunayVolumeNode};
    let node = |index: u32, point| {
        let mut identity = [0; 32];
        identity[28..].copy_from_slice(&index.to_be_bytes());
        DelaunayVolumeNode::explicit(StableDigest::from_bytes(identity), point)
    };
    let mut nodes = vec![node(1, [0.0, 0.0, 1.0]), node(2, [0.0; 3])];
    for index in 0..count {
        let angle = std::f64::consts::TAU * f64::from(index) / f64::from(count);
        nodes.push(node(index + 3, [angle.cos(), angle.sin(), 0.0]));
    }
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

#[test]
fn local_undo_membership_count_is_independent_of_vertex_degree() {
    let mut costs = Vec::new();
    for count in [8, 128, 2048] {
        let mut topology = cone(count);
        let original = topology.clone();
        assert_eq!(topology.vertex_stars[0].len(), count as usize);
        {
            let mut trial = topology.cavity_trial();
            let source = trial.cell(0).vertex_indices;
            let point = std::array::from_fn(|axis| {
                source
                    .iter()
                    .map(|vertex| trial.node(*vertex).coordinates_m.canonical_value()[axis])
                    .sum::<f64>()
                    / 4.0
            });
            let inserted = trial
                .append_node(DelaunayVolumeNode::explicit(
                    StableDigest::from_bytes([255; 32]),
                    point,
                ))
                .unwrap();
            let created = (0..4)
                .map(|opposite| {
                    let mut vertices = source;
                    vertices[opposite] = inserted;
                    (vertices, None)
                })
                .collect();
            trial.replace_cavity(&BTreeSet::from([0]), created).unwrap();
            costs.push(trial.trial_journals.last().unwrap().stars.len());
        }
        assert_storage(&topology, &original);
        topology
            .freeze(DelaunayTopologyOptions::default(), &NeverCancelled)
            .unwrap();
    }
    assert!(costs.iter().all(|count| *count == costs[0]));
    // Three unchanged memberships are removed from the net undo inventory.
    assert_eq!(costs[0], 14);
}
