use super::super::super::*;

pub(in crate::cavity::constrained::tests) fn octahedron_cavity() -> ConstrainedCavity {
    ConstrainedCavity {
        removed_tetrahedron_ids: vec![1, 2],
        boundary_faces: [
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
            [1, 0, 5],
            [2, 1, 5],
            [3, 2, 5],
            [0, 3, 5],
        ]
        .into_iter()
        .map(|node_ids| ConstrainedCavityBoundaryFace {
            node_ids,
            outside_tetrahedron_ids: Vec::new(),
            source_face_id: None,
            source_edge_ids: [None, None, None],
            region_ids: vec!["body".to_string()],
        })
        .collect(),
        protected_node_ids: Vec::new(),
        target_volume_m3: 4.0 / 3.0,
    }
}

pub(in crate::cavity::constrained::tests) fn octahedron_nodes() -> Vec<ConstrainedCavityNode> {
    vec![
        ConstrainedCavityNode {
            node_id: 0,
            coordinates_m: [1.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 1,
            coordinates_m: [0.0, 1.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 2,
            coordinates_m: [-1.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 3,
            coordinates_m: [0.0, -1.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 4,
            coordinates_m: [0.0, 0.0, 1.0],
        },
        ConstrainedCavityNode {
            node_id: 5,
            coordinates_m: [0.0, 0.0, -1.0],
        },
    ]
}

pub(in crate::cavity::constrained::tests) fn unit_cube_cavity() -> ConstrainedCavity {
    ConstrainedCavity {
        removed_tetrahedron_ids: vec![1],
        boundary_faces: [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ]
        .into_iter()
        .map(|node_ids| ConstrainedCavityBoundaryFace {
            node_ids,
            outside_tetrahedron_ids: Vec::new(),
            source_face_id: None,
            source_edge_ids: [None, None, None],
            region_ids: vec!["body".to_string()],
        })
        .collect(),
        protected_node_ids: Vec::new(),
        target_volume_m3: 1.0,
    }
}

pub(in crate::cavity::constrained::tests) fn unit_cube_nodes() -> Vec<ConstrainedCavityNode> {
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
    .into_iter()
    .enumerate()
    .map(|(node_id, coordinates_m)| ConstrainedCavityNode {
        node_id: node_id as u32,
        coordinates_m,
    })
    .collect()
}
