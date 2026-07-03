use super::*;

mod boundary_edge_recovery;
mod boundary_edge_refinement;
mod boundary_face_completion;
mod boundary_node_flips;
mod boundary_refinement;
mod component_retriangulation;
mod exact_cover;
mod extraction_validation;
mod interior_refill;
mod local_flips;
mod missing_face_caps;
mod refill_evaluation;
mod refill_rejections;
mod shared_face_split;
mod solid_empty;
mod validation;

fn tetrahedron_cavity() -> ConstrainedCavity {
    ConstrainedCavity {
        removed_tetrahedron_ids: vec![7],
        boundary_faces: vec![
            face([0, 1, 2]),
            face([0, 3, 1]),
            face([1, 3, 2]),
            face([2, 3, 0]),
        ],
        protected_node_ids: vec![0, 1],
        target_volume_m3: 1.0,
    }
}

fn face(node_ids: [u32; 3]) -> ConstrainedCavityBoundaryFace {
    ConstrainedCavityBoundaryFace {
        node_ids,
        outside_tetrahedron_ids: Vec::new(),
        source_face_id: None,
        source_edge_ids: [None, None, None],
        region_ids: Vec::new(),
    }
}

fn provenance_cavity() -> ConstrainedCavity {
    ConstrainedCavity {
        removed_tetrahedron_ids: vec![7],
        boundary_faces: vec![
            face_with_provenance(
                [0, 1, 2],
                10,
                [Some(100), Some(101), Some(102)],
                &["loaded", "fixed"],
            ),
            face_with_provenance([0, 3, 1], 11, [Some(103), Some(104), Some(100)], &["fixed"]),
            face_with_provenance([1, 3, 2], 12, [Some(104), Some(105), Some(101)], &["solid"]),
            face_with_provenance([2, 3, 0], 13, [Some(105), Some(103), Some(102)], &["solid"]),
        ],
        protected_node_ids: vec![0, 1],
        target_volume_m3: 1.0,
    }
}

fn unit_tetrahedron_cavity() -> ConstrainedCavity {
    ConstrainedCavity {
        removed_tetrahedron_ids: vec![1],
        boundary_faces: tetrahedron_faces([0, 1, 2, 3])
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
        target_volume_m3: 1.0 / 6.0,
    }
}

fn unit_tetrahedron_nodes() -> Vec<ConstrainedCavityNode> {
    vec![
        ConstrainedCavityNode {
            node_id: 0,
            coordinates_m: [0.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 1,
            coordinates_m: [1.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 2,
            coordinates_m: [0.0, 1.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 3,
            coordinates_m: [0.0, 0.0, 1.0],
        },
    ]
}

fn octahedron_cavity() -> ConstrainedCavity {
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

fn octahedron_nodes() -> Vec<ConstrainedCavityNode> {
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

fn unit_cube_cavity() -> ConstrainedCavity {
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

fn unit_cube_nodes() -> Vec<ConstrainedCavityNode> {
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

fn two_tetrahedron_bipyramid_cavity() -> ConstrainedCavity {
    ConstrainedCavity {
        removed_tetrahedron_ids: vec![1, 2],
        boundary_faces: [
            [0, 1, 3],
            [1, 2, 3],
            [0, 2, 3],
            [0, 2, 4],
            [1, 2, 4],
            [0, 1, 4],
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
        target_volume_m3: 1.0 / 3.0,
    }
}

fn two_tetrahedron_bipyramid_nodes() -> Vec<ConstrainedCavityNode> {
    vec![
        ConstrainedCavityNode {
            node_id: 0,
            coordinates_m: [0.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 1,
            coordinates_m: [1.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 2,
            coordinates_m: [0.0, 1.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 3,
            coordinates_m: [0.0, 0.0, 1.0],
        },
        ConstrainedCavityNode {
            node_id: 4,
            coordinates_m: [0.0, 0.0, -1.0],
        },
    ]
}

fn two_tetrahedron_face_flip_nodes() -> Vec<ConstrainedCavityNode> {
    vec![
        ConstrainedCavityNode {
            node_id: 0,
            coordinates_m: [0.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 1,
            coordinates_m: [1.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 2,
            coordinates_m: [0.0, 1.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 3,
            coordinates_m: [1.0 / 3.0, 1.0 / 3.0, 1.0],
        },
        ConstrainedCavityNode {
            node_id: 4,
            coordinates_m: [1.0 / 3.0, 1.0 / 3.0, -1.0],
        },
    ]
}

fn triangular_edge_ring_nodes() -> Vec<ConstrainedCavityNode> {
    vec![
        ConstrainedCavityNode {
            node_id: 0,
            coordinates_m: [0.0, 0.0, -1.0],
        },
        ConstrainedCavityNode {
            node_id: 3,
            coordinates_m: [0.0, 0.0, 1.0],
        },
        ConstrainedCavityNode {
            node_id: 4,
            coordinates_m: [1.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 5,
            coordinates_m: [-0.5, 0.8660254037844386, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 6,
            coordinates_m: [-0.5, -0.8660254037844386, 0.0],
        },
    ]
}

fn refill_options() -> ConstrainedCavityRefillOptions {
    ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        volume_relative_tolerance: 1.0e-12,
        ..ConstrainedCavityRefillOptions::default()
    }
}

fn protected_refill_options() -> ConstrainedCavityRefillOptions {
    ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        volume_relative_tolerance: 1.0e-12,
        min_protected_node_distance_m: 0.10,
        ..ConstrainedCavityRefillOptions::default()
    }
}

fn synthetic_refill_tetrahedron(
    node_ids: [u32; 4],
    volume_m3: f64,
) -> ConstrainedCavityRefillTetrahedron {
    ConstrainedCavityRefillTetrahedron {
        node_ids,
        volume_m3,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 1.0,
    }
}

fn face_with_provenance(
    node_ids: [u32; 3],
    source_face_id: u32,
    source_edge_ids: [Option<u32>; 3],
    region_ids: &[&str],
) -> ConstrainedCavityBoundaryFace {
    ConstrainedCavityBoundaryFace {
        node_ids,
        outside_tetrahedron_ids: Vec::new(),
        source_face_id: Some(source_face_id),
        source_edge_ids,
        region_ids: region_ids.iter().map(|region| region.to_string()).collect(),
    }
}

fn source_edge_for(face: &ConstrainedCavityBoundaryFace, edge: [u32; 2]) -> Option<u32> {
    face_edges(face.node_ids)
        .into_iter()
        .zip(face.source_edge_ids)
        .find_map(|(candidate_edge, source_edge_id)| {
            (sorted_edge(candidate_edge) == sorted_edge(edge)).then_some(source_edge_id)
        })
        .flatten()
}

fn candidate_tetrahedron(
    tetrahedron_id: u32,
    node_ids: [u32; 4],
    volume_m3: f64,
    region_ids: &[&str],
) -> CavityTetrahedron {
    CavityTetrahedron {
        tetrahedron_id,
        component_id: 0,
        node_ids,
        source_surface_element_id: 0,
        region_ids: region_ids.iter().map(|region| region.to_string()).collect(),
        volume_m3,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 1.0,
    }
}
