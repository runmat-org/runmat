//! Merge behavior independent of CAD production: exact ordering, accumulated
//! edge tolerances, coordinate conflicts, duplicate uses and hard limits.
use super::*;
use crate::{ExactFaceMeshEdgeParameter, EXACT_FACE_MESH_SCHEMA_VERSION};

fn face(topology: &ExactBRepTopology, index: usize, nodes: &[(u8, f64)]) -> ExactFaceMesh {
    let source = topology.faces[0].id.clone();
    ExactFaceMesh {
        schema_version: EXACT_FACE_MESH_SCHEMA_VERSION,
        source_face_id: source.clone(),
        nodes: nodes
            .iter()
            .map(|&(id, x)| ExactFaceMeshNode {
                node_id: StableDigest::from_bytes([id; 32]),
                point_m: [x, 0.0, 0.0],
                uses: vec![ExactFaceMeshNodeUse {
                    source_face_id: source.clone(),
                    chart_id: StableDigest::from_bytes([index as u8 + 1; 32]),
                    uv: [index as f64, 0.0],
                    evaluator_uv: [index as f64, 0.0],
                    exact_edge_parameters: vec![],
                }],
            })
            .collect(),
        triangles: vec![],
        boundary_segments: vec![],
        joined_chart_cuts: vec![],
        maximum_chordal_deviation_m: runmat_types::physical::Length::new(0.0),
        maximum_normal_deviation_rad: 0.0,
    }
}

#[test]
fn canonical_stream_merge_moves_original_nodes_and_orders_all_shared_uses() {
    let (_, topology, _) = runmat_geometry_fixtures::exact_circle();
    let mut faces = vec![
        face(&topology, 0, &[(1, 0.0), (3, 2.0)]),
        face(&topology, 1, &[(2, 1.0), (3, 2.0)]),
        face(&topology, 2, &[(3, 2.0), (4, 3.0)]),
    ];
    let expected_uses = faces
        .iter()
        .flat_map(|face| face.nodes.iter())
        .filter(|node| node.node_id == StableDigest::from_bytes([3; 32]))
        .flat_map(|node| node.uses.clone())
        .collect::<Vec<_>>();
    let original_use_buffer = faces[0].nodes[0].uses.as_ptr();
    let result = merge(
        &topology,
        &mut faces,
        ExactSurfaceJoinOptions::default(),
        None,
        None,
    )
    .unwrap();
    assert_eq!(
        result.iter().map(|node| node.node_id).collect::<Vec<_>>(),
        (1..=4)
            .map(|id| StableDigest::from_bytes([id; 32]))
            .collect::<Vec<_>>()
    );
    assert_eq!(result[2].uses, expected_uses);
    assert_eq!(
        result[0].uses.as_ptr(),
        original_use_buffer,
        "unshared original buffer must move"
    );
    assert!(faces.iter().all(|face| face.nodes.is_empty()));
}

#[test]
fn shared_coordinate_admission_uses_all_previously_merged_edge_tolerances() {
    let (_, mut topology, _) = runmat_geometry_fixtures::exact_circle();
    topology.edges[0].tolerance_m = runmat_types::physical::Length::new(0.1);
    let mut faces = vec![
        face(&topology, 0, &[(1, 0.0)]),
        face(&topology, 1, &[(1, 0.05)]),
        face(&topology, 2, &[(1, 0.075)]),
    ];
    faces[0].nodes[0].uses[0]
        .exact_edge_parameters
        .push(ExactFaceMeshEdgeParameter {
            source_coedge_id: topology.coedges[0].id.clone(),
            source_edge_id: topology.edges[0].id.clone(),
            parameter: 0.0,
        });
    let merged = merge(
        &topology,
        &mut faces,
        ExactSurfaceJoinOptions::default(),
        None,
        None,
    )
    .unwrap();
    assert_eq!(merged[0].point_m, [0.0; 3]);
    assert_eq!(merged[0].uses.len(), 3);
}

#[test]
fn conflicting_coordinates_duplicate_uses_and_unique_node_limit_reject() {
    let (_, topology, _) = runmat_geometry_fixtures::exact_circle();
    for (faces, maximum_nodes) in [
        (
            vec![
                face(&topology, 0, &[(1, 0.0)]),
                face(&topology, 1, &[(1, 1.0)]),
            ],
            10,
        ),
        (
            vec![
                face(&topology, 0, &[(1, 0.0)]),
                face(&topology, 0, &[(1, 0.0)]),
            ],
            10,
        ),
        (vec![face(&topology, 0, &[(1, 0.0), (2, 1.0)])], 1),
    ] {
        let mut faces = faces;
        let error = merge(
            &topology,
            &mut faces,
            ExactSurfaceJoinOptions {
                maximum_nodes,
                ..ExactSurfaceJoinOptions::default()
            },
            None,
            None,
        )
        .unwrap_err();
        assert_eq!(
            error.kind,
            if maximum_nodes == 1 {
                ExactSurfaceMeshErrorKind::ResourceLimit
            } else {
                ExactSurfaceMeshErrorKind::InvalidInput
            }
        );
    }
}
