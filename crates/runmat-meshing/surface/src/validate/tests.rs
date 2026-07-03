use super::*;
use crate::{
    discretize_topology_surfaces, math::distance, validate::geometry::sorted_edge,
    validate::source_edges::count_closed_source_edge_loops, SurfaceDiscretizationOptions,
    INTERNAL_SOURCE_EDGE_ID,
};
use runmat_meshing_cad::{
    SourceTopologyEdge, SourceTopologyFace, SourceTopologyModel, SourceTopologyVertex,
};
use std::collections::BTreeMap;

#[test]
fn validates_surface_projection_and_source_edge_loops() {
    let topology = cube_topology();
    let surface = discretize_topology_surfaces(&topology, SurfaceDiscretizationOptions::default())
        .expect("surface should discretize");

    let report =
        validate_surface_discretization(&topology, &surface, SurfaceValidationOptions::default())
            .expect("surface validation should pass");

    assert_eq!(report.source_face_count, 12);
    assert_eq!(report.surface_element_count, 12);
    assert_eq!(report.source_edge_loop_count, 1);
    assert_eq!(report.closed_source_edge_loop_count, 1);
    assert_eq!(report.conforming_source_edge_count, 18);
    assert_eq!(report.missing_source_edge_count, 0);
    assert_eq!(report.max_projection_error_m, 0.0);
    assert_eq!(report.face_coverage_ratio, 1.0);
}

#[test]
fn rejects_surface_projection_drift() {
    let topology = cube_topology();
    let mut surface =
        discretize_topology_surfaces(&topology, SurfaceDiscretizationOptions::default())
            .expect("surface should discretize");
    surface.nodes[0].coordinates_m = [0.0, 0.0, 0.1];

    let err = validate_surface_discretization(
        &topology,
        &surface,
        SurfaceValidationOptions {
            max_projection_error_m: 1.0e-6,
            ..SurfaceValidationOptions::default()
        },
    )
    .expect_err("projection drift should fail");

    assert!(matches!(
        err,
        SurfaceValidationError::ProjectionError { .. }
    ));
}

#[test]
fn orientation_mismatch_reports_source_face() {
    let topology = cube_topology();
    let mut surface =
        discretize_topology_surfaces(&topology, SurfaceDiscretizationOptions::default())
            .expect("surface should discretize");
    let source_face_id = surface.elements[0].source_face_id;
    surface.elements[0].node_ids.swap(1, 2);

    let err =
        validate_surface_discretization(&topology, &surface, SurfaceValidationOptions::default())
            .expect_err("flipped surface orientation should fail");

    assert_eq!(
        err,
        SurfaceValidationError::OrientationMismatch {
            element_id: 0,
            source_face_id,
            alignment: -1.0,
            min_alignment: SurfaceValidationOptions::default().min_orientation_alignment,
        }
    );
    assert!(err
        .to_string()
        .contains(&format!("source face {source_face_id}")));
}

#[test]
fn edge_conformity_failure_reports_recovered_segment_count() {
    let topology = cube_topology();
    let mut surface =
        discretize_topology_surfaces(&topology, SurfaceDiscretizationOptions::default())
            .expect("surface should discretize");
    let edge = topology.edges[0].clone();
    for element in &mut surface.elements {
        for source_edge_id in &mut element.source_edge_ids {
            if *source_edge_id == edge.edge_id {
                *source_edge_id = INTERNAL_SOURCE_EDGE_ID;
            }
        }
    }

    let err =
        validate_surface_discretization(&topology, &surface, SurfaceValidationOptions::default())
            .expect_err("missing source edge recovery should fail");

    assert_eq!(
        err,
        SurfaceValidationError::EdgeConformityFailed {
            source_edge_id: edge.edge_id,
            source_edge_node_ids: edge.node_ids,
            recovered_segment_count: 0,
        }
    );
    let message = err.to_string();
    assert!(message.contains(&format!("source edge {}", edge.edge_id)));
    assert!(message.contains("recovered surface segment count is 0"));
}

#[test]
fn rejects_open_source_loop() {
    let mut topology = cube_topology();
    topology.edges[0].node_ids = [100, 101];
    let surface = discretize_topology_surfaces(&topology, SurfaceDiscretizationOptions::default())
        .expect("surface should discretize");

    let err = validate_surface_discretization(
        &topology,
        &surface,
        SurfaceValidationOptions {
            require_source_edge_conformity: false,
            ..SurfaceValidationOptions::default()
        },
    )
    .expect_err("open source loop should fail");

    assert!(matches!(err, SurfaceValidationError::OpenSourceLoop { .. }));
}

#[test]
fn counts_disconnected_closed_source_edge_loop_components() {
    let edges = vec![
        source_edge(0, [0, 1]),
        source_edge(1, [1, 2]),
        source_edge(2, [2, 0]),
        source_edge(3, [3, 4]),
        source_edge(4, [4, 5]),
        source_edge(5, [5, 3]),
    ];

    let (loop_count, closed_loop_count) =
        count_closed_source_edge_loops(&edges).expect("closed loops should count");

    assert_eq!(loop_count, 2);
    assert_eq!(closed_loop_count, 2);
}

#[test]
fn rejects_disconnected_open_source_edge_component() {
    let edges = vec![
        source_edge(0, [0, 1]),
        source_edge(1, [1, 2]),
        source_edge(2, [2, 0]),
        source_edge(3, [3, 4]),
    ];

    let err =
        count_closed_source_edge_loops(&edges).expect_err("open component should fail closed");

    assert!(matches!(
        err,
        SurfaceValidationError::OpenSourceLoop {
            source_edge_id: 3,
            ..
        }
    ));
}

fn source_edge(edge_id: u32, node_ids: [u32; 2]) -> SourceTopologyEdge {
    SourceTopologyEdge {
        edge_id,
        node_ids,
        adjacent_face_ids: Vec::new(),
        region_ids: Vec::new(),
        length_m: 1.0,
    }
}

fn cube_topology() -> SourceTopologyModel {
    let vertices = [
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
    .map(|(vertex_id, coordinates_m)| SourceTopologyVertex {
        vertex_id: vertex_id as u32,
        coordinates_m,
    })
    .collect::<Vec<_>>();
    let face_nodes = [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ];
    let face_normals = [
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ];
    let mut edge_ids = BTreeMap::<[u32; 2], u32>::new();
    let mut edge_faces = BTreeMap::<[u32; 2], Vec<u32>>::new();
    let mut faces = Vec::<SourceTopologyFace>::new();
    for (face_id, node_ids) in face_nodes.into_iter().enumerate() {
        let mut face_edge_ids = [0_u32; 3];
        for (index, edge) in [
            sorted_edge(node_ids[0], node_ids[1]),
            sorted_edge(node_ids[1], node_ids[2]),
            sorted_edge(node_ids[2], node_ids[0]),
        ]
        .into_iter()
        .enumerate()
        {
            let next_edge_id = edge_ids.len() as u32;
            let edge_id = *edge_ids.entry(edge).or_insert(next_edge_id);
            face_edge_ids[index] = edge_id;
            edge_faces.entry(edge).or_default().push(face_id as u32);
        }
        faces.push(SourceTopologyFace {
            face_id: face_id as u32,
            source_triangle_id: face_id as u32,
            node_ids,
            edge_ids: face_edge_ids,
            region_ids: Vec::new(),
            area_m2: 0.5,
            unit_normal: face_normals[face_id],
        });
    }
    let mut edges = edge_ids
        .into_iter()
        .map(|(node_ids, edge_id)| SourceTopologyEdge {
            edge_id,
            node_ids,
            adjacent_face_ids: edge_faces.remove(&node_ids).unwrap_or_default(),
            region_ids: Vec::new(),
            length_m: distance(
                vertices[node_ids[0] as usize].coordinates_m,
                vertices[node_ids[1] as usize].coordinates_m,
            ),
        })
        .collect::<Vec<_>>();
    edges.sort_by_key(|edge| edge.edge_id);

    SourceTopologyModel {
        mesh_id: "surface_validate_cube".to_string(),
        source_geometry_id: "geo_surface_validate_cube".to_string(),
        source_geometry_revision: 1,
        source_geometry_sha256: None,
        vertices,
        edges,
        faces,
        bounds_min_m: [0.0, 0.0, 0.0],
        bounds_max_m: [1.0, 1.0, 1.0],
        region_ids: Vec::new(),
    }
}
