use super::*;
use crate::{discretize_topology_surfaces, SurfaceDiscretizationOptions};
use runmat_meshing_cad::{SourceTopologyFace, SourceTopologyModel, SourceTopologyVertex};

#[test]
fn validates_closed_surface_recovery() {
    let topology = cube_topology();
    let surface = discretize_topology_surfaces(&topology, SurfaceDiscretizationOptions::default())
        .expect("surface should discretize");

    let report = validate_surface_recovery(&topology, &surface, SurfaceRecoveryOptions::default())
        .expect("surface recovery should validate");

    assert_eq!(report.surface_element_count, 12);
    assert_eq!(report.open_edge_count, 0);
    assert_eq!(report.nonmanifold_edge_count, 0);
    assert_eq!(report.source_face_coverage_ratio, 1.0);
    assert!(report.min_normal_alignment >= 1.0 - 1.0e-8);
}

#[test]
fn rejects_surface_with_open_edge() {
    let topology = cube_topology();
    let mut surface =
        discretize_topology_surfaces(&topology, SurfaceDiscretizationOptions::default())
            .expect("surface should discretize");
    surface.elements.pop();

    let err = validate_surface_recovery(
        &topology,
        &surface,
        SurfaceRecoveryOptions {
            require_closed: true,
            ..SurfaceRecoveryOptions::default()
        },
    )
    .expect_err("open surface should fail recovery");

    assert!(matches!(
        err,
        SurfaceRecoveryError::UncoveredSourceFace { .. } | SurfaceRecoveryError::OpenEdge { .. }
    ));
}

#[test]
fn rejects_surface_area_mismatch() {
    let topology = cube_topology();
    let mut surface =
        discretize_topology_surfaces(&topology, SurfaceDiscretizationOptions::default())
            .expect("surface should discretize");
    surface.nodes[1].coordinates_m = [2.0, 0.0, 0.0];

    let err = validate_surface_recovery(
        &topology,
        &surface,
        SurfaceRecoveryOptions {
            max_area_relative_error: 1.0e-12,
            ..SurfaceRecoveryOptions::default()
        },
    )
    .expect_err("area mismatch should fail recovery");

    assert!(matches!(err, SurfaceRecoveryError::AreaMismatch { .. }));
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
    let faces = [
        ([0, 2, 1], [0.0, 0.0, -1.0]),
        ([0, 3, 2], [0.0, 0.0, -1.0]),
        ([4, 5, 6], [0.0, 0.0, 1.0]),
        ([4, 6, 7], [0.0, 0.0, 1.0]),
        ([0, 1, 5], [0.0, -1.0, 0.0]),
        ([0, 5, 4], [0.0, -1.0, 0.0]),
        ([1, 2, 6], [1.0, 0.0, 0.0]),
        ([1, 6, 5], [1.0, 0.0, 0.0]),
        ([2, 3, 7], [0.0, 1.0, 0.0]),
        ([2, 7, 6], [0.0, 1.0, 0.0]),
        ([3, 0, 4], [-1.0, 0.0, 0.0]),
        ([3, 4, 7], [-1.0, 0.0, 0.0]),
    ]
    .into_iter()
    .enumerate()
    .map(|(face_id, (node_ids, unit_normal))| SourceTopologyFace {
        face_id: face_id as u32,
        source_triangle_id: face_id as u32,
        node_ids,
        edge_ids: [
            face_id as u32 * 3,
            face_id as u32 * 3 + 1,
            face_id as u32 * 3 + 2,
        ],
        region_ids: Vec::new(),
        material_region_ids: Vec::new(),
        area_m2: 0.5,
        unit_normal,
    })
    .collect::<Vec<_>>();

    SourceTopologyModel {
        mesh_id: "surface_recovery_cube".to_string(),
        source_geometry_id: "geo_surface_recovery_cube".to_string(),
        source_geometry_revision: 1,
        source_geometry_sha256: None,
        vertices,
        edges: Vec::new(),
        faces,
        bounds_min_m: [0.0, 0.0, 0.0],
        bounds_max_m: [1.0, 1.0, 1.0],
        region_ids: Vec::new(),
        material_region_ids: Vec::new(),
    }
}
