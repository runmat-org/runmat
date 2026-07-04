use super::*;
use runmat_geometry_core::{
    GeometryAsset, GeometrySource, MeshDescriptor, MeshKind, Region, RegionEntityMapping,
    SourceGeometry, SourceGeometryKind, SurfaceMesh, TessellationProfile, UnitSystem,
};

#[test]
fn boundary_input_welds_face_local_duplicate_vertices() {
    let mut geometry = cube_geometry_with_shared_vertices();
    geometry.surface_meshes[0] = SurfaceMesh::new(
        "cube_surface",
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        vec![
            [0, 2, 1],
            [4, 6, 5],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18],
            [19, 20, 21],
            [22, 23, 24],
            [25, 26, 27],
            [28, 29, 30],
            [31, 32, 33],
            [34, 35, 36],
        ],
    );

    let input = boundary_input_from_geometry(&geometry)
        .expect("closed cube with face-local vertices should weld");

    assert_eq!(input.vertices.len(), 8);
    assert_eq!(input.triangles.len(), 12);
}

#[test]
fn boundary_input_rejects_open_shell_after_welding() {
    let mut geometry = cube_geometry_with_shared_vertices();
    geometry.surface_meshes[0].triangles.pop();

    let err = boundary_input_from_geometry(&geometry).expect_err("open shell should fail");

    assert!(matches!(
        err,
        BoundaryMeshInputError::OpenBoundaryEdge { .. }
    ));
}

#[test]
fn boundary_input_converts_millimeter_vertices_to_meters() {
    let mut geometry = cube_geometry_with_shared_vertices();
    geometry.units = UnitSystem::Millimeter;
    for vertex in &mut geometry.surface_meshes[0].vertices {
        for coordinate in vertex {
            *coordinate *= 1000.0;
        }
    }

    let input = boundary_input_from_geometry(&geometry)
        .expect("millimeter cube should convert to meter boundary input");

    assert_eq!(input.bounds_min_m, [0.0, 0.0, 0.0]);
    assert_eq!(input.bounds_max_m, [1.0, 1.0, 1.0]);
    assert!(input
        .vertices
        .iter()
        .any(|vertex| *vertex == [1.0, 1.0, 1.0]));
}

fn cube_geometry_with_shared_vertices() -> GeometryAsset {
    GeometryAsset {
        geometry_id: "geo_boundary_cube".to_string(),
        source: GeometrySource {
            path: "/fixtures/generic_cube.step".to_string(),
            sha256: "generic-cube".to_string(),
            importer_version: "test".to_string(),
        },
        source_geometry: SourceGeometry {
            kind: SourceGeometryKind::Cad,
            assembly: None,
            material_evidence: Vec::new(),
            cad_evaluators: Vec::new(),
        },
        tessellation_profile: TessellationProfile::default(),
        units: UnitSystem::Meter,
        revision: 1,
        meshes: vec![MeshDescriptor {
            mesh_id: "cube_surface".to_string(),
            kind: MeshKind::Surface,
            vertex_count: 8,
            element_count: 12,
        }],
        surface_meshes: vec![SurfaceMesh::new(
            "cube_surface",
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            vec![
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
            ],
        )],
        regions: vec![Region {
            region_id: "region_all".to_string(),
            name: "all".to_string(),
            tag: None,
            cad_ownership: None,
        }],
        region_entity_mappings: vec![RegionEntityMapping::all_faces(
            "region_all",
            "cube_surface",
            12,
        )],
        diagnostics: Vec::new(),
    }
}
