use super::*;
use runmat_geometry_core::{
    GeometryAsset, GeometrySource, MeshDescriptor, MeshKind, Region, RegionEntityMapping,
    SourceGeometry, SourceGeometryKind, SurfaceMesh, TessellationProfile, UnitSystem,
};
use runmat_meshing_core::{
    validate_analysis_mesh, MeshBackendKind, MeshTargetSize, QualityThresholds,
    VolumeMeshingOptions,
};

#[test]
fn auto_backend_runs_topology_first_solid_pipeline() {
    let mesh = generate_analysis_mesh(&cube_geometry(), VolumeMeshingOptions::default())
        .expect("auto backend should run the root solid pipeline");

    assert_eq!(mesh.backend.backend, "solid");
    assert_eq!(mesh.backend.algorithm, "topology_first_plc_tetrahedron/v1");
    assert!(!mesh.volume_elements.is_empty());
    assert!(!mesh.boundary_faces.is_empty());
    assert_eq!(mesh.backend.boundary_face_recovery_ratio, 1.0);
    validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect("root solid pipeline should produce a solve-ready mesh for a generic cube");
}

#[test]
fn explicit_sizing_generates_solve_ready_single_tetrahedron_mesh() {
    let mesh = generate_analysis_mesh(
        &tetrahedron_geometry(),
        VolumeMeshingOptions {
            target_size: MeshTargetSize::LengthM(10.0),
            ..VolumeMeshingOptions::default()
        },
    )
    .expect("tetrahedron PLC should run through the root solid pipeline");

    assert_eq!(mesh.backend.backend, "solid");
    assert_eq!(mesh.volume_elements.len(), 1);
    assert_eq!(mesh.boundary_faces.len(), 4);
    validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect("single Tetrahedron solid mesh should be solve-ready");
}

#[test]
fn explicit_sizing_generates_solve_ready_convex_octahedron_mesh() {
    let mesh = generate_analysis_mesh(
        &octahedron_geometry(),
        VolumeMeshingOptions {
            target_size: MeshTargetSize::LengthM(10.0),
            ..VolumeMeshingOptions::default()
        },
    )
    .expect("convex octahedron PLC should run through the root solid pipeline");

    assert_eq!(mesh.backend.backend, "solid");
    assert_eq!(mesh.volume_elements.len(), 8);
    assert_eq!(mesh.boundary_faces.len(), 8);
    validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect("convex octahedron solid mesh should be solve-ready");
}

#[test]
fn structured_fallback_still_uses_structured_stage_explicitly() {
    let mesh = generate_analysis_mesh(
        &cube_geometry(),
        VolumeMeshingOptions {
            backend: MeshBackendKind::StructuredTetrahedronFallback,
            ..VolumeMeshingOptions::default()
        },
    )
    .expect("explicit structured fallback should still be available");

    assert_eq!(mesh.backend.backend, "structured_tetrahedron_fallback");
}

fn cube_geometry() -> GeometryAsset {
    GeometryAsset {
        geometry_id: "geo_root_meshing_cube".to_string(),
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
            region_id: "region_boundary".to_string(),
            name: "boundary".to_string(),
            tag: Some("boundary".to_string()),
            cad_ownership: None,
        }],
        region_entity_mappings: vec![RegionEntityMapping::all_faces(
            "region_boundary",
            "cube_surface",
            12,
        )],
        diagnostics: Vec::new(),
    }
}

fn octahedron_geometry() -> GeometryAsset {
    GeometryAsset {
        geometry_id: "geo_root_meshing_octahedron".to_string(),
        source: GeometrySource {
            path: "/fixtures/generic_octahedron.step".to_string(),
            sha256: "generic-octahedron".to_string(),
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
            mesh_id: "octahedron_surface".to_string(),
            kind: MeshKind::Surface,
            vertex_count: 6,
            element_count: 8,
        }],
        surface_meshes: vec![SurfaceMesh::new(
            "octahedron_surface",
            vec![
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            vec![
                [0, 1, 2],
                [0, 2, 3],
                [0, 3, 4],
                [0, 4, 1],
                [5, 2, 1],
                [5, 3, 2],
                [5, 4, 3],
                [5, 1, 4],
            ],
        )],
        regions: vec![Region {
            region_id: "region_boundary".to_string(),
            name: "boundary".to_string(),
            tag: Some("boundary".to_string()),
            cad_ownership: None,
        }],
        region_entity_mappings: vec![RegionEntityMapping::all_faces(
            "region_boundary",
            "octahedron_surface",
            8,
        )],
        diagnostics: Vec::new(),
    }
}

fn tetrahedron_geometry() -> GeometryAsset {
    GeometryAsset {
        geometry_id: "geo_root_meshing_tetrahedron".to_string(),
        source: GeometrySource {
            path: "/fixtures/generic_tetrahedron.step".to_string(),
            sha256: "generic-tetrahedron".to_string(),
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
            mesh_id: "tetrahedron_surface".to_string(),
            kind: MeshKind::Surface,
            vertex_count: 4,
            element_count: 4,
        }],
        surface_meshes: vec![SurfaceMesh::new(
            "tetrahedron_surface",
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
        )],
        regions: vec![Region {
            region_id: "region_boundary".to_string(),
            name: "boundary".to_string(),
            tag: Some("boundary".to_string()),
            cad_ownership: None,
        }],
        region_entity_mappings: vec![RegionEntityMapping::all_faces(
            "region_boundary",
            "tetrahedron_surface",
            4,
        )],
        diagnostics: Vec::new(),
    }
}
