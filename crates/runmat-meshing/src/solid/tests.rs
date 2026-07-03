use super::*;
use runmat_geometry_core::{
    GeometryAsset, GeometrySource, MeshDescriptor, MeshKind, Region, RegionEntityMapping,
    SourceGeometry, SourceGeometryKind, SurfaceMesh, TessellationProfile, UnitSystem,
};
use runmat_meshing_core::{validate_analysis_mesh, QualityThresholds};

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
