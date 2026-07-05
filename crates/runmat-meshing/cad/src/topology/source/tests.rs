use super::*;
use crate::extract_source_topology;
use runmat_geometry_core::{
    CadEvaluatorSet, CadFaceEvaluator, CadLabelRef, CadRegionOwnership, CadSemanticKind,
    EntityIdRange, EntityKind, GeometrySource, MeshDescriptor, MeshKind, Region,
    RegionEntityMapping, SourceGeometry, SourceGeometryKind, SurfaceMesh, TessellationProfile,
    UnitSystem,
};

#[test]
fn builds_generic_cad_topology_from_source_triangles() {
    let geometry = cube_geometry(false);
    let topology = extract_source_topology(&geometry).expect("topology should extract");

    let cad = build_cad_topology(&geometry, &topology).expect("cad topology should build");

    assert_eq!(cad.source, CadTopologySource::GenericCadMesh);
    assert_eq!(cad.report.vertex_count, 8);
    assert_eq!(cad.report.edge_count, 18);
    assert_eq!(cad.report.face_count, 6);
    assert_eq!(cad.report.closed_shell_count, 1);
    assert_eq!(cad.report.volume_count, 1);
    assert_eq!(cad.report.semantic_face_count, 0);
    assert_eq!(cad.report.imported_face_count, 0);
    assert_eq!(cad.report.evaluator_face_count, 0);
    assert_eq!(cad.report.generic_face_count, 6);
    assert_eq!(cad.report.loop_count, 6);
    assert_eq!(cad.report.hole_loop_count, 0);
    assert_eq!(cad.loops.len(), 6);
    assert!(cad
        .loops
        .iter()
        .all(|cad_loop| cad_loop.is_outer && cad_loop.edge_ids.len() == 4));
    assert!(cad.faces.iter().all(|face| {
        face.source_face_ids.len() == 2 && face.loop_ids.len() == 1 && face.loop_edge_ids.len() == 4
    }));
}

#[test]
fn preserves_semantic_cad_face_regions() {
    let geometry = cube_geometry(true);
    let topology = extract_source_topology(&geometry).expect("topology should extract");

    let cad = build_cad_topology(&geometry, &topology).expect("cad topology should build");

    assert_eq!(cad.source, CadTopologySource::SemanticCad);
    assert_eq!(cad.report.face_count, 11);
    assert_eq!(cad.report.semantic_face_count, 1);
    assert_eq!(cad.report.imported_face_count, 1);
    assert_eq!(cad.report.evaluator_face_count, 1);
    let semantic_face = cad
        .faces
        .iter()
        .find(|face| face.entity_id.id == "face_000001")
        .expect("semantic face should be merged");
    assert_eq!(semantic_face.imported_face_id, Some(1));
    assert_eq!(
        semantic_face.evaluator_reference_point_m,
        Some([0.5, 0.5, 0.0])
    );
    assert_eq!(semantic_face.source_face_ids, vec![0, 1]);
    assert_eq!(semantic_face.source_edge_ids.len(), 5);
    assert_eq!(semantic_face.loop_ids.len(), 1);
    assert_eq!(semantic_face.loop_edge_ids.len(), 4);
    let semantic_loop = cad
        .loops
        .iter()
        .find(|cad_loop| cad_loop.face_id == semantic_face.entity_id.id)
        .expect("semantic face loop should be represented");
    assert!(semantic_loop.is_outer);
    assert_eq!(semantic_loop.edge_ids, semantic_face.loop_edge_ids);
    assert!((semantic_face.area_m2 - 1.0).abs() <= 1.0e-12);
    assert_eq!(semantic_face.unit_normal, [0.0, 0.0, -1.0]);
}

fn cube_geometry(with_semantic_face: bool) -> runmat_geometry_core::GeometryAsset {
    let face_region = Region {
        region_id: "face_000001".to_string(),
        name: "face".to_string(),
        tag: Some("cad_face".to_string()),
        cad_ownership: with_semantic_face.then(|| CadRegionOwnership {
            face_id: Some(1),
            label: Some(CadLabelRef {
                label_entry: "0:1:1".to_string(),
                name: "face".to_string(),
                kind: CadSemanticKind::Face,
            }),
            owner_path: Vec::new(),
            layers: Vec::new(),
            color: None,
            material: None,
        }),
    };
    runmat_geometry_core::GeometryAsset {
        geometry_id: "geo_cad_topology_cube".to_string(),
        source: GeometrySource {
            path: "/fixtures/generic_cube.step".to_string(),
            sha256: "generic-cube".to_string(),
            importer_version: "test".to_string(),
        },
        source_geometry: SourceGeometry {
            kind: SourceGeometryKind::Cad,
            assembly: None,
            material_evidence: Vec::new(),
            cad_evaluators: if with_semantic_face {
                vec![CadEvaluatorSet {
                    evaluator_id: "cad_evaluator_test".to_string(),
                    backend: "test".to_string(),
                    format_name: "step".to_string(),
                    requires_source_geometry: true,
                    faces: vec![CadFaceEvaluator {
                        evaluator_id: "cad_face_1".to_string(),
                        imported_face_id: 1,
                        name: "face".to_string(),
                        supports_point_evaluation: true,
                        supports_projection: true,
                        supports_normal: true,
                        supports_derivatives: true,
                        supports_curvature: true,
                        reference_point_m: Some([0.5, 0.5, 0.0]),
                        reference_unit_normal: Some([0.0, 0.0, 1.0]),
                        evaluation_samples: Vec::new(),
                    }],
                    curves: Vec::new(),
                }]
            } else {
                Vec::new()
            },
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
        regions: vec![face_region],
        region_entity_mappings: vec![RegionEntityMapping::new(
            "face_000001",
            "cube_surface",
            EntityKind::Face,
            vec![EntityIdRange::new(0, 2)],
        )],
        diagnostics: Vec::new(),
    }
}
