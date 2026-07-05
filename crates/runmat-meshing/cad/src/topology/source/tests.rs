use super::*;
use crate::extract_source_topology;
use runmat_geometry_core::{CadCurveEvaluationSample, CadCurveEvaluationSampleSource};
use runmat_geometry_core::{
    CadCurveEvaluator, CadEvaluatorSet, CadFaceEvaluator, CadLabelRef, CadRegionOwnership,
    CadSemanticKind, EntityIdRange, EntityKind, GeometrySource, MeshDescriptor, MeshKind, Region,
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
    assert_eq!(semantic_face.evaluator_id.as_deref(), Some("cad_face_1"));
    assert!(semantic_face.evaluator_supports_point_evaluation);
    assert!(semantic_face.evaluator_supports_projection);
    assert!(semantic_face.evaluator_supports_normal);
    assert!(semantic_face.evaluator_supports_derivatives);
    assert!(semantic_face.evaluator_supports_curvature);
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

#[test]
fn validates_normalized_cad_topology_references_and_report_counts() {
    let geometry = cube_geometry(false);
    let topology = extract_source_topology(&geometry).expect("topology should extract");
    let cad = build_cad_topology(&geometry, &topology).expect("cad topology should build");

    validate_cad_topology_model(&cad).expect("builder output should validate");
}

#[test]
fn rejects_duplicate_cad_entity_ids() {
    let geometry = cube_geometry(false);
    let topology = extract_source_topology(&geometry).expect("topology should extract");
    let mut cad = build_cad_topology(&geometry, &topology).expect("cad topology should build");
    cad.edges[1].entity_id.id = cad.edges[0].entity_id.id.clone();

    let err = validate_cad_topology_model(&cad).expect_err("duplicate edge IDs should fail");

    assert_eq!(
        err,
        CadTopologyError::DuplicateEntityId {
            kind: CadEntityKind::Edge,
            id: cad.edges[0].entity_id.id.clone()
        }
    );
}

#[test]
fn rejects_stale_cad_loop_edge_references() {
    let geometry = cube_geometry(false);
    let topology = extract_source_topology(&geometry).expect("topology should extract");
    let mut cad = build_cad_topology(&geometry, &topology).expect("cad topology should build");
    cad.loops[0].edge_ids[0] = "cad_edge_missing".to_string();

    let err = validate_cad_topology_model(&cad).expect_err("stale loop edge should fail");

    assert_eq!(
        err,
        CadTopologyError::MissingEntityReference {
            owner_kind: CadEntityKind::Loop,
            owner_id: cad.loops[0].entity_id.id.clone(),
            reference_kind: CadEntityKind::Edge,
            reference_id: "cad_edge_missing".to_string(),
        }
    );
}

#[test]
fn rejects_cad_loop_listed_by_wrong_face() {
    let geometry = cube_geometry(false);
    let topology = extract_source_topology(&geometry).expect("topology should extract");
    let mut cad = build_cad_topology(&geometry, &topology).expect("cad topology should build");
    let wrong_loop_id = cad.faces[1].loop_ids[0].clone();
    cad.faces[0].loop_ids = vec![wrong_loop_id.clone()];

    let err = validate_cad_topology_model(&cad).expect_err("wrong face-loop owner should fail");

    assert_eq!(
        err,
        CadTopologyError::MissingFaceLoopReference {
            face_id: cad.loops[0].face_id.clone(),
            loop_id: cad.loops[0].entity_id.id.clone(),
        }
    );
}

#[test]
fn rejects_cad_loop_with_stale_owning_face_reference() {
    let geometry = cube_geometry(false);
    let topology = extract_source_topology(&geometry).expect("topology should extract");
    let mut cad = build_cad_topology(&geometry, &topology).expect("cad topology should build");
    let original_face_id = cad.loops[0].face_id.clone();
    cad.loops[0].face_id = cad.faces[1].entity_id.id.clone();

    let err = validate_cad_topology_model(&cad).expect_err("stale loop owner should fail");

    assert_eq!(
        err,
        CadTopologyError::MissingFaceLoopReference {
            face_id: cad.loops[0].face_id.clone(),
            loop_id: cad.loops[0].entity_id.id.clone(),
        }
    );
    assert_ne!(original_face_id, cad.loops[0].face_id);
}

#[test]
fn rejects_cad_loop_owned_by_a_different_face() {
    let geometry = cube_geometry(false);
    let topology = extract_source_topology(&geometry).expect("topology should extract");
    let mut cad = build_cad_topology(&geometry, &topology).expect("cad topology should build");
    let wrong_loop_id = cad.faces[1].loop_ids[0].clone();
    let wrong_loop_face_id = cad.loops[1].face_id.clone();
    cad.faces[0].loop_ids.push(wrong_loop_id.clone());

    let err = validate_cad_topology_model(&cad).expect_err("foreign loop should fail");

    assert_eq!(
        err,
        CadTopologyError::LoopFaceMismatch {
            loop_id: wrong_loop_id,
            expected_face_id: cad.faces[0].entity_id.id.clone(),
            actual_face_id: wrong_loop_face_id,
        }
    );
}

#[test]
fn rejects_cad_topology_report_count_mismatch() {
    let geometry = cube_geometry(false);
    let topology = extract_source_topology(&geometry).expect("topology should extract");
    let mut cad = build_cad_topology(&geometry, &topology).expect("cad topology should build");
    cad.report.loop_count += 1;

    let err = validate_cad_topology_model(&cad).expect_err("stale report count should fail");

    assert_eq!(
        err,
        CadTopologyError::ReportCountMismatch {
            field: "loop_count",
            expected: cad.loops.len(),
            actual: cad.report.loop_count,
        }
    );
}

#[test]
fn rejects_cad_topology_imported_face_count_mismatch() {
    let geometry = cube_geometry(true);
    let topology = extract_source_topology(&geometry).expect("topology should extract");
    let mut cad = build_cad_topology(&geometry, &topology).expect("cad topology should build");
    cad.report.imported_face_count += 1;

    let err = validate_cad_topology_model(&cad).expect_err("stale imported count should fail");

    assert_eq!(
        err,
        CadTopologyError::ReportCountMismatch {
            field: "imported_face_count",
            expected: 1,
            actual: 2,
        }
    );
}

#[test]
fn rejects_cad_topology_evaluator_face_count_mismatch() {
    let geometry = cube_geometry(true);
    let topology = extract_source_topology(&geometry).expect("topology should extract");
    let mut cad = build_cad_topology(&geometry, &topology).expect("cad topology should build");
    cad.report.evaluator_face_count = 0;

    let err = validate_cad_topology_model(&cad).expect_err("stale evaluator count should fail");

    assert_eq!(
        err,
        CadTopologyError::ReportCountMismatch {
            field: "evaluator_face_count",
            expected: 1,
            actual: 0,
        }
    );
}

#[test]
fn rejects_evaluator_metadata_without_imported_face_handle() {
    let geometry = cube_geometry(true);
    let topology = extract_source_topology(&geometry).expect("topology should extract");
    let mut cad = build_cad_topology(&geometry, &topology).expect("cad topology should build");
    let face = cad
        .faces
        .iter_mut()
        .find(|face| face.evaluator_id.is_some())
        .expect("evaluator-backed face should exist");
    let face_id = face.entity_id.id.clone();
    face.imported_face_id = None;

    let err = validate_cad_topology_model(&cad).expect_err("missing imported handle should fail");

    assert_eq!(
        err,
        CadTopologyError::EvaluatorMetadataWithoutImportedFace { face_id }
    );
}

#[test]
fn rejects_evaluator_capability_without_evaluator_id() {
    let geometry = cube_geometry(true);
    let topology = extract_source_topology(&geometry).expect("topology should extract");
    let mut cad = build_cad_topology(&geometry, &topology).expect("cad topology should build");
    let face = cad
        .faces
        .iter_mut()
        .find(|face| face.evaluator_id.is_some())
        .expect("evaluator-backed face should exist");
    let face_id = face.entity_id.id.clone();
    face.evaluator_id = None;

    let err = validate_cad_topology_model(&cad).expect_err("missing evaluator id should fail");

    assert_eq!(
        err,
        CadTopologyError::EvaluatorCapabilityWithoutEvaluator {
            face_id,
            capability: "point_evaluation",
        }
    );
}

#[test]
fn preserves_imported_cad_curve_handles_and_evaluator_capabilities() {
    let geometry = cube_geometry_with_curve_evaluator();
    let topology = extract_source_topology(&geometry).expect("topology should extract");

    let cad = build_cad_topology(&geometry, &topology).expect("cad topology should build");

    let curve_edge = cad
        .edges
        .iter()
        .find(|edge| edge.source_edge_id == 0)
        .expect("mapped curve edge should exist");
    assert_eq!(cad.report.imported_curve_count, 1);
    assert_eq!(cad.report.evaluator_curve_count, 1);
    assert_eq!(curve_edge.imported_curve_id, Some(4));
    assert_eq!(curve_edge.evaluator_id.as_deref(), Some("cad_curve_4"));
    assert!(curve_edge.evaluator_supports_point_evaluation);
    assert!(curve_edge.evaluator_supports_projection);
    assert!(curve_edge.evaluator_supports_tangent);
    assert!(curve_edge.evaluator_supports_curvature);
    assert_eq!(curve_edge.evaluator_samples.len(), 1);
    assert_eq!(curve_edge.evaluator_samples[0].parameter, 0.5);
    assert_eq!(curve_edge.evaluator_samples[0].point_m, [0.5, 0.1, 0.0]);
}

#[test]
fn rejects_cad_topology_imported_curve_count_mismatch() {
    let geometry = cube_geometry_with_curve_evaluator();
    let topology = extract_source_topology(&geometry).expect("topology should extract");
    let mut cad = build_cad_topology(&geometry, &topology).expect("cad topology should build");
    cad.report.imported_curve_count = 0;

    let err =
        validate_cad_topology_model(&cad).expect_err("stale imported curve count should fail");

    assert_eq!(
        err,
        CadTopologyError::ReportCountMismatch {
            field: "imported_curve_count",
            expected: 1,
            actual: 0,
        }
    );
}

#[test]
fn rejects_cad_topology_evaluator_curve_count_mismatch() {
    let geometry = cube_geometry_with_curve_evaluator();
    let topology = extract_source_topology(&geometry).expect("topology should extract");
    let mut cad = build_cad_topology(&geometry, &topology).expect("cad topology should build");
    cad.report.evaluator_curve_count = 0;

    let err =
        validate_cad_topology_model(&cad).expect_err("stale evaluator curve count should fail");

    assert_eq!(
        err,
        CadTopologyError::ReportCountMismatch {
            field: "evaluator_curve_count",
            expected: 1,
            actual: 0,
        }
    );
}

#[test]
fn rejects_curve_evaluator_metadata_without_imported_curve_handle() {
    let geometry = cube_geometry_with_curve_evaluator();
    let topology = extract_source_topology(&geometry).expect("topology should extract");
    let mut cad = build_cad_topology(&geometry, &topology).expect("cad topology should build");
    let edge = cad
        .edges
        .iter_mut()
        .find(|edge| edge.evaluator_id.is_some())
        .expect("evaluator-backed edge should exist");
    let edge_id = edge.entity_id.id.clone();
    edge.imported_curve_id = None;

    let err = validate_cad_topology_model(&cad).expect_err("missing curve handle should fail");

    assert_eq!(
        err,
        CadTopologyError::EvaluatorMetadataWithoutImportedCurve { edge_id }
    );
}

#[test]
fn rejects_curve_evaluator_capability_without_evaluator_id() {
    let geometry = cube_geometry_with_curve_evaluator();
    let topology = extract_source_topology(&geometry).expect("topology should extract");
    let mut cad = build_cad_topology(&geometry, &topology).expect("cad topology should build");
    let edge = cad
        .edges
        .iter_mut()
        .find(|edge| edge.evaluator_id.is_some())
        .expect("evaluator-backed edge should exist");
    let edge_id = edge.entity_id.id.clone();
    edge.evaluator_id = None;

    let err =
        validate_cad_topology_model(&cad).expect_err("missing curve evaluator id should fail");

    assert_eq!(
        err,
        CadTopologyError::CurveEvaluatorCapabilityWithoutEvaluator {
            edge_id,
            capability: "point_evaluation",
        }
    );
}

#[test]
fn rejects_curve_evaluator_sample_with_invalid_parameter() {
    let geometry = cube_geometry_with_curve_evaluator();
    let topology = extract_source_topology(&geometry).expect("topology should extract");
    let mut cad = build_cad_topology(&geometry, &topology).expect("cad topology should build");
    let edge = cad
        .edges
        .iter_mut()
        .find(|edge| !edge.evaluator_samples.is_empty())
        .expect("sample-backed edge should exist");
    let edge_id = edge.entity_id.id.clone();
    edge.evaluator_samples[0].parameter = 1.5;

    let err = validate_cad_topology_model(&cad).expect_err("invalid parameter should fail");

    assert_eq!(
        err,
        CadTopologyError::InvalidCurveEvaluatorSample {
            edge_id,
            sample_index: 0,
            reason: "parameter must be finite and in [0, 1]",
        }
    );
}

#[test]
fn rejects_curve_evaluator_sample_with_invalid_projection_error() {
    let geometry = cube_geometry_with_curve_evaluator();
    let topology = extract_source_topology(&geometry).expect("topology should extract");
    let mut cad = build_cad_topology(&geometry, &topology).expect("cad topology should build");
    let edge = cad
        .edges
        .iter_mut()
        .find(|edge| !edge.evaluator_samples.is_empty())
        .expect("sample-backed edge should exist");
    let edge_id = edge.entity_id.id.clone();
    edge.evaluator_samples[0].projection_error_m = Some(-1.0);

    let err = validate_cad_topology_model(&cad).expect_err("negative projection error should fail");

    assert_eq!(
        err,
        CadTopologyError::InvalidCurveEvaluatorSample {
            edge_id,
            sample_index: 0,
            reason: "projection error must be finite and non-negative",
        }
    );
}

fn cube_geometry_with_curve_evaluator() -> runmat_geometry_core::GeometryAsset {
    let mut geometry = cube_geometry(true);
    geometry.regions.push(Region {
        region_id: "curve_000004".to_string(),
        name: "Curve 4".to_string(),
        tag: Some("cad_curve".to_string()),
        cad_ownership: Some(CadRegionOwnership {
            face_id: None,
            curve_id: Some(4),
            label: Some(CadLabelRef {
                label_entry: "0:1:curve:4".to_string(),
                name: "Curve 4".to_string(),
                kind: CadSemanticKind::Subshape,
            }),
            owner_path: Vec::new(),
            layers: Vec::new(),
            color: None,
            material: None,
        }),
    });
    geometry
        .region_entity_mappings
        .push(RegionEntityMapping::new(
            "curve_000004",
            "cube_surface",
            EntityKind::Edge,
            vec![EntityIdRange::new(0, 1)],
        ));
    geometry.source_geometry.cad_evaluators[0]
        .curves
        .push(CadCurveEvaluator {
            evaluator_id: "cad_curve_4".to_string(),
            imported_curve_id: 4,
            name: "Curve 4".to_string(),
            supports_point_evaluation: true,
            supports_projection: true,
            supports_tangent: true,
            supports_curvature: true,
            evaluation_samples: vec![CadCurveEvaluationSample {
                source: CadCurveEvaluationSampleSource::BackendQuery,
                parameter: 0.5,
                point_m: [0.5, 0.1, 0.0],
                projected_point_m: Some([0.5, 0.12, 0.0]),
                tangent_m: Some([1.0, 0.0, 0.0]),
                curvature_1_per_m: Some(0.25),
                projection_error_m: Some(0.02),
            }],
        });
    geometry
}

fn cube_geometry(with_semantic_face: bool) -> runmat_geometry_core::GeometryAsset {
    let face_region = Region {
        region_id: "face_000001".to_string(),
        name: "face".to_string(),
        tag: Some("cad_face".to_string()),
        cad_ownership: with_semantic_face.then(|| CadRegionOwnership {
            face_id: Some(1),
            curve_id: None,
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
