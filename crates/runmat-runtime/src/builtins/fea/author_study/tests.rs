use futures::executor::block_on;
use runmat_value::{Tensor, Value};

use super::super::contracts::descriptors::ERROR_INTERNAL;
use super::super::contracts::identities::{
    FEA_PAYLOAD_JSON_PROPERTY, FEA_RUN_RESULT_CLASS, FEA_STUDY_CLASS, FEA_STUDY_SPEC_JSON_PROPERTY,
};
use super::super::entrypoints::{fea_run_builtin, fea_validate_builtin};
use super::super::integer_serialization::serializable_to_object;
use super::*;
use crate::builtins::geometry::{GEOMETRY_ASSET_CLASS, GEOMETRY_ASSET_JSON_PROPERTY};

fn authoring_summary_value() -> Value {
    crate::builtins::io::json::jsondecode::value_from_json(&serde_json::json!({
        "mesh_authoring_summary": {
            "schema_version": "mesh-authoring-summary/v1",
            "mesh_id": "mesh_authoring_fixture",
            "solve_ready": true,
            "backend": "solid",
            "tetrahedron_generation_family": "structured_box",
            "tetrahedron_generation_attempted_family_count": 2,
            "tetrahedron_generation_rejected_family_count": 1,
            "tetrahedron_generation_selected_family_index": 2,
            "tetrahedron_generation_interior_support_candidate_count": 17,
            "tetrahedron_generation_interior_support_accepted_count": 1,
            "topology": {
                "node_count": 4,
                "volume_element_count": 1,
                "boundary_face_count": 2,
                "boundary_edge_count": 3,
                "adaptive_iteration_count": 0
            },
            "quality": {
                "meets_quality_thresholds": true,
                "min_scaled_jacobian": 0.5,
                "min_exact_scaled_jacobian": 0.45,
                "max_aspect_ratio": 2.0,
                "max_boundary_projection_error_m": 0.0,
                "inverted_element_count": 0,
                "sliver_count": 0,
                "sliver_removed_count": 0,
                "unrepaired_exact_quality_count": 0
            },
            "recovery": {
                "boundary_face_recovery_ratio": 1.0,
                "boundary_edge_recovery_ratio": 1.0,
                "recovery_item_count": 2,
                "recovered_item_count": 2,
                "missing_recovery_item_count": 0,
                "unrecovered_tetrahedron_component_count": 0
            },
            "regions": {
                "material_regions": [
                    {
                        "region_id": "solid",
                        "element_count": 1,
                        "volume_m3": 0.16666666666666666,
                        "required": true
                    }
                ],
                "boundary_regions": [
                    {
                        "region_id": "root",
                        "face_count": 1,
                        "recovered_face_count": 1,
                        "edge_count": 3,
                        "fully_recovered": true,
                        "required": true
                    },
                    {
                        "region_id": "tip",
                        "face_count": 1,
                        "recovered_face_count": 1,
                        "edge_count": 3,
                        "fully_recovered": true,
                        "required": true
                    }
                ],
                "required_material_region_ids": ["solid"],
                "required_boundary_region_ids": ["root", "tip"]
            }
        }
    }))
    .expect("authoring summary value should convert")
}

fn generic_authoring_geometry_value() -> Value {
    use runmat_geometry_core::{
        EntityIdRange, EntityKind, GeometryAsset, GeometrySource, MeshDescriptor, MeshKind, Region,
        RegionEntityMapping, SourceGeometry, SourceGeometryKind, SurfaceMesh, TessellationProfile,
        UnitSystem,
    };

    let asset = GeometryAsset {
        geometry_id: "geo:authoring_fixture".to_string(),
        source: GeometrySource {
            path: "/fixtures/authoring.step".to_string(),
            sha256: "hash-authoring".to_string(),
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
            mesh_id: "surface".to_string(),
            kind: MeshKind::Surface,
            vertex_count: 4,
            element_count: 2,
        }],
        surface_meshes: vec![SurfaceMesh::new(
            "surface",
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            vec![[0, 1, 2], [0, 1, 3]],
        )],
        regions: vec![
            Region {
                region_id: "root".to_string(),
                name: "root".into(),
                tag: Some("fixed".to_string()),
                cad_ownership: None,
            },
            Region {
                region_id: "tip".to_string(),
                name: "tip".into(),
                tag: Some("load".to_string()),
                cad_ownership: None,
            },
            Region {
                region_id: "solid".to_string(),
                name: "solid".into(),
                tag: Some("material".to_string()),
                cad_ownership: None,
            },
        ],
        region_entity_mappings: vec![
            RegionEntityMapping::new(
                "root",
                "surface",
                EntityKind::Face,
                vec![EntityIdRange::new(0, 1)],
            ),
            RegionEntityMapping::new(
                "tip",
                "surface",
                EntityKind::Face,
                vec![EntityIdRange::new(1, 1)],
            ),
            RegionEntityMapping::all_faces("solid", "surface", 2),
        ],
        diagnostics: Vec::new(),
    };

    serializable_to_object(
        AUTHOR_STUDY_NAME,
        &ERROR_INTERNAL,
        GEOMETRY_ASSET_CLASS,
        &asset,
        Some(GEOMETRY_ASSET_JSON_PROPERTY),
    )
    .expect("geometry asset should convert")
}

fn diagram_observation_value() -> Value {
    crate::builtins::io::json::jsondecode::value_from_json(&serde_json::json!({
        "artifact_path": "diagram://fixture/free-body.png",
        "source_mime_type": "image/png",
        "summary": "boundary condition on tip and driving condition on root",
        "material_region_id": "solid",
        "boundary_condition_region_id": "tip",
        "driving_condition_region_id": "root",
        "structural_force_n": [12.0, -3.0, 4.0],
        "confidence": 0.82
    }))
    .expect("diagram observation should convert")
}

fn authoring_analysis_mesh_artifacts(dir: &std::path::Path) -> (String, String, Value) {
    use runmat_meshing_core::{
        contracts::{
            artifact::ANALYSIS_MESH_SCHEMA_VERSION, AnalysisBoundaryEdge, AnalysisBoundaryFace,
            AnalysisMeshArtifact, AnalysisMeshNode, AnalysisMeshProvenance, AnalysisVolumeElement,
            BoundaryElementKind, MeshBackendSummary, VolumeElementKind,
        },
        quality::{AnalysisMeshQualityReport, ElementQuality},
        AnalysisMeshValidationOptions, MeshSizingField,
    };

    let mut mesh = AnalysisMeshArtifact {
        schema_version: ANALYSIS_MESH_SCHEMA_VERSION.to_string(),
        mesh_id: "mesh_authoring_fixture".to_string(),
        nodes: vec![
            AnalysisMeshNode {
                node_id: 1,
                coordinates_m: [0.0, 0.0, 0.0],
                provenance: Vec::new(),
            },
            AnalysisMeshNode {
                node_id: 2,
                coordinates_m: [1.0, 0.0, 0.0],
                provenance: Vec::new(),
            },
            AnalysisMeshNode {
                node_id: 3,
                coordinates_m: [0.0, 1.0, 0.0],
                provenance: Vec::new(),
            },
            AnalysisMeshNode {
                node_id: 4,
                coordinates_m: [0.0, 0.0, 1.0],
                provenance: Vec::new(),
            },
        ],
        volume_elements: vec![AnalysisVolumeElement {
            element_id: "tetrahedron_1".to_string(),
            kind: VolumeElementKind::Tetrahedron4,
            node_ids: vec![1, 2, 3, 4],
            material_region_id: "solid".to_string(),
            provenance: Vec::new(),
        }],
        boundary_faces: vec![
            AnalysisBoundaryFace {
                face_id: "face_root".to_string(),
                kind: BoundaryElementKind::Tri3,
                node_ids: vec![1, 2, 3],
                adjacent_volume_element_ids: vec!["tetrahedron_1".to_string()],
                region_ids: vec!["root".to_string()],
                provenance: Vec::new(),
            },
            AnalysisBoundaryFace {
                face_id: "face_tip".to_string(),
                kind: BoundaryElementKind::Tri3,
                node_ids: vec![1, 2, 4],
                adjacent_volume_element_ids: vec!["tetrahedron_1".to_string()],
                region_ids: vec!["tip".to_string()],
                provenance: Vec::new(),
            },
            AnalysisBoundaryFace {
                face_id: "face_side_a".to_string(),
                kind: BoundaryElementKind::Tri3,
                node_ids: vec![1, 3, 4],
                adjacent_volume_element_ids: vec!["tetrahedron_1".to_string()],
                region_ids: Vec::new(),
                provenance: Vec::new(),
            },
            AnalysisBoundaryFace {
                face_id: "face_side_b".to_string(),
                kind: BoundaryElementKind::Tri3,
                node_ids: vec![2, 3, 4],
                adjacent_volume_element_ids: vec!["tetrahedron_1".to_string()],
                region_ids: Vec::new(),
                provenance: Vec::new(),
            },
        ],
        boundary_edges: vec![
            AnalysisBoundaryEdge {
                edge_id: "edge_1_2".to_string(),
                node_ids: [1, 2],
                adjacent_boundary_face_ids: vec!["face_root".to_string(), "face_tip".to_string()],
                region_ids: vec!["root".to_string(), "tip".to_string()],
                provenance: Vec::new(),
            },
            AnalysisBoundaryEdge {
                edge_id: "edge_1_3".to_string(),
                node_ids: [1, 3],
                adjacent_boundary_face_ids: vec![
                    "face_root".to_string(),
                    "face_side_a".to_string(),
                ],
                region_ids: vec!["root".to_string()],
                provenance: Vec::new(),
            },
            AnalysisBoundaryEdge {
                edge_id: "edge_2_3".to_string(),
                node_ids: [2, 3],
                adjacent_boundary_face_ids: vec![
                    "face_root".to_string(),
                    "face_side_b".to_string(),
                ],
                region_ids: vec!["root".to_string()],
                provenance: Vec::new(),
            },
            AnalysisBoundaryEdge {
                edge_id: "edge_1_4".to_string(),
                node_ids: [1, 4],
                adjacent_boundary_face_ids: vec!["face_tip".to_string(), "face_side_a".to_string()],
                region_ids: vec!["tip".to_string()],
                provenance: Vec::new(),
            },
            AnalysisBoundaryEdge {
                edge_id: "edge_2_4".to_string(),
                node_ids: [2, 4],
                adjacent_boundary_face_ids: vec!["face_tip".to_string(), "face_side_b".to_string()],
                region_ids: vec!["tip".to_string()],
                provenance: Vec::new(),
            },
            AnalysisBoundaryEdge {
                edge_id: "edge_3_4".to_string(),
                node_ids: [3, 4],
                adjacent_boundary_face_ids: vec![
                    "face_side_a".to_string(),
                    "face_side_b".to_string(),
                ],
                region_ids: Vec::new(),
                provenance: Vec::new(),
            },
        ],
        quality: AnalysisMeshQualityReport {
            min_scaled_jacobian: 0.5,
            min_exact_scaled_jacobian: 0.45,
            mean_aspect_ratio: 2.0,
            max_aspect_ratio: 2.0,
            inverted_element_count: 0,
            mean_boundary_projection_error_m: 0.0,
            max_boundary_projection_error_m: 0.0,
            elements: vec![ElementQuality {
                element_id: "tetrahedron_1".to_string(),
                scaled_jacobian: 0.5,
                exact_scaled_jacobian: 0.45,
                aspect_ratio: 2.0,
                volume_m3: 1.0 / 6.0,
            }],
        },
        sizing: MeshSizingField::default(),
        field_topology: Vec::new(),
        backend: MeshBackendSummary {
            backend: "artifact_fixture".to_string(),
            algorithm: "artifact_fixture".to_string(),
            tetrahedron_generation_family: "artifact_fixture".to_string(),
            tetrahedron_element_count: 1,
            tetrahedron_material_region_count: 1,
            tetrahedron_recovered_component_ratio: 1.0,
            tetrahedron_recovered_boundary_face_count: 4,
            ..MeshBackendSummary::default()
        },
        adaptive_iterations: Vec::new(),
        provenance: AnalysisMeshProvenance {
            algorithm: "artifact_fixture".to_string(),
            source_geometry_id: "geo:authoring_fixture".to_string(),
            source_geometry_revision: 1,
            source_geometry_sha256: Some("hash-authoring".to_string()),
        },
    };
    mesh.refresh_field_topology();

    let validation = AnalysisMeshValidationOptions {
        required_boundary_region_ids: vec!["root".to_string(), "tip".to_string()],
        required_material_region_ids: vec!["solid".to_string()],
        ..AnalysisMeshValidationOptions::default()
    };
    runmat_meshing_core::validate_analysis_mesh_with_options(&mesh, validation.clone())
        .expect("artifact-backed authoring mesh should validate");
    let evidence = runmat_meshing_evidence::build_mesh_evidence_artifact(&mesh, &validation);
    let summary = runmat_meshing_evidence::build_mesh_authoring_summary(&evidence);

    let evidence_path = dir.join("mesh_evidence.json");
    let mesh_path = dir.join("analysis_mesh.json");
    std::fs::write(
        &evidence_path,
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema_version": "fea_study_mesh_evidence_artifact/v1",
            "mesh_validation_options": validation,
            "mesh_authoring_summary": summary,
            "mesh_evidence": evidence,
        }))
        .expect("evidence payload should encode"),
    )
    .expect("evidence artifact should write");
    std::fs::write(
        &mesh_path,
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema_version": "fea_study_analysis_mesh_artifact/v1",
            "mesh_evidence_artifact_path": evidence_path.to_string_lossy(),
            "mesh_validation_options": validation,
            "mesh": mesh,
        }))
        .expect("mesh payload should encode"),
    )
    .expect("mesh artifact should write");

    let summary_value =
        crate::builtins::io::json::jsondecode::value_from_json(&serde_json::json!({
            "mesh_authoring_summary": summary,
        }))
        .expect("summary value should convert");
    (
        mesh_path.to_string_lossy().to_string(),
        evidence_path.to_string_lossy().to_string(),
        summary_value,
    )
}

#[test]
fn builds_study_from_mesh_authoring_summary() {
    let study = create_author_study_object_from_args(vec![
        Value::String("authored_static".to_string()),
        generic_authoring_geometry_value(),
        authoring_summary_value(),
        Value::String("Profile".to_string()),
        Value::String("linear_static_structural".to_string()),
        Value::String("StructuralForceN".to_string()),
        Value::Tensor(
            Tensor::new_2d(vec![25.0, -50.0, 0.0], 1, 3).expect("force tensor should build"),
        ),
    ])
    .expect("authoring should produce a study");

    let Value::Object(study_object) = study.clone() else {
        panic!("expected authored study object");
    };
    assert!(study_object.class_name.is(FEA_STUDY_CLASS));
    let Some(Value::String(payload)) = study_object.properties.get(FEA_STUDY_SPEC_JSON_PROPERTY)
    else {
        panic!("expected study JSON payload");
    };
    let decoded: crate::analysis::AnalysisStudySpec =
        serde_json::from_str(payload).expect("authored study should decode");
    let model = decoded.model.expect("authored study should include model");
    assert_eq!(model.material_assignments[0].region_id, "solid");
    assert_eq!(model.boundary_conditions[0].region_id, "root");
    assert_eq!(model.loads[0].region_id, "tip");

    let validation = block_on(fea_validate_builtin(study)).expect("authored study should validate");
    let Value::Object(validation_object) = validation else {
        panic!("expected validation object");
    };
    assert_eq!(
        validation_object.properties.get("valid"),
        Some(&Value::Bool(true))
    );
}

#[test]
fn runs_with_analysis_mesh_artifact() {
    let tmp = tempfile::tempdir().expect("tempdir should be created");
    let (mesh_path, evidence_path, summary) = authoring_analysis_mesh_artifacts(tmp.path());
    let study = create_author_study_object_from_args(vec![
        Value::String("authored_run_static".to_string()),
        generic_authoring_geometry_value(),
        summary,
        Value::String("Profile".to_string()),
        Value::String("linear_static_structural".to_string()),
        Value::String("AnalysisMeshArtifactPath".to_string()),
        Value::String(mesh_path.clone()),
        Value::String("AnalysisMeshEvidenceArtifactPath".to_string()),
        Value::String(evidence_path.clone()),
        Value::String("StructuralForceN".to_string()),
        Value::Tensor(
            Tensor::new_2d(vec![10.0, 0.0, -5.0], 1, 3).expect("force tensor should build"),
        ),
    ])
    .expect("authoring should produce a study");

    let Value::Object(study_object) = study.clone() else {
        panic!("expected authored study object");
    };
    let Some(Value::String(study_payload)) =
        study_object.properties.get(FEA_STUDY_SPEC_JSON_PROPERTY)
    else {
        panic!("expected study payload");
    };
    let decoded_study: crate::analysis::AnalysisStudySpec =
        serde_json::from_str(study_payload).expect("authored study should decode");
    assert_eq!(
        decoded_study.analysis_mesh_artifact_path.as_deref(),
        Some(mesh_path.as_str())
    );

    let run = block_on(fea_run_builtin(study)).expect("authored study should run");
    let Value::Object(run_object) = run else {
        panic!("expected run result object");
    };
    assert!(run_object.class_name.is(FEA_RUN_RESULT_CLASS));
    let Some(Value::String(run_payload)) = run_object.properties.get(FEA_PAYLOAD_JSON_PROPERTY)
    else {
        panic!("expected run result payload");
    };
    let run_data: crate::analysis::AnalysisStudyRunData =
        serde_json::from_str(run_payload).expect("run result should decode");
    assert_eq!(run_data.run_kind, AnalysisRunKind::LinearStatic);
    assert_eq!(run_data.run_status, crate::analysis::RunStatus::Publishable);
    assert!(run_data.publishable);
    assert_eq!(run_data.quality_reasons.len(), 0);
    assert_eq!(
        run_data.analysis_mesh_artifact_path.as_deref(),
        Some(mesh_path.as_str())
    );
    assert_eq!(
        run_data.analysis_mesh_evidence_artifact_path.as_deref(),
        Some(evidence_path.as_str())
    );
}

#[test]
fn builds_study_from_diagram_observation() {
    let study = create_author_study_object_from_args(vec![
        Value::String("authored_diagram_static".to_string()),
        generic_authoring_geometry_value(),
        authoring_summary_value(),
        Value::String("Profile".to_string()),
        Value::String("linear_static_structural".to_string()),
        Value::String("DiagramObservation".to_string()),
        diagram_observation_value(),
    ])
    .expect("diagram observation should author a study");

    let Value::Object(study_object) = study else {
        panic!("expected authored study object");
    };
    let Some(Value::String(payload)) = study_object.properties.get(FEA_STUDY_SPEC_JSON_PROPERTY)
    else {
        panic!("expected study JSON payload");
    };
    let decoded: crate::analysis::AnalysisStudySpec =
        serde_json::from_str(payload).expect("authored study should decode");
    let model = decoded.model.expect("authored study should include model");
    assert_eq!(model.material_assignments[0].region_id, "solid");
    assert_eq!(model.boundary_conditions[0].region_id, "tip");
    assert_eq!(model.loads[0].region_id, "root");
    let runmat_analysis_core::LoadKind::Force { fx, fy, fz } = model.loads[0].kind else {
        panic!("diagram-authored study should use a force load");
    };
    assert_eq!([fx, fy, fz], [12.0, -3.0, 4.0]);
}

#[test]
fn runs_generic_study_from_minimal_authoring_inputs() {
    let tmp = tempfile::tempdir().expect("tempdir should be created");
    let (mesh_path, evidence_path, summary) = authoring_analysis_mesh_artifacts(tmp.path());
    let study = create_author_study_object_from_args(vec![
        Value::String("authored_minimal_static".to_string()),
        generic_authoring_geometry_value(),
        summary,
        Value::String("Profile".to_string()),
        Value::String("linear_static_structural".to_string()),
        Value::String("AnalysisMeshArtifactPath".to_string()),
        Value::String(mesh_path.clone()),
        Value::String("AnalysisMeshEvidenceArtifactPath".to_string()),
        Value::String(evidence_path.clone()),
    ])
    .expect("minimal authoring inputs should produce a runnable generic study");

    let Value::Object(study_object) = study.clone() else {
        panic!("expected authored study object");
    };
    let Some(Value::String(study_payload)) =
        study_object.properties.get(FEA_STUDY_SPEC_JSON_PROPERTY)
    else {
        panic!("expected study payload");
    };
    let decoded_study: crate::analysis::AnalysisStudySpec =
        serde_json::from_str(study_payload).expect("authored study should decode");
    let model = decoded_study
        .model
        .as_ref()
        .expect("minimal authored study should include a model");
    assert_eq!(model.material_assignments[0].region_id, "solid");
    assert_eq!(model.boundary_conditions[0].region_id, "root");
    assert_eq!(model.loads[0].region_id, "tip");
    let runmat_analysis_core::LoadKind::Force { fx, fy, fz } = model.loads[0].kind else {
        panic!("minimal authored study should default to a force load");
    };
    assert_eq!([fx, fy, fz], [0.0, -1000.0, 0.0]);
    assert_eq!(
        decoded_study.analysis_mesh_artifact_path.as_deref(),
        Some(mesh_path.as_str())
    );
    assert_eq!(
        decoded_study
            .analysis_mesh_evidence_artifact_path
            .as_deref(),
        Some(evidence_path.as_str())
    );

    let run = block_on(fea_run_builtin(study)).expect("minimal authored study should run");
    let Value::Object(run_object) = run else {
        panic!("expected run result object");
    };
    assert!(run_object.class_name.is(FEA_RUN_RESULT_CLASS));
    let Some(Value::String(run_payload)) = run_object.properties.get(FEA_PAYLOAD_JSON_PROPERTY)
    else {
        panic!("expected run result payload");
    };
    let run_data: crate::analysis::AnalysisStudyRunData =
        serde_json::from_str(run_payload).expect("run result should decode");
    assert_eq!(run_data.run_kind, AnalysisRunKind::LinearStatic);
    assert_eq!(run_data.run_status, crate::analysis::RunStatus::Publishable);
    assert!(run_data.publishable);
    assert_eq!(run_data.quality_reasons.len(), 0);
    assert_eq!(
        run_data.analysis_mesh_artifact_path.as_deref(),
        Some(mesh_path.as_str())
    );
    assert_eq!(
        run_data.analysis_mesh_evidence_artifact_path.as_deref(),
        Some(evidence_path.as_str())
    );
}

#[test]
fn requires_profile() {
    let err = create_author_study_object_from_args(vec![
        Value::String("missing_profile".to_string()),
        generic_authoring_geometry_value(),
        authoring_summary_value(),
    ])
    .expect_err("missing profile should fail");
    assert_eq!(err.identifier(), Some("RunMat:fea:InvalidInput"));
    assert!(err.message().contains("fea.authorStudy requires Profile"));
}

#[test]
fn requires_geometry_asset() {
    let err = create_author_study_object_from_args(vec![
        Value::String("bad".to_string()),
        Value::Num(1.0),
        authoring_summary_value(),
    ])
    .expect_err("invalid geometry should fail");
    assert_eq!(err.identifier(), Some("RunMat:fea:InvalidInput"));
    assert!(err.message().contains("fea.authorStudy geometry"));
}
