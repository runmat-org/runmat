use runmat_analysis_fea::ComputeBackend;
use runmat_builtins::Value;

use crate::analysis::{
    analysis_author_study_op, AnalysisCreateModelProfile, AnalysisRunKind,
    AnalysisStudyAuthoringIntent,
};
use crate::operations::OperationContext;
use crate::BuiltinResult;

use super::{
    builtin_error, builtin_error_with_source, geometry_asset_from_value_with_builtin,
    json_deserialize, operation_error, option_key, parse_scalar_enum, scalar_string,
    study_to_object, value_to_json, AUTHOR_STUDY_NAME, ERROR_INPUT, ERROR_OPERATION,
};

pub(super) fn create_author_study_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() < 3 {
        return Err(builtin_error(
            AUTHOR_STUDY_NAME,
            &ERROR_INPUT,
            "fea.authorStudy requires id, geometry, and mesh authoring summary arguments",
        ));
    }
    if !(args.len() - 3).is_multiple_of(2) {
        return Err(builtin_error(
            AUTHOR_STUDY_NAME,
            &ERROR_INPUT,
            "fea.authorStudy options must be Name, Value pairs",
        ));
    }

    let study_id = scalar_string(&args[0], AUTHOR_STUDY_NAME, &ERROR_INPUT)?;
    let geometry = geometry_asset_from_value_with_builtin(&args[1], AUTHOR_STUDY_NAME)?;
    let mesh_authoring_summary = mesh_authoring_summary_from_value(&args[2])?;
    let mut profile = AnalysisCreateModelProfile::LinearStaticStructural;
    let mut run_kind = AnalysisRunKind::LinearStatic;
    let mut backend = ComputeBackend::Cpu;
    let mut model_id = None::<String>;
    let mut material_region_id = None::<String>;
    let mut fixed_boundary_region_id = None::<String>;
    let mut load_boundary_region_id = None::<String>;
    let mut force_n = None::<[f64; 3]>;
    let mut analysis_mesh_artifact_path = None::<String>;
    let mut analysis_mesh_evidence_artifact_path = None::<String>;

    for pair in args[3..].chunks(2) {
        let key = option_key(&pair[0], AUTHOR_STUDY_NAME)?;
        match key.as_str() {
            "profile" => {
                let text = scalar_string(&pair[1], AUTHOR_STUDY_NAME, &ERROR_INPUT)?;
                profile = parse_scalar_enum(&text, "Profile")?;
            }
            "runkind" | "kind" => {
                let text = scalar_string(&pair[1], AUTHOR_STUDY_NAME, &ERROR_INPUT)?;
                run_kind = parse_scalar_enum(&text, "RunKind")?;
            }
            "backend" => {
                let text = scalar_string(&pair[1], AUTHOR_STUDY_NAME, &ERROR_INPUT)?;
                backend = parse_scalar_enum(&text, "Backend")?;
            }
            "modelid" => {
                model_id = Some(scalar_string(&pair[1], AUTHOR_STUDY_NAME, &ERROR_INPUT)?);
            }
            "materialregion" | "materialregionid" => {
                material_region_id =
                    Some(scalar_string(&pair[1], AUTHOR_STUDY_NAME, &ERROR_INPUT)?);
            }
            "fixedregion" | "fixedregionid" | "fixedboundaryregion" | "fixedboundaryregionid" => {
                fixed_boundary_region_id =
                    Some(scalar_string(&pair[1], AUTHOR_STUDY_NAME, &ERROR_INPUT)?);
            }
            "loadregion" | "loadregionid" | "loadboundaryregion" | "loadboundaryregionid" => {
                load_boundary_region_id =
                    Some(scalar_string(&pair[1], AUTHOR_STUDY_NAME, &ERROR_INPUT)?);
            }
            "forcen" | "force" | "forcevector" => {
                force_n = Some(vector3_from_value(AUTHOR_STUDY_NAME, &pair[1], "ForceN")?);
            }
            "analysismeshartifactpath" | "meshartifactpath" => {
                analysis_mesh_artifact_path =
                    Some(scalar_string(&pair[1], AUTHOR_STUDY_NAME, &ERROR_INPUT)?);
            }
            "analysismeshevidenceartifactpath" | "meshevidenceartifactpath" => {
                analysis_mesh_evidence_artifact_path =
                    Some(scalar_string(&pair[1], AUTHOR_STUDY_NAME, &ERROR_INPUT)?);
            }
            other => {
                return Err(builtin_error(
                    AUTHOR_STUDY_NAME,
                    &ERROR_INPUT,
                    format!("unsupported fea.authorStudy option `{other}`"),
                ));
            }
        }
    }

    let authored = analysis_author_study_op(
        AnalysisStudyAuthoringIntent {
            study_id,
            model_id,
            geometry,
            mesh_authoring_summary,
            profile,
            run_kind,
            backend,
            analysis_mesh_artifact_path,
            analysis_mesh_evidence_artifact_path,
            material_region_id,
            fixed_boundary_region_id,
            load_boundary_region_id,
            force_n,
        },
        OperationContext::new(None, None),
    )
    .map_err(|err| operation_error(AUTHOR_STUDY_NAME, &ERROR_OPERATION, err))?;

    study_to_object(authored.data.study)
}

fn mesh_authoring_summary_from_value(
    value: &Value,
) -> BuiltinResult<runmat_meshing_evidence::MeshAuthoringSummary> {
    if let Ok(text) = scalar_string(value, AUTHOR_STUDY_NAME, &ERROR_INPUT) {
        let json: serde_json::Value = serde_json::from_str(&text).map_err(|err| {
            builtin_error_with_source(AUTHOR_STUDY_NAME, &ERROR_INPUT, err.to_string(), err)
        })?;
        return mesh_authoring_summary_from_json(json);
    }

    let json = value_to_json(AUTHOR_STUDY_NAME, value)?;
    mesh_authoring_summary_from_json(json)
}

fn mesh_authoring_summary_from_json(
    json: serde_json::Value,
) -> BuiltinResult<runmat_meshing_evidence::MeshAuthoringSummary> {
    let mut summary_json = json.get("mesh_authoring_summary").cloned().unwrap_or(json);
    normalize_mesh_authoring_summary_json(&mut summary_json);
    json_deserialize(AUTHOR_STUDY_NAME, summary_json, "mesh authoring summary")
}

fn normalize_mesh_authoring_summary_json(value: &mut serde_json::Value) {
    normalize_integral_json_numbers(value);
    wrap_scalar_string_arrays(value);
}

fn normalize_integral_json_numbers(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Array(values) => {
            for value in values {
                normalize_integral_json_numbers(value);
            }
        }
        serde_json::Value::Object(fields) => {
            for value in fields.values_mut() {
                normalize_integral_json_numbers(value);
            }
        }
        serde_json::Value::Number(number) => {
            let Some(value) = number.as_f64() else {
                return;
            };
            if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
                return;
            }
            if value > u64::MAX as f64 {
                return;
            }
            *number = serde_json::Number::from(value as u64);
        }
        _ => {}
    }
}

fn wrap_scalar_string_arrays(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Array(values) => {
            for value in values {
                wrap_scalar_string_arrays(value);
            }
        }
        serde_json::Value::Object(fields) => {
            for key in [
                "required_material_region_ids",
                "missing_required_material_region_ids",
                "required_boundary_region_ids",
                "missing_required_boundary_region_ids",
            ] {
                if let Some(serde_json::Value::String(region_id)) = fields.get(key).cloned() {
                    fields.insert(
                        key.to_string(),
                        serde_json::Value::Array(vec![serde_json::Value::String(region_id)]),
                    );
                }
            }
            for value in fields.values_mut() {
                wrap_scalar_string_arrays(value);
            }
        }
        _ => {}
    }
}

fn vector3_from_value(builtin: &'static str, value: &Value, key: &str) -> BuiltinResult<[f64; 3]> {
    let values: Vec<f64> =
        serde_json::from_value(value_to_json(builtin, value)?).map_err(|err| {
            builtin_error(
                builtin,
                &ERROR_INPUT,
                format!("invalid vector option `{key}`: {err}"),
            )
        })?;
    if values.len() != 3 {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("vector option `{key}` must contain exactly 3 values"),
        ));
    }
    Ok([values[0], values[1], values[2]])
}

#[cfg(test)]
mod tests {
    use futures::executor::block_on;
    use runmat_builtins::{Tensor, Value};

    use super::super::{
        fea_run_builtin, fea_validate_builtin, serializable_to_object, ERROR_INTERNAL,
        FEA_PAYLOAD_JSON_PROPERTY, FEA_RUN_RESULT_CLASS, FEA_STUDY_CLASS,
        FEA_STUDY_SPEC_JSON_PROPERTY, GEOMETRY_ASSET_CLASS, GEOMETRY_ASSET_JSON_PROPERTY,
    };
    use super::*;

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
            EntityIdRange, EntityKind, GeometryAsset, GeometrySource, MeshDescriptor, MeshKind,
            Region, RegionEntityMapping, SourceGeometry, SourceGeometryKind, SurfaceMesh,
            TessellationProfile, UnitSystem,
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
                    name: "root".to_string(),
                    tag: Some("fixed".to_string()),
                    cad_ownership: None,
                },
                Region {
                    region_id: "tip".to_string(),
                    name: "tip".to_string(),
                    tag: Some("load".to_string()),
                    cad_ownership: None,
                },
                Region {
                    region_id: "solid".to_string(),
                    name: "solid".to_string(),
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

    fn authoring_analysis_mesh_artifacts(dir: &std::path::Path) -> (String, String, Value) {
        use runmat_meshing_core::{
            contracts::{
                artifact::ANALYSIS_MESH_SCHEMA_VERSION, AnalysisBoundaryEdge, AnalysisBoundaryFace,
                AnalysisMeshArtifact, AnalysisMeshNode, AnalysisMeshProvenance,
                AnalysisVolumeElement, BoundaryElementKind, MeshBackendSummary, VolumeElementKind,
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
                    adjacent_boundary_face_ids: vec![
                        "face_root".to_string(),
                        "face_tip".to_string(),
                    ],
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
                    adjacent_boundary_face_ids: vec![
                        "face_tip".to_string(),
                        "face_side_a".to_string(),
                    ],
                    region_ids: vec!["tip".to_string()],
                    provenance: Vec::new(),
                },
                AnalysisBoundaryEdge {
                    edge_id: "edge_2_4".to_string(),
                    node_ids: [2, 4],
                    adjacent_boundary_face_ids: vec![
                        "face_tip".to_string(),
                        "face_side_b".to_string(),
                    ],
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
            Value::String("ForceN".to_string()),
            Value::Tensor(
                Tensor::new_2d(vec![25.0, -50.0, 0.0], 1, 3).expect("force tensor should build"),
            ),
        ])
        .expect("authoring should produce a study");

        let Value::Object(study_object) = study.clone() else {
            panic!("expected authored study object");
        };
        assert_eq!(study_object.class_name, FEA_STUDY_CLASS);
        let Some(Value::String(payload)) =
            study_object.properties.get(FEA_STUDY_SPEC_JSON_PROPERTY)
        else {
            panic!("expected study JSON payload");
        };
        let decoded: crate::analysis::AnalysisStudySpec =
            serde_json::from_str(payload).expect("authored study should decode");
        let model = decoded.model.expect("authored study should include model");
        assert_eq!(model.material_assignments[0].region_id, "solid");
        assert_eq!(model.boundary_conditions[0].region_id, "root");
        assert_eq!(model.loads[0].region_id, "tip");

        let validation =
            block_on(fea_validate_builtin(study)).expect("authored study should validate");
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
            Value::String("AnalysisMeshArtifactPath".to_string()),
            Value::String(mesh_path.clone()),
            Value::String("AnalysisMeshEvidenceArtifactPath".to_string()),
            Value::String(evidence_path.clone()),
            Value::String("ForceN".to_string()),
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
        assert_eq!(run_object.class_name, FEA_RUN_RESULT_CLASS);
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
}
