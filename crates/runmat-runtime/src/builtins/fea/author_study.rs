use runmat_analysis_fea::ComputeBackend;
use runmat_value::Value;

use crate::analysis::{
    analysis_author_study_op, AnalysisCreateModelProfile, AnalysisRunKind,
    AnalysisStudyAuthoringIntent, AnalysisStudyDiagramObservation,
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
    let mut profile = None::<AnalysisCreateModelProfile>;
    let mut run_kind = None::<AnalysisRunKind>;
    let mut backend = ComputeBackend::Cpu;
    let mut model_id = None::<String>;
    let mut material_region_id = None::<String>;
    let mut boundary_condition_region_id = None::<String>;
    let mut driving_condition_region_id = None::<String>;
    let mut structural_force_n = None::<[f64; 3]>;
    let mut solver_mesh_artifact_path = None::<String>;
    let mut meshing_evidence_artifact_path = None::<String>;
    let mut diagram_observation = None::<AnalysisStudyDiagramObservation>;

    for pair in args[3..].chunks(2) {
        let key = option_key(&pair[0], AUTHOR_STUDY_NAME)?;
        match key.as_str() {
            "profile" => {
                let text = scalar_string(&pair[1], AUTHOR_STUDY_NAME, &ERROR_INPUT)?;
                profile = Some(parse_scalar_enum(&text, "Profile")?);
            }
            "runkind" | "kind" => {
                let text = scalar_string(&pair[1], AUTHOR_STUDY_NAME, &ERROR_INPUT)?;
                run_kind = Some(parse_scalar_enum(&text, "RunKind")?);
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
            "boundaryconditionregion"
            | "boundaryconditionregionid"
            | "boundaryregion"
            | "boundaryregionid" => {
                boundary_condition_region_id =
                    Some(scalar_string(&pair[1], AUTHOR_STUDY_NAME, &ERROR_INPUT)?);
            }
            "drivingconditionregion"
            | "drivingconditionregionid"
            | "driverregion"
            | "driverregionid" => {
                driving_condition_region_id =
                    Some(scalar_string(&pair[1], AUTHOR_STUDY_NAME, &ERROR_INPUT)?);
            }
            "structuralforcen" | "structuralforce" | "structuralforcevector" => {
                structural_force_n = Some(vector3_from_value(
                    AUTHOR_STUDY_NAME,
                    &pair[1],
                    "StructuralForceN",
                )?);
            }
            "analysismeshartifactpath" | "meshartifactpath" => {
                solver_mesh_artifact_path =
                    Some(scalar_string(&pair[1], AUTHOR_STUDY_NAME, &ERROR_INPUT)?);
            }
            "analysismeshevidenceartifactpath" | "meshevidenceartifactpath" => {
                meshing_evidence_artifact_path =
                    Some(scalar_string(&pair[1], AUTHOR_STUDY_NAME, &ERROR_INPUT)?);
            }
            "diagramobservation" | "diagramevidence" | "diagram" => {
                diagram_observation = Some(diagram_observation_from_value(&pair[1])?);
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
    let profile = profile.ok_or_else(|| {
        builtin_error(
            AUTHOR_STUDY_NAME,
            &ERROR_INPUT,
            "fea.authorStudy requires Profile; choose a physics profile from fea.capabilities().physicsProfiles",
        )
    })?;
    let derived_run_kind = profile.derived_run_kind();
    let run_kind = match run_kind {
        Some(explicit_run_kind) if explicit_run_kind != derived_run_kind => {
            return Err(builtin_error(
                AUTHOR_STUDY_NAME,
                &ERROR_INPUT,
                format!(
                    "explicit solver {} does not match Profile {}; omit RunKind or choose a matching Profile",
                    explicit_run_kind.as_snake_case(),
                    profile.as_snake_case()
                ),
            ));
        }
        Some(explicit_run_kind) => explicit_run_kind,
        None => derived_run_kind,
    };

    let authored = analysis_author_study_op(
        AnalysisStudyAuthoringIntent {
            study_id,
            model_id,
            geometry,
            mesh_authoring_summary,
            profile,
            run_kind,
            backend,
            solver_mesh_artifact_path,
            meshing_evidence_artifact_path,
            material_region_id,
            boundary_condition_region_id,
            driving_condition_region_id,
            structural_force_n,
            diagram_observation,
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

fn diagram_observation_from_value(value: &Value) -> BuiltinResult<AnalysisStudyDiagramObservation> {
    if let Ok(text) = scalar_string(value, AUTHOR_STUDY_NAME, &ERROR_INPUT) {
        let json: serde_json::Value = serde_json::from_str(&text).map_err(|err| {
            builtin_error_with_source(AUTHOR_STUDY_NAME, &ERROR_INPUT, err.to_string(), err)
        })?;
        return json_deserialize(AUTHOR_STUDY_NAME, json, "diagram observation");
    }

    json_deserialize(
        AUTHOR_STUDY_NAME,
        value_to_json(AUTHOR_STUDY_NAME, value)?,
        "diagram observation",
    )
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
    use runmat_value::{Tensor, Value};

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

    fn canonical_authoring_solver_mesh(dir: &std::path::Path) -> (String, Value) {
        use runmat_meshing_core::{
            CanonicalMeshingContract, ElementOrder, PersistentEntityId, PersistentEntityKind,
            StableDigest,
        };

        let mut mesh =
            runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
        for (face_index, region_id) in [(0, "root"), (1, "tip")] {
            mesh.topology.boundary_faces[face_index]
                .provenance
                .push(PersistentEntityId {
                    kind: PersistentEntityKind::Face,
                    source_topology_id: region_id.to_owned(),
                    assembly_path: vec!["root".to_owned()],
                });
            mesh.topology.boundary_faces[face_index].provenance.sort();
        }
        mesh.topology.volume_elements[0].material_id = "mat_default_steel".to_owned();
        mesh.topology.regions[0].material_id = "mat_default_steel".to_owned();
        mesh.canonical_digest = StableDigest::ZERO;
        mesh.seal_canonical_digest()
            .expect("canonical authoring mesh should seal");

        let mesh_path = dir.join("solver_mesh.cbor");
        std::fs::write(
            &mesh_path,
            mesh.canonical_encode()
                .expect("canonical authoring mesh should encode"),
        )
        .expect("canonical authoring mesh should write");

        (
            mesh_path.to_string_lossy().to_string(),
            authoring_summary_value(),
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
    fn runs_with_canonical_solver_mesh_artifact() {
        let tmp = tempfile::tempdir().expect("tempdir should be created");
        let (mesh_path, summary) = canonical_authoring_solver_mesh(tmp.path());
        let study = create_author_study_object_from_args(vec![
            Value::String("authored_run_static".to_string()),
            generic_authoring_geometry_value(),
            summary,
            Value::String("Profile".to_string()),
            Value::String("linear_static_structural".to_string()),
            Value::String("SolverMeshArtifactPath".to_string()),
            Value::String(mesh_path.clone()),
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
            decoded_study.solver_mesh_artifact_path.as_deref(),
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
            run_data.solver_mesh_artifact_path.as_deref(),
            Some(mesh_path.as_str())
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
        let Some(Value::String(payload)) =
            study_object.properties.get(FEA_STUDY_SPEC_JSON_PROPERTY)
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
        let (mesh_path, summary) = canonical_authoring_solver_mesh(tmp.path());
        let study = create_author_study_object_from_args(vec![
            Value::String("authored_minimal_static".to_string()),
            generic_authoring_geometry_value(),
            summary,
            Value::String("Profile".to_string()),
            Value::String("linear_static_structural".to_string()),
            Value::String("SolverMeshArtifactPath".to_string()),
            Value::String(mesh_path.clone()),
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
            decoded_study.solver_mesh_artifact_path.as_deref(),
            Some(mesh_path.as_str())
        );

        let run = block_on(fea_run_builtin(study)).expect("minimal authored study should run");
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
            run_data.solver_mesh_artifact_path.as_deref(),
            Some(mesh_path.as_str())
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
}
