use super::support::assert_object_class;
use super::*;

#[test]
fn fea_study_requires_geometry_asset() {
    let err = block_on(fea_study_builtin(vec![
        Value::String("demo".to_string()),
        Value::Num(1.0),
    ]))
    .expect_err("invalid geometry should fail");
    assert_eq!(err.identifier(), Some("RunMat:fea:InvalidInput"));
}

#[test]
fn fea_study_requires_profile() {
    let tmp = tempfile::tempdir().expect("tempdir should be created");
    let geometry_path = tmp.path().join("part.step");
    std::fs::write(&geometry_path, SIMPLE_STEP).expect("geometry fixture should write");
    let geometry = block_on(crate::builtins::geometry::geometry_load_builtin(
        geometry_path.to_string_lossy().to_string(),
    ))
    .expect("geometry should load");

    let err = block_on(fea_study_builtin(vec![
        Value::String("missing_profile".to_string()),
        geometry,
    ]))
    .expect_err("missing profile should fail");
    assert_eq!(err.identifier(), Some("RunMat:fea:InvalidInput"));
    assert!(err.message().contains("fea.study requires Profile"));
}

#[test]
fn fea_model_requires_profile() {
    let tmp = tempfile::tempdir().expect("tempdir should be created");
    let geometry_path = tmp.path().join("part.step");
    std::fs::write(&geometry_path, SIMPLE_STEP).expect("geometry fixture should write");
    let geometry = block_on(crate::builtins::geometry::geometry_load_builtin(
        geometry_path.to_string_lossy().to_string(),
    ))
    .expect("geometry should load");

    let err = block_on(fea_model_builtin(vec![
        Value::String("missing_profile_model".to_string()),
        geometry,
    ]))
    .expect_err("missing profile should fail");
    assert_eq!(err.identifier(), Some("RunMat:fea:InvalidInput"));
    assert!(err.message().contains("fea.model requires Profile"));
}

#[test]
fn fea_load_validate_and_plan_document_workflow() {
    let tmp = tempfile::tempdir().expect("tempdir should be created");
    std::fs::write(tmp.path().join("part.stl"), TRIANGLE_STL)
        .expect("geometry fixture should write");
    let fea_path = tmp.path().join("bracket.fea");
    std::fs::write(
        &fea_path,
        r#"
version: 1
kind: study
id: bracket_static
geometry:
  path: part.stl
  units: meter
model:
  profile: linear_static_structural
run:
  backend: cpu
"#,
    )
    .expect("FEA fixture should write");

    let study = block_on(fea_load_builtin(fea_path.to_string_lossy().to_string()))
        .expect("FEA document should load");
    let Value::Object(study_object) = study.clone() else {
        panic!("expected loaded FEA study object");
    };
    assert!(study_object.class_name.is(FEA_STUDY_CLASS));
    assert!(study_object
        .properties
        .contains_key(FEA_STUDY_SPEC_JSON_PROPERTY));

    let validation =
        block_on(fea_validate_builtin(study.clone())).expect("FEA study should validate");
    let Value::Object(validation_object) = validation else {
        panic!("expected validation object");
    };
    assert!(validation_object.class_name.is(FEA_VALIDATION_CLASS));
    assert_eq!(
        validation_object.properties.get("valid"),
        Some(&Value::Bool(true))
    );

    let plan = block_on(fea_plan_builtin(study)).expect("FEA study should plan");
    let Value::Object(plan_object) = plan else {
        panic!("expected plan object");
    };
    assert!(plan_object.class_name.is(FEA_PLAN_CLASS));
    assert!(plan_object.properties.contains_key("operation_sequence"));
}

#[test]
fn fea_load_case_accepts_moment_and_torque_alias() {
    for kind in ["moment", "torque"] {
        let load = block_on(fea_load_case_builtin(vec![
            Value::String(format!("tip_{kind}")),
            Value::String("tip_node".to_string()),
            Value::String(kind.to_string()),
            Value::String("Vector".to_string()),
            moment_vector(),
        ]))
        .expect("moment load should build");
        assert_object_class(&load, FEA_LOAD_CASE_CLASS);

        let Value::Object(object) = load else {
            panic!("expected load object");
        };
        let Some(Value::String(payload)) = object.properties.get(FEA_PAYLOAD_JSON_PROPERTY) else {
            panic!("expected load JSON payload");
        };
        let decoded: LoadCase = serde_json::from_str(payload).expect("load payload should decode");
        assert_eq!(decoded.load_id, format!("tip_{kind}"));
        assert_eq!(decoded.region_id, "tip_node");
        assert!(matches!(
            decoded.kind,
            LoadKind::Moment {
                mx: 10.0,
                my: 20.0,
                mz: 30.0
            }
        ));
    }
}

#[test]
fn fea_load_case_doc_keywords_include_moment_and_torque() {
    let doc = runmat_builtins::builtin_docs()
        .into_iter()
        .find(|doc| doc.name == "fea.loadCase")
        .expect("fea.loadCase doc metadata should be registered");
    let keywords = doc
        .keywords
        .expect("fea.loadCase should advertise keywords");
    let keyword_set = keywords
        .split(',')
        .map(str::trim)
        .collect::<std::collections::BTreeSet<_>>();

    assert!(keyword_set.contains("moment"));
    assert!(keyword_set.contains("torque"));
}

#[test]
fn fea_boundary_condition_accepts_prescribed_rotation() {
    let boundary = block_on(fea_boundary_condition_builtin(vec![
        Value::String("tip_rotation".to_string()),
        Value::String("tip_node".to_string()),
        Value::String("prescribedRotation".to_string()),
        Value::String("rx".to_string()),
        Value::Num(0.1),
        Value::String("ry".to_string()),
        Value::Num(0.2),
        Value::String("rz".to_string()),
        Value::Num(0.3),
    ]))
    .expect("prescribed rotation boundary condition should build");
    assert_object_class(&boundary, FEA_BOUNDARY_CONDITION_CLASS);

    let Value::Object(object) = boundary else {
        panic!("expected boundary condition object");
    };
    let Some(Value::String(payload)) = object.properties.get(FEA_PAYLOAD_JSON_PROPERTY) else {
        panic!("expected boundary condition JSON payload");
    };
    let decoded: BoundaryCondition =
        serde_json::from_str(payload).expect("boundary condition payload should decode");
    assert_eq!(decoded.bc_id, "tip_rotation");
    assert_eq!(decoded.region_id, "tip_node");
    assert!(matches!(
        decoded.kind,
        BoundaryConditionKind::PrescribedRotation {
            rx: 0.1,
            ry: 0.2,
            rz: 0.3
        }
    ));
}

#[test]
fn fea_boundary_condition_accepts_integer_fields_for_all_numeric_kinds() {
    let rotation = boundary_payload(
        block_on(fea_boundary_condition_builtin(boundary_args(
            "prescribedRotation",
            vec![
                ("rx", Value::Int(IntValue::I8(1))),
                ("ry", Value::Int(IntValue::U16(2))),
                ("rz", Value::Int(IntValue::I32(3))),
            ],
        )))
        .unwrap(),
    );
    assert!(matches!(
        rotation.kind,
        BoundaryConditionKind::PrescribedRotation {
            rx: 1.0,
            ry: 2.0,
            rz: 3.0
        }
    ));

    let cases = [
        (
            "acousticImpedance",
            "specificImpedancePaSPerM",
            IntValue::U32(4),
        ),
        (
            "thermalPrescribedTemperature",
            "temperatureK",
            IntValue::I64(5),
        ),
        ("thermalHeatFlux", "heatFluxWPerM2", IntValue::U64(6)),
        ("cfdInletVelocity", "velocityMPerS", IntValue::I16(7)),
        ("cfdOutletPressure", "pressurePa", IntValue::U8(8)),
    ];
    for (kind, field, value) in cases {
        boundary_payload(
            block_on(fea_boundary_condition_builtin(boundary_args(
                kind,
                vec![(field, Value::Int(value))],
            )))
            .unwrap(),
        );
    }

    let convection = boundary_payload(
        block_on(fea_boundary_condition_builtin(boundary_args(
            "thermalConvection",
            vec![
                ("ambientTemperatureK", Value::Int(IntValue::U8(9))),
                ("coefficientWPerM2K", Value::Int(IntValue::I16(10))),
            ],
        )))
        .unwrap(),
    );
    assert!(matches!(
        convection.kind,
        BoundaryConditionKind::ThermalConvection {
            ambient_temperature_k: 9.0,
            coefficient_w_per_m2k: 10.0
        }
    ));
}

#[test]
fn fea_boundary_condition_converts_every_integer_class_at_binary64_boundary() {
    for value in [
        IntValue::I8(1),
        IntValue::I16(1),
        IntValue::I32(1),
        IntValue::I64(1),
        IntValue::U8(1),
        IntValue::U16(1),
        IntValue::U32(1),
        IntValue::U64(u64::MAX),
    ] {
        let expected = boundary_integer_to_f64(&value);
        let boundary = boundary_payload(
            block_on(fea_boundary_condition_builtin(boundary_args(
                "thermalHeatFlux",
                vec![("heatFluxWPerM2", Value::Int(value))],
            )))
            .unwrap(),
        );
        let BoundaryConditionKind::ThermalHeatFlux { heat_flux_w_per_m2 } = boundary.kind else {
            panic!("expected thermal heat-flux boundary");
        };
        assert_eq!(heat_flux_w_per_m2, expected);
    }
}

#[test]
fn fea_boundary_condition_rejects_nonscalar_numeric_fields() {
    let values =
        Tensor::new_integer(runmat_value::IntegerStorage::U8(vec![1, 2]), vec![1, 2]).unwrap();
    let err = block_on(fea_boundary_condition_builtin(boundary_args(
        "thermalHeatFlux",
        vec![("heatFluxWPerM2", Value::Tensor(values))],
    )))
    .expect_err("nonscalar field must fail");
    assert_eq!(err.identifier(), Some("RunMat:fea:InvalidInput"));
    assert_eq!(
        err.context.builtin.as_deref(),
        Some(BOUNDARY_CONDITION_NAME)
    );
}

#[test]
fn fea_boundary_condition_declares_seven_integer_forms() {
    assert_eq!(FEA_BOUNDARY_CONDITION_INTEGER_CAPABILITIES.len(), 7);
    assert!(FEA_BOUNDARY_CONDITION_INTEGER_CAPABILITIES
        .iter()
        .flat_map(|capability| capability.inputs)
        .all(|input| input.classes.len() == 8));
}

#[test]
fn typed_constructors_build_full_study_and_sweep_objects() {
    let tmp = tempfile::tempdir().expect("tempdir should be created");
    let geometry_path = tmp.path().join("part.step");
    std::fs::write(&geometry_path, SIMPLE_STEP).expect("geometry fixture should write");

    let geometry = block_on(crate::builtins::geometry::geometry_load_builtin(
        geometry_path.to_string_lossy().to_string(),
    ))
    .expect("geometry should load");
    let asset =
        geometry_asset_from_value(MODEL_NAME, &geometry).expect("geometry payload should decode");
    let region_id = asset
        .regions
        .first()
        .expect("fixture should import a region")
        .region_id
        .clone();

    let material = block_on(fea_material_builtin(vec![
        Value::String("steel".to_string()),
        Value::String("YoungsModulusPa".to_string()),
        Value::Num(200e9),
        Value::String("PoissonRatio".to_string()),
        Value::Num(0.30),
    ]))
    .expect("material should build");
    assert_object_class(&material, FEA_MATERIAL_CLASS);

    let assignment = block_on(fea_material_assignment_builtin(vec![
        Value::String(region_id.clone()),
        Value::String("steel".to_string()),
    ]))
    .expect("material assignment should build");
    assert_object_class(&assignment, FEA_MATERIAL_ASSIGNMENT_CLASS);

    let fixed = block_on(fea_boundary_condition_builtin(vec![
        Value::String("fixed_base".to_string()),
        Value::String(region_id.clone()),
        Value::String("fixed".to_string()),
    ]))
    .expect("boundary condition should build");
    assert_object_class(&fixed, FEA_BOUNDARY_CONDITION_CLASS);

    let load = block_on(fea_load_case_builtin(vec![
        Value::String("tip_force".to_string()),
        Value::String(region_id.clone()),
        Value::String("force".to_string()),
        Value::String("Vector".to_string()),
        force_vector(),
    ]))
    .expect("load case should build");
    assert_object_class(&load, FEA_LOAD_CASE_CLASS);

    let step = block_on(fea_step_builtin(vec![
        Value::String("static_step".to_string()),
        Value::String("static".to_string()),
    ]))
    .expect("analysis step should build");
    assert_object_class(&step, FEA_STEP_CLASS);

    let selector = format!("id:{region_id}");
    let mut regional_delta = StructValue::new();
    regional_delta.insert("region_id", Value::String(selector.clone()));
    regional_delta.insert("temperature_delta_k", Value::Int(IntValue::I8(5)));
    let mut field_source = StructValue::new();
    field_source.insert("source_id", Value::String("temperature-map".into()));
    field_source.insert("revision", Value::Int(IntValue::U32(7)));
    field_source.insert("expected_region_ids", cell(vec![Value::String(selector)]));
    let domain = block_on(fea_domain_builtin(vec![
        Value::String("thermoMechanical".into()),
        Value::String("RegionTemperatureDeltas".into()),
        cell(vec![Value::Struct(regional_delta)]),
        Value::String("FieldSource".into()),
        Value::Struct(field_source),
    ]))
    .expect("thermo-mechanical domain should build");

    let model = block_on(fea_model_builtin(vec![
        Value::String("bracket_static_model".to_string()),
        geometry.clone(),
        Value::String("Defaults".to_string()),
        Value::String("none".to_string()),
        Value::String("Profile".to_string()),
        Value::String("linear_static_structural".to_string()),
        Value::String("Materials".to_string()),
        cell(vec![material]),
        Value::String("MaterialAssignments".to_string()),
        cell(vec![assignment]),
        Value::String("BoundaryConditions".to_string()),
        cell(vec![fixed]),
        Value::String("Loads".to_string()),
        cell(vec![load]),
        Value::String("Steps".to_string()),
        cell(vec![step]),
        Value::String("Domains".to_string()),
        cell(vec![domain]),
    ]))
    .expect("model should build");
    assert_object_class(&model, FEA_MODEL_CLASS);
    let decoded_model: AnalysisModel = object_payload(&model);
    let thermo = decoded_model
        .thermo_mechanical
        .expect("thermo-mechanical domain");
    assert_eq!(thermo.region_temperature_deltas[0].region_id, region_id);
    assert_eq!(
        thermo
            .field_source
            .expect("field source")
            .expected_region_ids,
        vec![region_id.clone()]
    );

    let run_options = block_on(fea_run_options_builtin(vec![
        Value::String("linear_static".to_string()),
        Value::String("DeterministicMode".to_string()),
        Value::Bool(true),
        Value::String("PrecisionMode".to_string()),
        Value::String("fp64".to_string()),
        Value::String("QualityPolicy".to_string()),
        Value::String("balanced".to_string()),
    ]))
    .expect("run options should build");
    assert_object_class(&run_options, FEA_RUN_OPTIONS_CLASS);

    let study = block_on(fea_study_builtin(vec![
        Value::String("bracket_static".to_string()),
        geometry,
        Value::String("Profile".to_string()),
        Value::String("linear_static_structural".to_string()),
        Value::String("Backend".to_string()),
        Value::String("cpu".to_string()),
        Value::String("Model".to_string()),
        model,
        Value::String("RunOptions".to_string()),
        run_options,
    ]))
    .expect("study should build");
    assert_object_class(&study, FEA_STUDY_CLASS);

    let sweep = block_on(fea_sweep_builtin(vec![
        Value::String("bracket_sweep".to_string()),
        cell(vec![study]),
        Value::String("FailFast".to_string()),
        Value::Bool(false),
    ]))
    .expect("sweep should build");
    assert_object_class(&sweep, FEA_SWEEP_CLASS);
}
