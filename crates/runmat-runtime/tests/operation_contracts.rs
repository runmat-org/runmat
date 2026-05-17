use runmat_analysis_core::{AnalysisFieldValues, ElectromagneticDomain, ReferenceFrame};
use runmat_analysis_fea::fixtures::{fixture_model, FixtureId};
use runmat_analysis_fea::ComputeBackend;
use runmat_geometry_core::EntityKind;
use runmat_geometry_core::UnitSystem;
use runmat_runtime::analysis::{
    analysis_create_model_op, analysis_plan_study_op, analysis_results_by_run_id_op,
    analysis_results_compare_op, analysis_results_op, analysis_run_acoustic_op,
    analysis_run_electromagnetic_op, analysis_run_fsi_op, analysis_run_linear_static_op,
    analysis_run_linear_static_with_options, analysis_run_modal_op,
    analysis_run_modal_with_options_op, analysis_run_nonlinear_op,
    analysis_run_nonlinear_with_options_op, analysis_run_study_op, analysis_run_transient_op,
    analysis_run_transient_with_options_op, analysis_trends_op, analysis_validate,
    analysis_validate_study_op, AnalysisCreateModelIntentSpec, AnalysisCreateModelProfile,
    AnalysisModalRunOptions, AnalysisNonlinearRunOptions, AnalysisResultsCompareQuery,
    AnalysisResultsQuery, AnalysisRunKind, AnalysisRunOptions, AnalysisStudySpec,
    AnalysisTransientRunOptions, AnalysisTrendsQuery, ModalFrequencyBasis, ModalFrequencyUnits,
    PrecisionMode, PreconditionerMode, QualityPolicy, QualityReasonCode, RunStatus,
};
use runmat_runtime::geometry::{
    geometry_capture_view_op, geometry_inspect_op, geometry_list_regions_op, geometry_load_op,
    geometry_prep_artifact_health_op, geometry_prep_for_analysis_op, geometry_query_entities_op,
    GeometryCaptureViewSpec, GeometryEntityQuery, GeometryPrepArtifactHealthQuery,
    GeometryPrepForAnalysisSpec, GeometryPrepProfile,
};
use runmat_runtime::operations::OperationContext;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

const TRIANGLE_STL: &str = "solid tri\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid tri\n";
const SIMPLE_STEP: &str = "ISO-10303-21;\nHEADER;\nFILE_NAME('Assembly_A');\nENDSEC;\nDATA;\n#10=PRODUCT('Base_Mount','',(#1));\n#11=PRODUCT('Tip_Load','',(#1));\n#20=MATERIAL_DESIGNATION('Aluminum 6061');\nENDSEC;\nEND-ISO-10303-21;\n";

struct EnvVarRestoreGuard {
    key: &'static str,
    previous: Option<String>,
}

impl Drop for EnvVarRestoreGuard {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.as_ref() {
            std::env::set_var(self.key, previous);
        } else {
            std::env::remove_var(self.key);
        }
    }
}

fn sorted_object_keys(value: &Value) -> Vec<String> {
    let mut keys = value
        .as_object()
        .expect("snapshot target must be object")
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    keys.sort();
    keys
}

fn assert_fallback_event_schema(event: &str) {
    let parts: Vec<&str> = event.splitn(3, ':').collect();
    assert_eq!(parts.len(), 3, "fallback event must have 3 parts");

    assert!(
        matches!(
            parts[0],
            "BACKEND_NO_PROVIDER" | "BACKEND_UPLOAD_FAILED" | "SOLVER_BACKEND_FALLBACK"
        ),
        "unexpected fallback category: {}",
        parts[0]
    );
    if parts[0] == "SOLVER_BACKEND_FALLBACK" {
        assert!(
            parts[1].starts_with("requested="),
            "unexpected solver backend fallback stage: {}",
            parts[1]
        );
    } else {
        assert!(
            matches!(parts[1], "displacement" | "von_mises"),
            "unexpected fallback stage: {}",
            parts[1]
        );
    }
    assert!(!parts[2].is_empty(), "fallback reason must be non-empty");
}

#[test]
fn geometry_operation_contracts_are_v1_and_versioned() {
    let context = OperationContext::new(Some("trace-contract-1".to_string()), None);
    let inspect = geometry_inspect_op("/part.stl", TRIANGLE_STL.as_bytes(), context.clone())
        .expect("inspect should succeed");
    assert_eq!(inspect.operation, "geometry.inspect");
    assert_eq!(inspect.op_version, "geometry.inspect/v1");

    let load = geometry_load_op("/part.stl", TRIANGLE_STL.as_bytes(), context)
        .expect("load should succeed");
    assert_eq!(load.operation, "geometry.load");
    assert_eq!(load.op_version, "geometry.load/v1");

    let list_regions = geometry_list_regions_op(
        &load.data,
        OperationContext::new(Some("trace-contract-1c".to_string()), None),
    )
    .expect("list regions should succeed");
    assert_eq!(list_regions.operation, "geometry.list_regions");
    assert_eq!(list_regions.op_version, "geometry.list_regions/v1");

    let query_entities = geometry_query_entities_op(
        &load.data,
        GeometryEntityQuery {
            region_id: None,
            mesh_id: None,
            entity_kind: EntityKind::Node,
            limit: Some(2),
        },
        OperationContext::new(Some("trace-contract-1d".to_string()), None),
    )
    .expect("query entities should succeed");
    assert_eq!(query_entities.operation, "geometry.query_entities");
    assert_eq!(query_entities.op_version, "geometry.query_entities/v1");

    let capture_view = geometry_capture_view_op(
        &load.data,
        GeometryCaptureViewSpec {
            format: "png".to_string(),
            width: 800,
            height: 600,
        },
        OperationContext::new(Some("trace-contract-1e".to_string()), None),
    )
    .expect_err("capture view is not wired yet");
    assert_eq!(capture_view.operation, "geometry.capture_view");
    assert_eq!(capture_view.op_version, "geometry.capture_view/v1");
    assert_eq!(capture_view.error_code, "GEOMETRY_CAPTURE_UNSUPPORTED");

    let svg_capture = geometry_capture_view_op(
        &load.data,
        GeometryCaptureViewSpec {
            format: "svg".to_string(),
            width: 800,
            height: 600,
        },
        OperationContext::new(Some("trace-contract-1e-svg".to_string()), None),
    )
    .expect("svg capture should succeed via default adapter");
    assert_eq!(svg_capture.operation, "geometry.capture_view");
    assert_eq!(svg_capture.op_version, "geometry.capture_view/v1");

    let prep = geometry_prep_for_analysis_op(
        &load.data,
        GeometryPrepForAnalysisSpec {
            profile: GeometryPrepProfile::AnalysisReady,
            target_element_budget: 100_000,
        },
        OperationContext::new(Some("trace-contract-1f".to_string()), None),
    )
    .expect("prep for analysis should succeed");
    assert_eq!(prep.operation, "geometry.prep_for_analysis");
    assert_eq!(prep.op_version, "geometry.prep_for_analysis/v1");
    assert!(!prep.data.prep_artifact_id.is_empty());
    assert!(!prep.data.prep.prepared_meshes.is_empty());

    let prep_health = geometry_prep_artifact_health_op(
        GeometryPrepArtifactHealthQuery::default(),
        OperationContext::new(Some("trace-contract-1g".to_string()), None),
    )
    .expect("prep artifact health should succeed");
    assert_eq!(prep_health.operation, "geometry.prep_artifact_health");
    assert_eq!(prep_health.op_version, "geometry.prep_artifact_health/v1");
    assert_eq!(
        prep_health.data.schema_version,
        "geometry-prep-artifact-health/v1"
    );
    assert_eq!(svg_capture.data.format, "svg");

    let invalid_capture_view = geometry_capture_view_op(
        &load.data,
        GeometryCaptureViewSpec {
            format: "png".to_string(),
            width: 0,
            height: 600,
        },
        OperationContext::new(Some("trace-contract-1f".to_string()), None),
    )
    .expect_err("invalid capture dimensions should fail");
    assert_eq!(invalid_capture_view.operation, "geometry.capture_view");
    assert_eq!(invalid_capture_view.op_version, "geometry.capture_view/v1");
    assert_eq!(
        invalid_capture_view.error_code,
        "GEOMETRY_CAPTURE_INVALID_SPEC"
    );
}

#[test]
fn analysis_validate_contract_is_v1_and_maps_codes() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let context = OperationContext::new(Some("trace-contract-2".to_string()), None);
    let envelope = analysis_validate(&model, UnitSystem::Meter, &ReferenceFrame::Global, context)
        .expect("validate should succeed");
    assert_eq!(envelope.operation, "analysis.validate");
    assert_eq!(envelope.op_version, "analysis.validate/v1");

    let mut invalid = model;
    invalid.materials.clear();
    let err = analysis_validate(
        &invalid,
        UnitSystem::Meter,
        &ReferenceFrame::Global,
        OperationContext::new(None, None),
    )
    .expect_err("validate should fail");
    assert_eq!(err.operation, "analysis.validate");
    assert_eq!(err.op_version, "analysis.validate/v1");
    assert_eq!(err.error_code, "ANALYSIS_VALIDATION_MISSING_MATERIALS");
}

#[test]
fn analysis_create_model_contract_is_v1_and_maps_codes() {
    let geometry = geometry_load_op(
        "/part.stl",
        TRIANGLE_STL.as_bytes(),
        OperationContext::new(Some("trace-contract-create-1".to_string()), None),
    )
    .expect("geometry load should succeed");

    let envelope = analysis_create_model_op(
        &geometry.data,
        AnalysisCreateModelIntentSpec {
            model_id: "contract_generated_model".to_string(),
            profile: AnalysisCreateModelProfile::LinearStaticStructural,
            prep_context: None,
        },
        OperationContext::new(Some("trace-contract-create-2".to_string()), None),
    )
    .expect("create model should succeed");
    assert_eq!(envelope.operation, "analysis.create_model");
    assert_eq!(envelope.op_version, "analysis.create_model/v1");
    assert_eq!(envelope.data.model_id.0, "contract_generated_model");

    let err = analysis_create_model_op(
        &geometry.data,
        AnalysisCreateModelIntentSpec {
            model_id: "".to_string(),
            profile: AnalysisCreateModelProfile::LinearStaticStructural,
            prep_context: None,
        },
        OperationContext::new(Some("trace-contract-create-3".to_string()), None),
    )
    .expect_err("create model should fail for empty model id");
    assert_eq!(err.operation, "analysis.create_model");
    assert_eq!(err.op_version, "analysis.create_model/v1");
    assert_eq!(err.error_code, "ANALYSIS_CREATE_MODEL_INVALID_INTENT");

    let modal = analysis_create_model_op(
        &geometry.data,
        AnalysisCreateModelIntentSpec {
            model_id: "contract_modal_model".to_string(),
            profile: AnalysisCreateModelProfile::ModalStructural,
            prep_context: None,
        },
        OperationContext::new(Some("trace-contract-create-4-modal".to_string()), None),
    )
    .expect("modal profile should be supported");
    assert_eq!(modal.operation, "analysis.create_model");
    assert_eq!(modal.op_version, "analysis.create_model/v1");
    assert_eq!(
        modal.data.steps[0].kind,
        runmat_analysis_core::AnalysisStepKind::Modal
    );

    let acoustic = analysis_create_model_op(
        &geometry.data,
        AnalysisCreateModelIntentSpec {
            model_id: "contract_acoustic_harmonic_model".to_string(),
            profile: AnalysisCreateModelProfile::AcousticHarmonic,
            prep_context: None,
        },
        OperationContext::new(Some("trace-contract-create-4-acoustic".to_string()), None),
    )
    .expect("acoustic harmonic profile should be supported");
    assert_eq!(acoustic.operation, "analysis.create_model");
    assert_eq!(acoustic.op_version, "analysis.create_model/v1");
    assert_eq!(
        acoustic.data.steps[0].kind,
        runmat_analysis_core::AnalysisStepKind::Modal
    );

    let transient = analysis_create_model_op(
        &geometry.data,
        AnalysisCreateModelIntentSpec {
            model_id: "contract_transient_model".to_string(),
            profile: AnalysisCreateModelProfile::TransientStructural,
            prep_context: None,
        },
        OperationContext::new(Some("trace-contract-create-4-transient".to_string()), None),
    )
    .expect("transient profile should be supported");
    assert_eq!(transient.operation, "analysis.create_model");
    assert_eq!(transient.op_version, "analysis.create_model/v1");
    assert_eq!(
        transient.data.steps[0].kind,
        runmat_analysis_core::AnalysisStepKind::Transient
    );

    let nonlinear = analysis_create_model_op(
        &geometry.data,
        AnalysisCreateModelIntentSpec {
            model_id: "contract_nonlinear_model".to_string(),
            profile: AnalysisCreateModelProfile::NonlinearStructural,
            prep_context: None,
        },
        OperationContext::new(Some("trace-contract-create-4-nonlinear".to_string()), None),
    )
    .expect("nonlinear profile should be supported");
    assert_eq!(nonlinear.operation, "analysis.create_model");
    assert_eq!(nonlinear.op_version, "analysis.create_model/v1");
    assert_eq!(
        nonlinear.data.steps[0].kind,
        runmat_analysis_core::AnalysisStepKind::Nonlinear
    );

    let electromagnetic = analysis_create_model_op(
        &geometry.data,
        AnalysisCreateModelIntentSpec {
            model_id: "contract_electromagnetic_model".to_string(),
            profile: AnalysisCreateModelProfile::ElectromagneticStatic,
            prep_context: None,
        },
        OperationContext::new(
            Some("trace-contract-create-4-electromagnetic".to_string()),
            None,
        ),
    )
    .expect("electromagnetic profile should be supported");
    assert_eq!(electromagnetic.operation, "analysis.create_model");
    assert_eq!(electromagnetic.op_version, "analysis.create_model/v1");
    assert_eq!(
        electromagnetic.data.steps[0].kind,
        runmat_analysis_core::AnalysisStepKind::Electromagnetic
    );
    let electromagnetic_domain = electromagnetic
        .data
        .electromagnetic
        .as_ref()
        .expect("electromagnetic profile should seed electromagnetic domain");
    assert!(electromagnetic_domain.enabled);
    assert_eq!(electromagnetic_domain.reference_frequency_hz, 60.0);
    assert_eq!(electromagnetic_domain.applied_current_a, 100.0);
}

#[test]
fn analysis_study_workflow_contract_persists_evidence_artifacts() {
    static NEXT_TMP_ID: AtomicU64 = AtomicU64::new(1);
    let evidence_root = std::env::temp_dir().join(format!(
        "runmat-study-contract-artifacts-{}-{}",
        std::process::id(),
        NEXT_TMP_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&evidence_root);
    let env_guard = EnvVarRestoreGuard {
        key: "RUNMAT_ANALYSIS_STUDY_ARTIFACT_ROOT",
        previous: std::env::var("RUNMAT_ANALYSIS_STUDY_ARTIFACT_ROOT").ok(),
    };
    std::env::set_var(
        "RUNMAT_ANALYSIS_STUDY_ARTIFACT_ROOT",
        evidence_root.display().to_string(),
    );

    let geometry = geometry_load_op(
        "/part.stl",
        TRIANGLE_STL.as_bytes(),
        OperationContext::new(Some("trace-contract-study-1".to_string()), None),
    )
    .expect("geometry load should succeed");
    let spec = AnalysisStudySpec {
        study_id: "contract_study_linear_static".to_string(),
        geometry: geometry.data.clone(),
        create_model_intent: AnalysisCreateModelIntentSpec {
            model_id: "contract_study_model".to_string(),
            profile: AnalysisCreateModelProfile::LinearStaticStructural,
            prep_context: None,
        },
        run_kind: AnalysisRunKind::LinearStatic,
        backend: ComputeBackend::Cpu,
        electromagnetic_run_options: None,
    };

    let validate = analysis_validate_study_op(
        &spec,
        OperationContext::new(Some("trace-contract-study-2".to_string()), None),
    )
    .expect("validate study should succeed");
    assert_eq!(validate.operation, "analysis.validate_study");
    assert_eq!(validate.op_version, "analysis.validate_study/v1");
    assert!(validate.data.valid);
    assert_eq!(validate.data.issues.len(), 0);
    assert!(PathBuf::from(&validate.data.evidence_artifact_path).exists());

    let plan = analysis_plan_study_op(
        &spec,
        OperationContext::new(Some("trace-contract-study-3".to_string()), None),
    )
    .expect("plan study should succeed");
    assert_eq!(plan.operation, "analysis.plan_study");
    assert_eq!(plan.op_version, "analysis.plan_study/v1");
    assert!(plan.data.study_fingerprint.starts_with("sha256:"));
    assert!(plan.data.electromagnetic_run_options.is_none());
    assert_eq!(plan.data.run_operation, "analysis.run_linear_static");
    assert_eq!(plan.data.run_op_version, "analysis.run_linear_static/v1");
    assert!(PathBuf::from(&plan.data.evidence_artifact_path).exists());

    let run = analysis_run_study_op(
        &spec,
        OperationContext::new(Some("trace-contract-study-4".to_string()), None),
    )
    .expect("run study should succeed");
    assert_eq!(run.operation, "analysis.run_study");
    assert_eq!(run.op_version, "analysis.run_study/v1");
    assert_eq!(run.data.study_fingerprint, plan.data.study_fingerprint);
    assert!(run.data.electromagnetic_run_options.is_none());
    assert_eq!(run.data.run_operation, "analysis.run_linear_static");
    assert_eq!(run.data.run_op_version, "analysis.run_linear_static/v1");
    assert_eq!(run.data.quality_reasons.len(), 0);
    assert!(PathBuf::from(&run.data.evidence_artifact_path).exists());

    drop(env_guard);
    let _ = fs::remove_dir_all(&evidence_root);
}

#[test]
fn prep_to_create_model_to_validate_flow_is_contract_stable() {
    let geometry = geometry_load_op(
        "/assembly.step",
        SIMPLE_STEP.as_bytes(),
        OperationContext::new(Some("trace-contract-prep-flow-1".to_string()), None),
    )
    .expect("geometry load should succeed");
    let prep = geometry_prep_for_analysis_op(
        &geometry.data,
        GeometryPrepForAnalysisSpec {
            profile: GeometryPrepProfile::AnalysisReady,
            target_element_budget: 120_000,
        },
        OperationContext::new(Some("trace-contract-prep-flow-2".to_string()), None),
    )
    .expect("prep should succeed");
    assert_eq!(prep.operation, "geometry.prep_for_analysis");
    assert_eq!(prep.op_version, "geometry.prep_for_analysis/v1");

    let created = analysis_create_model_op(
        &geometry.data,
        AnalysisCreateModelIntentSpec {
            model_id: "contract_prep_model".to_string(),
            profile: AnalysisCreateModelProfile::LinearStaticStructural,
            prep_context: Some(runmat_runtime::analysis::AnalysisCreateModelPrepContext {
                source_geometry_id: prep.data.prep.provenance.source_geometry_id.clone(),
                source_geometry_revision: prep.data.prep.provenance.source_geometry_revision,
                region_mappings: prep.data.prep.region_mappings.clone(),
            }),
        },
        OperationContext::new(Some("trace-contract-prep-flow-3".to_string()), None),
    )
    .expect("create model should succeed");

    let validated = analysis_validate(
        &created.data,
        created.data.units,
        &runmat_analysis_core::ReferenceFrame::Global,
        OperationContext::new(Some("trace-contract-prep-flow-4".to_string()), None),
    )
    .expect("validate should succeed");
    assert_eq!(validated.operation, "analysis.validate");
    assert_eq!(validated.op_version, "analysis.validate/v1");
    assert!(validated.data.valid);
}

#[test]
fn analysis_create_model_infers_materials_from_step_metadata_contract() {
    let geometry = geometry_load_op(
        "/assembly.step",
        SIMPLE_STEP.as_bytes(),
        OperationContext::new(Some("trace-contract-create-step-1".to_string()), None),
    )
    .expect("step geometry load should succeed");

    let envelope = analysis_create_model_op(
        &geometry.data,
        AnalysisCreateModelIntentSpec {
            model_id: "contract_step_model".to_string(),
            profile: AnalysisCreateModelProfile::LinearStaticStructural,
            prep_context: None,
        },
        OperationContext::new(Some("trace-contract-create-step-2".to_string()), None),
    )
    .expect("create model should succeed");

    assert!(envelope
        .data
        .materials
        .iter()
        .any(|material| material.material_id == "mat_aluminum"));
    assert_eq!(envelope.data.boundary_conditions[0].region_id, "region_1");
    assert_eq!(envelope.data.loads[0].region_id, "region_2");
    assert_eq!(envelope.data.material_assignments.len(), 2);
}

#[test]
fn analysis_validate_maps_unit_and_frame_mismatch_codes() {
    let mut unit_mismatch = fixture_model(FixtureId::CantileverLinearStatic);
    unit_mismatch.units = UnitSystem::Inch;
    let unit_err = analysis_validate(
        &unit_mismatch,
        UnitSystem::Meter,
        &ReferenceFrame::Global,
        OperationContext::new(Some("trace-contract-2b".to_string()), None),
    )
    .expect_err("validate should fail for units");
    assert_eq!(unit_err.error_code, "ANALYSIS_VALIDATION_UNIT_MISMATCH");
    assert_eq!(
        unit_err.context.get("model_units").map(|s| s.as_str()),
        Some("Inch")
    );
    assert_eq!(
        unit_err.context.get("geometry_units").map(|s| s.as_str()),
        Some("Meter")
    );

    let mut frame_mismatch = fixture_model(FixtureId::CantileverLinearStatic);
    frame_mismatch.frame = ReferenceFrame::Local("fixture_frame".to_string());
    let frame_err = analysis_validate(
        &frame_mismatch,
        UnitSystem::Meter,
        &ReferenceFrame::Global,
        OperationContext::new(Some("trace-contract-2c".to_string()), None),
    )
    .expect_err("validate should fail for frame");
    assert_eq!(frame_err.error_code, "ANALYSIS_VALIDATION_FRAME_MISMATCH");
    assert_eq!(
        frame_err.context.get("model_frame").map(|s| s.as_str()),
        Some("Local(\"fixture_frame\")")
    );
    assert_eq!(
        frame_err.context.get("geometry_frame").map(|s| s.as_str()),
        Some("Global")
    );
}

#[test]
fn analysis_run_contract_is_v1_and_publishable_for_fixture() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let envelope = analysis_run_linear_static_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-contract-3".to_string()), None),
    )
    .expect("run should succeed");

    assert_eq!(envelope.operation, "analysis.run_linear_static");
    assert_eq!(envelope.op_version, "analysis.run_linear_static/v1");
    assert_eq!(envelope.data.run_status, RunStatus::Publishable);
    assert!(envelope.data.publishable);

    assert_eq!(
        envelope.data.run.displacement_field.field_id,
        "displacement"
    );
    assert_eq!(envelope.data.run.von_mises_field.field_id, "von_mises");
    assert!(matches!(
        envelope.data.run.displacement_field.values,
        AnalysisFieldValues::HostF64(_)
    ));
}

#[test]
fn analysis_run_modal_contract_is_v1_and_typed() {
    let geometry = geometry_load_op(
        "/part.stl",
        TRIANGLE_STL.as_bytes(),
        OperationContext::new(Some("trace-contract-modal-1".to_string()), None),
    )
    .expect("geometry load should succeed");

    let modal_model = analysis_create_model_op(
        &geometry.data,
        AnalysisCreateModelIntentSpec {
            model_id: "contract_modal_model".to_string(),
            profile: AnalysisCreateModelProfile::ModalStructural,
            prep_context: None,
        },
        OperationContext::new(Some("trace-contract-modal-2".to_string()), None),
    )
    .expect("modal model should be created");

    let modal_envelope = analysis_run_modal_op(
        &modal_model.data,
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-contract-modal-3".to_string()), None),
    )
    .expect("modal run should produce envelope");
    assert_eq!(modal_envelope.operation, "analysis.run_modal");
    assert_eq!(modal_envelope.op_version, "analysis.run_modal/v1");
    assert_eq!(
        modal_envelope.data.run.solver_method,
        "matrix_free_subspace_iteration"
    );
    let modal_results = modal_envelope
        .data
        .modal_results
        .as_ref()
        .expect("modal results payload should exist");
    assert!(!modal_results.eigenvalues_hz.is_empty());
    assert_eq!(
        modal_results.eigenvalues_hz.len(),
        modal_results.mode_shapes.len()
    );
    assert_eq!(modal_results.mode_shapes[0].field_id, "mode_shape_1");
    assert_eq!(modal_results.modal_payload_version, "modal_results/v1");
    assert_eq!(
        modal_results.eigenvalues_hz.len(),
        modal_results.residual_norms.len()
    );
    assert_eq!(modal_results.mode_units, ModalFrequencyUnits::Hz);
    assert_eq!(
        modal_results.frequency_basis,
        ModalFrequencyBasis::NativeEigenSolve
    );
    assert_eq!(modal_envelope.data.run_status, RunStatus::Degraded);
    assert!(!modal_envelope.data.publishable);
    assert!(!modal_envelope
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::ModalPlaceholder));
    assert!(modal_envelope
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::ModalResidualExceeded));

    let invalid = analysis_run_modal_op(
        &fixture_model(FixtureId::CantileverLinearStatic),
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-contract-modal-4".to_string()), None),
    )
    .expect_err("modal run should reject models without modal step");
    assert_eq!(invalid.operation, "analysis.run_modal");
    assert_eq!(invalid.op_version, "analysis.run_modal/v1");
    assert_eq!(invalid.error_code, "ANALYSIS_RUN_MODAL_INVALID_MODEL");
}

#[test]
fn analysis_run_modal_with_options_contract_controls_mode_budget() {
    let geometry = geometry_load_op(
        "/part.stl",
        TRIANGLE_STL.as_bytes(),
        OperationContext::new(Some("trace-contract-modal-opts-1".to_string()), None),
    )
    .expect("geometry load should succeed");

    let modal_model = analysis_create_model_op(
        &geometry.data,
        AnalysisCreateModelIntentSpec {
            model_id: "contract_modal_model_opts".to_string(),
            profile: AnalysisCreateModelProfile::ModalStructural,
            prep_context: None,
        },
        OperationContext::new(Some("trace-contract-modal-opts-2".to_string()), None),
    )
    .expect("modal model should be created");

    let envelope = analysis_run_modal_with_options_op(
        &modal_model.data,
        ComputeBackend::Cpu,
        AnalysisModalRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Balanced,
            mode_count: 2,
            residual_warn_threshold: 1.0e-2,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
        },
        OperationContext::new(Some("trace-contract-modal-opts-3".to_string()), None),
    )
    .expect("modal run with options should succeed");

    assert_eq!(envelope.operation, "analysis.run_modal");
    assert_eq!(envelope.op_version, "analysis.run_modal/v1");
    let modal = envelope
        .data
        .modal_results
        .as_ref()
        .expect("modal payload should exist");
    assert!(modal.eigenvalues_hz.len() > 0);
    assert!(modal.eigenvalues_hz.len() <= 2);
    assert!(envelope.data.provenance.deterministic_mode);

    let invalid = analysis_run_modal_with_options_op(
        &modal_model.data,
        ComputeBackend::Cpu,
        AnalysisModalRunOptions {
            deterministic_mode: false,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Balanced,
            mode_count: 0,
            residual_warn_threshold: 1.0e-3,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
        },
        OperationContext::new(Some("trace-contract-modal-opts-4".to_string()), None),
    )
    .expect_err("mode_count=0 should fail");
    assert_eq!(invalid.error_code, "ANALYSIS_RUN_MODAL_INVALID_OPTIONS");
    assert_eq!(invalid.operation, "analysis.run_modal");
    assert_eq!(invalid.op_version, "analysis.run_modal/v1");
}

#[test]
fn analysis_run_acoustic_contract_is_v1_and_typed() {
    let geometry = geometry_load_op(
        "/part.stl",
        TRIANGLE_STL.as_bytes(),
        OperationContext::new(Some("trace-contract-acoustic-1".to_string()), None),
    )
    .expect("geometry load should succeed");

    let acoustic_model = analysis_create_model_op(
        &geometry.data,
        AnalysisCreateModelIntentSpec {
            model_id: "contract_acoustic_model".to_string(),
            profile: AnalysisCreateModelProfile::AcousticHarmonic,
            prep_context: None,
        },
        OperationContext::new(Some("trace-contract-acoustic-2".to_string()), None),
    )
    .expect("acoustic model should be created");

    let envelope = analysis_run_acoustic_op(
        &acoustic_model.data,
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-contract-acoustic-3".to_string()), None),
    )
    .expect("acoustic run should produce envelope");
    assert_eq!(envelope.operation, "analysis.run_acoustic");
    assert_eq!(envelope.op_version, "analysis.run_acoustic/v1");
    assert!(envelope.data.modal_results.is_some());
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_ACOUSTIC_PLACEHOLDER"));

    let invalid = analysis_run_acoustic_op(
        &fixture_model(FixtureId::CantileverLinearStatic),
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-contract-acoustic-4".to_string()), None),
    )
    .expect_err("acoustic run should reject models without modal step");
    assert_eq!(invalid.operation, "analysis.run_acoustic");
    assert_eq!(invalid.op_version, "analysis.run_acoustic/v1");
    assert_eq!(invalid.error_code, "ANALYSIS_RUN_ACOUSTIC_INVALID_MODEL");
}

#[test]
fn analysis_run_transient_contract_is_v1_and_typed() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.steps = vec![runmat_analysis_core::AnalysisStep {
        step_id: "transient_1".to_string(),
        kind: runmat_analysis_core::AnalysisStepKind::Transient,
    }];

    let envelope = analysis_run_transient_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-contract-transient-1".to_string()), None),
    )
    .expect("transient run should return envelope");
    assert_eq!(envelope.operation, "analysis.run_transient");
    assert_eq!(envelope.op_version, "analysis.run_transient/v1");
    assert_eq!(envelope.data.run.solver_method, "implicit_euler_pcg");
    assert_eq!(envelope.data.run_status, RunStatus::Publishable);
    assert!(envelope.data.publishable);
    assert!(!envelope
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::TransientPlaceholder));
    let transient = envelope
        .data
        .transient_results
        .as_ref()
        .expect("transient payload should exist");
    assert_eq!(
        transient.integration_method,
        runmat_runtime::analysis::TransientIntegrationMethod::ImplicitEuler
    );
    assert_eq!(
        transient.time_points_s.len(),
        transient.displacement_snapshots.len()
    );

    let invalid = analysis_run_transient_op(
        &fixture_model(FixtureId::CantileverLinearStatic),
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-contract-transient-2".to_string()), None),
    )
    .expect_err("transient run should reject models without transient step");
    assert_eq!(invalid.operation, "analysis.run_transient");
    assert_eq!(invalid.op_version, "analysis.run_transient/v1");
    assert_eq!(invalid.error_code, "ANALYSIS_RUN_TRANSIENT_INVALID_MODEL");
}

#[test]
fn analysis_run_fsi_contract_is_v1_and_typed() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.steps = vec![
        runmat_analysis_core::AnalysisStep {
            step_id: "fsi_structure".to_string(),
            kind: runmat_analysis_core::AnalysisStepKind::Transient,
        },
        runmat_analysis_core::AnalysisStep {
            step_id: "fsi_flow".to_string(),
            kind: runmat_analysis_core::AnalysisStepKind::Cfd,
        },
    ];
    model.cfd = Some(runmat_analysis_core::CfdDomain {
        enabled: true,
        solve_family: runmat_analysis_core::CfdSolveFamily::Transient,
        reference_density_kg_per_m3: 1.225,
        dynamic_viscosity_pa_s: 1.81e-5,
        inlet_velocity_m_per_s: 4.0,
        turbulence_intensity: 0.06,
        time_profile: vec![
            runmat_analysis_core::CfdTimeProfilePoint {
                normalized_time: 0.0,
                inlet_scale: 0.6,
            },
            runmat_analysis_core::CfdTimeProfilePoint {
                normalized_time: 1.0,
                inlet_scale: 1.0,
            },
        ],
    });

    let envelope = analysis_run_fsi_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-contract-fsi-1".to_string()), None),
    )
    .expect("fsi run should return envelope");
    assert_eq!(envelope.operation, "analysis.run_fsi");
    assert_eq!(envelope.op_version, "analysis.run_fsi/v1");
    assert_eq!(envelope.data.run.solver_method, "implicit_euler_pcg");
    assert!(envelope.data.transient_results.is_some());
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_CFD_FLOW"));
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_FSI_COUPLING"));

    let mut invalid_model = model.clone();
    invalid_model.steps = vec![runmat_analysis_core::AnalysisStep {
        step_id: "fsi_flow_only".to_string(),
        kind: runmat_analysis_core::AnalysisStepKind::Cfd,
    }];
    let invalid = analysis_run_fsi_op(
        &invalid_model,
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-contract-fsi-2".to_string()), None),
    )
    .expect_err("fsi run should reject missing transient step");
    assert_eq!(invalid.operation, "analysis.run_fsi");
    assert_eq!(invalid.op_version, "analysis.run_fsi/v1");
    assert_eq!(invalid.error_code, "ANALYSIS_RUN_FSI_INVALID_MODEL");
}

#[test]
fn analysis_run_transient_thermo_field_reference_errors_are_typed() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.steps = vec![runmat_analysis_core::AnalysisStep {
        step_id: "transient_1".to_string(),
        kind: runmat_analysis_core::AnalysisStepKind::Transient,
    }];
    model.thermo_mechanical = Some(runmat_analysis_core::ThermoMechanicalDomain {
        enabled: true,
        reference_temperature_k: 293.15,
        applied_temperature_delta_k: 70.0,
        field_artifact_id: Some("missing_thermo_field".to_string()),
        field_source: None,
        region_temperature_deltas: Vec::new(),
        time_profile: Vec::new(),
    });

    let invalid = analysis_run_transient_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisTransientRunOptions::default(),
        OperationContext::new(
            Some("trace-contract-transient-thermo-field-1".to_string()),
            None,
        ),
    )
    .expect_err("missing thermo field artifact should fail");
    assert_eq!(invalid.operation, "analysis.run_transient");
    assert_eq!(invalid.op_version, "analysis.run_transient/v1");
    assert_eq!(invalid.error_code, "ANALYSIS_RUN_THERMO_FIELD_NOT_FOUND");
}

#[test]
fn analysis_run_nonlinear_contract_is_v1_and_typed() {
    let model = fixture_model(FixtureId::NonlinearAssembly);
    let envelope = analysis_run_nonlinear_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-contract-nonlinear-1".to_string()), None),
    )
    .expect("nonlinear run should succeed for nonlinear fixture");

    assert_eq!(envelope.operation, "analysis.run_nonlinear");
    assert_eq!(envelope.op_version, "analysis.run_nonlinear/v1");
    assert!(envelope.data.nonlinear_results.is_some());
    let nonlinear = envelope
        .data
        .nonlinear_results
        .as_ref()
        .expect("nonlinear payload should be present");
    assert_eq!(nonlinear.load_factors.len(), nonlinear.residual_norms.len());
    assert_eq!(
        nonlinear.residual_norms.len(),
        nonlinear.increment_norms.len()
    );
    assert_eq!(
        nonlinear.increment_norms.len(),
        nonlinear.iteration_counts.len()
    );
    assert!(nonlinear.max_line_search_backtracks_per_increment > 0);
    assert!(nonlinear.iteration_spike_count <= nonlinear.load_factors.len());
    assert!(nonlinear.backtrack_burst_count > 0);
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_NONLINEAR_CONVERGENCE"));
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_NONLINEAR_COST"));

    let results = analysis_results_op(
        &envelope.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(Some("trace-contract-nonlinear-results-1".to_string()), None),
    )
    .expect("analysis.results should succeed for nonlinear run");
    assert!(results.data.summary.increment_count > 0);
    assert!(results.data.summary.failed_increment_count.is_some());
    assert!(results.data.summary.max_nonlinear_increment_norm.is_some());
    assert!(results.data.summary.max_nonlinear_iteration_count.is_some());
    assert!(results
        .data
        .summary
        .nonlinear_line_search_backtracks
        .is_some());
    assert!(results
        .data
        .summary
        .nonlinear_max_backtracks_per_increment
        .is_some());
    assert!(results
        .data
        .summary
        .nonlinear_tangent_rebuild_count
        .is_some());
    assert!(results
        .data
        .summary
        .nonlinear_iteration_spike_count
        .is_some());
    assert!(results
        .data
        .summary
        .nonlinear_convergence_stall_count
        .is_some());
    assert!(results
        .data
        .summary
        .nonlinear_backtrack_burst_count
        .is_some());

    let invalid = analysis_run_nonlinear_op(
        &fixture_model(FixtureId::CantileverLinearStatic),
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-contract-nonlinear-2".to_string()), None),
    )
    .expect_err("nonlinear run should reject models without nonlinear step");
    assert_eq!(invalid.operation, "analysis.run_nonlinear");
    assert_eq!(invalid.op_version, "analysis.run_nonlinear/v1");
    assert_eq!(invalid.error_code, "ANALYSIS_RUN_NONLINEAR_INVALID_MODEL");
}

#[test]
fn analysis_run_electromagnetic_contract_is_v1_typed_payload() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.steps[0].kind = runmat_analysis_core::AnalysisStepKind::Electromagnetic;
    model.electromagnetic = Some(ElectromagneticDomain {
        enabled: true,
        reference_frequency_hz: 60.0,
        applied_current_a: 120.0,
    });

    let envelope = analysis_run_electromagnetic_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-contract-run-em-1".to_string()), None),
    )
    .expect("electromagnetic run should return typed payload");
    assert_eq!(envelope.operation, "analysis.run_electromagnetic");
    assert_eq!(envelope.op_version, "analysis.run_electromagnetic/v1");
    assert_ne!(envelope.data.run_status, RunStatus::Rejected);
    assert!(envelope.data.electromagnetic_results.is_some());
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_EM_STATIC"));
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_EM_SWEEP"));
    let em_payload = envelope
        .data
        .electromagnetic_results
        .as_ref()
        .expect("electromagnetic payload expected");
    assert_eq!(em_payload.sweep_frequency_hz.len(), 1);
    assert_eq!(em_payload.sweep_peak_flux_density.len(), 1);
    assert_eq!(em_payload.sweep_solve_quality.len(), 1);
    assert!(em_payload.resonance_peak_frequency_hz.is_some());
    assert!(em_payload.resonance_q_proxy.is_none());
}

#[test]
fn analysis_results_can_filter_nonlinear_diagnostics_by_code() {
    let model = fixture_model(FixtureId::NonlinearLoadPathMix);
    let envelope = analysis_run_nonlinear_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(
            Some("trace-contract-nonlinear-diagnostics-1".to_string()),
            None,
        ),
    )
    .expect("nonlinear run should succeed");

    let results = analysis_results_op(
        &envelope.data,
        AnalysisResultsQuery {
            include_fields: Vec::new(),
            include_diagnostics: true,
            diagnostic_codes: vec![
                "FEA_NONLINEAR_CONVERGENCE".to_string(),
                "FEA_NONLINEAR_COST".to_string(),
            ],
            include_modal_results: true,
            mode_indices: Vec::new(),
            include_transient_results: true,
            transient_snapshot_indices: Vec::new(),
            include_nonlinear_results: true,
            include_electromagnetic_results: true,
        },
        OperationContext::new(
            Some("trace-contract-nonlinear-diagnostics-2".to_string()),
            None,
        ),
    )
    .expect("results query should succeed");

    let diagnostics = results
        .data
        .diagnostics
        .as_ref()
        .expect("diagnostics should be included");
    assert!(!diagnostics.is_empty());
    assert!(diagnostics.iter().all(|diag| {
        diag.code == "FEA_NONLINEAR_CONVERGENCE" || diag.code == "FEA_NONLINEAR_COST"
    }));
}

#[test]
fn electromagnetic_contract_snapshot_matches_expected_shape() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.steps[0].kind = runmat_analysis_core::AnalysisStepKind::Electromagnetic;
    model.electromagnetic = Some(ElectromagneticDomain {
        enabled: true,
        reference_frequency_hz: 60.0,
        applied_current_a: 120.0,
    });

    let envelope = analysis_run_electromagnetic_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-contract-em-snapshot-1".to_string()), None),
    )
    .expect("electromagnetic run should succeed");
    let results = analysis_results_op(
        &envelope.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(Some("trace-contract-em-snapshot-2".to_string()), None),
    )
    .expect("electromagnetic results should succeed");

    let snapshot_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/electromagnetic_contract_snapshot.json");
    let expected: Value = serde_json::from_str(
        &fs::read_to_string(&snapshot_path).expect("read electromagnetic contract snapshot"),
    )
    .expect("parse electromagnetic contract snapshot");

    let electromagnetic = serde_json::to_value(
        results
            .data
            .electromagnetic_results
            .as_ref()
            .expect("electromagnetic payload expected"),
    )
    .expect("serialize electromagnetic payload");
    let summary = serde_json::to_value(&results.data.summary).expect("serialize summary");

    let expected_em_keys = sorted_object_keys(
        expected
            .get("electromagnetic_results")
            .expect("snapshot electromagnetic_results"),
    );
    let expected_summary_keys =
        sorted_object_keys(expected.get("summary").expect("snapshot summary"));

    assert_eq!(sorted_object_keys(&electromagnetic), expected_em_keys);
    assert_eq!(sorted_object_keys(&summary), expected_summary_keys);
}

#[test]
fn nonlinear_contract_snapshot_matches_expected_shape() {
    let model = fixture_model(FixtureId::NonlinearLoadPathMix);
    let envelope = analysis_run_nonlinear_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(
            Some("trace-contract-nonlinear-snapshot-1".to_string()),
            None,
        ),
    )
    .expect("nonlinear run should succeed");
    let results = analysis_results_op(
        &envelope.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(
            Some("trace-contract-nonlinear-snapshot-2".to_string()),
            None,
        ),
    )
    .expect("results should succeed");

    let snapshot_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/nonlinear_contract_snapshot.json");
    let expected: Value = serde_json::from_str(
        &fs::read_to_string(&snapshot_path).expect("read nonlinear contract snapshot"),
    )
    .expect("parse nonlinear contract snapshot");

    let nonlinear = serde_json::to_value(
        results
            .data
            .nonlinear_results
            .as_ref()
            .expect("nonlinear payload expected"),
    )
    .expect("serialize nonlinear payload");
    let summary = serde_json::to_value(&results.data.summary).expect("serialize summary");

    let expected_nonlinear_keys = sorted_object_keys(
        expected
            .get("nonlinear_results")
            .expect("snapshot nonlinear_results"),
    );
    let expected_summary_keys =
        sorted_object_keys(expected.get("summary").expect("snapshot summary"));

    assert_eq!(sorted_object_keys(&nonlinear), expected_nonlinear_keys);
    assert_eq!(sorted_object_keys(&summary), expected_summary_keys);
}

#[test]
fn analysis_run_nonlinear_strict_iteration_cap_sets_degraded_status() {
    let model = fixture_model(FixtureId::NonlinearAssembly);
    let envelope = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions {
            quality_policy: QualityPolicy::Strict,
            max_newton_iters: 1,
            line_search: false,
            ..AnalysisNonlinearRunOptions::balanced()
        },
        OperationContext::new(
            Some("trace-contract-nonlinear-strict-cap".to_string()),
            None,
        ),
    )
    .expect("nonlinear strict run should return envelope");

    assert_eq!(envelope.operation, "analysis.run_nonlinear");
    assert_eq!(envelope.op_version, "analysis.run_nonlinear/v1");
    assert_eq!(envelope.data.run_status, RunStatus::Degraded);
    assert!(!envelope.data.publishable);
    assert!(envelope
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::NonlinearIncrementFailure));
}

#[test]
fn analysis_run_nonlinear_policy_contract_divergence_is_explicit() {
    let model = fixture_model(FixtureId::NonlinearAssembly);
    let run_with_policy = |policy| {
        analysis_run_nonlinear_with_options_op(
            &model,
            ComputeBackend::Cpu,
            AnalysisNonlinearRunOptions {
                quality_policy: policy,
                max_newton_iters: 1,
                line_search: false,
                max_line_search_backtracks: 0,
                ..AnalysisNonlinearRunOptions::balanced()
            },
            OperationContext::new(
                Some(format!("trace-contract-nonlinear-policy-{policy:?}")),
                None,
            ),
        )
        .expect("nonlinear run should produce typed envelope")
    };

    let exploratory = run_with_policy(QualityPolicy::Exploratory);
    let balanced = run_with_policy(QualityPolicy::Balanced);
    let strict = run_with_policy(QualityPolicy::Strict);

    assert!(exploratory.data.publishable);
    assert_eq!(exploratory.data.run_status, RunStatus::Publishable);

    for degraded in [balanced, strict] {
        assert!(!degraded.data.publishable);
        assert_eq!(degraded.data.run_status, RunStatus::Degraded);
        assert!(degraded
            .data
            .quality_reasons
            .iter()
            .any(|reason| reason.code == QualityReasonCode::NonlinearIncrementFailure));
    }
}

#[test]
fn analysis_run_nonlinear_policy_diverges_on_harder_fixture_profile() {
    let model = fixture_model(FixtureId::NonlinearSofteningProxy);
    let run_with_policy = |policy| {
        analysis_run_nonlinear_with_options_op(
            &model,
            ComputeBackend::Cpu,
            AnalysisNonlinearRunOptions {
                quality_policy: policy,
                max_newton_iters: 3,
                ..AnalysisNonlinearRunOptions::production_recommended()
            },
            OperationContext::new(
                Some(format!("trace-contract-nonlinear-hard-policy-{policy:?}")),
                None,
            ),
        )
        .expect("hard nonlinear fixture should produce typed envelope")
    };

    let exploratory = run_with_policy(QualityPolicy::Exploratory);
    let balanced = run_with_policy(QualityPolicy::Balanced);
    let strict = run_with_policy(QualityPolicy::Strict);

    assert!(exploratory.data.publishable);
    assert_eq!(exploratory.data.run_status, RunStatus::Publishable);
    assert!(!balanced.data.publishable);
    assert_eq!(balanced.data.run_status, RunStatus::Degraded);
    assert!(!strict.data.publishable);
    assert_eq!(strict.data.run_status, RunStatus::Degraded);
}

#[test]
fn analysis_run_nonlinear_prep_reference_errors_are_typed() {
    let model = fixture_model(FixtureId::NonlinearAssembly);
    let missing = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions {
            prep_artifact_id: Some("prep:missing".to_string()),
            ..AnalysisNonlinearRunOptions::production_recommended()
        },
        OperationContext::new(Some("trace-contract-prep-ref-1".to_string()), None),
    )
    .expect_err("missing prep artifact should fail");
    assert_eq!(missing.operation, "analysis.run_nonlinear");
    assert_eq!(missing.op_version, "analysis.run_nonlinear/v1");
    assert_eq!(missing.error_code, "ANALYSIS_RUN_PREP_NOT_FOUND");
}

#[test]
fn analysis_run_nonlinear_stale_prep_reference_is_typed() {
    runmat_runtime::geometry::reset_prep_artifact_store_for_tests();
    std::env::set_var("RUNMAT_GEOMETRY_PREP_REQUIRE_LATEST_REVISION", "true");

    let mut geometry_v1 = geometry_load_op(
        "/assembly.step",
        SIMPLE_STEP.as_bytes(),
        OperationContext::new(Some("trace-contract-prep-stale-1".to_string()), None),
    )
    .expect("geometry load should succeed")
    .data;
    geometry_v1.revision = 1;
    let mut geometry_v2 = geometry_v1.clone();
    geometry_v2.revision = 2;

    let prep_v1 = geometry_prep_for_analysis_op(
        &geometry_v1,
        GeometryPrepForAnalysisSpec::default(),
        OperationContext::new(Some("trace-contract-prep-stale-2".to_string()), None),
    )
    .expect("prep v1 should succeed");
    let _prep_v2 = geometry_prep_for_analysis_op(
        &geometry_v2,
        GeometryPrepForAnalysisSpec::default(),
        OperationContext::new(Some("trace-contract-prep-stale-3".to_string()), None),
    )
    .expect("prep v2 should succeed");

    let model = analysis_create_model_op(
        &geometry_v1,
        AnalysisCreateModelIntentSpec {
            model_id: "contract_stale_prep_model".to_string(),
            profile: AnalysisCreateModelProfile::NonlinearStructural,
            prep_context: None,
        },
        OperationContext::new(Some("trace-contract-prep-stale-4".to_string()), None),
    )
    .expect("create model should succeed")
    .data;

    let stale = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions {
            prep_artifact_id: Some(prep_v1.data.prep_artifact_id),
            ..AnalysisNonlinearRunOptions::production_recommended()
        },
        OperationContext::new(Some("trace-contract-prep-stale-5".to_string()), None),
    )
    .expect_err("stale prep artifact should fail");
    assert_eq!(stale.operation, "analysis.run_nonlinear");
    assert_eq!(stale.op_version, "analysis.run_nonlinear/v1");
    assert_eq!(stale.error_code, "ANALYSIS_RUN_PREP_STALE");

    std::env::remove_var("RUNMAT_GEOMETRY_PREP_REQUIRE_LATEST_REVISION");
    runmat_runtime::geometry::reset_prep_artifact_store_for_tests();
}

#[test]
fn analysis_results_compare_contract_is_v1_and_handles_missing_run_ids() {
    let model = fixture_model(FixtureId::NonlinearAssembly);
    let baseline = analysis_run_nonlinear_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-contract-compare-1".to_string()), None),
    )
    .expect("baseline nonlinear run should succeed");
    let candidate = analysis_run_nonlinear_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-contract-compare-2".to_string()), None),
    )
    .expect("candidate nonlinear run should succeed");

    let compare = analysis_results_compare_op(
        AnalysisResultsCompareQuery {
            baseline_run_id: baseline.data.run_id.clone(),
            candidate_run_id: candidate.data.run_id.clone(),
        },
        OperationContext::new(Some("trace-contract-compare-3".to_string()), None),
    )
    .expect("compare should succeed");
    assert_eq!(compare.operation, "analysis.results_compare");
    assert_eq!(compare.op_version, "analysis.results_compare/v1");
    assert!(compare.data.solve_ms_delta.is_some());

    let missing = analysis_results_compare_op(
        AnalysisResultsCompareQuery {
            baseline_run_id: "missing-run-id".to_string(),
            candidate_run_id: candidate.data.run_id,
        },
        OperationContext::new(Some("trace-contract-compare-4".to_string()), None),
    )
    .expect_err("missing baseline run should fail");
    assert_eq!(missing.operation, "analysis.results_compare");
    assert_eq!(missing.op_version, "analysis.results_compare/v1");
    assert_eq!(missing.error_code, "ANALYSIS_RESULTS_RUN_NOT_FOUND");
}

#[test]
fn analysis_trends_contract_is_v1_and_typed() {
    let model = fixture_model(FixtureId::NonlinearAssembly);
    for idx in 0..3 {
        let _ = analysis_run_nonlinear_op(
            &model,
            ComputeBackend::Cpu,
            OperationContext::new(Some(format!("trace-contract-trends-{idx}")), None),
        )
        .expect("nonlinear run should succeed for trends");
    }

    let trends = analysis_trends_op(
        AnalysisTrendsQuery { window_size: 2 },
        OperationContext::new(Some("trace-contract-trends-op".to_string()), None),
    )
    .expect("trends operation should succeed");
    assert_eq!(trends.operation, "analysis.trends");
    assert_eq!(trends.op_version, "analysis.trends/v1");
    assert_eq!(trends.data.window_size, 2);
    let nonlinear = trends
        .data
        .summaries
        .iter()
        .find(|summary| summary.run_kind == AnalysisRunKind::Nonlinear)
        .expect("nonlinear trend summary should be present");
    assert_eq!(nonlinear.sample_count, 2);
    assert!(nonlinear.median_solve_ms.is_some());
    assert!(nonlinear.p95_solve_ms.is_some());
    assert!(nonlinear.thermo_coupling_enabled_rate.is_none());
    assert!(nonlinear.thermo_transient_warn_rate.is_none());
    assert!(nonlinear.thermo_nonlinear_warn_rate.is_none());
    assert!(nonlinear.thermo_spread_breach_rate.is_none());
    assert!(nonlinear.thermo_heterogeneity_breach_rate.is_none());
}

#[test]
fn analysis_run_transient_with_options_contract_controls_execution_window() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.steps = vec![runmat_analysis_core::AnalysisStep {
        step_id: "transient_1".to_string(),
        kind: runmat_analysis_core::AnalysisStepKind::Transient,
    }];

    let envelope = analysis_run_transient_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisTransientRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Balanced,
            time_step_s: 1.5e-3,
            min_time_step_s: 1.5e-3,
            max_time_step_s: 1.5e-3,
            step_count: 4,
            max_linear_iters: 64,
            tolerance: 1.0e-8,
            residual_target: 1.0e-6,
            adaptive_time_step: false,
            max_step_retries: 0,
            adapt_min_scale: 0.8,
            adapt_max_scale: 1.25,
            adapt_growth_exponent: 0.35,
            adapt_retry_growth_cap: 1.05,
            adapt_nonconverged_shrink: 0.75,
            dt_bucket_rel_tolerance: 0.0,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
        },
        OperationContext::new(Some("trace-contract-transient-opts-1".to_string()), None),
    )
    .expect("transient run with options should succeed");

    assert_eq!(envelope.operation, "analysis.run_transient");
    assert_eq!(envelope.op_version, "analysis.run_transient/v1");
    let transient = envelope
        .data
        .transient_results
        .as_ref()
        .expect("transient payload should exist");
    assert_eq!(transient.time_points_s.len(), 5);
    assert!((transient.time_points_s[4] - 6.0e-3).abs() < 1.0e-12);
    assert!(envelope.data.provenance.deterministic_mode);
}

#[test]
fn analysis_modal_large_fixture_quality_signals_propagate_to_results() {
    let model = fixture_model(FixtureId::ModalLarge);
    let envelope = analysis_run_modal_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisModalRunOptions {
            mode_count: 8,
            ..AnalysisModalRunOptions::balanced()
        },
        OperationContext::new(Some("trace-contract-modal-large-1".to_string()), None),
    )
    .expect("modal large fixture run should succeed");

    let has_orthogonality_warning = envelope.data.run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_MODAL_ORTHOGONALITY"
            && diag.severity == runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Warning
    });
    let has_separation_warning = envelope.data.run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_MODAL_SEPARATION"
            && diag.severity == runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Warning
    });

    assert_eq!(
        envelope
            .data
            .quality_reasons
            .iter()
            .any(|reason| reason.code == QualityReasonCode::ModalOrthogonalityExceeded),
        has_orthogonality_warning
    );
    assert_eq!(
        envelope
            .data
            .quality_reasons
            .iter()
            .any(|reason| reason.code == QualityReasonCode::ModalSeparationLow),
        has_separation_warning
    );
    if has_orthogonality_warning || has_separation_warning {
        assert!(!envelope.data.publishable);
        assert_eq!(envelope.data.run_status, RunStatus::Degraded);
    }

    let results = analysis_results_op(
        &envelope.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(Some("trace-contract-modal-large-2".to_string()), None),
    )
    .expect("modal large fixture results should succeed");

    assert!(results.data.summary.mode_count >= 2);
    assert_eq!(
        results.data.summary.mode_count,
        results.data.summary.available_mode_indices.len()
    );
    assert!(results.data.summary.max_modal_residual_norm.is_some());
    assert!(results.data.summary.first_mode_converged.is_some());
    assert!(results
        .data
        .diagnostics
        .as_ref()
        .expect("diagnostics should be included")
        .iter()
        .any(|diag| diag.code == "FEA_MODAL_ORTHOGONALITY"));
    assert!(results
        .data
        .diagnostics
        .as_ref()
        .expect("diagnostics should be included")
        .iter()
        .any(|diag| diag.code == "FEA_MODAL_SEPARATION"));
}

#[test]
fn analysis_transient_long_fixture_quality_signals_propagate_to_results() {
    let model = fixture_model(FixtureId::TransientLong);
    let envelope = analysis_run_transient_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisTransientRunOptions {
            step_count: 24,
            ..AnalysisTransientRunOptions::balanced()
        },
        OperationContext::new(Some("trace-contract-transient-long-1".to_string()), None),
    )
    .expect("transient long fixture run should succeed");

    let has_stability_warning = envelope.data.run.diagnostics.iter().any(|diag| {
        (diag.code == "FEA_TRANSIENT_STABILITY" || diag.code == "FEA_TRANSIENT_ENERGY")
            && diag.severity == runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Warning
    });
    let has_step_failure_warning = envelope.data.run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_TRANSIENT_STEP_FAILURE"
            && diag.severity == runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Warning
    });

    assert_eq!(
        envelope
            .data
            .quality_reasons
            .iter()
            .any(|reason| reason.code == QualityReasonCode::TransientStabilityExceeded),
        has_stability_warning
    );
    assert_eq!(
        envelope
            .data
            .quality_reasons
            .iter()
            .any(|reason| reason.code == QualityReasonCode::TransientStepFailure),
        has_step_failure_warning
    );
    if has_stability_warning || has_step_failure_warning {
        assert!(!envelope.data.publishable);
        assert_eq!(envelope.data.run_status, RunStatus::Degraded);
    }

    let results = analysis_results_op(
        &envelope.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(Some("trace-contract-transient-long-2".to_string()), None),
    )
    .expect("transient long fixture results should succeed");

    assert!(results.data.summary.snapshot_count > 8);
    assert_eq!(results.data.summary.time_start_s, Some(0.0));
    assert!(results.data.summary.time_end_s.unwrap_or(0.0) > 0.0);
    assert!(results.data.summary.max_transient_residual_norm.is_some());
    assert!(results.data.summary.final_step_converged.is_some());
    assert!(results
        .data
        .diagnostics
        .as_ref()
        .expect("diagnostics should be included")
        .iter()
        .any(|diag| diag.code == "FEA_TRANSIENT_STABILITY"));
    assert!(results
        .data
        .diagnostics
        .as_ref()
        .expect("diagnostics should be included")
        .iter()
        .any(|diag| diag.code == "FEA_TRANSIENT_ENERGY"));
}

#[test]
fn analysis_results_contract_is_v1_and_filterable() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let run = analysis_run_linear_static_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-contract-3b-run".to_string()), None),
    )
    .expect("run should succeed");

    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery {
            include_fields: vec!["von_mises".to_string()],
            include_diagnostics: false,
            diagnostic_codes: Vec::new(),
            include_modal_results: true,
            mode_indices: Vec::new(),
            include_transient_results: true,
            transient_snapshot_indices: Vec::new(),
            include_nonlinear_results: true,
            include_electromagnetic_results: true,
        },
        OperationContext::new(Some("trace-contract-3b-results".to_string()), None),
    )
    .expect("results should succeed");

    assert_eq!(results.operation, "analysis.results");
    assert_eq!(results.op_version, "analysis.results/v1");
    assert_eq!(results.data.fields.len(), 1);
    assert_eq!(results.data.fields[0].field_id, "von_mises");
    assert!(results.data.diagnostics.is_none());
}

#[test]
fn analysis_results_unknown_field_maps_typed_error_contract() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let run = analysis_run_linear_static_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-contract-3c-run".to_string()), None),
    )
    .expect("run should succeed");

    let err = analysis_results_op(
        &run.data,
        AnalysisResultsQuery {
            include_fields: vec!["nonexistent".to_string()],
            include_diagnostics: true,
            diagnostic_codes: Vec::new(),
            include_modal_results: true,
            mode_indices: Vec::new(),
            include_transient_results: true,
            transient_snapshot_indices: Vec::new(),
            include_nonlinear_results: true,
            include_electromagnetic_results: true,
        },
        OperationContext::new(Some("trace-contract-3c-results".to_string()), None),
    )
    .expect_err("results should fail");

    assert_eq!(err.operation, "analysis.results");
    assert_eq!(err.op_version, "analysis.results/v1");
    assert_eq!(err.error_code, "ANALYSIS_RESULTS_FIELD_NOT_FOUND");
}

#[test]
fn analysis_results_modal_query_controls_are_typed() {
    let geometry = geometry_load_op(
        "/part.stl",
        TRIANGLE_STL.as_bytes(),
        OperationContext::new(Some("trace-contract-modal-results-1".to_string()), None),
    )
    .expect("geometry load should succeed");
    let modal_model = analysis_create_model_op(
        &geometry.data,
        AnalysisCreateModelIntentSpec {
            model_id: "contract_modal_results_model".to_string(),
            profile: AnalysisCreateModelProfile::ModalStructural,
            prep_context: None,
        },
        OperationContext::new(Some("trace-contract-modal-results-2".to_string()), None),
    )
    .expect("modal model should be created");
    let modal_run = analysis_run_modal_op(
        &modal_model.data,
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-contract-modal-results-3".to_string()), None),
    )
    .expect("modal run should succeed");

    let excluded = analysis_results_op(
        &modal_run.data,
        AnalysisResultsQuery {
            include_fields: Vec::new(),
            include_diagnostics: true,
            diagnostic_codes: Vec::new(),
            include_modal_results: false,
            mode_indices: Vec::new(),
            include_transient_results: true,
            transient_snapshot_indices: Vec::new(),
            include_nonlinear_results: true,
            include_electromagnetic_results: true,
        },
        OperationContext::new(Some("trace-contract-modal-results-4".to_string()), None),
    )
    .expect("results should succeed");
    assert!(excluded.data.modal_results.is_none());
    assert!(excluded.data.summary.mode_count > 0);
    assert_eq!(
        excluded.data.summary.mode_count,
        excluded.data.summary.available_mode_indices.len()
    );
    assert!(excluded.data.summary.min_frequency_hz.is_some());
    assert!(excluded.data.summary.max_frequency_hz.is_some());
    assert!(excluded.data.summary.max_modal_residual_norm.is_some());
    assert!(excluded.data.summary.first_mode_converged.is_some());
    assert_eq!(excluded.data.summary.snapshot_count, 0);
    assert_eq!(excluded.data.summary.time_start_s, None);
    assert_eq!(excluded.data.summary.time_end_s, None);
    assert_eq!(excluded.data.summary.max_transient_residual_norm, None);
    assert_eq!(excluded.data.summary.final_step_converged, None);

    let invalid_mode = analysis_results_op(
        &modal_run.data,
        AnalysisResultsQuery {
            include_fields: Vec::new(),
            include_diagnostics: true,
            diagnostic_codes: Vec::new(),
            include_modal_results: true,
            mode_indices: vec![99],
            include_transient_results: true,
            transient_snapshot_indices: Vec::new(),
            include_nonlinear_results: true,
            include_electromagnetic_results: true,
        },
        OperationContext::new(Some("trace-contract-modal-results-5".to_string()), None),
    )
    .expect_err("unknown mode index should fail");
    assert_eq!(invalid_mode.error_code, "ANALYSIS_RESULTS_MODE_NOT_FOUND");
    assert_eq!(invalid_mode.operation, "analysis.results");
    assert_eq!(invalid_mode.op_version, "analysis.results/v1");
}

#[test]
fn analysis_results_transient_query_controls_are_typed() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.steps = vec![runmat_analysis_core::AnalysisStep {
        step_id: "transient_1".to_string(),
        kind: runmat_analysis_core::AnalysisStepKind::Transient,
    }];
    let transient_run = analysis_run_transient_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-contract-transient-results-1".to_string()), None),
    )
    .expect("transient run should succeed");

    let excluded = analysis_results_op(
        &transient_run.data,
        AnalysisResultsQuery {
            include_fields: Vec::new(),
            include_diagnostics: true,
            diagnostic_codes: Vec::new(),
            include_modal_results: true,
            mode_indices: Vec::new(),
            include_transient_results: false,
            transient_snapshot_indices: Vec::new(),
            include_nonlinear_results: true,
            include_electromagnetic_results: true,
        },
        OperationContext::new(Some("trace-contract-transient-results-2".to_string()), None),
    )
    .expect("results should succeed");
    assert!(excluded.data.transient_results.is_none());
    assert!(excluded.data.summary.snapshot_count > 0);
    assert_eq!(excluded.data.summary.time_start_s, Some(0.0));
    assert!(excluded.data.summary.time_end_s.unwrap_or(0.0) > 0.0);
    assert!(excluded.data.summary.max_transient_residual_norm.is_some());
    assert!(excluded.data.summary.final_step_converged.is_some());

    let invalid_snapshot = analysis_results_op(
        &transient_run.data,
        AnalysisResultsQuery {
            include_fields: Vec::new(),
            include_diagnostics: true,
            diagnostic_codes: Vec::new(),
            include_modal_results: true,
            mode_indices: Vec::new(),
            include_transient_results: true,
            transient_snapshot_indices: vec![999],
            include_nonlinear_results: true,
            include_electromagnetic_results: true,
        },
        OperationContext::new(Some("trace-contract-transient-results-3".to_string()), None),
    )
    .expect_err("unknown transient snapshot index should fail");
    assert_eq!(
        invalid_snapshot.error_code,
        "ANALYSIS_RESULTS_TRANSIENT_SNAPSHOT_NOT_FOUND"
    );
    assert_eq!(invalid_snapshot.operation, "analysis.results");
    assert_eq!(invalid_snapshot.op_version, "analysis.results/v1");
}

#[test]
fn analysis_results_by_run_id_contract_roundtrip() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let run = analysis_run_linear_static_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-contract-results-id-run".to_string()), None),
    )
    .expect("run should succeed");

    let results = analysis_results_by_run_id_op(
        &run.data.run_id,
        AnalysisResultsQuery::default(),
        OperationContext::new(Some("trace-contract-results-id-get".to_string()), None),
    )
    .expect("results by run id should succeed");

    assert_eq!(results.operation, "analysis.results");
    assert_eq!(results.op_version, "analysis.results/v1");
    assert_eq!(results.data.summary.field_count, 2);
    assert_eq!(results.data.summary.mode_count, 0);
    assert!(results.data.summary.available_mode_indices.is_empty());
    assert_eq!(results.data.summary.min_frequency_hz, None);
    assert_eq!(results.data.summary.max_frequency_hz, None);
    assert_eq!(results.data.summary.max_modal_residual_norm, None);
    assert_eq!(results.data.summary.first_mode_converged, None);
    assert_eq!(results.data.summary.snapshot_count, 0);
    assert_eq!(results.data.summary.time_start_s, None);
    assert_eq!(results.data.summary.time_end_s, None);
    assert_eq!(results.data.summary.max_transient_residual_norm, None);
    assert_eq!(results.data.summary.final_step_converged, None);
    assert_eq!(results.data.summary.thermo_coupling_enabled, None);
    assert_eq!(results.data.summary.thermo_coupling_fingerprint, None);
    assert_eq!(
        results.data.summary.thermo_constitutive_temperature_factor,
        None
    );
    assert_eq!(results.data.summary.thermo_effective_modulus_scale, None);
    assert_eq!(
        results
            .data
            .summary
            .thermo_constitutive_material_spread_ratio,
        None
    );
    assert_eq!(
        results.data.summary.thermo_assignment_heterogeneity_index,
        None
    );
    assert_eq!(results.data.summary.thermo_transient_severity, None);
    assert_eq!(results.data.summary.thermo_nonlinear_severity, None);
}

#[test]
fn analysis_results_by_run_id_missing_maps_typed_error_contract() {
    let err = analysis_results_by_run_id_op(
        "run_does_not_exist",
        AnalysisResultsQuery::default(),
        OperationContext::new(Some("trace-contract-results-id-missing".to_string()), None),
    )
    .expect_err("results by run id should fail");

    assert_eq!(err.operation, "analysis.results");
    assert_eq!(err.op_version, "analysis.results/v1");
    assert_eq!(err.error_code, "ANALYSIS_RESULTS_RUN_NOT_FOUND");
}

#[test]
fn analysis_run_contract_maps_fixture_validation_failures() {
    let missing_materials = fixture_model(FixtureId::MissingMaterials);
    let err = analysis_run_linear_static_op(
        &missing_materials,
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-contract-4".to_string()), None),
    )
    .expect_err("run should fail");
    assert_eq!(err.operation, "analysis.run_linear_static");
    assert_eq!(err.op_version, "analysis.run_linear_static/v1");
    assert_eq!(err.error_code, "SOLVER_MODEL_INVALID");
    assert_eq!(
        err.context.get("analysis_model_id").map(|s| s.as_str()),
        Some("missing_materials")
    );

    let missing_loads = fixture_model(FixtureId::MissingLoads);
    let err = analysis_run_linear_static_op(
        &missing_loads,
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-contract-5".to_string()), None),
    )
    .expect_err("run should fail");
    assert_eq!(err.error_code, "SOLVER_MODEL_INVALID");
    assert_eq!(
        err.context.get("analysis_model_id").map(|s| s.as_str()),
        Some("missing_loads")
    );
}

#[test]
fn multi_material_confidence_mismatch_degrades_publishability() {
    let model = fixture_model(FixtureId::MultiMaterialAssembly);
    let envelope = analysis_run_linear_static_op(
        &model,
        ComputeBackend::Gpu,
        OperationContext::new(Some("trace-contract-multi-material".to_string()), None),
    )
    .expect("run should succeed");

    assert_eq!(envelope.data.run_status, RunStatus::Degraded);
    assert!(!envelope.data.publishable);
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "ANALYSIS_MATERIAL_ASSIGNMENT_CONFLICT_INFERRED"));
}

#[test]
fn analysis_run_deterministic_contract_is_stable_across_replays() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let options = AnalysisRunOptions {
        deterministic_mode: true,
        precision_mode: PrecisionMode::Fp64,
        preconditioner_mode: PreconditionerMode::Auto,
        quality_policy: QualityPolicy::Balanced,
        prep_context: None,
        prep_artifact_id: None,
        prep_calibration_profile: None,
    };

    let first = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        options.clone(),
        OperationContext::new(Some("trace-contract-6a".to_string()), None),
    )
    .expect("first deterministic run should succeed");
    let second = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        options.clone(),
        OperationContext::new(Some("trace-contract-6b".to_string()), None),
    )
    .expect("second deterministic run should succeed");

    assert_eq!(first.data.run_status, RunStatus::Publishable);
    assert_eq!(second.data.run_status, RunStatus::Publishable);
    assert!(first.data.publishable);
    assert!(second.data.publishable);
    assert_eq!(first.data.provenance.precision_mode, "fp64");
    assert_eq!(second.data.provenance.precision_mode, "fp64");
    assert!(first.data.provenance.deterministic_mode);
    assert!(second.data.provenance.deterministic_mode);
    assert_eq!(
        first.data.run.displacement_field,
        second.data.run.displacement_field
    );
    assert_eq!(
        first.data.run.von_mises_field,
        second.data.run.von_mises_field
    );
    assert_eq!(first.data.run.diagnostics, second.data.run.diagnostics);
}

#[test]
fn analysis_run_backend_selection_is_recorded_in_provenance() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let options = AnalysisRunOptions::default();

    let cpu = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        options.clone(),
        OperationContext::new(Some("trace-contract-7a".to_string()), None),
    )
    .expect("cpu run should succeed");

    let gpu = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Gpu,
        options.clone(),
        OperationContext::new(Some("trace-contract-7b".to_string()), None),
    )
    .expect("gpu run should succeed");

    assert_eq!(cpu.data.provenance.backend, ComputeBackend::Cpu);
    assert_eq!(gpu.data.provenance.backend, ComputeBackend::Gpu);
    assert_eq!(cpu.data.provenance.solver_device_apply_k_ratio, 0.0);
    assert_eq!(cpu.data.provenance.solver_host_sync_count, 0);
    assert!(
        (0.0..=1.0).contains(&gpu.data.provenance.solver_device_apply_k_ratio),
        "gpu ratio must be in [0,1]"
    );
    assert!(matches!(
        gpu.data.provenance.solver_backend.as_str(),
        "runtime_tensor" | "cpu_reference"
    ));
    assert_eq!(cpu.data.run_status, RunStatus::Publishable);
    assert_eq!(gpu.data.run_status, RunStatus::Publishable);
}

#[test]
fn analysis_run_gpu_without_provider_records_fallback_contract() {
    let _guard = runmat_accelerate_api::ThreadProviderGuard::set(None);
    let model = fixture_model(FixtureId::CantileverLinearStatic);

    let envelope = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Gpu,
        AnalysisRunOptions::default(),
        OperationContext::new(Some("trace-contract-8".to_string()), None),
    )
    .expect("gpu run should succeed with host fallback");

    if envelope.data.provenance.solver_backend == "cpu_reference" {
        assert!(envelope
            .data
            .provenance
            .fallback_events
            .iter()
            .any(|event| event.starts_with("SOLVER_BACKEND_FALLBACK")));
        assert_eq!(envelope.data.provenance.solver_device_apply_k_ratio, 0.0);
    } else {
        assert_eq!(envelope.data.provenance.solver_backend, "runtime_tensor");
    }
    for event in &envelope.data.provenance.fallback_events {
        assert_fallback_event_schema(event);
    }
    assert!(matches!(
        envelope.data.run.displacement_field.values,
        AnalysisFieldValues::HostF64(_)
    ));
}

#[test]
fn analysis_run_gpu_with_provider_emits_device_ref_contract() {
    struct ContractTestProvider;

    impl runmat_accelerate_api::AccelProvider for ContractTestProvider {
        fn upload(
            &self,
            host: &runmat_accelerate_api::HostTensorView,
        ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
            static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(2000);
            Ok(runmat_accelerate_api::GpuTensorHandle {
                shape: host.shape.to_vec(),
                device_id: 11,
                buffer_id: NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed),
            })
        }

        fn download<'a>(
            &'a self,
            h: &'a runmat_accelerate_api::GpuTensorHandle,
        ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
            Box::pin(async move {
                Ok(runmat_accelerate_api::HostTensorOwned {
                    data: vec![0.0; h.shape.iter().product()],
                    shape: h.shape.clone(),
                })
            })
        }

        fn free(&self, _h: &runmat_accelerate_api::GpuTensorHandle) -> anyhow::Result<()> {
            Ok(())
        }

        fn device_info(&self) -> String {
            "contract-test-provider".to_string()
        }

        fn device_id(&self) -> u32 {
            11
        }

        fn device_info_struct(&self) -> runmat_accelerate_api::ApiDeviceInfo {
            runmat_accelerate_api::ApiDeviceInfo {
                device_id: 11,
                name: "contract-test-provider".to_string(),
                vendor: "runmat-tests".to_string(),
                memory_bytes: None,
                backend: Some("contract_gpu".to_string()),
            }
        }
    }

    static PROVIDER: ContractTestProvider = ContractTestProvider;
    let _guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(&PROVIDER));
    let model = fixture_model(FixtureId::CantileverLinearStatic);

    let envelope = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Gpu,
        AnalysisRunOptions::default(),
        OperationContext::new(Some("trace-contract-9".to_string()), None),
    )
    .expect("gpu run should succeed");

    assert!(!envelope
        .data
        .provenance
        .fallback_events
        .iter()
        .any(|event| event.starts_with("BACKEND_NO_PROVIDER")
            || event.starts_with("BACKEND_UPLOAD_FAILED")));
    if envelope.data.provenance.solver_backend == "cpu_reference" {
        assert!(envelope
            .data
            .provenance
            .fallback_events
            .iter()
            .any(|event| event.starts_with("SOLVER_BACKEND_FALLBACK")));
    }
    assert!(matches!(
        envelope.data.run.displacement_field.values,
        AnalysisFieldValues::DeviceRef(_)
    ));
    assert!(matches!(
        envelope.data.run.von_mises_field.values,
        AnalysisFieldValues::DeviceRef(_)
    ));
}

#[test]
fn analysis_run_gpu_upload_failure_records_fallback_contract() {
    struct UploadFailProvider;

    impl runmat_accelerate_api::AccelProvider for UploadFailProvider {
        fn upload(
            &self,
            _host: &runmat_accelerate_api::HostTensorView,
        ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
            Err(anyhow::anyhow!("forced-upload-failure"))
        }

        fn download<'a>(
            &'a self,
            h: &'a runmat_accelerate_api::GpuTensorHandle,
        ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
            Box::pin(async move {
                Ok(runmat_accelerate_api::HostTensorOwned {
                    data: vec![0.0; h.shape.iter().product()],
                    shape: h.shape.clone(),
                })
            })
        }

        fn free(&self, _h: &runmat_accelerate_api::GpuTensorHandle) -> anyhow::Result<()> {
            Ok(())
        }

        fn device_info(&self) -> String {
            "upload-fail-provider".to_string()
        }
    }

    static PROVIDER: UploadFailProvider = UploadFailProvider;
    let _guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(&PROVIDER));
    let model = fixture_model(FixtureId::CantileverLinearStatic);

    let envelope = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Gpu,
        AnalysisRunOptions::default(),
        OperationContext::new(Some("trace-contract-10".to_string()), None),
    )
    .expect("gpu run should still succeed with fallback");

    assert!(envelope
        .data
        .provenance
        .fallback_events
        .iter()
        .any(|event| event.starts_with("BACKEND_UPLOAD_FAILED:displacement")));
    for event in &envelope.data.provenance.fallback_events {
        assert_fallback_event_schema(event);
    }
    assert!(matches!(
        envelope.data.run.displacement_field.values,
        AnalysisFieldValues::HostF64(_)
    ));
}

#[test]
fn strict_policy_quality_reasons_propagate_to_results_contracts() {
    let model = fixture_model(FixtureId::MultiMaterialAssembly);
    let run = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        AnalysisRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            preconditioner_mode: PreconditionerMode::Auto,
            quality_policy: QualityPolicy::Strict,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
        },
        OperationContext::new(Some("trace-contract-11-run".to_string()), None),
    )
    .expect("run should succeed");

    assert_eq!(run.data.run_status, RunStatus::Degraded);
    assert!(!run.data.publishable);
    assert_eq!(run.data.provenance.quality_policy, "strict");
    assert!(run
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::MaterialAssignmentConflict));

    let direct_results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(Some("trace-contract-11-results".to_string()), None),
    )
    .expect("results should succeed");
    assert_eq!(direct_results.data.run_status, RunStatus::Degraded);
    assert!(!direct_results.data.publishable);
    assert_eq!(direct_results.data.provenance.quality_policy, "strict");
    assert!(direct_results
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::MaterialAssignmentConflict));

    let by_id_results = analysis_results_by_run_id_op(
        &run.data.run_id,
        AnalysisResultsQuery::default(),
        OperationContext::new(Some("trace-contract-11-results-id".to_string()), None),
    )
    .expect("results by run_id should succeed");
    assert_eq!(by_id_results.data.run_status, RunStatus::Degraded);
    assert!(!by_id_results.data.publishable);
    assert_eq!(by_id_results.data.provenance.quality_policy, "strict");
    assert!(by_id_results
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::MaterialAssignmentConflict));
}

#[test]
fn balanced_and_strict_diverge_for_same_field_promotion_fallback() {
    struct UploadFailProvider;

    impl runmat_accelerate_api::AccelProvider for UploadFailProvider {
        fn upload(
            &self,
            _host: &runmat_accelerate_api::HostTensorView,
        ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
            Err(anyhow::anyhow!("forced-upload-failure"))
        }

        fn download<'a>(
            &'a self,
            h: &'a runmat_accelerate_api::GpuTensorHandle,
        ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
            Box::pin(async move {
                Ok(runmat_accelerate_api::HostTensorOwned {
                    data: vec![0.0; h.shape.iter().product()],
                    shape: h.shape.clone(),
                })
            })
        }

        fn free(&self, _h: &runmat_accelerate_api::GpuTensorHandle) -> anyhow::Result<()> {
            Ok(())
        }

        fn device_info(&self) -> String {
            "upload-fail-provider".to_string()
        }
    }

    static PROVIDER: UploadFailProvider = UploadFailProvider;
    let _guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(&PROVIDER));
    let model = fixture_model(FixtureId::CantileverLinearStatic);

    let balanced = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Gpu,
        AnalysisRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            preconditioner_mode: PreconditionerMode::Auto,
            quality_policy: QualityPolicy::Balanced,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
        },
        OperationContext::new(Some("trace-contract-12-balanced".to_string()), None),
    )
    .expect("balanced run should succeed");

    let strict = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Gpu,
        AnalysisRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            preconditioner_mode: PreconditionerMode::Auto,
            quality_policy: QualityPolicy::Strict,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
        },
        OperationContext::new(Some("trace-contract-12-strict".to_string()), None),
    )
    .expect("strict run should succeed");

    assert_eq!(
        balanced.data.solver_convergence,
        strict.data.solver_convergence
    );
    assert_eq!(balanced.data.result_quality, strict.data.result_quality);
    assert_eq!(balanced.data.provenance.quality_policy, "balanced");
    assert_eq!(strict.data.provenance.quality_policy, "strict");
    assert!(balanced
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::FieldPromotionFallback));
    assert!(strict
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::FieldPromotionFallback));

    assert!(balanced.data.publishable);
    assert_eq!(balanced.data.run_status, RunStatus::Publishable);
    assert!(!strict.data.publishable);
    assert_eq!(strict.data.run_status, RunStatus::Degraded);
}

#[test]
fn balanced_and_strict_divergence_propagates_through_results_endpoints() {
    struct UploadFailProvider;

    impl runmat_accelerate_api::AccelProvider for UploadFailProvider {
        fn upload(
            &self,
            _host: &runmat_accelerate_api::HostTensorView,
        ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
            Err(anyhow::anyhow!("forced-upload-failure"))
        }

        fn download<'a>(
            &'a self,
            h: &'a runmat_accelerate_api::GpuTensorHandle,
        ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
            Box::pin(async move {
                Ok(runmat_accelerate_api::HostTensorOwned {
                    data: vec![0.0; h.shape.iter().product()],
                    shape: h.shape.clone(),
                })
            })
        }

        fn free(&self, _h: &runmat_accelerate_api::GpuTensorHandle) -> anyhow::Result<()> {
            Ok(())
        }

        fn device_info(&self) -> String {
            "upload-fail-provider".to_string()
        }
    }

    static PROVIDER: UploadFailProvider = UploadFailProvider;
    let _guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(&PROVIDER));
    let model = fixture_model(FixtureId::CantileverLinearStatic);

    let balanced_run = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Gpu,
        AnalysisRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            preconditioner_mode: PreconditionerMode::Auto,
            quality_policy: QualityPolicy::Balanced,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
        },
        OperationContext::new(Some("trace-contract-13-balanced-run".to_string()), None),
    )
    .expect("balanced run should succeed");
    let strict_run = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Gpu,
        AnalysisRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            preconditioner_mode: PreconditionerMode::Auto,
            quality_policy: QualityPolicy::Strict,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
        },
        OperationContext::new(Some("trace-contract-13-strict-run".to_string()), None),
    )
    .expect("strict run should succeed");

    let balanced_results = analysis_results_op(
        &balanced_run.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(Some("trace-contract-13-balanced-results".to_string()), None),
    )
    .expect("balanced results should succeed");
    let strict_results = analysis_results_op(
        &strict_run.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(Some("trace-contract-13-strict-results".to_string()), None),
    )
    .expect("strict results should succeed");

    assert_eq!(balanced_results.data.run_status, RunStatus::Publishable);
    assert!(balanced_results.data.publishable);
    assert_eq!(balanced_results.data.provenance.quality_policy, "balanced");
    assert!(balanced_results
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::FieldPromotionFallback));

    assert_eq!(strict_results.data.run_status, RunStatus::Degraded);
    assert!(!strict_results.data.publishable);
    assert_eq!(strict_results.data.provenance.quality_policy, "strict");
    assert!(strict_results
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::FieldPromotionFallback));

    let balanced_by_id = analysis_results_by_run_id_op(
        &balanced_run.data.run_id,
        AnalysisResultsQuery::default(),
        OperationContext::new(Some("trace-contract-13-balanced-by-id".to_string()), None),
    )
    .expect("balanced by-id results should succeed");
    let strict_by_id = analysis_results_by_run_id_op(
        &strict_run.data.run_id,
        AnalysisResultsQuery::default(),
        OperationContext::new(Some("trace-contract-13-strict-by-id".to_string()), None),
    )
    .expect("strict by-id results should succeed");

    assert_eq!(balanced_by_id.data.run_status, RunStatus::Publishable);
    assert!(balanced_by_id.data.publishable);
    assert_eq!(balanced_by_id.data.provenance.quality_policy, "balanced");
    assert!(balanced_by_id
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::FieldPromotionFallback));

    assert_eq!(strict_by_id.data.run_status, RunStatus::Degraded);
    assert!(!strict_by_id.data.publishable);
    assert_eq!(strict_by_id.data.provenance.quality_policy, "strict");
    assert!(strict_by_id
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::FieldPromotionFallback));
}
