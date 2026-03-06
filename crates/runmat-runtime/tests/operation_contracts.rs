use runmat_analysis_core::{AnalysisFieldValues, ReferenceFrame};
use runmat_geometry_core::EntityKind;
use runmat_analysis_fea::fixtures::{fixture_model, FixtureId};
use runmat_analysis_fea::ComputeBackend;
use runmat_geometry_core::UnitSystem;
use runmat_runtime::analysis::{
    analysis_create_model_op, analysis_results_by_run_id_op, analysis_results_op,
    analysis_run_linear_static_op, analysis_run_linear_static_with_options, analysis_run_modal_op,
    analysis_validate, AnalysisCreateModelIntentSpec, AnalysisCreateModelProfile,
    AnalysisResultsQuery, AnalysisRunOptions, ModalFrequencyBasis, ModalFrequencyUnits,
    PrecisionMode, PreconditionerMode, QualityPolicy, QualityReasonCode, RunStatus,
};
use runmat_runtime::geometry::{
    geometry_capture_view_op, geometry_inspect_op, geometry_list_regions_op, geometry_load_op,
    geometry_query_entities_op, GeometryCaptureViewSpec, GeometryEntityQuery,
};
use runmat_runtime::operations::OperationContext;
use std::sync::atomic::{AtomicU64, Ordering};

const TRIANGLE_STL: &str = "solid tri\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid tri\n";
const SIMPLE_STEP: &str = "ISO-10303-21;\nHEADER;\nFILE_NAME('Assembly_A');\nENDSEC;\nDATA;\n#10=PRODUCT('Base_Mount','',(#1));\n#11=PRODUCT('Tip_Load','',(#1));\n#20=MATERIAL_DESIGNATION('Aluminum 6061');\nENDSEC;\nEND-ISO-10303-21;\n";

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
    assert_eq!(invalid_capture_view.error_code, "GEOMETRY_CAPTURE_INVALID_SPEC");
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
        },
        OperationContext::new(Some("trace-contract-create-4-modal".to_string()), None),
    )
    .expect("modal profile should be supported");
    assert_eq!(modal.operation, "analysis.create_model");
    assert_eq!(modal.op_version, "analysis.create_model/v1");
    assert_eq!(modal.data.steps[0].kind, runmat_analysis_core::AnalysisStepKind::Modal);

    for profile in [
        AnalysisCreateModelProfile::TransientStructural,
        AnalysisCreateModelProfile::NonlinearStructural,
    ] {
        let unsupported_profile = analysis_create_model_op(
            &geometry.data,
            AnalysisCreateModelIntentSpec {
                model_id: "contract_unsupported_model".to_string(),
                profile,
            },
            OperationContext::new(Some("trace-contract-create-4".to_string()), None),
        )
        .expect_err("profile should be unsupported");
        assert_eq!(unsupported_profile.operation, "analysis.create_model");
        assert_eq!(unsupported_profile.op_version, "analysis.create_model/v1");
        assert_eq!(
            unsupported_profile.error_code,
            "ANALYSIS_CREATE_MODEL_PROFILE_UNSUPPORTED"
        );
    }
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
        "diag_generalized_eigen"
    );
    let modal_results = modal_envelope
        .data
        .modal_results
        .as_ref()
        .expect("modal results payload should exist");
    assert!(!modal_results.eigenvalues_hz.is_empty());
    assert_eq!(modal_results.eigenvalues_hz.len(), modal_results.mode_shapes.len());
    assert_eq!(modal_results.mode_shapes[0].field_id, "mode_shape_1");
    assert_eq!(modal_results.modal_payload_version, "modal_results/v1");
    assert_eq!(modal_results.mode_units, ModalFrequencyUnits::Hz);
    assert_eq!(
        modal_results.frequency_basis,
        ModalFrequencyBasis::NativeEigenSolve
    );
    assert_eq!(modal_envelope.data.run_status, RunStatus::Publishable);
    assert!(modal_envelope.data.publishable);
    assert!(!modal_envelope
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::ModalPlaceholder));

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
            include_modal_results: true,
            mode_indices: Vec::new(),
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
            include_modal_results: true,
            mode_indices: Vec::new(),
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
            include_modal_results: false,
            mode_indices: Vec::new(),
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

    let invalid_mode = analysis_results_op(
        &modal_run.data,
        AnalysisResultsQuery {
            include_fields: Vec::new(),
            include_diagnostics: true,
            include_modal_results: true,
            mode_indices: vec![99],
        },
        OperationContext::new(Some("trace-contract-modal-results-5".to_string()), None),
    )
    .expect_err("unknown mode index should fail");
    assert_eq!(invalid_mode.error_code, "ANALYSIS_RESULTS_MODE_NOT_FOUND");
    assert_eq!(invalid_mode.operation, "analysis.results");
    assert_eq!(invalid_mode.op_version, "analysis.results/v1");
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
    };

    let first = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        options,
        OperationContext::new(Some("trace-contract-6a".to_string()), None),
    )
    .expect("first deterministic run should succeed");
    let second = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        options,
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
        options,
        OperationContext::new(Some("trace-contract-7a".to_string()), None),
    )
    .expect("cpu run should succeed");

    let gpu = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Gpu,
        options,
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
        },
        OperationContext::new(Some("trace-contract-12-strict".to_string()), None),
    )
    .expect("strict run should succeed");

    assert_eq!(balanced.data.solver_convergence, strict.data.solver_convergence);
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
