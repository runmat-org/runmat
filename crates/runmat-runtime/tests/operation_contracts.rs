use runmat_analysis_core::{AnalysisFieldValues, ReferenceFrame};
use runmat_analysis_fea::fixtures::{fixture_model, FixtureId};
use runmat_analysis_fea::ComputeBackend;
use runmat_geometry_core::UnitSystem;
use runmat_runtime::analysis::{
    analysis_run_linear_static_op, analysis_run_linear_static_with_options, analysis_validate,
    AnalysisRunOptions, PrecisionMode, RunStatus,
};
use runmat_runtime::geometry::{geometry_inspect_op, geometry_load_op};
use runmat_runtime::operations::OperationContext;

const TRIANGLE_STL: &str = "solid tri\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid tri\n";

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
fn analysis_run_deterministic_contract_is_stable_across_replays() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let options = AnalysisRunOptions {
        deterministic_mode: true,
        precision_mode: PrecisionMode::Fp64,
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
    assert_eq!(cpu.data.run_status, RunStatus::Publishable);
    assert_eq!(gpu.data.run_status, RunStatus::Publishable);
}
