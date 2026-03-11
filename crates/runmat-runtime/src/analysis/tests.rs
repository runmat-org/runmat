use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::{fs, path::PathBuf};

use chrono::Utc;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, ApiDeviceInfo, GpuTensorHandle, HostTensorOwned,
    HostTensorView,
};
use runmat_analysis_core::{
    AnalysisFieldValues, AnalysisModel, AnalysisModelId, AnalysisStep, AnalysisStepKind,
    BoundaryCondition, BoundaryConditionKind, EvidenceConfidence, LoadCase, LoadKind,
    MaterialAssignment, MaterialModel, ReferenceFrame,
};
use runmat_analysis_fea::ComputeBackend;
use runmat_geometry_core::{
    GeometryAsset, GeometrySource, MaterialEvidence, MaterialEvidenceConfidence, MeshDescriptor,
    MeshKind, Region, SourceGeometry, SourceGeometryKind, TessellationProfile, UnitSystem,
};

use super::*;

fn analysis_test_guard() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn sample_model() -> AnalysisModel {
    AnalysisModel {
        model_id: AnalysisModelId("beam_model".to_string()),
        geometry_id: "geo:beam".to_string(),
        geometry_revision: 1,
        units: UnitSystem::Meter,
        frame: ReferenceFrame::Global,
        materials: vec![MaterialModel {
            material_id: "mat_steel".to_string(),
            name: "Steel".to_string(),
            youngs_modulus_pa: 200e9,
            poisson_ratio: 0.3,
            reference_temperature_k: 293.15,
            modulus_temp_coeff_per_k: -2.5e-4,
        }],
        material_assignments: Vec::new(),
        boundary_conditions: vec![BoundaryCondition {
            bc_id: "bc_root".to_string(),
            region_id: "root".to_string(),
            kind: BoundaryConditionKind::Fixed,
        }],
        loads: vec![LoadCase {
            load_id: "load_tip".to_string(),
            region_id: "tip".to_string(),
            kind: LoadKind::Force {
                fx: 0.0,
                fy: -1000.0,
                fz: 0.0,
            },
        }],
        steps: vec![AnalysisStep {
            step_id: "step_static".to_string(),
            kind: AnalysisStepKind::Static,
        }],
    }
}

fn sample_model_with_material_assignment_mismatch() -> AnalysisModel {
    let mut model = sample_model();
    model.materials.push(MaterialModel {
        material_id: "mat_polymer".to_string(),
        name: "Polymer".to_string(),
        youngs_modulus_pa: 3.2e9,
        poisson_ratio: 0.37,
        reference_temperature_k: 293.15,
        modulus_temp_coeff_per_k: -7.0e-4,
    });
    model.material_assignments = vec![MaterialAssignment {
        region_id: "tip".to_string(),
        expected_material_id: "mat_steel".to_string(),
        assigned_material_id: "mat_polymer".to_string(),
        confidence: EvidenceConfidence::Verified,
    }];
    model
}

fn sample_geometry_asset() -> GeometryAsset {
    GeometryAsset {
        geometry_id: "geo:beam".to_string(),
        source: GeometrySource {
            path: "/fixtures/beam.stl".to_string(),
            sha256: "hash-beam".to_string(),
            importer_version: "stl/v1".to_string(),
        },
        source_geometry: SourceGeometry {
            kind: SourceGeometryKind::Mesh,
            assembly: None,
            material_evidence: Vec::new(),
        },
        tessellation_profile: TessellationProfile::default(),
        units: UnitSystem::Meter,
        revision: 2,
        meshes: vec![MeshDescriptor {
            mesh_id: "mesh_1".to_string(),
            kind: MeshKind::Surface,
            vertex_count: 3,
            element_count: 1,
        }],
        regions: Vec::new(),
        diagnostics: Vec::new(),
    }
}

fn sample_step_like_geometry_asset() -> GeometryAsset {
    let mut asset = sample_geometry_asset();
    asset.source_geometry.kind = SourceGeometryKind::Cad;
    asset.source_geometry.material_evidence = vec![MaterialEvidence {
        source_key: "STEP:MATERIAL".to_string(),
        normalized_key: "material_name".to_string(),
        value: "Aluminum 6061".to_string(),
        confidence: MaterialEvidenceConfidence::High,
        unit_basis: None,
        assumptions: vec!["imported".to_string()],
    }];
    asset.regions = vec![
        Region {
            region_id: "region_root".to_string(),
            name: "Base_Mount".to_string(),
            tag: Some("fixed".to_string()),
        },
        Region {
            region_id: "region_tip".to_string(),
            name: "Tip_Load".to_string(),
            tag: Some("load".to_string()),
        },
    ];
    asset
}

#[test]
fn analysis_create_model_returns_v1_envelope() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let envelope = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "model_from_geo".to_string(),
            profile: AnalysisCreateModelProfile::LinearStaticStructural,
            prep_context: None,
        },
        OperationContext::new(Some("trace-create-1".to_string()), None),
    )
    .expect("create model should pass");

    assert_eq!(envelope.operation, "analysis.create_model");
    assert_eq!(envelope.op_version, "analysis.create_model/v1");
    assert_eq!(envelope.data.model_id.0, "model_from_geo");
    assert_eq!(envelope.data.geometry_id, "geo:beam");
    assert_eq!(envelope.data.geometry_revision, 2);
    assert_eq!(envelope.data.units, UnitSystem::Meter);
    assert_eq!(envelope.data.frame, ReferenceFrame::Global);
    assert!(!envelope.data.materials.is_empty());
    assert!(!envelope.data.boundary_conditions.is_empty());
    assert!(!envelope.data.loads.is_empty());
    assert!(!envelope.data.steps.is_empty());
    assert_eq!(envelope.data.steps[0].kind, AnalysisStepKind::Static);
}

#[test]
fn transient_run_option_presets_are_ordered_for_cost_vs_accuracy() {
    let coarse = AnalysisTransientRunOptions::coarse();
    let balanced = AnalysisTransientRunOptions::balanced();
    let production = AnalysisTransientRunOptions::production_recommended();
    let high_accuracy = AnalysisTransientRunOptions::high_accuracy();

    assert!(coarse.step_count < balanced.step_count);
    assert!(balanced.step_count < high_accuracy.step_count);

    assert!(coarse.tolerance > balanced.tolerance);
    assert!(balanced.tolerance > high_accuracy.tolerance);

    assert!(coarse.time_step_s > balanced.time_step_s);
    assert!(balanced.time_step_s > high_accuracy.time_step_s);

    assert_eq!(production.quality_policy, QualityPolicy::Balanced);
    assert!(production.deterministic_mode);
    assert_eq!(production.precision_mode, PrecisionMode::Fp64);
    assert_eq!(production.dt_bucket_rel_tolerance, 0.01);
}

#[test]
fn modal_run_option_presets_are_ordered_for_cost_vs_accuracy() {
    let coarse = AnalysisModalRunOptions::coarse();
    let balanced = AnalysisModalRunOptions::balanced();
    let high_accuracy = AnalysisModalRunOptions::high_accuracy();

    assert!(coarse.mode_count < balanced.mode_count);
    assert!(balanced.mode_count < high_accuracy.mode_count);
    assert!(coarse.residual_warn_threshold > balanced.residual_warn_threshold);
    assert!(balanced.residual_warn_threshold > high_accuracy.residual_warn_threshold);
}

#[test]
fn nonlinear_run_option_presets_are_ordered_for_cost_vs_accuracy() {
    let coarse = AnalysisNonlinearRunOptions::coarse();
    let balanced = AnalysisNonlinearRunOptions::balanced();
    let production = AnalysisNonlinearRunOptions::production_recommended();
    let high_accuracy = AnalysisNonlinearRunOptions::high_accuracy();

    assert!(coarse.increment_count < balanced.increment_count);
    assert!(balanced.increment_count <= production.increment_count);
    assert!(production.increment_count <= high_accuracy.increment_count);

    assert!(coarse.max_newton_iters < balanced.max_newton_iters);
    assert!(balanced.max_newton_iters <= production.max_newton_iters);
    assert!(production.max_newton_iters <= high_accuracy.max_newton_iters);

    assert!(coarse.tolerance > balanced.tolerance);
    assert!(balanced.tolerance >= production.tolerance);
    assert!(production.tolerance >= high_accuracy.tolerance);

    assert_eq!(production.quality_policy, QualityPolicy::Balanced);
    assert!(production.deterministic_mode);
    assert_eq!(production.precision_mode, PrecisionMode::Fp64);
    assert!(production.line_search);
    assert!(production.max_line_search_backtracks >= balanced.max_line_search_backtracks);
}

#[test]
fn analysis_create_model_maps_invalid_intent_error() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let err = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "   ".to_string(),
            profile: AnalysisCreateModelProfile::LinearStaticStructural,
            prep_context: None,
        },
        OperationContext::new(None, None),
    )
    .expect_err("create model should fail");

    assert_eq!(err.error_code, "ANALYSIS_CREATE_MODEL_INVALID_INTENT");
    assert_eq!(err.operation, "analysis.create_model");
    assert_eq!(err.op_version, "analysis.create_model/v1");
}

#[test]
fn analysis_create_model_supports_nonlinear_profile_template() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let envelope = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "nonlinear_model".to_string(),
            profile: AnalysisCreateModelProfile::NonlinearStructural,
            prep_context: None,
        },
        OperationContext::new(None, None),
    )
    .expect("nonlinear profile should be supported");

    assert_eq!(envelope.data.model_id.0, "nonlinear_model");
    assert_eq!(envelope.data.steps[0].kind, AnalysisStepKind::Nonlinear);
    assert_eq!(
        envelope.data.loads[0].load_id,
        "load_default_nonlinear_force"
    );
}

#[test]
fn analysis_create_model_accepts_prep_context_and_validates_model() {
    let _guard = analysis_test_guard();
    let geometry = sample_step_like_geometry_asset();
    let prep = crate::geometry::geometry_prep_for_analysis_op(
        &geometry,
        crate::geometry::GeometryPrepForAnalysisSpec::default(),
        OperationContext::new(None, None),
    )
    .expect("prep for analysis should succeed");

    let created = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "prep_model".to_string(),
            profile: AnalysisCreateModelProfile::LinearStaticStructural,
            prep_context: Some(AnalysisCreateModelPrepContext {
                source_geometry_id: prep.data.prep.provenance.source_geometry_id.clone(),
                source_geometry_revision: prep.data.prep.provenance.source_geometry_revision,
                region_mappings: prep.data.prep.region_mappings.clone(),
            }),
        },
        OperationContext::new(None, None),
    )
    .expect("create model with prep context should succeed");

    analysis_validate(
        &created.data,
        geometry.units,
        &ReferenceFrame::Global,
        OperationContext::new(None, None),
    )
    .expect("prep-aware created model should validate");
    assert_eq!(created.data.boundary_conditions[0].region_id, "region_root");
    assert_eq!(created.data.loads[0].region_id, "region_tip");
    assert!(created
        .data
        .material_assignments
        .iter()
        .all(|assignment| assignment.confidence
            == runmat_analysis_core::EvidenceConfidence::Verified));
}

#[test]
fn analysis_create_model_rejects_mismatched_prep_context() {
    let _guard = analysis_test_guard();
    let geometry = sample_step_like_geometry_asset();
    let error = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "bad_prep_model".to_string(),
            profile: AnalysisCreateModelProfile::LinearStaticStructural,
            prep_context: Some(AnalysisCreateModelPrepContext {
                source_geometry_id: "geo:other".to_string(),
                source_geometry_revision: geometry.revision,
                region_mappings: Vec::new(),
            }),
        },
        OperationContext::new(None, None),
    )
    .expect_err("mismatched prep context should fail");
    assert_eq!(error.error_code, "ANALYSIS_CREATE_MODEL_PREP_MISMATCH");
}

#[test]
fn analysis_create_model_supports_transient_profile_template() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let envelope = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "transient_model".to_string(),
            profile: AnalysisCreateModelProfile::TransientStructural,
            prep_context: None,
        },
        OperationContext::new(None, None),
    )
    .expect("transient profile should be supported");

    assert_eq!(envelope.data.model_id.0, "transient_model");
    assert_eq!(envelope.data.steps[0].kind, AnalysisStepKind::Transient);
    assert_eq!(
        envelope.data.loads[0].load_id,
        "load_default_transient_force"
    );
}

#[test]
fn analysis_create_model_supports_modal_profile_template() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let envelope = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "modal_model".to_string(),
            profile: AnalysisCreateModelProfile::ModalStructural,
            prep_context: None,
        },
        OperationContext::new(None, None),
    )
    .expect("modal profile should be supported");

    assert_eq!(envelope.data.model_id.0, "modal_model");
    assert_eq!(envelope.data.steps[0].kind, AnalysisStepKind::Modal);
    assert_eq!(envelope.data.loads[0].load_id, "load_default_modal_seed");
}

#[test]
fn analysis_create_model_infers_materials_and_assignments_from_geometry_evidence() {
    let _guard = analysis_test_guard();
    let geometry = sample_step_like_geometry_asset();
    let envelope = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "model_from_step_like".to_string(),
            profile: AnalysisCreateModelProfile::LinearStaticStructural,
            prep_context: None,
        },
        OperationContext::new(None, None),
    )
    .expect("create model should succeed");

    assert!(envelope
        .data
        .materials
        .iter()
        .any(|material| material.material_id == "mat_aluminum"));
    assert_eq!(envelope.data.material_assignments.len(), 2);
    assert!(envelope
        .data
        .material_assignments
        .iter()
        .all(|assignment| assignment.assigned_material_id == "mat_aluminum"));
    assert_eq!(
        envelope.data.boundary_conditions[0].region_id,
        "region_root"
    );
    assert_eq!(envelope.data.loads[0].region_id, "region_tip");
}

#[test]
fn analysis_validate_returns_typed_envelope() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let context =
        OperationContext::new(Some("trace-a1".to_string()), Some("request-a1".to_string()));
    let envelope = analysis_validate(&model, UnitSystem::Meter, &ReferenceFrame::Global, context)
        .expect("validation should pass");

    assert_eq!(envelope.operation, "analysis.validate");
    assert_eq!(envelope.op_version, "analysis.validate/v1");
    assert!(envelope.data.valid);
    assert_eq!(envelope.trace_id.as_deref(), Some("trace-a1"));
}

#[test]
fn analysis_validate_maps_typed_error_code() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.materials.clear();
    let context = OperationContext::new(None, None);
    let error = analysis_validate(&model, UnitSystem::Meter, &ReferenceFrame::Global, context)
        .expect_err("validation should fail");

    assert_eq!(error.error_code, "ANALYSIS_VALIDATION_MISSING_MATERIALS");
    assert_eq!(error.operation, "analysis.validate");
    assert_eq!(error.op_version, "analysis.validate/v1");
}

#[test]
fn analysis_run_linear_static_returns_typed_envelope() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let context =
        OperationContext::new(Some("trace-a2".to_string()), Some("request-a2".to_string()));
    let envelope = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        AnalysisRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            preconditioner_mode: PreconditionerMode::Auto,
            quality_policy: QualityPolicy::Balanced,
            thermo_mechanical_coupling: None,
            electro_thermal_coupling: None,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
        },
        context,
    )
    .expect("run should pass");

    assert_eq!(envelope.operation, "analysis.run_linear_static");
    assert_eq!(envelope.op_version, "analysis.run_linear_static/v1");
    assert_eq!(envelope.data.run.backend, ComputeBackend::Cpu);
    assert!(!envelope.data.run.displacement_field.is_empty());
    assert_eq!(envelope.data.run_status, RunStatus::Publishable);
    assert!(envelope.data.publishable);
    assert!(envelope.data.modal_results.is_none());
    assert_eq!(envelope.data.solver_convergence, QualityGate::Pass);
    assert!(envelope.data.provenance.deterministic_mode);
    assert_eq!(envelope.data.provenance.precision_mode, "fp64");
    assert_eq!(envelope.data.provenance.solver_method, "matrix_free_pcg");
    assert_eq!(envelope.data.provenance.preconditioner, "jacobi");
    assert_eq!(envelope.data.provenance.quality_policy, "balanced");
    assert_eq!(envelope.data.provenance.solver_device_apply_k_ratio, 0.0);
    assert_eq!(envelope.data.provenance.solver_host_sync_count, 0);
}

#[test]
fn gpu_run_without_provider_records_fallback_event() {
    let _guard = analysis_test_guard();
    let _guard = runmat_accelerate_api::ThreadProviderGuard::set(None);
    let model = sample_model();
    let envelope = analysis_run_linear_static_op(
        &model,
        ComputeBackend::Gpu,
        OperationContext::new(None, None),
    )
    .expect("run should pass");

    if envelope.data.provenance.solver_backend == "cpu_reference" {
        assert!(envelope
            .data
            .provenance
            .fallback_events
            .iter()
            .any(|event| event.starts_with("SOLVER_BACKEND_FALLBACK")));
        assert_eq!(envelope.data.provenance.solver_device_apply_k_ratio, 0.0);
        assert!(matches!(
            envelope.data.run.displacement_field.values,
            AnalysisFieldValues::HostF64(_)
        ));
    } else {
        assert_eq!(envelope.data.provenance.solver_backend, "runtime_tensor");
    }
}

#[test]
fn gpu_run_with_provider_emits_device_refs() {
    let _guard = analysis_test_guard();
    static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(1000);

    struct AnalysisTestProvider;

    impl AccelProvider for AnalysisTestProvider {
        fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            Ok(GpuTensorHandle {
                shape: host.shape.to_vec(),
                device_id: 7,
                buffer_id: NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed),
            })
        }

        fn download<'a>(&'a self, h: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async move {
                Ok(HostTensorOwned {
                    data: vec![0.0; h.shape.iter().product()],
                    shape: h.shape.clone(),
                })
            })
        }

        fn free(&self, _h: &GpuTensorHandle) -> anyhow::Result<()> {
            Ok(())
        }

        fn device_info(&self) -> String {
            "analysis-test-provider".to_string()
        }

        fn device_id(&self) -> u32 {
            7
        }

        fn device_info_struct(&self) -> ApiDeviceInfo {
            ApiDeviceInfo {
                device_id: 7,
                name: "analysis-test-provider".to_string(),
                vendor: "runmat-tests".to_string(),
                memory_bytes: None,
                backend: Some("test_gpu".to_string()),
            }
        }
    }

    static PROVIDER: AnalysisTestProvider = AnalysisTestProvider;
    let _guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(&PROVIDER));

    let model = sample_model();
    let envelope = analysis_run_linear_static_op(
        &model,
        ComputeBackend::Gpu,
        OperationContext::new(None, None),
    )
    .expect("run should pass");

    assert!(!envelope
        .data
        .provenance
        .fallback_events
        .iter()
        .any(|event| event.starts_with("BACKEND_NO_PROVIDER")
            || event.starts_with("BACKEND_UPLOAD_FAILED")));
    assert!(matches!(
        envelope.data.provenance.solver_backend.as_str(),
        "runtime_tensor" | "cpu_reference"
    ));
    if envelope.data.provenance.solver_backend == "cpu_reference" {
        assert!(envelope
            .data
            .provenance
            .fallback_events
            .iter()
            .any(|event| event.starts_with("SOLVER_BACKEND_FALLBACK")));
    }
    assert!(
        (0.0..=1.0).contains(&envelope.data.provenance.solver_device_apply_k_ratio),
        "ratio must be in [0,1]"
    );
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
fn analysis_results_returns_filtered_fields_and_metadata() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let run = analysis_run_linear_static_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-results-1".to_string()), None),
    )
    .expect("run should pass");

    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery {
            include_fields: vec!["displacement".to_string()],
            include_diagnostics: false,
            diagnostic_codes: Vec::new(),
            include_modal_results: true,
            mode_indices: Vec::new(),
            include_transient_results: true,
            transient_snapshot_indices: Vec::new(),
            include_nonlinear_results: true,
        },
        OperationContext::new(Some("trace-results-2".to_string()), None),
    )
    .expect("results should pass");

    assert_eq!(results.operation, "analysis.results");
    assert_eq!(results.op_version, "analysis.results/v1");
    assert_eq!(results.data.fields.len(), 1);
    assert_eq!(results.data.fields[0].field_id, "displacement");
    assert!(results.data.diagnostics.is_none());
    assert_eq!(results.data.summary.field_count, 1);
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
}

#[test]
fn analysis_results_unknown_field_maps_typed_error() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let run = analysis_run_linear_static_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-results-3".to_string()), None),
    )
    .expect("run should pass");

    let err = analysis_results_op(
        &run.data,
        AnalysisResultsQuery {
            include_fields: vec!["strain_energy".to_string()],
            include_diagnostics: true,
            diagnostic_codes: Vec::new(),
            include_modal_results: true,
            mode_indices: Vec::new(),
            include_transient_results: true,
            transient_snapshot_indices: Vec::new(),
            include_nonlinear_results: true,
        },
        OperationContext::new(Some("trace-results-4".to_string()), None),
    )
    .expect_err("results should fail");

    assert_eq!(err.operation, "analysis.results");
    assert_eq!(err.op_version, "analysis.results/v1");
    assert_eq!(err.error_code, "ANALYSIS_RESULTS_FIELD_NOT_FOUND");
}

#[test]
fn analysis_results_by_run_id_roundtrip_works() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let model = sample_model();
    let run = analysis_run_linear_static_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-results-by-id-run".to_string()), None),
    )
    .expect("run should pass");

    let fetched = analysis_results_by_run_id_op(
        &run.data.run_id,
        AnalysisResultsQuery::default(),
        OperationContext::new(Some("trace-results-by-id-fetch".to_string()), None),
    )
    .expect("results by id should pass");

    assert_eq!(fetched.operation, "analysis.results");
    assert_eq!(fetched.op_version, "analysis.results/v1");
    assert_eq!(fetched.data.summary.field_count, 2);
    assert_eq!(fetched.data.summary.mode_count, 0);
    assert!(fetched.data.summary.available_mode_indices.is_empty());
    assert_eq!(fetched.data.summary.min_frequency_hz, None);
    assert_eq!(fetched.data.summary.max_frequency_hz, None);
    assert_eq!(fetched.data.summary.max_modal_residual_norm, None);
    assert_eq!(fetched.data.summary.first_mode_converged, None);
    assert_eq!(fetched.data.summary.snapshot_count, 0);
    assert_eq!(fetched.data.summary.time_start_s, None);
    assert_eq!(fetched.data.summary.time_end_s, None);
    assert_eq!(fetched.data.summary.max_transient_residual_norm, None);
    assert_eq!(fetched.data.summary.final_step_converged, None);

    storage::reset_artifact_store_for_tests();
}

#[test]
fn analysis_results_by_run_id_missing_maps_typed_error() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let err = analysis_results_by_run_id_op(
        "run_missing",
        AnalysisResultsQuery::default(),
        OperationContext::new(Some("trace-results-by-id-missing".to_string()), None),
    )
    .expect_err("missing run id should fail");

    assert_eq!(err.error_code, "ANALYSIS_RESULTS_RUN_NOT_FOUND");
    storage::reset_artifact_store_for_tests();
}

#[test]
fn analysis_results_compare_reports_typed_deltas() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();

    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];
    let baseline = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions {
            max_newton_iters: 1,
            line_search: false,
            ..AnalysisNonlinearRunOptions::balanced()
        },
        OperationContext::new(None, None),
    )
    .expect("baseline nonlinear run should succeed");
    let candidate = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions::production_recommended(),
        OperationContext::new(None, None),
    )
    .expect("candidate nonlinear run should succeed");

    let compare = analysis_results_compare_op(
        AnalysisResultsCompareQuery {
            baseline_run_id: baseline.data.run_id.clone(),
            candidate_run_id: candidate.data.run_id.clone(),
        },
        OperationContext::new(None, None),
    )
    .expect("compare operation should succeed");

    assert_eq!(compare.operation, "analysis.results_compare");
    assert_eq!(compare.op_version, "analysis.results_compare/v1");
    assert!(compare.data.failed_increment_delta.is_some());
    assert!(compare.data.max_iteration_delta.is_some());
    assert!(compare.data.solve_ms_delta.is_some());

    storage::reset_artifact_store_for_tests();
}

#[test]
fn analysis_trends_summarizes_recent_nonlinear_runs() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();

    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];
    for _ in 0..4 {
        let _ = analysis_run_nonlinear_op(
            &model,
            ComputeBackend::Cpu,
            OperationContext::new(None, None),
        )
        .expect("nonlinear run should persist for trends");
    }

    let trends = analysis_trends_op(
        AnalysisTrendsQuery { window_size: 3 },
        OperationContext::new(None, None),
    )
    .expect("trends should succeed");

    assert_eq!(trends.operation, "analysis.trends");
    assert_eq!(trends.op_version, "analysis.trends/v1");
    let nonlinear = trends
        .data
        .summaries
        .iter()
        .find(|summary| summary.run_kind == AnalysisRunKind::Nonlinear)
        .expect("nonlinear trend summary should exist");
    assert_eq!(nonlinear.sample_count, 3);
    assert!(nonlinear.median_solve_ms.is_some());
    assert!(nonlinear.p95_solve_ms.is_some());
    assert!(nonlinear.failed_increment_rate.is_some());
    assert!(nonlinear.thermo_coupling_enabled_rate.is_none());
    assert!(nonlinear.thermo_transient_warn_rate.is_none());
    assert!(nonlinear.thermo_nonlinear_warn_rate.is_none());
    assert!(nonlinear.thermo_spread_breach_rate.is_none());
    assert!(nonlinear.thermo_heterogeneity_breach_rate.is_none());
    assert!(nonlinear.electro_thermal_coupling_enabled_rate.is_none());
    assert!(nonlinear.electro_transient_warn_rate.is_none());
    assert!(nonlinear.electro_nonlinear_warn_rate.is_none());
    assert!(nonlinear.plastic_nonlinear_warn_rate.is_none());

    storage::reset_artifact_store_for_tests();
}

#[test]
fn analysis_results_summary_surfaces_thermo_transient_metrics() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "transient_1".to_string(),
        kind: AnalysisStepKind::Transient,
    }];

    let run = analysis_run_transient_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisTransientRunOptions {
            thermo_mechanical_coupling: Some(ThermoMechanicalCouplingOptions {
                enabled: true,
                reference_temperature_k: 293.15,
                applied_temperature_delta_k: 65.0,
                thermal_expansion_coefficient: 1.2e-5,
                field_artifact_id: None,
                field_source: None,
                region_temperature_deltas: Vec::new(),
                time_profile: Vec::new(),
            }),
            electro_thermal_coupling: Some(ElectroThermalCouplingOptions {
                enabled: true,
                reference_temperature_k: 293.15,
                applied_voltage_v: 36.0,
                base_electrical_conductivity_s_per_m: 3.5e7,
                resistive_heating_coefficient: 4.0e-4,
                region_conductivity_scales: Vec::new(),
                time_profile: Vec::new(),
            }),
            ..AnalysisTransientRunOptions::default()
        },
        OperationContext::new(None, None),
    )
    .expect("transient run should succeed");

    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("results should succeed");

    assert_eq!(results.data.summary.thermo_coupling_enabled, Some(true));
    assert!(results.data.summary.thermo_coupling_fingerprint.is_some());
    assert!(results
        .data
        .summary
        .thermo_constitutive_temperature_factor
        .is_some());
    assert!(results
        .data
        .summary
        .thermo_effective_modulus_scale
        .is_some());
    assert!(results
        .data
        .summary
        .thermo_constitutive_material_spread_ratio
        .is_some());
    assert!(results
        .data
        .summary
        .thermo_assignment_heterogeneity_index
        .is_some());
    assert!(results.data.summary.thermo_transient_severity.is_some());
    assert!(results.data.summary.thermo_nonlinear_severity.is_none());
    assert_eq!(results.data.summary.electro_thermal_coupling_enabled, Some(true));
    assert!(results
        .data
        .summary
        .electro_thermal_coupling_fingerprint
        .is_some());
    assert!(results.data.summary.electro_joule_heating_scale.is_some());
    assert!(results
        .data
        .summary
        .electro_conductivity_spread_ratio
        .is_some());
    assert!(results.data.summary.electro_transient_severity.is_some());
    assert!(results.data.summary.electro_nonlinear_severity.is_none());
    assert!(results.data.summary.plastic_nonlinear_severity.is_none());
}

#[test]
fn analysis_results_summary_surfaces_thermo_nonlinear_metrics() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];

    let run = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions {
            thermo_mechanical_coupling: Some(ThermoMechanicalCouplingOptions {
                enabled: true,
                reference_temperature_k: 293.15,
                applied_temperature_delta_k: 80.0,
                thermal_expansion_coefficient: 1.2e-5,
                field_artifact_id: None,
                field_source: None,
                region_temperature_deltas: Vec::new(),
                time_profile: Vec::new(),
            }),
            electro_thermal_coupling: Some(ElectroThermalCouplingOptions {
                enabled: true,
                reference_temperature_k: 293.15,
                applied_voltage_v: 82.0,
                base_electrical_conductivity_s_per_m: 2.6e7,
                resistive_heating_coefficient: 6.0e-4,
                region_conductivity_scales: Vec::new(),
                time_profile: Vec::new(),
            }),
            ..AnalysisNonlinearRunOptions::production_recommended()
        },
        OperationContext::new(None, None),
    )
    .expect("nonlinear run should succeed");

    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("results should succeed");

    assert_eq!(results.data.summary.thermo_coupling_enabled, Some(true));
    assert!(results.data.summary.thermo_coupling_fingerprint.is_some());
    assert!(results
        .data
        .summary
        .thermo_constitutive_temperature_factor
        .is_some());
    assert!(results
        .data
        .summary
        .thermo_effective_modulus_scale
        .is_some());
    assert!(results
        .data
        .summary
        .thermo_constitutive_material_spread_ratio
        .is_some());
    assert!(results
        .data
        .summary
        .thermo_assignment_heterogeneity_index
        .is_some());
    assert!(results.data.summary.thermo_nonlinear_severity.is_some());
    assert!(results.data.summary.thermo_transient_severity.is_some());
    assert_eq!(results.data.summary.electro_thermal_coupling_enabled, Some(true));
    assert!(results
        .data
        .summary
        .electro_thermal_coupling_fingerprint
        .is_some());
    assert!(results.data.summary.electro_joule_heating_scale.is_some());
    assert!(results
        .data
        .summary
        .electro_conductivity_spread_ratio
        .is_some());
    assert!(results.data.summary.electro_nonlinear_severity.is_some());
    assert!(results.data.summary.electro_transient_severity.is_some());
    assert!(results.data.summary.plastic_nonlinear_severity.is_none());
}

#[test]
fn analysis_trends_handles_mixed_schema_and_noisy_samples() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("trends-mixed-schema");
    let _ = fs::remove_dir_all(&root);
    storage::configure_artifact_store(storage::AnalysisArtifactStoreConfig::Filesystem {
        root: root.clone(),
    })
    .expect("configure filesystem artifact store");

    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];
    let run = analysis_run_nonlinear_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("seed nonlinear run should succeed");

    let run_path = root.join("runs").join(format!("{}.json", run.data.run_id));
    let raw = fs::read_to_string(&run_path).expect("read wrapped artifact");
    let wrapped: serde_json::Value = serde_json::from_str(&raw).expect("parse wrapped artifact");
    let mut legacy = wrapped
        .get("run")
        .cloned()
        .expect("wrapped artifact should have run payload");
    legacy["run_id"] = serde_json::json!(format!("{}_legacy", run.data.run_id));
    fs::write(
        root.join("runs")
            .join(format!("{}_legacy.json", run.data.run_id)),
        serde_json::to_vec_pretty(&legacy).expect("encode legacy artifact"),
    )
    .expect("write legacy artifact");

    let trends = analysis_trends_op(
        AnalysisTrendsQuery { window_size: 8 },
        OperationContext::new(None, None),
    )
    .expect("trends should succeed on mixed schema artifacts");
    let nonlinear = trends
        .data
        .summaries
        .iter()
        .find(|summary| summary.run_kind == AnalysisRunKind::Nonlinear)
        .expect("nonlinear summary should be present");
    assert!(nonlinear.sample_count >= 2);
    assert!(nonlinear.p95_solve_ms.unwrap_or(0.0) >= nonlinear.median_solve_ms.unwrap_or(0.0));

    storage::reset_artifact_store_for_tests();
    let _ = fs::remove_dir_all(&root);
}

fn temp_artifact_root(test_name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "runmat-analysis-tests-{}-{}",
        test_name,
        Utc::now().timestamp_nanos_opt().unwrap_or_default()
    ))
}

#[test]
fn analysis_results_by_run_id_legacy_nonlinear_artifacts_remain_loadable() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("legacy-loadable");
    let _ = fs::remove_dir_all(&root);
    storage::configure_artifact_store(storage::AnalysisArtifactStoreConfig::Filesystem {
        root: root.clone(),
    })
    .expect("configure filesystem artifact store");

    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];
    let run = analysis_run_nonlinear_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("nonlinear run should succeed");
    let run_id = run.data.run_id.clone();
    let run_path = root.join("runs").join(format!("{run_id}.json"));

    let mut legacy_value = serde_json::to_value(&run.data).expect("serialize nonlinear run");
    let nonlinear = legacy_value
        .get_mut("nonlinear_results")
        .and_then(|value| value.as_object_mut())
        .expect("nonlinear results should be object");
    nonlinear.remove("increment_norms");
    nonlinear.remove("iteration_counts");
    nonlinear.remove("failed_increments");
    nonlinear.remove("line_search_backtracks");
    nonlinear.remove("max_line_search_backtracks_per_increment");
    nonlinear.remove("tangent_rebuild_count");
    nonlinear.remove("iteration_spike_count");
    nonlinear.remove("convergence_stall_count");
    nonlinear.remove("backtrack_burst_count");
    fs::write(
        &run_path,
        serde_json::to_vec_pretty(&legacy_value).expect("encode legacy artifact"),
    )
    .expect("write legacy artifact");

    let fetched = analysis_results_by_run_id_op(
        &run_id,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("legacy nonlinear artifact should still load");
    assert_eq!(fetched.data.summary.failed_increment_count, Some(0));
    assert_eq!(
        fetched.data.summary.nonlinear_iteration_spike_count,
        Some(0)
    );

    storage::reset_artifact_store_for_tests();
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn analysis_results_by_run_id_future_artifact_extra_fields_are_ignored() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("future-extra-fields");
    let _ = fs::remove_dir_all(&root);
    storage::configure_artifact_store(storage::AnalysisArtifactStoreConfig::Filesystem {
        root: root.clone(),
    })
    .expect("configure filesystem artifact store");

    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];
    let run = analysis_run_nonlinear_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("nonlinear run should succeed");
    let run_id = run.data.run_id.clone();
    let run_path = root.join("runs").join(format!("{run_id}.json"));

    let mut wrapped = serde_json::json!({
        "schema_version": "analysis_run_artifact/v1",
        "created_at": Utc::now().to_rfc3339(),
        "op_version": "analysis.run_nonlinear/v1",
        "run": run.data,
        "future_metadata": {
            "schema_hint": "analysis_run_artifact/v2",
            "opaque": [1, 2, 3]
        }
    });
    wrapped["run"]["nonlinear_results"]["future_spatial_difficulty"] =
        serde_json::json!([0.1, 0.2, 0.3]);
    fs::write(
        &run_path,
        serde_json::to_vec_pretty(&wrapped).expect("encode future artifact"),
    )
    .expect("write future artifact");

    let fetched = analysis_results_by_run_id_op(
        &run_id,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("future nonlinear artifact should still load");
    assert!(fetched.data.summary.increment_count > 0);
    assert!(fetched.data.summary.max_nonlinear_iteration_count.is_some());

    storage::reset_artifact_store_for_tests();
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn analysis_artifact_retention_prunes_old_runs_per_kind() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("retention-prune");
    let _ = fs::remove_dir_all(&root);
    storage::configure_artifact_store(storage::AnalysisArtifactStoreConfig::Filesystem {
        root: root.clone(),
    })
    .expect("configure filesystem artifact store");
    std::env::set_var("RUNMAT_ANALYSIS_ARTIFACT_MAX_RUNS_PER_KIND", "2");
    std::env::remove_var("RUNMAT_ANALYSIS_ARTIFACT_MAX_RUNS");

    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];
    let mut run_ids = Vec::new();
    for _ in 0..5 {
        let run = analysis_run_nonlinear_op(
            &model,
            ComputeBackend::Cpu,
            OperationContext::new(None, None),
        )
        .expect("nonlinear run should succeed");
        run_ids.push(run.data.run_id.clone());
    }

    let run_dir = root.join("runs");
    let kept_files = fs::read_dir(&run_dir)
        .expect("read run dir")
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().and_then(|ext| ext.to_str()) == Some("json"))
        .count();
    assert!(kept_files <= 2);
    assert!(storage::load_run_result(&run_ids[0])
        .expect("load pruned result")
        .is_none());
    assert!(
        storage::load_run_result(run_ids.last().expect("latest run id"))
            .expect("load latest result")
            .is_some()
    );

    std::env::remove_var("RUNMAT_ANALYSIS_ARTIFACT_MAX_RUNS_PER_KIND");
    storage::reset_artifact_store_for_tests();
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn analysis_results_by_run_id_filesystem_replay_is_stable() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("filesystem-replay");
    let _ = fs::remove_dir_all(&root);
    storage::configure_artifact_store(storage::AnalysisArtifactStoreConfig::Filesystem {
        root: root.clone(),
    })
    .expect("configure filesystem artifact store");

    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];
    let run = analysis_run_nonlinear_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("nonlinear run should succeed");

    let first = analysis_results_by_run_id_op(
        &run.data.run_id,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("load first replay");
    let second = analysis_results_by_run_id_op(
        &run.data.run_id,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("load second replay");

    assert_eq!(first.data.summary, second.data.summary);
    assert_eq!(first.data.run_status, second.data.run_status);
    assert_eq!(first.data.publishable, second.data.publishable);
    assert_eq!(first.data.quality_reasons, second.data.quality_reasons);

    storage::reset_artifact_store_for_tests();
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn requested_preconditioner_fallback_is_recorded() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let envelope = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        AnalysisRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            preconditioner_mode: PreconditionerMode::Amg,
            quality_policy: QualityPolicy::Balanced,
            thermo_mechanical_coupling: None,
            electro_thermal_coupling: None,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
        },
        OperationContext::new(Some("trace-preconditioner-fallback".to_string()), None),
    )
    .expect("run should succeed");

    assert_eq!(envelope.data.provenance.preconditioner, "jacobi");
    assert!(envelope
        .data
        .provenance
        .fallback_events
        .iter()
        .any(|event| event.starts_with("SOLVER_PRECONDITIONER_FALLBACK")));
}

#[test]
fn ilu_preconditioner_request_is_honored_without_fallback() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let envelope = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        AnalysisRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            preconditioner_mode: PreconditionerMode::Ilu,
            quality_policy: QualityPolicy::Balanced,
            thermo_mechanical_coupling: None,
            electro_thermal_coupling: None,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
        },
        OperationContext::new(Some("trace-preconditioner-ilu".to_string()), None),
    )
    .expect("run should succeed");

    assert_eq!(envelope.data.provenance.preconditioner, "ilu0");
    assert!(!envelope
        .data
        .provenance
        .fallback_events
        .iter()
        .any(|event| event.starts_with("SOLVER_PRECONDITIONER_FALLBACK")));
}

#[test]
fn quality_policy_exploratory_allows_publishable_warn_path() {
    let _guard = analysis_test_guard();
    let model = runmat_analysis_fea::fixtures::fixture_model(
        runmat_analysis_fea::fixtures::FixtureId::MultiMaterialAssembly,
    );
    let envelope = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        AnalysisRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            preconditioner_mode: PreconditionerMode::Auto,
            quality_policy: QualityPolicy::Exploratory,
            thermo_mechanical_coupling: None,
            electro_thermal_coupling: None,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
        },
        OperationContext::new(Some("trace-quality-policy-exploratory".to_string()), None),
    )
    .expect("run should succeed");

    assert!(envelope.data.publishable);
    assert_eq!(envelope.data.run_status, RunStatus::Publishable);
    assert!(envelope
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::MaterialAssignmentConflict));
    assert_eq!(envelope.data.provenance.quality_policy, "exploratory");
}

#[test]
fn quality_policy_balanced_allows_publishable_with_quality_reasons() {
    let _guard = analysis_test_guard();

    struct UploadFailProvider;

    impl AccelProvider for UploadFailProvider {
        fn upload(&self, _host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            Err(anyhow::anyhow!("forced-upload-failure"))
        }

        fn download<'a>(&'a self, h: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async move {
                Ok(HostTensorOwned {
                    data: vec![0.0; h.shape.iter().product()],
                    shape: h.shape.clone(),
                })
            })
        }

        fn free(&self, _h: &GpuTensorHandle) -> anyhow::Result<()> {
            Ok(())
        }

        fn device_info(&self) -> String {
            "upload-fail-provider".to_string()
        }
    }

    static PROVIDER: UploadFailProvider = UploadFailProvider;
    let _provider_guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(&PROVIDER));

    let model = sample_model();
    let envelope = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Gpu,
        AnalysisRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            preconditioner_mode: PreconditionerMode::Auto,
            quality_policy: QualityPolicy::Balanced,
            thermo_mechanical_coupling: None,
            electro_thermal_coupling: None,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
        },
        OperationContext::new(Some("trace-quality-policy-balanced".to_string()), None),
    )
    .expect("run should succeed");

    assert_eq!(envelope.data.solver_convergence, QualityGate::Pass);
    assert_eq!(envelope.data.result_quality, QualityGate::Pass);
    assert!(envelope.data.publishable);
    assert_eq!(envelope.data.run_status, RunStatus::Publishable);
    assert!(envelope
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::FieldPromotionFallback));
    assert_eq!(envelope.data.provenance.quality_policy, "balanced");
}

#[test]
fn quality_policy_strict_rejects_publishable_with_quality_reasons() {
    let _guard = analysis_test_guard();

    struct UploadFailProvider;

    impl AccelProvider for UploadFailProvider {
        fn upload(&self, _host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            Err(anyhow::anyhow!("forced-upload-failure"))
        }

        fn download<'a>(&'a self, h: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async move {
                Ok(HostTensorOwned {
                    data: vec![0.0; h.shape.iter().product()],
                    shape: h.shape.clone(),
                })
            })
        }

        fn free(&self, _h: &GpuTensorHandle) -> anyhow::Result<()> {
            Ok(())
        }

        fn device_info(&self) -> String {
            "upload-fail-provider".to_string()
        }
    }

    static PROVIDER: UploadFailProvider = UploadFailProvider;
    let _provider_guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(&PROVIDER));

    let model = sample_model();
    let envelope = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Gpu,
        AnalysisRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            preconditioner_mode: PreconditionerMode::Auto,
            quality_policy: QualityPolicy::Strict,
            thermo_mechanical_coupling: None,
            electro_thermal_coupling: None,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
        },
        OperationContext::new(Some("trace-quality-policy-strict".to_string()), None),
    )
    .expect("run should succeed");

    assert_eq!(envelope.data.solver_convergence, QualityGate::Pass);
    assert_eq!(envelope.data.result_quality, QualityGate::Pass);
    assert!(!envelope.data.publishable);
    assert_eq!(envelope.data.run_status, RunStatus::Degraded);
    assert!(envelope
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::FieldPromotionFallback));
    assert_eq!(envelope.data.provenance.quality_policy, "strict");
}

#[test]
fn analysis_run_modal_rejects_models_without_modal_step() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let err = analysis_run_modal_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect_err("modal run should fail for missing modal step");

    assert_eq!(err.operation, "analysis.run_modal");
    assert_eq!(err.op_version, "analysis.run_modal/v1");
    assert_eq!(err.error_code, "ANALYSIS_RUN_MODAL_INVALID_MODEL");
}

#[test]
fn analysis_run_transient_rejects_models_without_transient_step() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let err = analysis_run_transient_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect_err("transient run should fail for missing transient step");

    assert_eq!(err.operation, "analysis.run_transient");
    assert_eq!(err.op_version, "analysis.run_transient/v1");
    assert_eq!(err.error_code, "ANALYSIS_RUN_TRANSIENT_INVALID_MODEL");
}

#[test]
fn analysis_run_nonlinear_rejects_models_without_nonlinear_step() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let err = analysis_run_nonlinear_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect_err("nonlinear run should reject models without nonlinear step");
    assert_eq!(err.operation, "analysis.run_nonlinear");
    assert_eq!(err.op_version, "analysis.run_nonlinear/v1");
    assert_eq!(err.error_code, "ANALYSIS_RUN_NONLINEAR_INVALID_MODEL");
}

#[test]
fn analysis_run_nonlinear_returns_native_nonlinear_result() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];
    let envelope = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions {
            increment_count: 16,
            ..AnalysisNonlinearRunOptions::balanced()
        },
        OperationContext::new(None, None),
    )
    .expect("nonlinear run should succeed");

    assert_eq!(envelope.operation, "analysis.run_nonlinear");
    assert_eq!(envelope.op_version, "analysis.run_nonlinear/v1");
    let nonlinear = envelope
        .data
        .nonlinear_results
        .as_ref()
        .expect("nonlinear payload should exist");
    assert_eq!(nonlinear.method, NonlinearMethod::IncrementalNewtonRaphson);
    assert_eq!(nonlinear.load_factors.len(), 16);
    assert_eq!(nonlinear.load_factors.len(), nonlinear.residual_norms.len());
    assert_eq!(
        nonlinear.residual_norms.len(),
        nonlinear.increment_norms.len()
    );
    assert_eq!(
        nonlinear.residual_norms.len(),
        nonlinear.iteration_counts.len()
    );
    assert!(nonlinear.tangent_rebuild_count > 0);
    assert!(nonlinear.iteration_spike_count <= nonlinear.load_factors.len());
    assert!(nonlinear.max_line_search_backtracks_per_increment > 0);
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_NONLINEAR_CONVERGENCE"));
}

#[test]
fn analysis_run_nonlinear_strict_rejects_iteration_cap_exhaustion() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];

    let envelope = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions {
            quality_policy: QualityPolicy::Strict,
            max_newton_iters: 1,
            line_search: false,
            ..AnalysisNonlinearRunOptions::balanced()
        },
        OperationContext::new(None, None),
    )
    .expect("nonlinear run should produce envelope");

    assert_eq!(envelope.data.run_status, RunStatus::Degraded);
    assert!(!envelope.data.publishable);
    assert!(envelope
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::NonlinearIncrementFailure));
}

#[test]
fn analysis_run_nonlinear_rejects_missing_prep_artifact_reference() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];

    let error = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions {
            prep_artifact_id: Some("prep:missing".to_string()),
            ..AnalysisNonlinearRunOptions::production_recommended()
        },
        OperationContext::new(None, None),
    )
    .expect_err("missing prep artifact reference should fail");
    assert_eq!(error.error_code, "ANALYSIS_RUN_PREP_NOT_FOUND");
}

#[test]
fn analysis_run_nonlinear_rejects_mismatched_prep_artifact_reference() {
    let _guard = analysis_test_guard();
    let geometry = sample_step_like_geometry_asset();
    let prep = crate::geometry::geometry_prep_for_analysis_op(
        &geometry,
        crate::geometry::GeometryPrepForAnalysisSpec::default(),
        OperationContext::new(None, None),
    )
    .expect("prep should succeed");

    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];

    let error = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions {
            prep_artifact_id: Some(prep.data.prep_artifact_id.clone()),
            ..AnalysisNonlinearRunOptions::production_recommended()
        },
        OperationContext::new(None, None),
    )
    .expect_err("mismatched prep artifact reference should fail");
    assert_eq!(error.error_code, "ANALYSIS_RUN_PREP_MISMATCH");
}

#[test]
fn analysis_run_nonlinear_rejects_stale_prep_artifact_when_newer_revision_exists() {
    let _guard = analysis_test_guard();
    crate::geometry::reset_prep_artifact_store_for_tests();
    std::env::set_var("RUNMAT_GEOMETRY_PREP_REQUIRE_LATEST_REVISION", "true");

    let mut geometry_v1 = sample_step_like_geometry_asset();
    geometry_v1.revision = 1;
    let mut geometry_v2 = geometry_v1.clone();
    geometry_v2.revision = 2;

    let prep_v1 = crate::geometry::geometry_prep_for_analysis_op(
        &geometry_v1,
        crate::geometry::GeometryPrepForAnalysisSpec::default(),
        OperationContext::new(None, None),
    )
    .expect("prep v1 should succeed");
    let _prep_v2 = crate::geometry::geometry_prep_for_analysis_op(
        &geometry_v2,
        crate::geometry::GeometryPrepForAnalysisSpec::default(),
        OperationContext::new(None, None),
    )
    .expect("prep v2 should succeed");

    let created = analysis_create_model_op(
        &geometry_v1,
        AnalysisCreateModelIntentSpec {
            model_id: "stale_prep_model".to_string(),
            profile: AnalysisCreateModelProfile::NonlinearStructural,
            prep_context: None,
        },
        OperationContext::new(None, None),
    )
    .expect("create model should succeed");

    let error = analysis_run_nonlinear_with_options_op(
        &created.data,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions {
            prep_artifact_id: Some(prep_v1.data.prep_artifact_id),
            ..AnalysisNonlinearRunOptions::production_recommended()
        },
        OperationContext::new(None, None),
    )
    .expect_err("stale prep artifact should fail");
    assert_eq!(error.error_code, "ANALYSIS_RUN_PREP_STALE");

    let health = crate::geometry::geometry_prep_artifact_health_op(
        crate::geometry::GeometryPrepArtifactHealthQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("prep health should be queryable");
    assert!(health.data.metrics.stale_reject_count >= 1);

    std::env::remove_var("RUNMAT_GEOMETRY_PREP_REQUIRE_LATEST_REVISION");
    crate::geometry::reset_prep_artifact_store_for_tests();
}

#[test]
fn nonlinear_quality_policy_diverges_for_increment_failures() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];

    let run_with_policy = |quality_policy| {
        analysis_run_nonlinear_with_options_op(
            &model,
            ComputeBackend::Cpu,
            AnalysisNonlinearRunOptions {
                quality_policy,
                max_newton_iters: 1,
                line_search: false,
                max_line_search_backtracks: 0,
                ..AnalysisNonlinearRunOptions::balanced()
            },
            OperationContext::new(None, None),
        )
        .expect("nonlinear run should return envelope")
    };

    let exploratory = run_with_policy(QualityPolicy::Exploratory);
    let balanced = run_with_policy(QualityPolicy::Balanced);
    let strict = run_with_policy(QualityPolicy::Strict);

    assert!(exploratory.data.publishable);
    assert_eq!(exploratory.data.run_status, RunStatus::Publishable);
    assert!(balanced
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::NonlinearIncrementFailure));
    assert!(!balanced.data.publishable);
    assert_eq!(balanced.data.run_status, RunStatus::Degraded);

    assert!(strict
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::NonlinearIncrementFailure));
    assert!(!strict.data.publishable);
    assert_eq!(strict.data.run_status, RunStatus::Degraded);
}

#[test]
fn nonlinear_balanced_degrades_when_thermo_mechanical_severity_is_high() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];

    let run = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions {
            quality_policy: QualityPolicy::Balanced,
            thermo_mechanical_coupling: Some(ThermoMechanicalCouplingOptions {
                enabled: true,
                reference_temperature_k: 293.15,
                applied_temperature_delta_k: 90.0,
                thermal_expansion_coefficient: 1.0e-3,
                field_artifact_id: None,
                field_source: None,
                region_temperature_deltas: Vec::new(),
                time_profile: Vec::new(),
            }),
            ..AnalysisNonlinearRunOptions::production_recommended()
        },
        OperationContext::new(None, None),
    )
    .expect("nonlinear run should return envelope");

    assert!(!run.data.publishable);
    assert_eq!(run.data.run_status, RunStatus::Degraded);
    assert!(run
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::ThermoMechanicalNonlinearStress));
}

#[test]
fn nonlinear_balanced_degrades_when_thermo_heterogeneity_is_high() {
    let _guard = analysis_test_guard();
    let mut model = sample_model_with_material_assignment_mismatch();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];

    let run = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions {
            quality_policy: QualityPolicy::Balanced,
            thermo_mechanical_coupling: Some(ThermoMechanicalCouplingOptions {
                enabled: true,
                reference_temperature_k: 293.15,
                applied_temperature_delta_k: 90.0,
                thermal_expansion_coefficient: 1.2e-5,
                field_artifact_id: None,
                field_source: None,
                region_temperature_deltas: Vec::new(),
                time_profile: Vec::new(),
            }),
            ..AnalysisNonlinearRunOptions::production_recommended()
        },
        OperationContext::new(None, None),
    )
    .expect("nonlinear run should return envelope");

    assert!(!run.data.publishable);
    assert_eq!(run.data.run_status, RunStatus::Degraded);
    assert!(run.data.quality_reasons.iter().any(|reason| {
        reason.code == QualityReasonCode::ThermoMechanicalConstitutiveSpreadHigh
            || reason.code == QualityReasonCode::ThermoMechanicalAssignmentHeterogeneityHigh
    }));
}

#[test]
fn analysis_results_query_can_exclude_nonlinear_payload() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];
    let run = analysis_run_nonlinear_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("nonlinear run should succeed");

    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery {
            include_fields: Vec::new(),
            include_diagnostics: true,
            diagnostic_codes: Vec::new(),
            include_modal_results: true,
            mode_indices: Vec::new(),
            include_transient_results: true,
            transient_snapshot_indices: Vec::new(),
            include_nonlinear_results: false,
        },
        OperationContext::new(None, None),
    )
    .expect("results should succeed");

    assert!(results.data.nonlinear_results.is_none());
    assert!(results.data.summary.increment_count > 0);
    assert!(results.data.summary.failed_increment_count.is_some());
    assert!(results.data.summary.max_nonlinear_residual_norm.is_some());
    assert!(results.data.summary.max_nonlinear_increment_norm.is_some());
    assert!(results.data.summary.max_nonlinear_iteration_count.is_some());
    assert!(results.data.summary.final_increment_converged.is_some());
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
}

#[test]
fn nonlinear_results_deserialize_with_missing_new_fields() {
    let payload = serde_json::json!({
        "nonlinear_payload_version": "nonlinear_results/v1",
        "load_factors": [0.5, 1.0],
        "displacement_snapshots": [],
        "residual_norms": [1.0e-6, 5.0e-7],
        "method": "incremental_newton_raphson"
    });
    let parsed: NonlinearResultsData =
        serde_json::from_value(payload).expect("legacy nonlinear payload should deserialize");

    assert_eq!(parsed.increment_norms.len(), 0);
    assert_eq!(parsed.iteration_counts.len(), 0);
    assert_eq!(parsed.failed_increments, 0);
    assert_eq!(parsed.line_search_backtracks, 0);
    assert_eq!(parsed.max_line_search_backtracks_per_increment, 0);
    assert_eq!(parsed.tangent_rebuild_count, 0);
    assert_eq!(parsed.iteration_spike_count, 0);
    assert_eq!(parsed.convergence_stall_count, 0);
    assert_eq!(parsed.backtrack_burst_count, 0);
}

#[test]
fn analysis_run_transient_returns_native_transient_result() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "transient_1".to_string(),
        kind: AnalysisStepKind::Transient,
    }];

    let envelope = analysis_run_transient_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("transient run should return envelope");

    assert_eq!(envelope.operation, "analysis.run_transient");
    assert_eq!(envelope.op_version, "analysis.run_transient/v1");
    assert_eq!(envelope.data.run.solver_method, "implicit_euler_pcg");
    assert_eq!(envelope.data.provenance.solver_method, "implicit_euler_pcg");
    assert_eq!(envelope.data.run_status, RunStatus::Publishable);
    assert!(envelope.data.publishable);
    assert!(!envelope
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::TransientPlaceholder));
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_TRANSIENT_CONVERGENCE"));
    let transient = envelope
        .data
        .transient_results
        .as_ref()
        .expect("transient payload should exist");
    assert_eq!(
        transient.integration_method,
        TransientIntegrationMethod::ImplicitEuler
    );
    assert!(!transient.time_points_s.is_empty());
    assert_eq!(
        transient.time_points_s.len(),
        transient.displacement_snapshots.len()
    );
}

#[test]
fn analysis_run_transient_with_options_controls_timeline() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "transient_1".to_string(),
        kind: AnalysisStepKind::Transient,
    }];

    let envelope = analysis_run_transient_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisTransientRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Balanced,
            time_step_s: 2.0e-3,
            min_time_step_s: 2.0e-3,
            max_time_step_s: 2.0e-3,
            step_count: 3,
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
            thermo_mechanical_coupling: None,
            electro_thermal_coupling: None,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
        },
        OperationContext::new(None, None),
    )
    .expect("transient run should succeed with options");

    let transient = envelope
        .data
        .transient_results
        .as_ref()
        .expect("transient payload should exist");
    assert_eq!(transient.time_points_s.len(), 4);
    assert_eq!(transient.time_points_s[0], 0.0);
    assert!((transient.time_points_s[3] - 6.0e-3).abs() < 1.0e-12);
    assert_eq!(envelope.data.provenance.deterministic_mode, true);
}

#[test]
fn analysis_run_transient_rejects_non_monotonic_thermo_time_profile() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "transient_1".to_string(),
        kind: AnalysisStepKind::Transient,
    }];

    let err = analysis_run_transient_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisTransientRunOptions {
            thermo_mechanical_coupling: Some(ThermoMechanicalCouplingOptions {
                enabled: true,
                reference_temperature_k: 293.15,
                applied_temperature_delta_k: 70.0,
                thermal_expansion_coefficient: 1.1e-5,
                field_artifact_id: None,
                field_source: Some(ThermoFieldSource {
                    source_id: "field/transient-a".to_string(),
                    revision: 1,
                    interpolation_mode: Some(ThermoFieldInterpolationMode::Linear),
                    expected_region_ids: vec!["tip".to_string()],
                }),
                region_temperature_deltas: vec![ThermoRegionTemperatureDelta {
                    region_id: "tip".to_string(),
                    temperature_delta_k: 70.0,
                }],
                time_profile: vec![
                    ThermoTimeProfilePoint {
                        normalized_time: 0.8,
                        scale: 1.0,
                    },
                    ThermoTimeProfilePoint {
                        normalized_time: 0.5,
                        scale: 0.9,
                    },
                ],
            }),
            ..AnalysisTransientRunOptions::default()
        },
        OperationContext::new(None, None),
    )
    .expect_err("non-monotonic thermo time profile should be rejected");

    assert_eq!(err.error_code, "ANALYSIS_RUN_TRANSIENT_INVALID_OPTIONS");
}

#[test]
fn analysis_run_nonlinear_rejects_unknown_thermo_expected_region_ids() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];

    let err = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions {
            thermo_mechanical_coupling: Some(ThermoMechanicalCouplingOptions {
                enabled: true,
                reference_temperature_k: 293.15,
                applied_temperature_delta_k: 80.0,
                thermal_expansion_coefficient: 1.2e-5,
                field_artifact_id: None,
                field_source: Some(ThermoFieldSource {
                    source_id: "field/nonlinear-a".to_string(),
                    revision: 2,
                    interpolation_mode: Some(ThermoFieldInterpolationMode::Step),
                    expected_region_ids: vec!["missing_region".to_string()],
                }),
                region_temperature_deltas: Vec::new(),
                time_profile: Vec::new(),
            }),
            ..AnalysisNonlinearRunOptions::production_recommended()
        },
        OperationContext::new(None, None),
    )
    .expect_err("unknown thermo expected region should be rejected");

    assert_eq!(err.error_code, "ANALYSIS_RUN_NONLINEAR_INVALID_OPTIONS");
}

#[test]
fn analysis_run_nonlinear_rejects_invalid_plasticity_proxy_options() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];

    let err = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions {
            plasticity_proxy: Some(PlasticityProxyOptions {
                enabled: true,
                yield_strain: -1.0,
                hardening_modulus_ratio: 0.1,
                saturation_exponent: 1.0,
            }),
            ..AnalysisNonlinearRunOptions::default()
        },
        OperationContext::new(None, None),
    )
    .expect_err("nonlinear run should reject invalid plasticity options");

    assert_eq!(err.error_code, "ANALYSIS_RUN_NONLINEAR_INVALID_OPTIONS");
}

#[test]
fn analysis_run_nonlinear_rejects_invalid_contact_proxy_options() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];

    let err = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions {
            contact_proxy: Some(ContactProxyOptions {
                enabled: true,
                penalty_stiffness_scale: 0.0,
                max_penetration_ratio: 0.01,
                friction_coefficient: 0.0,
            }),
            ..AnalysisNonlinearRunOptions::default()
        },
        OperationContext::new(None, None),
    )
    .expect_err("nonlinear run should reject invalid contact options");

    assert_eq!(err.error_code, "ANALYSIS_RUN_NONLINEAR_INVALID_OPTIONS");
}

#[test]
fn analysis_run_transient_can_resolve_thermo_field_artifact() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "transient_1".to_string(),
        kind: AnalysisStepKind::Transient,
    }];

    let root = PathBuf::from("target/runmat-analysis-artifacts/thermo-fields");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).expect("create thermo field artifact root");
    let mut field_artifact = serde_json::json!({
        "schema_version": "analysis_thermo_field_artifact/v1",
        "source_geometry_id": model.geometry_id,
        "source_geometry_revision": model.geometry_revision,
        "artifact_status": "approved",
        "approved_by": "release-bot",
        "field_source": {
            "source_id": "artifact/transient-field",
            "revision": 1,
            "interpolation_mode": "linear",
            "expected_region_ids": [],
        },
        "region_temperature_deltas": [
            {"region_id": "tip", "temperature_delta_k": 72.0}
        ],
        "time_profile": [
            {"normalized_time": 0.0, "scale": 0.5},
            {"normalized_time": 1.0, "scale": 1.0}
        ]
    });
    let artifact_hash = thermo_field_payload_hash(
        &serde_json::from_value(field_artifact.clone()).expect("decode artifact struct"),
    );
    field_artifact["payload_hash"] = serde_json::Value::String(artifact_hash.clone());
    field_artifact["signature"] = serde_json::Value::String(thermo_field_signature(
        &artifact_hash,
        "release-bot",
        "runmat-dev-thermo-signing-key",
    ));
    fs::write(
        root.join("field_ok.json"),
        serde_json::to_vec_pretty(&field_artifact).expect("encode thermo field artifact"),
    )
    .expect("write thermo field artifact");
    let run = analysis_run_transient_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisTransientRunOptions {
            thermo_mechanical_coupling: Some(ThermoMechanicalCouplingOptions {
                enabled: true,
                reference_temperature_k: 293.15,
                applied_temperature_delta_k: 70.0,
                thermal_expansion_coefficient: 1.1e-5,
                field_artifact_id: Some("field_ok".to_string()),
                field_source: None,
                region_temperature_deltas: Vec::new(),
                time_profile: Vec::new(),
            }),
            ..AnalysisTransientRunOptions::default()
        },
        OperationContext::new(None, None),
    )
    .expect("transient run should resolve thermo field artifact");

    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("results should succeed");

    let _ = fs::remove_dir_all(&root);

    assert!(results.data.summary.thermo_spatial_coverage_ratio.is_some());
    assert_eq!(results.data.summary.thermo_region_delta_count, Some(1.0));
}

#[test]
fn analysis_run_transient_rejects_missing_thermo_field_artifact() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "transient_1".to_string(),
        kind: AnalysisStepKind::Transient,
    }];

    let root = PathBuf::from("target/runmat-analysis-artifacts/thermo-fields");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).expect("create empty thermo field artifact root");

    let err = analysis_run_transient_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisTransientRunOptions {
            thermo_mechanical_coupling: Some(ThermoMechanicalCouplingOptions {
                enabled: true,
                reference_temperature_k: 293.15,
                applied_temperature_delta_k: 70.0,
                thermal_expansion_coefficient: 1.1e-5,
                field_artifact_id: Some("missing".to_string()),
                field_source: None,
                region_temperature_deltas: Vec::new(),
                time_profile: Vec::new(),
            }),
            ..AnalysisTransientRunOptions::default()
        },
        OperationContext::new(None, None),
    )
    .expect_err("missing thermo field artifact should be rejected");
    let _ = fs::remove_dir_all(&root);

    assert_eq!(err.error_code, "ANALYSIS_RUN_THERMO_FIELD_NOT_FOUND");
}

#[test]
fn analysis_run_transient_artifact_backed_thermo_matches_inline_profile() {
    let _guard = analysis_test_guard();
    let mut model = sample_model_with_material_assignment_mismatch();
    model.steps = vec![AnalysisStep {
        step_id: "transient_1".to_string(),
        kind: AnalysisStepKind::Transient,
    }];

    let root = PathBuf::from("target/runmat-analysis-artifacts/thermo-fields");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).expect("create thermo field artifact root");
    let mut inline_equivalent_artifact = serde_json::json!({
        "schema_version": "analysis_thermo_field_artifact/v1",
        "source_geometry_id": model.geometry_id,
        "source_geometry_revision": model.geometry_revision,
        "artifact_status": "approved",
        "approved_by": "release-bot",
        "field_source": {
            "source_id": "artifact/inline-equivalent",
            "revision": 1,
            "interpolation_mode": "linear",
            "expected_region_ids": []
        },
        "region_temperature_deltas": [
            {"region_id": "tip", "temperature_delta_k": 90.0}
        ],
        "time_profile": [
            {"normalized_time": 0.0, "scale": 0.4},
            {"normalized_time": 1.0, "scale": 1.0}
        ]
    });
    let inline_hash = thermo_field_payload_hash(
        &serde_json::from_value(inline_equivalent_artifact.clone())
            .expect("decode inline artifact struct"),
    );
    inline_equivalent_artifact["payload_hash"] = serde_json::Value::String(inline_hash.clone());
    inline_equivalent_artifact["signature"] = serde_json::Value::String(thermo_field_signature(
        &inline_hash,
        "release-bot",
        "runmat-dev-thermo-signing-key",
    ));
    fs::write(
        root.join("inline_equivalent.json"),
        serde_json::to_vec_pretty(&inline_equivalent_artifact).expect("encode artifact"),
    )
    .expect("write artifact");

    let inline = analysis_run_transient_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisTransientRunOptions {
            thermo_mechanical_coupling: Some(ThermoMechanicalCouplingOptions {
                enabled: true,
                reference_temperature_k: 293.15,
                applied_temperature_delta_k: 90.0,
                thermal_expansion_coefficient: 1.2e-5,
                field_artifact_id: None,
                field_source: None,
                region_temperature_deltas: vec![ThermoRegionTemperatureDelta {
                    region_id: "tip".to_string(),
                    temperature_delta_k: 90.0,
                }],
                time_profile: vec![
                    ThermoTimeProfilePoint {
                        normalized_time: 0.0,
                        scale: 0.4,
                    },
                    ThermoTimeProfilePoint {
                        normalized_time: 1.0,
                        scale: 1.0,
                    },
                ],
            }),
            ..AnalysisTransientRunOptions::default()
        },
        OperationContext::new(None, None),
    )
    .expect("inline thermo run should succeed");

    let artifact_backed = analysis_run_transient_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisTransientRunOptions {
            thermo_mechanical_coupling: Some(ThermoMechanicalCouplingOptions {
                enabled: true,
                reference_temperature_k: 293.15,
                applied_temperature_delta_k: 90.0,
                thermal_expansion_coefficient: 1.2e-5,
                field_artifact_id: Some("inline_equivalent".to_string()),
                field_source: None,
                region_temperature_deltas: Vec::new(),
                time_profile: Vec::new(),
            }),
            ..AnalysisTransientRunOptions::default()
        },
        OperationContext::new(None, None),
    )
    .expect("artifact-backed thermo run should succeed");
    let _ = fs::remove_dir_all(&root);

    let inline_transient = inline
        .data
        .transient_results
        .as_ref()
        .expect("inline transient payload");
    let artifact_transient = artifact_backed
        .data
        .transient_results
        .as_ref()
        .expect("artifact transient payload");
    let inline_final = inline_transient
        .displacement_snapshots
        .last()
        .and_then(|field| field.as_host_f64())
        .expect("inline host displacement");
    let artifact_final = artifact_transient
        .displacement_snapshots
        .last()
        .and_then(|field| field.as_host_f64())
        .expect("artifact host displacement");
    assert_eq!(inline_final.len(), artifact_final.len());
    let mut max_abs = 0.0_f64;
    for (lhs, rhs) in inline_final.iter().zip(artifact_final.iter()) {
        max_abs = max_abs.max((lhs - rhs).abs());
    }
    assert!(max_abs <= 1.0e-12);
}

#[test]
fn transient_balanced_degrades_when_thermo_mechanical_severity_is_high() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "transient_1".to_string(),
        kind: AnalysisStepKind::Transient,
    }];

    let run = analysis_run_transient_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisTransientRunOptions {
            quality_policy: QualityPolicy::Balanced,
            adaptive_time_step: true,
            step_count: 8,
            thermo_mechanical_coupling: Some(ThermoMechanicalCouplingOptions {
                enabled: true,
                reference_temperature_k: 293.15,
                applied_temperature_delta_k: 90.0,
                thermal_expansion_coefficient: 1.0e-3,
                field_artifact_id: None,
                field_source: None,
                region_temperature_deltas: Vec::new(),
                time_profile: Vec::new(),
            }),
            ..AnalysisTransientRunOptions::default()
        },
        OperationContext::new(None, None),
    )
    .expect("transient run should return envelope");

    assert!(!run.data.publishable);
    assert_eq!(run.data.run_status, RunStatus::Degraded);
    assert!(run
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::ThermoMechanicalTransientStress));
}

#[test]
fn transient_balanced_degrades_when_thermo_heterogeneity_is_high() {
    let _guard = analysis_test_guard();
    let mut model = sample_model_with_material_assignment_mismatch();
    model.steps = vec![AnalysisStep {
        step_id: "transient_1".to_string(),
        kind: AnalysisStepKind::Transient,
    }];

    let run = analysis_run_transient_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisTransientRunOptions {
            quality_policy: QualityPolicy::Balanced,
            adaptive_time_step: true,
            step_count: 8,
            thermo_mechanical_coupling: Some(ThermoMechanicalCouplingOptions {
                enabled: true,
                reference_temperature_k: 293.15,
                applied_temperature_delta_k: 90.0,
                thermal_expansion_coefficient: 1.2e-5,
                field_artifact_id: None,
                field_source: None,
                region_temperature_deltas: Vec::new(),
                time_profile: Vec::new(),
            }),
            ..AnalysisTransientRunOptions::default()
        },
        OperationContext::new(None, None),
    )
    .expect("transient run should return envelope");

    assert!(!run.data.publishable);
    assert_eq!(run.data.run_status, RunStatus::Degraded);
    assert!(run.data.quality_reasons.iter().any(|reason| {
        reason.code == QualityReasonCode::ThermoMechanicalConstitutiveSpreadHigh
            || reason.code == QualityReasonCode::ThermoMechanicalAssignmentHeterogeneityHigh
    }));
}

#[test]
fn nonlinear_balanced_degrades_when_plasticity_severity_is_high() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];

    let run = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions {
            quality_policy: QualityPolicy::Balanced,
            plasticity_proxy: Some(PlasticityProxyOptions {
                enabled: true,
                yield_strain: 2.0e-4,
                hardening_modulus_ratio: 0.2,
                saturation_exponent: 4.0,
            }),
            ..AnalysisNonlinearRunOptions::balanced()
        },
        OperationContext::new(None, None),
    )
    .expect("nonlinear run should return envelope");

    assert!(!run.data.publishable);
    assert_eq!(run.data.run_status, RunStatus::Degraded);
    assert!(run
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::PlasticityNonlinearStress));
    assert!(run
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_PLASTIC_NONLINEAR"));
}

#[test]
fn analysis_run_modal_returns_native_modal_result() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let modal_model = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "modal_model_run".to_string(),
            profile: AnalysisCreateModelProfile::ModalStructural,
            prep_context: None,
        },
        OperationContext::new(None, None),
    )
    .expect("modal model should be created");

    let envelope = analysis_run_modal_op(
        &modal_model.data,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("modal run should produce modal result");

    assert_eq!(envelope.operation, "analysis.run_modal");
    assert_eq!(envelope.op_version, "analysis.run_modal/v1");
    assert_eq!(
        envelope.data.run.solver_method,
        "matrix_free_subspace_iteration"
    );
    assert_eq!(
        envelope.data.provenance.solver_method,
        "matrix_free_subspace_iteration"
    );
    assert_eq!(envelope.data.run_status, RunStatus::Degraded);
    assert!(!envelope.data.publishable);
    let modal = envelope
        .data
        .modal_results
        .as_ref()
        .expect("modal payload should exist");
    assert!(!modal.eigenvalues_hz.is_empty());
    assert_eq!(modal.eigenvalues_hz.len(), modal.mode_shapes.len());
    assert_eq!(modal.mode_shapes[0].field_id, "mode_shape_1");
    assert_eq!(modal.eigenvalues_hz.len(), modal.residual_norms.len());
    assert!(modal.residual_norms.iter().all(|value| value.is_finite()));
    assert_eq!(modal.modal_payload_version, "modal_results/v1");
    assert_eq!(modal.mode_units, ModalFrequencyUnits::Hz);
    assert_eq!(modal.frequency_basis, ModalFrequencyBasis::NativeEigenSolve);
    assert!(!envelope
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::ModalPlaceholder));
    assert!(envelope
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::ModalResidualExceeded));
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_MODAL_CONVERGENCE"));
}

#[test]
fn analysis_run_modal_with_options_controls_requested_mode_count() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let modal_model = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "modal_model_run_opts".to_string(),
            profile: AnalysisCreateModelProfile::ModalStructural,
            prep_context: None,
        },
        OperationContext::new(None, None),
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
            thermo_mechanical_coupling: None,
            electro_thermal_coupling: None,
            prep_context: None,
            prep_artifact_id: None,
            prep_calibration_profile: None,
        },
        OperationContext::new(None, None),
    )
    .expect("modal run should succeed with options");

    let modal = envelope
        .data
        .modal_results
        .as_ref()
        .expect("modal payload should exist");
    assert!(modal.eigenvalues_hz.len() > 0);
    assert!(modal.eigenvalues_hz.len() <= 2);
    assert!(envelope.data.provenance.deterministic_mode);
}

#[test]
fn analysis_results_include_modal_payload_for_modal_runs() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let modal_model = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "modal_model_results".to_string(),
            profile: AnalysisCreateModelProfile::ModalStructural,
            prep_context: None,
        },
        OperationContext::new(None, None),
    )
    .expect("modal model should be created");

    let run = analysis_run_modal_op(
        &modal_model.data,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("modal run should succeed");

    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("results should succeed");

    let modal = results
        .data
        .modal_results
        .as_ref()
        .expect("modal payload should propagate to results");
    assert!(!modal.eigenvalues_hz.is_empty());
    assert_eq!(modal.eigenvalues_hz.len(), modal.mode_shapes.len());
    assert_eq!(modal.eigenvalues_hz.len(), modal.residual_norms.len());
    assert_eq!(modal.modal_payload_version, "modal_results/v1");
    assert_eq!(modal.mode_units, ModalFrequencyUnits::Hz);
    assert_eq!(modal.frequency_basis, ModalFrequencyBasis::NativeEigenSolve);
    assert!(results.data.summary.mode_count > 0);
    assert_eq!(
        results.data.summary.mode_count,
        results.data.summary.available_mode_indices.len()
    );
    assert!(results.data.summary.min_frequency_hz.is_some());
    assert!(results.data.summary.max_frequency_hz.is_some());
    assert!(results.data.summary.max_modal_residual_norm.is_some());
    assert!(results.data.summary.first_mode_converged.is_some());
}

#[test]
fn analysis_results_query_can_exclude_modal_payload() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let modal_model = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "modal_model_results_filter".to_string(),
            profile: AnalysisCreateModelProfile::ModalStructural,
            prep_context: None,
        },
        OperationContext::new(None, None),
    )
    .expect("modal model should be created");
    let run = analysis_run_modal_op(
        &modal_model.data,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("modal run should succeed");

    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery {
            include_fields: Vec::new(),
            include_diagnostics: true,
            diagnostic_codes: Vec::new(),
            include_modal_results: false,
            mode_indices: Vec::new(),
            include_transient_results: true,
            transient_snapshot_indices: Vec::new(),
            include_nonlinear_results: true,
        },
        OperationContext::new(None, None),
    )
    .expect("results should succeed");

    assert!(results.data.modal_results.is_none());
}

#[test]
fn analysis_results_query_rejects_unknown_modal_mode_index() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let modal_model = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "modal_model_results_index".to_string(),
            profile: AnalysisCreateModelProfile::ModalStructural,
            prep_context: None,
        },
        OperationContext::new(None, None),
    )
    .expect("modal model should be created");
    let run = analysis_run_modal_op(
        &modal_model.data,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("modal run should succeed");

    let err = analysis_results_op(
        &run.data,
        AnalysisResultsQuery {
            include_fields: Vec::new(),
            include_diagnostics: true,
            diagnostic_codes: Vec::new(),
            include_modal_results: true,
            mode_indices: vec![10],
            include_transient_results: true,
            transient_snapshot_indices: Vec::new(),
            include_nonlinear_results: true,
        },
        OperationContext::new(None, None),
    )
    .expect_err("results should fail for unknown mode index");

    assert_eq!(err.error_code, "ANALYSIS_RESULTS_MODE_NOT_FOUND");
    assert_eq!(err.operation, "analysis.results");
    assert_eq!(err.op_version, "analysis.results/v1");
}

#[test]
fn analysis_results_include_transient_payload_for_transient_runs() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "transient_1".to_string(),
        kind: AnalysisStepKind::Transient,
    }];
    let run = analysis_run_transient_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("transient run should succeed");

    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("results should succeed");

    let transient = results
        .data
        .transient_results
        .as_ref()
        .expect("transient payload should propagate");
    assert_eq!(
        transient.integration_method,
        TransientIntegrationMethod::ImplicitEuler
    );
    assert!(!transient.time_points_s.is_empty());
    assert_eq!(
        transient.time_points_s.len(),
        transient.displacement_snapshots.len()
    );
}

#[test]
fn analysis_results_query_can_exclude_transient_payload() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "transient_1".to_string(),
        kind: AnalysisStepKind::Transient,
    }];
    let run = analysis_run_transient_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("transient run should succeed");

    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery {
            include_fields: Vec::new(),
            include_diagnostics: true,
            diagnostic_codes: Vec::new(),
            include_modal_results: true,
            mode_indices: Vec::new(),
            include_transient_results: false,
            transient_snapshot_indices: Vec::new(),
            include_nonlinear_results: true,
        },
        OperationContext::new(None, None),
    )
    .expect("results should succeed");

    assert!(results.data.transient_results.is_none());
    assert!(results.data.summary.snapshot_count > 0);
    assert_eq!(results.data.summary.time_start_s, Some(0.0));
    assert!(results.data.summary.time_end_s.unwrap_or(0.0) > 0.0);
    assert!(results.data.summary.max_transient_residual_norm.is_some());
    assert!(results.data.summary.final_step_converged.is_some());
}

#[test]
fn analysis_results_query_rejects_unknown_transient_snapshot_index() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "transient_1".to_string(),
        kind: AnalysisStepKind::Transient,
    }];
    let run = analysis_run_transient_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("transient run should succeed");

    let err = analysis_results_op(
        &run.data,
        AnalysisResultsQuery {
            include_fields: Vec::new(),
            include_diagnostics: true,
            diagnostic_codes: Vec::new(),
            include_modal_results: true,
            mode_indices: Vec::new(),
            include_transient_results: true,
            transient_snapshot_indices: vec![999],
            include_nonlinear_results: true,
        },
        OperationContext::new(None, None),
    )
    .expect_err("results should fail for unknown transient snapshot index");

    assert_eq!(
        err.error_code,
        "ANALYSIS_RESULTS_TRANSIENT_SNAPSHOT_NOT_FOUND"
    );
    assert_eq!(err.operation, "analysis.results");
    assert_eq!(err.op_version, "analysis.results/v1");
}
