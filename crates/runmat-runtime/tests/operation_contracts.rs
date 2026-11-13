#[path = "support/fea_admission.rs"]
mod fea_admission_support;
#[path = "support/nonlinear_solid.rs"]
mod nonlinear_solid;
use runmat_analysis_core::{
    AnalysisExcitation, AnalysisFieldValues, AnalysisModel, AnalysisModelId, AnalysisProcedure,
    AnalysisSequence, CoordinateFrame, EvidenceConfidence, ExcitationAction, ExcitationCase,
    ExcitationCaseActivation, ExcitationCaseId, ExcitationDistribution, ExcitationId,
    ExcitationKinematics, ExcitationTarget, Force, MaterialAssignment, PhysicalSignal,
    PhysicsFamily, Pressure, ProcedureStep, ProjectionRequirement, RegionTarget, Signal,
    SignalApproximationEvidence, SignalExtrapolation, SignalId, SignalInterpolation, SignalKnot,
    SignalOrdinate, SignalPrecision, SignalProvenance, SignalRepresentation, SignalShape,
    SignalValue, TabulatedSignal, TabulatedSignalDomain, Velocity,
};

fn exact_signal_evidence(sample_count: usize) -> SignalApproximationEvidence {
    SignalApproximationEvidence {
        max_absolute_error: Some(0.0),
        max_relative_error: Some(0.0),
        derivative_max_absolute_error: None,
        sample_count,
    }
}

fn excitation(
    id: &str,
    target: ExcitationTarget,
    signal_id: SignalId,
    distribution: ExcitationDistribution,
) -> AnalysisExcitation {
    AnalysisExcitation {
        excitation_id: ExcitationId(id.to_string()),
        target,
        frame: CoordinateFrame::Global,
        direction: None,
        distribution,
        kinematics: ExcitationKinematics::FixedFrame,
        signal_id,
        active_step_ids: Vec::new(),
        combination_group: None,
        projection: ProjectionRequirement {
            preserve_resultant: distribution == ExcitationDistribution::PreserveResultant,
            preserve_moment: false,
            relative_tolerance: 1.0e-10,
        },
    }
}

fn replace_with_force(model: &mut AnalysisModel, id: &str, region_id: &str, force_n: [f64; 3]) {
    let signal_id = SignalId(format!("{id}.signal"));
    let case_id = ExcitationCaseId(format!("{id}.case"));
    model.signals = vec![PhysicalSignal::Force(Signal {
        signal_id: signal_id.clone(),
        shape: SignalShape::Vector3,
        frame: Some(CoordinateFrame::Global),
        representation: SignalRepresentation::Constant(SignalValue::Vector3(
            force_n.map(Force::new),
        )),
        precision: SignalPrecision::Real64,
        evidence: exact_signal_evidence(1),
        provenance: SignalProvenance::Authored,
    })];
    model.excitations = vec![ExcitationAction::Force(excitation(
        id,
        ExcitationTarget::Region(RegionTarget::authored(
            runmat_meshing_core::fixtures::fixture_boundary_selection(region_id),
        )),
        signal_id,
        ExcitationDistribution::PreserveResultant,
    ))];
    model.excitation_cases = vec![ExcitationCase {
        case_id,
        action_ids: vec![ExcitationId(id.to_string())],
        activation: ExcitationCaseActivation::Always,
    }];
}

fn append_inlet_velocity_history(model: &mut AnalysisModel, id: &str, samples: [(f64, f64); 2]) {
    let signal_id = SignalId(id.to_string());
    let excitation_id = ExcitationId(format!("{id}.excitation"));
    let case_id = ExcitationCaseId(format!("{id}.case"));
    model.signals.push(PhysicalSignal::Velocity(Signal {
        signal_id: signal_id.clone(),
        shape: SignalShape::Scalar,
        frame: None,
        representation: SignalRepresentation::Tabulated(TabulatedSignal {
            domain: TabulatedSignalDomain::TimeSeconds,
            knots: samples
                .into_iter()
                .map(|sample| {
                    SignalKnot::continuous(
                        sample.0,
                        SignalOrdinate::Real(SignalValue::Scalar(Velocity::new(sample.1))),
                    )
                })
                .collect(),
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::HoldEndpoint,
        }),
        precision: SignalPrecision::Real64,
        evidence: exact_signal_evidence(samples.len()),
        provenance: SignalProvenance::Authored,
    }));
    let mut inlet = excitation(
        &excitation_id.0,
        ExcitationTarget::Region(RegionTarget::boundary(
            runmat_meshing_core::fixtures::fixture_face("fluid_inlet"),
        )),
        signal_id,
        ExcitationDistribution::TopologyField,
    );
    inlet.direction = Some(runmat_analysis_core::Direction {
        vector: [1.0, 0.0, 0.0],
    });
    model
        .excitations
        .push(ExcitationAction::InletVelocity(inlet));
    model.excitation_cases.push(ExcitationCase {
        case_id,
        action_ids: vec![excitation_id],
        activation: ExcitationCaseActivation::Always,
    });
}
use runmat_analysis_fea::fixtures::{fixture_model, FixtureId};
use runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind;
use runmat_analysis_fea::{
    fea_modal_mode_shape_field_id, fea_nonlinear_load_factor_field_id,
    fea_nonlinear_residual_norm_field_id, fea_transient_residual_norm_field_id, ComputeBackend,
    ExecutionBackend, SolverMethod, FEA_FIELD_ACOUSTIC_PARTICLE_VELOCITY,
    FEA_FIELD_ACOUSTIC_PRESSURE_MAGNITUDE, FEA_FIELD_MODAL_EIGENVALUE,
    FEA_FIELD_MODAL_FREQUENCY_HZ, FEA_FIELD_MODAL_MODAL_MASS, FEA_FIELD_MODAL_MODAL_STIFFNESS,
    FEA_FIELD_MODAL_M_ORTHOGONALITY, FEA_FIELD_MODAL_PARTICIPATION_FACTOR,
    FEA_FIELD_MODAL_RELATIVE_FREQUENCY_SEPARATION, FEA_FIELD_MODAL_RESIDUAL_NORM,
    FEA_FIELD_STRUCTURAL_BEAM_BENDING_STRESS, FEA_FIELD_STRUCTURAL_DISPLACEMENT,
    FEA_FIELD_STRUCTURAL_EQUATION_SCALE, FEA_FIELD_STRUCTURAL_RESIDUAL_NORM,
    FEA_FIELD_STRUCTURAL_VON_MISES,
};
use runmat_geometry_core::{EntityKind, GeometryAsset};
use runmat_meshing_core::{CanonicalMeshingContract, ElementOrder, StableDigest};
use runmat_runtime::analysis::storage;
use runmat_runtime::analysis::{
    analysis_plan_study_op, analysis_plan_study_sweep_op, analysis_results_by_run_id_op,
    analysis_results_compare_op, analysis_results_op, analysis_run_acoustic_with_options_op,
    analysis_run_cfd_with_options_op, analysis_run_cht_op, analysis_run_fsi_op,
    analysis_run_linear_static_op, analysis_run_linear_static_with_options,
    analysis_run_modal_with_options_op, analysis_run_nonlinear_with_options_op,
    analysis_run_study_op, analysis_run_study_sweep_op, analysis_run_thermal_with_options_op,
    analysis_run_transient_with_options_op, analysis_trends_op, analysis_validate,
    analysis_validate_study_op, analysis_validate_study_sweep_op, AnalysisExecutionPlan,
    AnalysisExecutionStep, AnalysisModalRunOptions, AnalysisNonlinearRunOptions,
    AnalysisResultsCompareQuery, AnalysisResultsQuery, AnalysisRunOptions, AnalysisRunResult,
    AnalysisSolverOptions, AnalysisStructuralTimeStepPolicy, AnalysisStudySpec,
    AnalysisStudySweepSpec, AnalysisTransientRunOptions, AnalysisTrendsQuery, ModalFrequencyBasis,
    ModalFrequencyUnits, PrecisionMode, PreconditionerMode, QualityGate, QualityPolicy,
    QualityReasonCode, RunStatus,
};
use runmat_runtime::geometry::{
    geometry_capture_view_op, geometry_inspect_op, geometry_list_regions_op, geometry_load_op,
    geometry_query_entities_op, GeometryCaptureViewSpec, GeometryEntityQuery,
};
use runmat_runtime::operations::OperationContext;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

const TRIANGLE_STL: &str = "solid tri\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid tri\n";
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

fn study_artifact_env_guard() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
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

fn assert_static_gpu_cpu_fallback_is_truthful(trace_id: &str) {
    let envelope = analysis_run_linear_static_with_options(
        &contract_linear_static_model(),
        ComputeBackend::Gpu,
        AnalysisRunOptions::default(),
        OperationContext::new(Some(trace_id.to_string()), None),
    )
    .expect("default accelerator policy permits a truthfully reported CPU fallback");
    assert_eq!(envelope.operation, "fea.run_linear_static");
    assert_eq!(envelope.op_version, "fea.run_linear_static/v1");
    assert_eq!(envelope.trace_id.as_deref(), Some(trace_id));
    let result = &envelope.data;
    assert_eq!(result.provenance.requested_backend, ComputeBackend::Gpu);
    assert_eq!(result.provenance.execution_backend, ExecutionBackend::Cpu);
    assert_eq!(
        result.run.solver_backend,
        LinearAlgebraBackendKind::CpuReference
    );
    assert!(result.run.accelerator_evidence.is_empty());
    assert_eq!(result.provenance.solver_device_apply_k_ratio, 0.0);
    assert_eq!(result.provenance.solver_host_sync_count, 0);
    assert!(!result.provenance.fallback_events.is_empty());
    assert!(result
        .run
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "FEA_ACCELERATOR_ADMISSION_FALLBACK"));
    assert!(result.numerically_acceptable);
    let wire = serde_json::to_value(&envelope).expect("serialize typed fallback result");
    assert_eq!(
        serde_json::from_value::<runmat_runtime::operations::OperationEnvelope<AnalysisRunResult>>(
            wire
        )
        .expect("fallback schema roundtrip"),
        envelope
    );
}

fn assert_registered_result_identity(
    result: &AnalysisRunResult,
    family: PhysicsFamily,
    procedure: AnalysisProcedure,
) {
    assert_eq!(result.run.identity.family(), family);
    assert_eq!(result.run.identity.procedure(), procedure);
    assert_eq!(
        result.run.identity.solution_space(),
        runmat_analysis_core::SolutionSpace::Direct
    );
}

fn write_model_solver_mesh(model: &AnalysisModel) -> (tempfile::TempDir, PathBuf) {
    let artifact_root = tempfile::tempdir().expect("create contract solver-mesh artifact root");
    let mesh_path = artifact_root.path().join("solver_mesh.cbor");
    let mut mesh =
        runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let volume_region = model.material_assignments[0]
        .selected_region_id()
        .map(|region_id| {
            runmat_meshing_core::fixtures::fixture_entity_for_selection(region_id)
                .expect("contract volume target is a canonical fixture selector")
        })
        .unwrap_or_else(|| mesh.topology.regions[0].region_id.clone());
    mesh.topology.regions[0].region_id = volume_region.clone();
    for element in &mut mesh.topology.volume_elements {
        element.region_id = volume_region.clone();
    }
    if let Some(boundary_region_id) = model.constraints.iter().find_map(|constraint| {
        let runmat_analysis_core::AnalysisConstraint::Thermal(constraint) = constraint else {
            return None;
        };
        let runmat_analysis_core::ThermalConstraintKind::Prescribed { target, .. } =
            &constraint.kind
        else {
            return None;
        };
        Some(target.region_id.as_str())
    }) {
        mesh.topology.boundary_faces[0].provenance.push(
            runmat_meshing_core::fixtures::fixture_entity_for_selection(boundary_region_id)
                .expect("contract thermal target is a canonical fixture face selector"),
        );
        mesh.topology.boundary_faces[0].provenance.sort();
    }
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest()
        .expect("seal canonical contract solver mesh");
    fs::write(
        &mesh_path,
        mesh.canonical_encode()
            .expect("encode canonical contract solver mesh"),
    )
    .expect("write canonical contract solver mesh");
    (artifact_root, mesh_path)
}

fn contract_study_spec(
    study_id: &str,
    model_id: &str,
    geometry: GeometryAsset,
    procedure_step: ProcedureStep,
) -> AnalysisStudySpec {
    let mut model = contract_linear_static_model();
    model.model_id = AnalysisModelId(model_id.to_string());
    model.geometry_id = geometry.geometry_id.clone();
    model.geometry_revision = geometry.revision;
    model.sequence =
        AnalysisSequence::from_steps(format!("{model_id}.sequence"), vec![procedure_step.clone()]);
    AnalysisStudySpec {
        study_id: study_id.to_string(),
        geometry,
        model_source_evidence: None,
        model,
        execution: AnalysisExecutionPlan {
            steps: vec![AnalysisExecutionStep {
                step_id: procedure_step.step_id.clone(),
                backend: ComputeBackend::Cpu,
                accelerator_admission: Default::default(),
                solver_options: AnalysisSolverOptions::defaults_for(&procedure_step).unwrap(),
            }],
        },
        meshing_settings: None,
        immutable_values: Vec::new(),
        solver_mesh_artifact_path: None,
        meshing_evidence_artifact_path: None,
    }
}

fn contract_linear_static_model() -> AnalysisModel {
    let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    replace_with_force(&mut model, "tip_force_y", "node:2", [0.0, -750.0, 0.0]);
    model
}

fn contract_modal_model(model_id: &str) -> AnalysisModel {
    let mut model = fixture_model(FixtureId::ModalLarge);
    model.model_id = AnalysisModelId(model_id.to_string());
    model
}

fn contract_acoustic_model(model_id: &str) -> AnalysisModel {
    let mut model = contract_linear_static_model();
    model.model_id = AnalysisModelId(model_id.to_string());
    model.structural = None;
    model.sequence = AnalysisSequence::from_steps(
        format!("{model_id}.sequence"),
        vec![ProcedureStep::acoustic_harmonic("acoustic_harmonic")],
    );
    for material in &mut model.materials {
        material.acoustic = Some(runmat_analysis_core::MaterialAcousticModel {
            speed_of_sound_m_per_s: runmat_types::physical::SpeedOfSound::new(343.0),
            damping_ratio: runmat_types::physical::DampingRatio::new(0.02),
        });
    }
    model.material_assignments = vec![MaterialAssignment {
        target: runmat_analysis_core::PropertyRegionTarget::all_cells(),
        expected_material_id: model.materials[0].material_id.clone(),
        assigned_material_id: model.materials[0].material_id.clone(),
        confidence: EvidenceConfidence::Verified,
    }];
    model.constraints.clear();
    model.boundaries = vec![runmat_analysis_core::AnalysisBoundary {
        boundary_id: "acoustic_rigid_wall".to_string(),
        target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("root")),
        operator: runmat_analysis_core::BoundaryOperator::AcousticRigidWall,
    }];
    let id = "acoustic_pressure";
    let signal_id = SignalId(format!("{id}.signal"));
    model.signals = vec![PhysicalSignal::Pressure(Signal {
        signal_id: signal_id.clone(),
        shape: SignalShape::Scalar,
        frame: None,
        representation: SignalRepresentation::Constant(SignalValue::Scalar(Pressure::new(1.0))),
        precision: SignalPrecision::Real64,
        evidence: exact_signal_evidence(1),
        provenance: SignalProvenance::Authored,
    })];
    model.excitations = vec![ExcitationAction::AcousticPressure(excitation(
        id,
        ExcitationTarget::Region(RegionTarget::boundary(
            runmat_meshing_core::fixtures::fixture_face("tip"),
        )),
        signal_id,
        ExcitationDistribution::PreserveResultant,
    ))];
    model.excitation_cases = vec![ExcitationCase {
        case_id: ExcitationCaseId(format!("{id}.case")),
        action_ids: vec![ExcitationId(id.to_string())],
        activation: ExcitationCaseActivation::Always,
    }];
    model
}

fn contract_thermal_model(model_id: &str, procedure: AnalysisProcedure) -> AnalysisModel {
    let mut model = fixture_model(FixtureId::ThermoRampSmooth);
    model.model_id = AnalysisModelId(model_id.to_string());
    let material_id = model.materials[0].material_id.clone();
    model.material_assignments = vec![MaterialAssignment {
        target: runmat_analysis_core::PropertyRegionTarget::all_solids(),
        expected_material_id: material_id.clone(),
        assigned_material_id: material_id,
        confidence: EvidenceConfidence::Verified,
    }];
    model.thermal = Some(runmat_analysis_core::ThermalDomain {
        reference_temperature: runmat_analysis_core::Temperature::new(293.15),
    });
    model.constraints = vec![runmat_analysis_core::AnalysisConstraint::Thermal(
        runmat_analysis_core::ThermalConstraint {
            constraint_id: "thermal_reference".to_string(),
            kind: runmat_analysis_core::ThermalConstraintKind::Prescribed {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("root")),
                value: runmat_analysis_core::PrescribedValue::Constant(
                    runmat_analysis_core::Temperature::new(293.15),
                ),
            },
        },
    )];
    model.sequence = AnalysisSequence::from_steps(
        format!("{model_id}.sequence"),
        vec![match procedure {
            AnalysisProcedure::Static => ProcedureStep::thermal_steady("thermal_steady"),
            AnalysisProcedure::TransientResponse => {
                ProcedureStep::thermal_transient("thermal_transient")
            }
            other => panic!("unsupported thermal contract procedure {other:?}"),
        }],
    );
    model
}

fn contract_cht_model(model_id: &str) -> AnalysisModel {
    let mut model = fixture_model(FixtureId::ThermoMechanicalKickoff);
    model.model_id = AnalysisModelId(model_id.to_string());
    let material_id = model.materials[0].material_id.clone();
    model.material_assignments = vec![MaterialAssignment {
        target: runmat_analysis_core::PropertyRegionTarget::all_solids(),
        expected_material_id: material_id.clone(),
        assigned_material_id: material_id,
        confidence: EvidenceConfidence::Verified,
    }];
    model.thermal = Some(runmat_analysis_core::ThermalDomain {
        reference_temperature: runmat_analysis_core::Temperature::new(293.15),
    });
    model.thermo_mechanical = Some(runmat_analysis_core::ThermoMechanicalDomain {
        reference_temperature: runmat_analysis_core::Temperature::new(293.15),
    });
    model.sequence = AnalysisSequence::from_steps(
        format!("{model_id}.sequence"),
        vec![ProcedureStep::coupled(
            "cht",
            runmat_analysis_core::CoupledPhysicsFamily::ConjugateHeatTransfer,
            AnalysisProcedure::TransientResponse,
        )],
    );
    model.cfd = Some(runmat_analysis_core::CfdDomain {
        density: runmat_analysis_core::MassDensity::new(1.225),
        dynamic_viscosity: runmat_analysis_core::DynamicViscosity::new(1.81e-5),
        turbulence_intensity: runmat_analysis_core::TurbulenceIntensity::new(0.06),
        rheology: None,
        porous_zones: Vec::new(),
    });
    model
}

fn material_conflict_contract_model() -> AnalysisModel {
    let mut model = contract_linear_static_model();
    let mut polymer = model.materials[0].clone();
    polymer.material_id = "mat_polymer".to_string();
    polymer.name = "Polymer".to_string();
    polymer
        .require_mechanical_mut()
        .expect("structural fixture material")
        .youngs_modulus_pa = runmat_types::physical::YoungsModulus::new(3.0e9);
    model.materials.push(polymer);
    model.material_assignments = vec![MaterialAssignment {
        target: runmat_analysis_core::PropertyRegionTarget::beam(
            runmat_meshing_core::fixtures::fixture_edge("beam_span"),
        ),
        expected_material_id: "mat_steel".to_string(),
        assigned_material_id: "mat_polymer".to_string(),
        confidence: EvidenceConfidence::Inferred,
    }];
    model
}

#[test]
fn geometry_operation_contracts_are_v1_and_versioned() {
    let _fea_admission = fea_admission_support::guard();
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
    assert_eq!(
        capture_view.error_code,
        "RM.GEOMETRY.CAPTURE_VIEW.UNSUPPORTED"
    );

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
    assert_eq!(
        invalid_capture_view.error_code,
        "RM.GEOMETRY.CAPTURE_VIEW.INVALID_SPEC"
    );
}

#[test]
fn analysis_validate_contract_is_v1_and_maps_codes() {
    let _fea_admission = fea_admission_support::guard();
    let model = contract_linear_static_model();
    let context = OperationContext::new(Some("trace-contract-2".to_string()), None);
    let envelope = analysis_validate(&model, &CoordinateFrame::Global, context)
        .expect("validate should succeed");
    assert_eq!(envelope.operation, "fea.validate");
    assert_eq!(envelope.op_version, "fea.validate/v1");

    let mut invalid = model;
    invalid.materials.clear();
    let err = analysis_validate(
        &invalid,
        &CoordinateFrame::Global,
        OperationContext::new(None, None),
    )
    .expect_err("validate should fail");
    assert_eq!(err.operation, "fea.validate");
    assert_eq!(err.op_version, "fea.validate/v1");
    assert_eq!(err.error_code, "RM.FEA.VALIDATE.MISSING_MATERIALS");
}

#[test]
fn analysis_study_workflow_contract_persists_evidence_artifacts() {
    let _fea_admission = fea_admission_support::guard();
    {
        let _study_artifact_env_guard = study_artifact_env_guard();
        static NEXT_TMP_ID: AtomicU64 = AtomicU64::new(1);
        let evidence_root = std::env::temp_dir().join(format!(
            "runmat-study-contract-artifacts-{}-{}",
            std::process::id(),
            NEXT_TMP_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&evidence_root);
        let env_guard = EnvVarRestoreGuard {
            key: "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
            previous: std::env::var("RUNMAT_FEA_STUDY_ARTIFACT_ROOT").ok(),
        };
        std::env::set_var(
            "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
            evidence_root.display().to_string(),
        );

        let geometry = geometry_load_op(
            "/part.stl",
            TRIANGLE_STL.as_bytes(),
            OperationContext::new(Some("trace-contract-study-1".to_string()), None),
        )
        .expect("geometry load should succeed");
        let spec = contract_study_spec(
            "contract_study_linear_static",
            "contract_study_model",
            geometry.data.clone(),
            ProcedureStep::structural_static("step_default_static"),
        );

        let validate = analysis_validate_study_op(
            &spec,
            OperationContext::new(Some("trace-contract-study-2".to_string()), None),
        )
        .expect("validate study should succeed");
        assert_eq!(validate.operation, "fea.validate_study");
        assert_eq!(validate.op_version, "fea.validate_study/v1");
        assert!(validate.data.valid);
        assert_eq!(validate.data.issues.len(), 0);
        assert!(PathBuf::from(&validate.data.evidence_artifact_path).exists());

        let plan = analysis_plan_study_op(
            &spec,
            OperationContext::new(Some("trace-contract-study-3".to_string()), None),
        )
        .expect("plan study should succeed");
        assert_eq!(plan.operation, "fea.plan_study");
        assert_eq!(plan.op_version, "fea.plan_study/v1");
        assert!(plan.data.study_fingerprint.starts_with("sha256:"));
        assert!(matches!(
            plan.data.steps[0].execution.solver_options,
            AnalysisSolverOptions::StructuralLinear(_)
        ));
        assert_eq!(plan.data.steps[0].run_operation, "fea.run_linear_static");
        assert_eq!(
            plan.data.steps[0].run_op_version,
            "fea.run_linear_static/v1"
        );
        assert!(PathBuf::from(&plan.data.evidence_artifact_path).exists());

        let run = analysis_run_study_op(
            &spec,
            OperationContext::new(Some("trace-contract-study-4".to_string()), None),
        )
        .expect("run study should succeed");
        assert_eq!(run.operation, "fea.run_study");
        assert_eq!(run.op_version, "fea.run_study/v1");
        assert_eq!(run.data.study_fingerprint, plan.data.study_fingerprint);
        assert!(matches!(
            run.data.steps[0].execution.solver_options,
            AnalysisSolverOptions::StructuralLinear(_)
        ));
        assert_eq!(run.data.steps[0].run_operation, "fea.run_linear_static");
        assert_eq!(run.data.steps[0].run_op_version, "fea.run_linear_static/v1");
        assert_eq!(run.data.solver_convergence, QualityGate::Pass);
        assert_eq!(run.data.result_quality, QualityGate::Pass);
        assert_eq!(
            run.data.steps[0].identity.family(),
            PhysicsFamily::Structural
        );
        assert_eq!(
            run.data.steps[0].identity.procedure(),
            AnalysisProcedure::Static
        );
        assert!(run.data.numerically_acceptable);
        assert_eq!(run.data.steps[0].quality_reasons.len(), 0);
        assert_eq!(
            run.data.steps[0].provenance.requested_backend,
            ComputeBackend::Cpu
        );
        assert!(PathBuf::from(&run.data.evidence_artifact_path).exists());

        drop(env_guard);
        let _ = fs::remove_dir_all(&evidence_root);
    }
}

#[test]
fn analysis_study_sweep_contract_executes_multiple_studies() {
    let _fea_admission = fea_admission_support::guard();
    {
        let _study_artifact_env_guard = study_artifact_env_guard();
        static NEXT_TMP_ID: AtomicU64 = AtomicU64::new(1);
        let evidence_root = std::env::temp_dir().join(format!(
            "runmat-study-sweep-contract-artifacts-{}-{}",
            std::process::id(),
            NEXT_TMP_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&evidence_root);
        let env_guard = EnvVarRestoreGuard {
            key: "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
            previous: std::env::var("RUNMAT_FEA_STUDY_ARTIFACT_ROOT").ok(),
        };
        std::env::set_var(
            "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
            evidence_root.display().to_string(),
        );

        let geometry = geometry_load_op(
            "/part.stl",
            TRIANGLE_STL.as_bytes(),
            OperationContext::new(Some("trace-contract-study-sweep-1".to_string()), None),
        )
        .expect("geometry load should succeed");
        let studies = vec![
            contract_study_spec(
                "contract_study_sweep_linear",
                "contract_study_sweep_linear_model",
                geometry.data.clone(),
                ProcedureStep::structural_static("step_default_static"),
            ),
            contract_study_spec(
                "contract_study_sweep_secondary",
                "contract_study_sweep_secondary_model",
                geometry.data.clone(),
                ProcedureStep::structural_static("step_second_static"),
            ),
        ];
        let sweep = AnalysisStudySweepSpec {
            sweep_id: "contract_study_sweep_001".to_string(),
            studies,
            fail_fast: true,
        };

        let envelope = analysis_run_study_sweep_op(
            &sweep,
            OperationContext::new(Some("trace-contract-study-sweep-2".to_string()), None),
        )
        .expect("study sweep should succeed");
        assert_eq!(envelope.operation, "fea.run_study_sweep");
        assert_eq!(envelope.op_version, "fea.run_study_sweep/v1");
        assert_eq!(envelope.data.sweep_id, "contract_study_sweep_001");
        assert_eq!(envelope.data.study_count, 2);
        assert_eq!(envelope.data.success_count, 2);
        assert_eq!(envelope.data.failed_count, 0);
        assert!(envelope.data.failure_entries.is_empty());
        assert_eq!(envelope.data.run_entries.len(), 2);
        assert!(envelope.data.run_entries.iter().all(|entry| {
            entry.result.steps[0].run_id.starts_with("run_")
                && matches!(
                    entry.result.steps[0].procedure_step.family,
                    PhysicsFamily::Structural | PhysicsFamily::Electromagnetic
                )
        }));
        assert!(PathBuf::from(&envelope.data.evidence_artifact_path).exists());

        drop(env_guard);
        let _ = fs::remove_dir_all(&evidence_root);
    }
}

#[test]
fn analysis_study_sweep_contract_can_continue_on_failure() {
    let _fea_admission = fea_admission_support::guard();
    {
        let _study_artifact_env_guard = study_artifact_env_guard();
        static NEXT_TMP_ID: AtomicU64 = AtomicU64::new(1);
        let evidence_root = std::env::temp_dir().join(format!(
            "runmat-study-sweep-continue-contract-artifacts-{}-{}",
            std::process::id(),
            NEXT_TMP_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&evidence_root);
        let env_guard = EnvVarRestoreGuard {
            key: "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
            previous: std::env::var("RUNMAT_FEA_STUDY_ARTIFACT_ROOT").ok(),
        };
        std::env::set_var(
            "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
            evidence_root.display().to_string(),
        );

        let geometry = geometry_load_op(
            "/part.stl",
            TRIANGLE_STL.as_bytes(),
            OperationContext::new(
                Some("trace-contract-study-sweep-continue-1".to_string()),
                None,
            ),
        )
        .expect("geometry load should succeed");
        let mut invalid_study = contract_study_spec(
            "contract_study_sweep_invalid",
            "contract_study_sweep_invalid_model",
            geometry.data.clone(),
            ProcedureStep::structural_static("step_default_static"),
        );
        invalid_study.study_id = "   ".to_string();
        let sweep = AnalysisStudySweepSpec {
            sweep_id: "contract_study_sweep_continue_001".to_string(),
            studies: vec![
                contract_study_spec(
                    "contract_study_sweep_ok",
                    "contract_study_sweep_ok_model",
                    geometry.data.clone(),
                    ProcedureStep::structural_static("step_default_static"),
                ),
                invalid_study,
            ],
            fail_fast: false,
        };

        let envelope = analysis_run_study_sweep_op(
            &sweep,
            OperationContext::new(
                Some("trace-contract-study-sweep-continue-2".to_string()),
                None,
            ),
        )
        .expect("continue-on-failure sweep should succeed");
        assert_eq!(envelope.operation, "fea.run_study_sweep");
        assert_eq!(envelope.op_version, "fea.run_study_sweep/v1");
        assert_eq!(envelope.data.study_count, 2);
        assert_eq!(envelope.data.success_count, 1);
        assert_eq!(envelope.data.failed_count, 1);
        assert_eq!(envelope.data.run_entries.len(), 1);
        assert_eq!(envelope.data.failure_entries.len(), 1);
        assert_eq!(envelope.data.failure_entries[0].study_index, 1);
        assert_eq!(
            envelope.data.failure_entries[0].error_code,
            "RM.FEA.RUN_STUDY.INVALID_SPEC"
        );
        assert!(PathBuf::from(&envelope.data.evidence_artifact_path).exists());

        drop(env_guard);
        let _ = fs::remove_dir_all(&evidence_root);
    }
}

#[test]
fn analysis_study_sweep_validate_contract_is_v1_and_typed() {
    let _fea_admission = fea_admission_support::guard();
    let _study_artifact_env_guard = study_artifact_env_guard();
    static NEXT_TMP_ID: AtomicU64 = AtomicU64::new(1);
    let evidence_root = std::env::temp_dir().join(format!(
        "runmat-study-sweep-validate-contract-artifacts-{}-{}",
        std::process::id(),
        NEXT_TMP_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&evidence_root);
    let env_guard = EnvVarRestoreGuard {
        key: "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
        previous: std::env::var("RUNMAT_FEA_STUDY_ARTIFACT_ROOT").ok(),
    };
    std::env::set_var(
        "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
        evidence_root.display().to_string(),
    );

    let geometry = geometry_load_op(
        "/part.stl",
        TRIANGLE_STL.as_bytes(),
        OperationContext::new(
            Some("trace-contract-study-sweep-validate-1".to_string()),
            None,
        ),
    )
    .expect("geometry load should succeed");
    let mut invalid_study = contract_study_spec(
        "contract_study_sweep_invalid",
        "contract_study_sweep_invalid_model",
        geometry.data.clone(),
        ProcedureStep::structural_static("step_default_static"),
    );
    invalid_study.study_id = "   ".to_string();
    let sweep = AnalysisStudySweepSpec {
        sweep_id: "   ".to_string(),
        studies: vec![
            contract_study_spec(
                "contract_study_sweep_validate_ok",
                "contract_study_sweep_validate_ok_model",
                geometry.data.clone(),
                ProcedureStep::structural_static("step_default_static"),
            ),
            invalid_study,
        ],
        fail_fast: true,
    };

    let envelope = analysis_validate_study_sweep_op(
        &sweep,
        OperationContext::new(
            Some("trace-contract-study-sweep-validate-2".to_string()),
            None,
        ),
    )
    .expect("sweep validation should return a typed payload");
    assert_eq!(envelope.operation, "fea.validate_study_sweep");
    assert_eq!(envelope.op_version, "fea.validate_study_sweep/v1");
    assert!(!envelope.data.valid);
    assert_eq!(
        envelope.data.issue_codes,
        vec!["RM.FEA.STUDY_SWEEP.ID_EMPTY".to_string()]
    );
    assert_eq!(envelope.data.study_entries.len(), 2);
    assert!(envelope.data.study_entries.iter().any(|entry| {
        !entry.valid
            && entry
                .issue_codes
                .iter()
                .any(|code| code == "RM.FEA.STUDY.ID_EMPTY")
    }));
    assert!(PathBuf::from(&envelope.data.evidence_artifact_path).exists());

    drop(env_guard);
    let _ = fs::remove_dir_all(&evidence_root);
}

#[test]
fn analysis_study_sweep_plan_contract_is_v1_and_typed() {
    let _fea_admission = fea_admission_support::guard();
    let _study_artifact_env_guard = study_artifact_env_guard();
    static NEXT_TMP_ID: AtomicU64 = AtomicU64::new(1);
    let evidence_root = std::env::temp_dir().join(format!(
        "runmat-study-sweep-plan-contract-artifacts-{}-{}",
        std::process::id(),
        NEXT_TMP_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&evidence_root);
    let env_guard = EnvVarRestoreGuard {
        key: "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
        previous: std::env::var("RUNMAT_FEA_STUDY_ARTIFACT_ROOT").ok(),
    };
    std::env::set_var(
        "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
        evidence_root.display().to_string(),
    );

    let geometry = geometry_load_op(
        "/part.stl",
        TRIANGLE_STL.as_bytes(),
        OperationContext::new(Some("trace-contract-study-sweep-plan-1".to_string()), None),
    )
    .expect("geometry load should succeed");
    let sweep = AnalysisStudySweepSpec {
        sweep_id: "contract_study_sweep_plan_001".to_string(),
        studies: vec![
            contract_study_spec(
                "contract_study_sweep_plan_ok",
                "contract_study_sweep_plan_ok_model",
                geometry.data.clone(),
                ProcedureStep::structural_static("step_default_static"),
            ),
            contract_study_spec(
                "contract_study_sweep_plan_secondary",
                "contract_study_sweep_plan_secondary_model",
                geometry.data.clone(),
                ProcedureStep::structural_static("step_second_static"),
            ),
        ],
        fail_fast: true,
    };

    let envelope = analysis_plan_study_sweep_op(
        &sweep,
        OperationContext::new(Some("trace-contract-study-sweep-plan-2".to_string()), None),
    )
    .expect("sweep planning should return typed payload");
    assert_eq!(envelope.operation, "fea.plan_study_sweep");
    assert_eq!(envelope.op_version, "fea.plan_study_sweep/v1");
    assert_eq!(envelope.data.study_count, 2);
    assert_eq!(envelope.data.planned_count, 2);
    assert_eq!(envelope.data.failed_count, 0);
    assert_eq!(envelope.data.plan_entries.len(), 2);
    assert!(envelope.data.failure_entries.is_empty());
    assert!(envelope
        .data
        .plan_entries
        .iter()
        .any(
            |entry| entry.steps[0].procedure_step.family == PhysicsFamily::Structural
                && entry.steps[0].procedure_step.procedure == AnalysisProcedure::Static
        ));
    assert!(envelope.data.plan_entries.iter().any(|entry| {
        entry.steps[0].procedure_step.step_id.0 == "step_second_static"
            && entry.steps[0].procedure_step.family == PhysicsFamily::Structural
            && entry.steps[0].procedure_step.procedure == AnalysisProcedure::Static
    }));
    assert!(PathBuf::from(&envelope.data.evidence_artifact_path).exists());

    drop(env_guard);
    let _ = fs::remove_dir_all(&evidence_root);
}

#[test]
fn analysis_model_decode_rejects_noncanonical_units_and_validate_maps_frame_mismatch() {
    let _fea_admission = fea_admission_support::guard();
    let canonical = contract_linear_static_model();
    let mut wire = serde_json::to_value(&canonical).expect("encode canonical model");
    assert_eq!(
        wire["units"],
        serde_json::to_value(runmat_types::physical::LengthUnit::Meter).unwrap()
    );
    serde_json::from_value::<AnalysisModel>(wire.clone()).expect("decode canonical model");
    wire["units"] = serde_json::to_value(runmat_types::physical::LengthUnit::Inch).unwrap();
    let unit_err = serde_json::from_value::<AnalysisModel>(wire)
        .expect_err("canonical model admission must reject inch coordinates");
    assert!(unit_err
        .to_string()
        .contains("canonical solver coordinates must be meters"));

    let mut frame_mismatch = contract_linear_static_model();
    frame_mismatch.frame = CoordinateFrame::Cartesian(runmat_analysis_core::CartesianFrame {
        frame_id: "fixture_frame".to_string(),
        origin: runmat_analysis_core::Position::new([0.0, 0.0, 0.0]),
        x_axis: [1.0, 0.0, 0.0],
        y_axis: [0.0, 1.0, 0.0],
    });
    let frame_err = analysis_validate(
        &frame_mismatch,
        &CoordinateFrame::Global,
        OperationContext::new(Some("trace-contract-2c".to_string()), None),
    )
    .expect_err("validate should fail for frame");
    assert_eq!(frame_err.error_code, "RM.FEA.VALIDATE.FRAME_MISMATCH");
    assert_eq!(
        frame_err.context.get("model_frame").map(|s| s.as_str()),
        Some("cartesian:fixture_frame")
    );
    assert_eq!(
        frame_err.context.get("geometry_frame").map(|s| s.as_str()),
        Some("global")
    );
}

#[test]
fn analysis_run_contract_is_v1_and_numerically_accepted_for_fixture() {
    let _fea_admission = fea_admission_support::guard();
    {
        let model = contract_linear_static_model();
        let envelope = analysis_run_linear_static_with_options(
            &model,
            ComputeBackend::Cpu,
            Default::default(),
            OperationContext::new(Some("trace-contract-3".to_string()), None),
        )
        .expect("run should succeed");

        assert_eq!(envelope.operation, "fea.run_linear_static");
        assert_eq!(envelope.op_version, "fea.run_linear_static/v1");
        assert_eq!(envelope.data.run_status, RunStatus::Completed);
        assert!(envelope.data.numerically_acceptable);
        assert_registered_result_identity(
            &envelope.data,
            PhysicsFamily::Structural,
            AnalysisProcedure::Static,
        );

        assert!(envelope
            .data
            .run
            .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
            .is_some());
        assert!(envelope
            .data
            .run
            .field(FEA_FIELD_STRUCTURAL_BEAM_BENDING_STRESS)
            .is_some());
        assert!(envelope
            .data
            .run
            .field(FEA_FIELD_STRUCTURAL_VON_MISES)
            .is_none());
        assert!(envelope
            .data
            .run
            .field(FEA_FIELD_STRUCTURAL_RESIDUAL_NORM)
            .is_some());
        assert!(envelope
            .data
            .run
            .field(FEA_FIELD_STRUCTURAL_EQUATION_SCALE)
            .is_some());
        assert!(matches!(
            envelope
                .data
                .run
                .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
                .expect("structural displacement field should be present")
                .values,
            AnalysisFieldValues::HostF64(_)
        ));
    }
}

#[test]
fn convenience_run_accepts_production_qualified_beam_formulation() {
    let _fea_admission = fea_admission_support::guard();
    {
        let result = analysis_run_linear_static_op(
            &contract_linear_static_model(),
            ComputeBackend::Cpu,
            OperationContext::new(Some("trace-contract-production-gate".to_string()), None),
        )
        .expect("the beam fixture has a production-qualified exact formulation");

        assert_eq!(result.operation, "fea.run_linear_static");
        assert_eq!(result.op_version, "fea.run_linear_static/v1");
        assert!(result.data.numerically_acceptable);
        assert_eq!(result.data.run_status, RunStatus::Completed);
    }
}

#[test]
fn analysis_run_modal_contract_is_v1_and_typed() {
    let _fea_admission = fea_admission_support::guard();
    {
        let modal_model = contract_modal_model("contract_modal_model");

        let modal_envelope = analysis_run_modal_with_options_op(
            &modal_model,
            ComputeBackend::Cpu,
            Default::default(),
            OperationContext::new(Some("trace-contract-modal-3".to_string()), None),
        )
        .expect("modal run should produce envelope");
        assert_eq!(modal_envelope.operation, "fea.run_modal");
        assert_eq!(modal_envelope.op_version, "fea.run_modal/v1");
        assert_eq!(
            modal_envelope.data.run.solver_method,
            SolverMethod::ShiftInvertBlockLanczos
        );
        let modal_results = modal_envelope
            .data
            .modal_results
            .as_ref()
            .expect("modal results payload should exist");
        assert!(!modal_results.frequencies_hz().is_empty());
        assert_eq!(
            modal_results.frequencies_hz().len(),
            modal_results.mode_shapes.len()
        );
        assert_eq!(
            modal_results.mode_shapes[0].field_id,
            fea_modal_mode_shape_field_id(1)
        );
        assert_eq!(modal_results.mode_shapes[0].shape.len(), 2);
        assert_eq!(modal_results.mode_shapes[0].shape[1], 3);
        assert_eq!(
            modal_results.mode_indices.len(),
            modal_results.basis.modes().len()
        );
        assert_eq!(
            modal_results.frequencies_hz().len(),
            modal_results.residual_norms().len()
        );
        assert_eq!(modal_results.mode_units, ModalFrequencyUnits::Hz);
        assert_eq!(
            modal_results.frequency_basis,
            ModalFrequencyBasis::NativeEigenSolve
        );
        assert!(modal_envelope
            .data
            .run
            .field(FEA_FIELD_MODAL_FREQUENCY_HZ)
            .is_some());
        assert!(modal_envelope
            .data
            .run
            .field(FEA_FIELD_MODAL_EIGENVALUE)
            .is_some());
        assert!(modal_envelope
            .data
            .run
            .field(FEA_FIELD_MODAL_MODAL_MASS)
            .is_some());
        assert!(modal_envelope
            .data
            .run
            .field(FEA_FIELD_MODAL_MODAL_STIFFNESS)
            .is_some());
        assert!(modal_envelope
            .data
            .run
            .field(FEA_FIELD_MODAL_PARTICIPATION_FACTOR)
            .is_some());
        assert!(modal_envelope
            .data
            .run
            .field(FEA_FIELD_MODAL_RESIDUAL_NORM)
            .is_some());
        assert!(modal_envelope
            .data
            .run
            .field(FEA_FIELD_MODAL_RELATIVE_FREQUENCY_SEPARATION)
            .is_some());
        assert!(modal_envelope
            .data
            .run
            .field(FEA_FIELD_MODAL_M_ORTHOGONALITY)
            .is_some());
        assert_eq!(modal_envelope.data.run_status, RunStatus::Completed);
        assert!(modal_envelope.data.numerically_acceptable);
        assert!(!modal_envelope
            .data
            .quality_reasons
            .iter()
            .any(|reason| reason.code == QualityReasonCode::ModalResidualExceeded));

        let invalid = analysis_run_modal_with_options_op(
            &contract_linear_static_model(),
            ComputeBackend::Cpu,
            Default::default(),
            OperationContext::new(Some("trace-contract-modal-4".to_string()), None),
        )
        .expect_err("modal run should reject models without modal step");
        assert_eq!(invalid.operation, "fea.run_modal");
        assert_eq!(invalid.op_version, "fea.run_modal/v1");
        assert_eq!(invalid.error_code, "RM.FEA.RUN_MODAL.INVALID_MODEL");
    }
}

#[test]
fn analysis_run_modal_with_options_contract_controls_mode_budget() {
    let _fea_admission = fea_admission_support::guard();
    {
        let modal_model = contract_modal_model("contract_modal_model_opts");

        let envelope = analysis_run_modal_with_options_op(
            &modal_model,
            ComputeBackend::Cpu,
            AnalysisModalRunOptions {
                deterministic_mode: true,
                precision_mode: PrecisionMode::Fp64,
                quality_policy: QualityPolicy::Balanced,
                truncation: runmat_analysis_fea::ModalTruncationPolicy::requested(2),
                convergence: runmat_analysis_fea::ModalConvergencePolicy::production(),
                residual_vector_directions: Vec::new(),
                solver_mesh_artifact_path: None,
            },
            OperationContext::new(Some("trace-contract-modal-opts-3".to_string()), None),
        )
        .expect("modal run with options should succeed");

        assert_eq!(envelope.operation, "fea.run_modal");
        assert_eq!(envelope.op_version, "fea.run_modal/v1");
        let modal = envelope
            .data
            .modal_results
            .as_ref()
            .expect("modal payload should exist");
        assert!(!modal.frequencies_hz().is_empty());
        assert!(modal.frequencies_hz().len() >= 2);
        assert!(envelope.data.provenance.deterministic_mode);

        let invalid = analysis_run_modal_with_options_op(
            &modal_model,
            ComputeBackend::Cpu,
            AnalysisModalRunOptions {
                deterministic_mode: false,
                precision_mode: PrecisionMode::Fp64,
                quality_policy: QualityPolicy::Balanced,
                truncation: runmat_analysis_fea::ModalTruncationPolicy::requested(0),
                convergence: runmat_analysis_fea::ModalConvergencePolicy::production(),
                residual_vector_directions: Vec::new(),
                solver_mesh_artifact_path: None,
            },
            OperationContext::new(Some("trace-contract-modal-opts-4".to_string()), None),
        )
        .expect_err("requested modal count zero should fail");
        assert_eq!(invalid.error_code, "RM.FEA.RUN_MODAL.INVALID_OPTIONS");
        assert_eq!(invalid.operation, "fea.run_modal");
        assert_eq!(invalid.op_version, "fea.run_modal/v1");
    }
}

#[test]
fn analysis_run_acoustic_contract_is_v1_and_typed() {
    let _fea_admission = fea_admission_support::guard();
    {
        let acoustic_model = contract_acoustic_model("contract_acoustic_model");

        let envelope = analysis_run_acoustic_with_options_op(
            &acoustic_model,
            ComputeBackend::Cpu,
            Default::default(),
            OperationContext::new(Some("trace-contract-acoustic-3".to_string()), None),
        )
        .expect("acoustic run should produce envelope");
        assert_eq!(envelope.operation, "fea.run_acoustic");
        assert_eq!(envelope.op_version, "fea.run_acoustic/v1");
        assert!(envelope.data.modal_results.is_none());
        assert!(envelope
            .data
            .run
            .diagnostics
            .iter()
            .any(|diag| diag.code == "FEA_ACOUSTIC_HARMONIC_RESPONSE"));
        assert!(envelope
            .data
            .run
            .diagnostics
            .iter()
            .any(|diag| diag.code == "FEA_ACOUSTIC_DOMAIN_ASSEMBLY"));
        assert!(envelope
            .data
            .run
            .diagnostics
            .iter()
            .any(|diag| diag.code == "FEA_ACOUSTIC_HELMHOLTZ_RESIDUAL"));
        assert!(envelope
            .data
            .run
            .field(FEA_FIELD_ACOUSTIC_PRESSURE_MAGNITUDE)
            .is_some());
        assert!(envelope
            .data
            .run
            .field(FEA_FIELD_ACOUSTIC_PARTICLE_VELOCITY)
            .is_some());

        let invalid = analysis_run_acoustic_with_options_op(
            &contract_linear_static_model(),
            ComputeBackend::Cpu,
            Default::default(),
            OperationContext::new(Some("trace-contract-acoustic-4".to_string()), None),
        )
        .expect_err("acoustic run should reject models without acoustic harmonic step marker");
        assert_eq!(invalid.operation, "fea.run_acoustic");
        assert_eq!(invalid.op_version, "fea.run_acoustic/v1");
        assert_eq!(invalid.error_code, "RM.FEA.RUN_ACOUSTIC.INVALID_MODEL");
    }
}

#[test]
fn analysis_run_cfd_requires_authored_execution_without_proxy_fields() {
    let _fea_admission = fea_admission_support::guard();
    {
        let mut model = contract_linear_static_model();
        model.sequence.steps = vec![runmat_analysis_core::ProcedureStep::fluid_static(
            "cfd_steady".to_string(),
        )];
        model.cfd = Some(runmat_analysis_core::CfdDomain {
            density: runmat_analysis_core::MassDensity::new(1.225),
            dynamic_viscosity: runmat_analysis_core::DynamicViscosity::new(1.81e-5),
            turbulence_intensity: runmat_analysis_core::TurbulenceIntensity::new(0.06),
            rheology: None,
            porous_zones: Vec::new(),
        });
        model.boundaries.clear();
        // Explicit signed components are authored by this fixture, never inferred from speed.
        let velocity = [-4.25, 1.0, -2.0].map(Velocity::new);
        model.constraints = vec![runmat_analysis_core::AnalysisConstraint::Fluid(
            runmat_analysis_core::FluidConstraint {
                constraint_id: "bc_cfd_inlet".into(),
                kind: runmat_analysis_core::FluidConstraintKind::PrescribedVelocity {
                    target: RegionTarget::authored(runmat_meshing_core::fixtures::fixture_face(
                        "fluid_inlet",
                    )),
                    value: runmat_analysis_core::PrescribedValue::Constant(velocity),
                },
            },
        )];

        let envelope = analysis_run_cfd_with_options_op(
            &model,
            ComputeBackend::Cpu,
            Default::default(),
            OperationContext::new(Some("trace-contract-cfd-1".to_string()), None),
        )
        .expect_err("the removed reduced executor cannot publish fields");
        assert_eq!(envelope.operation, "fea.run_cfd");
        assert_eq!(envelope.op_version, "fea.run_cfd/v1");
        assert_eq!(
            envelope.error_code,
            "RM.FEA.RUN_CFD.AUTHORED_STUDY_REQUIRED"
        );
        assert!(!envelope.retryable);
        assert!(envelope.message.contains("canonical finite-volume"));
        let wire = serde_json::to_value(&envelope).unwrap();
        assert!(
            wire.get("data").is_none(),
            "rejection must not publish proxy fields"
        );
    }
}

#[test]
fn analysis_run_cht_contract_requires_an_authored_study() {
    let _fea_admission = fea_admission_support::guard();
    let model = contract_cht_model("contract_cht_model");
    let envelope = analysis_run_cht_op(
        &model,
        OperationContext::new(Some("trace-contract-cht-3".to_string()), None),
    )
    .expect_err("the removed reduced executor cannot publish fields");
    assert_eq!(envelope.operation, "fea.run_cht");
    assert_eq!(envelope.op_version, "fea.run_cht/v1");
    assert_eq!(
        envelope.error_code,
        "RM.FEA.RUN_CHT.AUTHORED_STUDY_REQUIRED"
    );
    assert!(!envelope.retryable);
    let wire = serde_json::to_value(&envelope).unwrap();
    assert!(
        wire.get("data").is_none(),
        "rejection must not publish proxy fields"
    );
}

#[test]
fn analysis_run_thermal_contract_preserves_steady_and_transient_identities() {
    let _fea_admission = fea_admission_support::guard();
    {
        let model = contract_thermal_model(
            "contract_thermal_model",
            AnalysisProcedure::TransientResponse,
        );
        let (_artifact_root, mesh_path) = write_model_solver_mesh(&model);

        let envelope = analysis_run_thermal_with_options_op(
            &model,
            ComputeBackend::Cpu,
            runmat_runtime::analysis::AnalysisThermalRunOptions {
                solver_mesh_artifact_path: Some(mesh_path.display().to_string()),
                ..Default::default()
            },
            OperationContext::new(Some("trace-contract-thermal-3".to_string()), None),
        )
        .expect("thermal transient run should succeed");
        assert_eq!(envelope.operation, "fea.run_thermal");
        assert_eq!(envelope.op_version, "fea.run_thermal/v1");
        assert_registered_result_identity(
            &envelope.data,
            PhysicsFamily::Thermal,
            AnalysisProcedure::TransientResponse,
        );

        let steady_model =
            contract_thermal_model("contract_steady_thermal_model", AnalysisProcedure::Static);
        let (_steady_artifact_root, steady_mesh_path) = write_model_solver_mesh(&steady_model);
        let steady = analysis_run_thermal_with_options_op(
            &steady_model,
            ComputeBackend::Cpu,
            runmat_runtime::analysis::AnalysisThermalRunOptions {
                solver_mesh_artifact_path: Some(steady_mesh_path.display().to_string()),
                ..Default::default()
            },
            OperationContext::new(Some("trace-contract-thermal-5".to_string()), None),
        )
        .expect("steady thermal run should succeed");
        assert_registered_result_identity(
            &steady.data,
            PhysicsFamily::Thermal,
            AnalysisProcedure::Static,
        );
        let steady_payload = steady
            .data
            .thermal_results
            .expect("steady run should publish thermal results");
        assert_eq!(steady_payload.time_points_s, vec![0.0]);
        assert_eq!(steady_payload.temperature_snapshots.len(), 1);
    }
}

#[test]
fn thermomechanical_transient_contract_fails_closed() {
    let _fea_admission = fea_admission_support::guard();
    {
        let model = fixture_model(FixtureId::ThermoMechanicalKickoff);

        let error = analysis_run_transient_with_options_op(
            &model,
            ComputeBackend::Cpu,
            Default::default(),
            OperationContext::new(Some("trace-contract-thermomechanical-3".to_string()), None),
        )
        .expect_err("thermomechanical transient response is not an admitted solver family");
        assert_eq!(error.operation, "fea.run_transient");
        assert_eq!(error.op_version, "fea.run_transient/v1");
        assert_eq!(error.error_code, "RM.FEA.RUN_TRANSIENT.INVALID_MODEL");
    }
}

#[test]
fn analysis_run_transient_contract_is_v1_and_typed() {
    let _fea_admission = fea_admission_support::guard();
    {
        let mut model = contract_linear_static_model();
        model.sequence.steps = vec![runmat_analysis_core::ProcedureStep::structural_transient(
            "transient_1".to_string(),
        )];

        let envelope = analysis_run_transient_with_options_op(
            &model,
            ComputeBackend::Cpu,
            Default::default(),
            OperationContext::new(Some("trace-contract-transient-1".to_string()), None),
        )
        .expect("transient run should return envelope");
        assert_eq!(envelope.operation, "fea.run_transient");
        assert_eq!(envelope.op_version, "fea.run_transient/v1");
        assert_eq!(
            envelope.data.run.solver_method,
            SolverMethod::NewmarkAverageAccelerationPcg
        );
        assert_eq!(envelope.data.run_status, RunStatus::Completed);
        assert!(envelope.data.numerically_acceptable);
        assert_eq!(envelope.data.provenance.precision_mode, "fp64");
        let transient = envelope
            .data
            .transient_results
            .as_ref()
            .expect("transient payload should exist");
        assert_eq!(
            transient.integration_method,
            runmat_runtime::analysis::TransientIntegrationMethod::NewmarkAverageAcceleration
        );
        assert_eq!(
            transient.time_points_s.len(),
            transient.displacement_snapshots.len()
        );
        assert_eq!(
            transient.time_points_s.len(),
            transient.residual_norm_snapshots.len()
        );
        assert!(transient.strain_snapshots.is_empty());
        assert!(transient.stress_snapshots.is_empty());
        assert!(transient.strain_energy_density_snapshots.is_empty());
        assert_eq!(
            transient.residual_norm_snapshots[1].field_id,
            fea_transient_residual_norm_field_id(1)
        );

        let invalid = analysis_run_transient_with_options_op(
            &contract_linear_static_model(),
            ComputeBackend::Cpu,
            Default::default(),
            OperationContext::new(Some("trace-contract-transient-2".to_string()), None),
        )
        .expect_err("transient run should reject models without transient step");
        assert_eq!(invalid.operation, "fea.run_transient");
        assert_eq!(invalid.op_version, "fea.run_transient/v1");
        assert_eq!(invalid.error_code, "RM.FEA.RUN_TRANSIENT.INVALID_MODEL");
    }
}

#[test]
fn analysis_run_fsi_contract_requires_an_authored_study() {
    let _fea_admission = fea_admission_support::guard();
    let mut model = contract_linear_static_model();
    model.sequence = runmat_analysis_core::AnalysisSequence::from_steps(
        "fsi_sequence",
        vec![runmat_analysis_core::ProcedureStep::coupled(
            "fsi",
            runmat_analysis_core::CoupledPhysicsFamily::FluidStructureInteraction,
            runmat_analysis_core::AnalysisProcedure::TransientResponse,
        )],
    );
    model.cfd = Some(runmat_analysis_core::CfdDomain {
        density: runmat_analysis_core::MassDensity::new(1.225),
        dynamic_viscosity: runmat_analysis_core::DynamicViscosity::new(1.81e-5),
        turbulence_intensity: runmat_analysis_core::TurbulenceIntensity::new(0.06),
        rheology: None,
        porous_zones: Vec::new(),
    });
    append_inlet_velocity_history(&mut model, "fsi_inlet_velocity", [(0.0, 2.4), (1.0, 4.0)]);

    let envelope = analysis_run_fsi_op(
        &model,
        OperationContext::new(Some("trace-contract-fsi-1".to_string()), None),
    )
    .expect_err("the removed reduced executor cannot publish fields");
    assert_eq!(envelope.operation, "fea.run_fsi");
    assert_eq!(envelope.op_version, "fea.run_fsi/v1");
    assert_eq!(
        envelope.error_code,
        "RM.FEA.RUN_FSI.AUTHORED_STUDY_REQUIRED"
    );
    let wire = serde_json::to_value(&envelope).unwrap();
    assert!(
        wire.get("data").is_none(),
        "rejection must not publish proxy fields"
    );
}

#[test]
fn analysis_run_nonlinear_contract_is_v1_and_typed() {
    let _fea_admission = fea_admission_support::guard();
    {
        let (_mesh_root, model, options) = nonlinear_solid::fixture();
        let envelope = analysis_run_nonlinear_with_options_op(
            &model,
            ComputeBackend::Cpu,
            options,
            OperationContext::new(Some("trace-contract-nonlinear-1".to_string()), None),
        )
        .expect("nonlinear run should succeed for nonlinear fixture");

        assert_eq!(envelope.operation, "fea.run_nonlinear");
        assert_eq!(envelope.op_version, "fea.run_nonlinear/v1");
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
        assert_eq!(
            nonlinear.load_factors.len(),
            nonlinear.load_factor_snapshots.len()
        );
        assert_eq!(
            nonlinear.residual_norms.len(),
            nonlinear.residual_norm_snapshots.len()
        );
        assert_eq!(
            nonlinear.load_factor_snapshots[0].field_id,
            fea_nonlinear_load_factor_field_id(0)
        );
        assert_eq!(
            nonlinear.residual_norm_snapshots[0].field_id,
            fea_nonlinear_residual_norm_field_id(0)
        );
        assert!(
            nonlinear.max_line_search_backtracks_per_increment <= nonlinear.line_search_backtracks
        );
        assert!(nonlinear.iteration_spike_count <= nonlinear.load_factors.len());
        assert_eq!(nonlinear.load_factors.last(), Some(&1.0));
        // Independent uniaxial J2 solution for the canonical four-node solid:
        // equivalent plastic strain = (stress - yield stress) / hardening.
        let plastic_strain = (5.6 - 5.0) / 30.0;
        let axial_strain = 5.6 / 1200.0 + plastic_strain;
        let lateral_strain = -0.25 * 5.6 / 1200.0 - plastic_strain / 2.0;
        let expected = [
            0.0,
            0.0,
            0.0,
            axial_strain,
            0.0,
            0.0,
            0.0,
            lateral_strain,
            0.0,
            0.0,
            0.0,
            lateral_strain,
        ];
        let actual = envelope
            .data
            .run
            .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
            .expect("canonical nonlinear solid displacement")
            .as_host_f64()
            .unwrap();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1e-9,
                "actual={actual}; expected={expected}"
            );
        }
        assert!(envelope
            .data
            .run
            .diagnostics
            .iter()
            .any(|diag| diag.code == "FEA_NONLINEAR_CONVERGENCE"));
        assert!(nonlinear.tangent_rebuild_count > 0);

        let results = analysis_results_op(
            &envelope.data,
            AnalysisResultsQuery::default(),
            OperationContext::new(Some("trace-contract-nonlinear-results-1".to_string()), None),
        )
        .expect("fea.results should succeed for nonlinear run");
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

        let invalid = analysis_run_nonlinear_with_options_op(
            &contract_linear_static_model(),
            ComputeBackend::Cpu,
            Default::default(),
            OperationContext::new(Some("trace-contract-nonlinear-2".to_string()), None),
        )
        .expect_err("nonlinear run should reject models without nonlinear step");
        assert_eq!(invalid.operation, "fea.run_nonlinear");
        assert_eq!(invalid.op_version, "fea.run_nonlinear/v1");
        assert_eq!(invalid.error_code, "RM.FEA.RUN_NONLINEAR.INVALID_MODEL");
    }
}

#[test]
fn analysis_results_can_filter_nonlinear_diagnostics_by_code() {
    let _fea_admission = fea_admission_support::guard();
    {
        let (_mesh_root, model, options) = nonlinear_solid::fixture();
        let envelope = analysis_run_nonlinear_with_options_op(
            &model,
            ComputeBackend::Cpu,
            options,
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
                include_field_values: true,

                include_diagnostics: true,
                diagnostic_codes: vec![
                    "FEA_NONLINEAR_CONVERGENCE".to_string(),
                    "FEA_NONLINEAR_COST".to_string(),
                ],
                include_modal_results: true,
                include_buckling_results: true,
                mode_indices: Vec::new(),
                include_transient_results: true,
                transient_snapshot_indices: Vec::new(),
                include_nonlinear_results: true,
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
}

#[test]
fn nonlinear_contract_snapshot_matches_expected_shape() {
    let _fea_admission = fea_admission_support::guard();
    {
        let (_mesh_root, model, options) = nonlinear_solid::fixture();
        let envelope = analysis_run_nonlinear_with_options_op(
            &model,
            ComputeBackend::Cpu,
            options,
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

        let contract_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/nonlinear_contract_snapshot.json");
        let expected: Value = serde_json::from_str(
            &fs::read_to_string(&contract_path).expect("read nonlinear contract snapshot"),
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
}

#[test]
fn analysis_run_nonlinear_strict_iteration_cap_refuses_unconverged_publication() {
    let _fea_admission = fea_admission_support::guard();
    {
        let (_mesh_root, model, options) = nonlinear_solid::fixture();
        let published = storage::list_run_metadata().unwrap().len();
        let error = analysis_run_nonlinear_with_options_op(
            &model,
            ComputeBackend::Cpu,
            AnalysisNonlinearRunOptions {
                quality_policy: QualityPolicy::Strict,
                increment_count: 1,
                max_newton_iters: 1,
                max_load_cutbacks: 0,
                line_search: false,
                ..options
            },
            OperationContext::new(
                Some("trace-contract-nonlinear-strict-cap".to_string()),
                None,
            ),
        )
        .expect_err("an exhausted physical solve must not publish a nonlinear result");

        assert_eq!(error.operation, "fea.run_nonlinear");
        assert_eq!(error.op_version, "fea.run_nonlinear/v1");
        assert_eq!(error.error_code, "RM.FEA.RUN_NONLINEAR.ASSEMBLY_FAILED");
        assert!(!error.retryable);
        assert_eq!(storage::list_run_metadata().unwrap().len(), published);
    }
}

#[test]
fn analysis_run_nonlinear_quality_policy_cannot_publish_unconverged_physical_solid() {
    let _fea_admission = fea_admission_support::guard();
    {
        let (_mesh_root, model, options) = nonlinear_solid::fixture();
        let published = storage::list_run_metadata().unwrap().len();
        let run_with_policy = |policy| {
            analysis_run_nonlinear_with_options_op(
                &model,
                ComputeBackend::Cpu,
                AnalysisNonlinearRunOptions {
                    quality_policy: policy,
                    increment_count: 1,
                    max_newton_iters: 1,
                    max_load_cutbacks: 0,
                    line_search: false,
                    max_line_search_backtracks: 0,
                    ..options.clone()
                },
                OperationContext::new(
                    Some(format!("trace-contract-nonlinear-policy-{policy:?}")),
                    None,
                ),
            )
            .expect_err("quality policy cannot authorize an unconverged physical result")
        };

        let exploratory = run_with_policy(QualityPolicy::Exploratory);
        let balanced = run_with_policy(QualityPolicy::Balanced);
        let strict = run_with_policy(QualityPolicy::Strict);

        for error in [exploratory, balanced, strict] {
            assert_eq!(error.operation, "fea.run_nonlinear");
            assert_eq!(error.error_code, "RM.FEA.RUN_NONLINEAR.ASSEMBLY_FAILED");
            assert!(!error.retryable);
        }
        assert_eq!(storage::list_run_metadata().unwrap().len(), published);
    }
}

#[test]
fn analysis_run_nonlinear_physical_solution_is_consistent_across_quality_policies() {
    let _fea_admission = fea_admission_support::guard();
    {
        let (_mesh_root, model, options) = nonlinear_solid::fixture();
        let run_with_policy = |policy| {
            analysis_run_nonlinear_with_options_op(
                &model,
                ComputeBackend::Cpu,
                AnalysisNonlinearRunOptions {
                    quality_policy: policy,
                    ..options.clone()
                },
                OperationContext::new(
                    Some(format!("trace-contract-nonlinear-hard-policy-{policy:?}")),
                    None,
                ),
            )
            .expect("canonical plastic solid should produce a physical nonlinear result")
        };

        let exploratory = run_with_policy(QualityPolicy::Exploratory);
        let balanced = run_with_policy(QualityPolicy::Balanced);
        let strict = run_with_policy(QualityPolicy::Strict);

        let mut reference_displacement = None;
        for result in [exploratory, balanced, strict] {
            assert!(result.data.numerically_acceptable);
            assert_eq!(result.data.run_status, RunStatus::Completed);
            assert_eq!(
                result
                    .data
                    .nonlinear_results
                    .as_ref()
                    .unwrap()
                    .load_factors
                    .last(),
                Some(&1.0)
            );
            let field = result
                .data
                .run
                .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
                .expect("physical nonlinear displacement");
            assert!(field
                .as_host_f64()
                .unwrap()
                .iter()
                .all(|value| value.is_finite()));
            if let Some(reference) = &reference_displacement {
                assert_eq!(field, reference);
            } else {
                reference_displacement = Some(field.clone());
            }
        }
    }
}

#[test]
fn analysis_results_compare_contract_is_v1_and_handles_missing_run_ids() {
    let _fea_admission = fea_admission_support::guard();
    {
        let _artifact_store_guard = study_artifact_env_guard();
        let (_mesh_root, model, options) = nonlinear_solid::fixture();
        let baseline = analysis_run_nonlinear_with_options_op(
            &model,
            ComputeBackend::Cpu,
            options.clone(),
            OperationContext::new(Some("trace-contract-compare-1".to_string()), None),
        )
        .expect("baseline nonlinear run should succeed");
        let candidate = analysis_run_nonlinear_with_options_op(
            &model,
            ComputeBackend::Cpu,
            options,
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
        assert_eq!(compare.operation, "fea.results_compare");
        assert_eq!(compare.op_version, "fea.results_compare/v1");
        // The physical solid does not publish the retired proxy's timing diagnostic.
        assert_eq!(compare.data.solve_ms_delta, None);
        assert_eq!(compare.data.baseline_run_id, baseline.data.run_id);
        assert_eq!(compare.data.candidate_run_id, candidate.data.run_id);
        assert!(!compare.data.run_status_changed);
        assert!(!compare.data.numerically_acceptable_changed);
        assert_eq!(compare.data.failed_increment_delta, Some(0));
        assert_eq!(compare.data.max_iteration_delta, Some(0));

        let missing = analysis_results_compare_op(
            AnalysisResultsCompareQuery {
                baseline_run_id: "missing-run-id".to_string(),
                candidate_run_id: candidate.data.run_id,
            },
            OperationContext::new(Some("trace-contract-compare-4".to_string()), None),
        )
        .expect_err("missing baseline run should fail");
        assert_eq!(missing.operation, "fea.results_compare");
        assert_eq!(missing.op_version, "fea.results_compare/v1");
        assert_eq!(missing.error_code, "RM.FEA.RESULTS_COMPARE.RUN_NOT_FOUND");
    }
}

#[test]
fn analysis_trends_contract_is_v1_and_typed() {
    let _fea_admission = fea_admission_support::guard();
    {
        let _artifact_store_guard = study_artifact_env_guard();
        storage::configure_artifact_store(storage::AnalysisArtifactStoreConfig::InMemory)
            .expect("configure isolated in-memory artifact store");
        let (_mesh_root, model, options) = nonlinear_solid::fixture();
        let mut seeded_identity = None;
        for idx in 0..3 {
            let run = analysis_run_nonlinear_with_options_op(
                &model,
                ComputeBackend::Cpu,
                options.clone(),
                OperationContext::new(Some(format!("trace-contract-trends-{idx}")), None),
            )
            .expect("nonlinear run should succeed for trends");
            match seeded_identity.as_ref() {
                Some(identity) => assert_eq!(&run.data.run.identity, identity),
                None => seeded_identity = Some(run.data.run.identity.clone()),
            }
        }
        let seeded_identity = seeded_identity.expect("a trend identity should be seeded");

        let trends = analysis_trends_op(
            AnalysisTrendsQuery { window_size: 2 },
            OperationContext::new(Some("trace-contract-trends-op".to_string()), None),
        )
        .expect("trends operation should succeed");
        assert_eq!(trends.operation, "fea.trends");
        assert_eq!(trends.op_version, "fea.trends/v1");
        assert_eq!(trends.data.window_size, 2);
        let nonlinear = trends
            .data
            .summaries
            .iter()
            .find(|summary| summary.identity == seeded_identity)
            .expect("seeded structural nonlinear trend summary should be present");
        assert_eq!(nonlinear.sample_count, 2);
        // Missing timing stays absent rather than becoming a fabricated zero sample.
        assert_eq!(nonlinear.median_solve_ms, None);
        assert_eq!(nonlinear.p95_solve_ms, None);
        assert!(nonlinear.thermo_coupling_enabled_rate.is_none());
        assert!(nonlinear.thermo_nonlinear_warn_rate.is_none());
        assert!(nonlinear.thermo_spread_breach_rate.is_none());
        assert!(nonlinear.thermo_heterogeneity_breach_rate.is_none());
    }
}

#[test]
fn analysis_run_transient_with_options_contract_controls_execution_window() {
    let _fea_admission = fea_admission_support::guard();
    {
        let mut model = contract_linear_static_model();
        model.sequence.steps = vec![runmat_analysis_core::ProcedureStep::structural_transient(
            "transient_1".to_string(),
        )];

        let envelope = analysis_run_transient_with_options_op(
            &model,
            ComputeBackend::Cpu,
            AnalysisTransientRunOptions {
                deterministic_mode: true,
                quality_policy: QualityPolicy::Balanced,
                time_step_policy: AnalysisStructuralTimeStepPolicy::Fixed { interval_count: 4 },
                residual_target: 1.0e-6,
                maximum_intervals_per_block: 256,
                solver_mesh_artifact_path: None,
            },
            OperationContext::new(Some("trace-contract-transient-opts-1".to_string()), None),
        )
        .expect("transient run with options should succeed");

        assert_eq!(envelope.operation, "fea.run_transient");
        assert_eq!(envelope.op_version, "fea.run_transient/v1");
        let transient = envelope
            .data
            .transient_results
            .as_ref()
            .expect("transient payload should exist");
        assert_eq!(transient.time_points_s.len(), 5);
        assert!((transient.time_points_s[4] - 1.0).abs() < 1.0e-12);
        assert!(envelope.data.provenance.deterministic_mode);
    }
}

#[test]
fn analysis_modal_large_fixture_quality_signals_propagate_to_results() {
    let _fea_admission = fea_admission_support::guard();
    {
        let model = fixture_model(FixtureId::ModalLarge);
        let envelope = analysis_run_modal_with_options_op(
            &model,
            ComputeBackend::Cpu,
            AnalysisModalRunOptions {
                truncation: runmat_analysis_fea::ModalTruncationPolicy::requested(8),
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
            assert!(!envelope.data.numerically_acceptable);
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
}

#[test]
fn analysis_transient_long_fixture_quality_signals_propagate_to_results() {
    let _fea_admission = fea_admission_support::guard();
    {
        let model = fixture_model(FixtureId::TransientLong);
        let envelope = analysis_run_transient_with_options_op(
            &model,
            ComputeBackend::Cpu,
            AnalysisTransientRunOptions {
                time_step_policy: AnalysisStructuralTimeStepPolicy::Fixed { interval_count: 24 },
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
            assert!(!envelope.data.numerically_acceptable);
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
}

#[test]
fn analysis_results_contract_is_v1_and_filterable() {
    let _fea_admission = fea_admission_support::guard();
    {
        let model = contract_linear_static_model();
        let run = analysis_run_linear_static_with_options(
            &model,
            ComputeBackend::Cpu,
            Default::default(),
            OperationContext::new(Some("trace-contract-3b-run".to_string()), None),
        )
        .expect("run should succeed");

        let results = analysis_results_op(
            &run.data,
            AnalysisResultsQuery {
                include_fields: vec![FEA_FIELD_STRUCTURAL_BEAM_BENDING_STRESS.to_string()],
                include_field_values: true,

                include_diagnostics: false,
                diagnostic_codes: Vec::new(),
                include_modal_results: true,
                include_buckling_results: true,
                mode_indices: Vec::new(),
                include_transient_results: true,
                transient_snapshot_indices: Vec::new(),
                include_nonlinear_results: true,
            },
            OperationContext::new(Some("trace-contract-3b-results".to_string()), None),
        )
        .expect("results should succeed");

        assert_eq!(results.operation, "fea.results");
        assert_eq!(results.op_version, "fea.results/v1");
        assert_eq!(results.data.fields.len(), 1);
        assert_eq!(
            results.data.fields[0].field_id,
            FEA_FIELD_STRUCTURAL_BEAM_BENDING_STRESS
        );
        assert!(results.data.diagnostics.is_none());
    }
}

#[test]
fn analysis_results_unknown_field_maps_typed_error_contract() {
    let _fea_admission = fea_admission_support::guard();
    {
        let model = contract_linear_static_model();
        let run = analysis_run_linear_static_with_options(
            &model,
            ComputeBackend::Cpu,
            Default::default(),
            OperationContext::new(Some("trace-contract-3c-run".to_string()), None),
        )
        .expect("run should succeed");

        let err = analysis_results_op(
            &run.data,
            AnalysisResultsQuery {
                include_fields: vec!["nonexistent".to_string()],
                include_field_values: true,

                include_diagnostics: true,
                diagnostic_codes: Vec::new(),
                include_modal_results: true,
                include_buckling_results: true,
                mode_indices: Vec::new(),
                include_transient_results: true,
                transient_snapshot_indices: Vec::new(),
                include_nonlinear_results: true,
            },
            OperationContext::new(Some("trace-contract-3c-results".to_string()), None),
        )
        .expect_err("results should fail");

        assert_eq!(err.operation, "fea.results");
        assert_eq!(err.op_version, "fea.results/v1");
        assert_eq!(err.error_code, "RM.FEA.RESULTS.FIELD_NOT_FOUND");
    }
}

#[test]
fn analysis_results_modal_query_controls_are_typed() {
    let _fea_admission = fea_admission_support::guard();
    {
        let modal_model = contract_modal_model("contract_modal_results_model");
        let modal_run = analysis_run_modal_with_options_op(
            &modal_model,
            ComputeBackend::Cpu,
            Default::default(),
            OperationContext::new(Some("trace-contract-modal-results-3".to_string()), None),
        )
        .expect("modal run should succeed");

        let excluded = analysis_results_op(
            &modal_run.data,
            AnalysisResultsQuery {
                include_fields: Vec::new(),
                include_field_values: true,

                include_diagnostics: true,
                diagnostic_codes: Vec::new(),
                include_modal_results: false,
                include_buckling_results: false,
                mode_indices: Vec::new(),
                include_transient_results: true,
                transient_snapshot_indices: Vec::new(),
                include_nonlinear_results: true,
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
                include_field_values: true,

                include_diagnostics: true,
                diagnostic_codes: Vec::new(),
                include_modal_results: true,
                include_buckling_results: true,
                mode_indices: vec![99],
                include_transient_results: true,
                transient_snapshot_indices: Vec::new(),
                include_nonlinear_results: true,
            },
            OperationContext::new(Some("trace-contract-modal-results-5".to_string()), None),
        )
        .expect_err("unknown mode index should fail");
        assert_eq!(invalid_mode.error_code, "RM.FEA.RESULTS.MODE_NOT_FOUND");
        assert_eq!(invalid_mode.operation, "fea.results");
        assert_eq!(invalid_mode.op_version, "fea.results/v1");
    }
}

#[test]
fn analysis_results_transient_query_controls_are_typed() {
    let _fea_admission = fea_admission_support::guard();
    {
        let mut model = contract_linear_static_model();
        model.sequence.steps = vec![runmat_analysis_core::ProcedureStep::structural_transient(
            "transient_1".to_string(),
        )];
        let transient_run = analysis_run_transient_with_options_op(
            &model,
            ComputeBackend::Cpu,
            Default::default(),
            OperationContext::new(Some("trace-contract-transient-results-1".to_string()), None),
        )
        .expect("transient run should succeed");

        let excluded = analysis_results_op(
            &transient_run.data,
            AnalysisResultsQuery {
                include_fields: Vec::new(),
                include_field_values: true,

                include_diagnostics: true,
                diagnostic_codes: Vec::new(),
                include_modal_results: true,
                include_buckling_results: true,
                mode_indices: Vec::new(),
                include_transient_results: false,
                transient_snapshot_indices: Vec::new(),
                include_nonlinear_results: true,
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
                include_field_values: true,

                include_diagnostics: true,
                diagnostic_codes: Vec::new(),
                include_modal_results: true,
                include_buckling_results: true,
                mode_indices: Vec::new(),
                include_transient_results: true,
                transient_snapshot_indices: vec![999],
                include_nonlinear_results: true,
            },
            OperationContext::new(Some("trace-contract-transient-results-3".to_string()), None),
        )
        .expect_err("unknown transient snapshot index should fail");
        assert_eq!(
            invalid_snapshot.error_code,
            "RM.FEA.RESULTS.TRANSIENT_SNAPSHOT_NOT_FOUND"
        );
        assert_eq!(invalid_snapshot.operation, "fea.results");
        assert_eq!(invalid_snapshot.op_version, "fea.results/v1");
    }
}

#[test]
fn analysis_results_by_run_id_contract_roundtrip() {
    let _fea_admission = fea_admission_support::guard();
    {
        let _artifact_store_guard = study_artifact_env_guard();
        let model = contract_linear_static_model();
        let run = analysis_run_linear_static_with_options(
            &model,
            ComputeBackend::Cpu,
            Default::default(),
            OperationContext::new(Some("trace-contract-results-id-run".to_string()), None),
        )
        .expect("run should succeed");

        let results = analysis_results_by_run_id_op(
            &run.data.run_id,
            AnalysisResultsQuery::default(),
            OperationContext::new(Some("trace-contract-results-id-get".to_string()), None),
        )
        .expect("results by run id should succeed");

        assert_eq!(results.operation, "fea.results");
        assert_eq!(results.op_version, "fea.results/v1");
        assert_eq!(results.data.summary.field_count, results.data.fields.len());
        assert_eq!(results.data.summary.field_count, run.data.run.fields.len());
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
        assert_eq!(results.data.summary.thermo_nonlinear_severity, None);
    }
}

#[test]
fn analysis_results_by_run_id_missing_maps_typed_error_contract() {
    let _fea_admission = fea_admission_support::guard();
    let err = analysis_results_by_run_id_op(
        "run_does_not_exist",
        AnalysisResultsQuery::default(),
        OperationContext::new(Some("trace-contract-results-id-missing".to_string()), None),
    )
    .expect_err("results by run id should fail");

    assert_eq!(err.operation, "fea.results");
    assert_eq!(err.op_version, "fea.results/v1");
    assert_eq!(err.error_code, "RM.FEA.RESULTS.RUN_NOT_FOUND");
}

#[test]
fn analysis_run_contract_maps_fixture_validation_failure() {
    let _fea_admission = fea_admission_support::guard();
    let missing_materials = fixture_model(FixtureId::MissingMaterials);
    let err = analysis_run_linear_static_with_options(
        &missing_materials,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(Some("trace-contract-4".to_string()), None),
    )
    .expect_err("run should fail");
    assert_eq!(err.operation, "fea.run_linear_static");
    assert_eq!(err.op_version, "fea.run_linear_static/v1");
    assert_eq!(
        err.error_code,
        "RM.FEA.RUN_LINEAR_STATIC.SOLVER_MODEL_INVALID"
    );
    assert_eq!(
        err.context.get("analysis_model_id").map(|s| s.as_str()),
        Some("missing_materials")
    );
}

#[test]
fn multi_material_confidence_mismatch_degrades_publishability() {
    let _fea_admission = fea_admission_support::guard();
    {
        let model = material_conflict_contract_model();
        let envelope = analysis_run_linear_static_with_options(
            &model,
            ComputeBackend::Cpu,
            Default::default(),
            OperationContext::new(Some("trace-contract-multi-material".to_string()), None),
        )
        .expect("run should succeed");

        assert_eq!(envelope.data.run_status, RunStatus::Degraded);
        assert!(!envelope.data.numerically_acceptable);
        assert!(envelope
            .data
            .quality_reasons
            .iter()
            .any(|reason| reason.code == QualityReasonCode::MaterialAssignmentConflict));
        assert!(envelope
            .data
            .run
            .diagnostics
            .iter()
            .any(|diag| diag.code == "ANALYSIS_MATERIAL_ASSIGNMENT_CONFLICT_INFERRED"));
    }
}

#[test]
fn analysis_run_deterministic_contract_is_stable_across_replays() {
    let _fea_admission = fea_admission_support::guard();
    {
        let model = contract_linear_static_model();
        let options = AnalysisRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            preconditioner_mode: PreconditionerMode::Auto,
            quality_policy: QualityPolicy::Balanced,
            solver_mesh_artifact_path: None,
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

        assert_eq!(first.data.run_status, RunStatus::Completed);
        assert_eq!(second.data.run_status, RunStatus::Completed);
        assert!(first.data.numerically_acceptable);
        assert!(second.data.numerically_acceptable);
        assert_eq!(first.data.provenance.precision_mode, "fp64");
        assert_eq!(second.data.provenance.precision_mode, "fp64");
        assert!(first.data.provenance.deterministic_mode);
        assert!(second.data.provenance.deterministic_mode);
        assert_eq!(first.data.run.fields, second.data.run.fields);
        assert_eq!(first.data.run.diagnostics, second.data.run.diagnostics);
    }
}

#[test]
fn analysis_run_cpu_provenance_and_gpu_admission_are_truthful() {
    let _fea_admission = fea_admission_support::guard();
    {
        let _guard = runmat_accelerate_api::ThreadProviderGuard::set(None);
        let cpu = analysis_run_linear_static_with_options(
            &contract_linear_static_model(),
            ComputeBackend::Cpu,
            AnalysisRunOptions::default(),
            OperationContext::new(Some("trace-contract-7a".to_string()), None),
        )
        .expect("cpu run should succeed");
        assert_eq!(cpu.data.provenance.requested_backend, ComputeBackend::Cpu);
        assert_eq!(cpu.data.provenance.execution_backend, ExecutionBackend::Cpu);
        assert_eq!(
            cpu.data.provenance.solver_backend,
            LinearAlgebraBackendKind::CpuReference
        );
        assert_eq!(cpu.data.provenance.solver_device_apply_k_ratio, 0.0);
        assert_eq!(cpu.data.provenance.solver_host_sync_count, 0);
        assert!(cpu.data.provenance.fallback_events.is_empty());
        assert_eq!(cpu.data.run_status, RunStatus::Completed);
        for field_id in [
            FEA_FIELD_STRUCTURAL_DISPLACEMENT,
            FEA_FIELD_STRUCTURAL_BEAM_BENDING_STRESS,
        ] {
            assert!(matches!(
                cpu.data
                    .run
                    .field(field_id)
                    .expect("CPU authoritative field")
                    .values,
                AnalysisFieldValues::HostF64(_)
            ));
        }
        assert!(cpu.data.run.field(FEA_FIELD_STRUCTURAL_VON_MISES).is_none());
        assert_static_gpu_cpu_fallback_is_truthful("trace-contract-7b");
    }
}

#[test]
fn analysis_run_gpu_without_provider_reports_cpu_fallback() {
    let _fea_admission = fea_admission_support::guard();
    {
        let _guard = runmat_accelerate_api::ThreadProviderGuard::set(None);
        assert_static_gpu_cpu_fallback_is_truthful("trace-contract-8");
    }
}

/// Counts device transfers; a CPU fallback must leave the device untouched.
struct AdmissionProbeProvider {
    uploads: AtomicU64,
    downloads: AtomicU64,
}

impl runmat_accelerate_api::AccelProvider for AdmissionProbeProvider {
    fn upload(
        &self,
        host: &runmat_accelerate_api::HostTensorView,
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        let _ = host;
        self.uploads.fetch_add(1, Ordering::Relaxed);
        Err(anyhow::anyhow!(
            "admission probe never allocates device memory"
        ))
    }

    fn download<'a>(
        &'a self,
        _handle: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::Relaxed);
        Box::pin(async { Err(anyhow::anyhow!("unadmitted download")) })
    }

    fn device_info(&self) -> String {
        "admission-probe-provider".to_string()
    }
}

#[test]
fn analysis_run_gpu_with_provider_falls_back_without_device_transfers() {
    let _fea_admission = fea_admission_support::guard();
    {
        static PROVIDER: AdmissionProbeProvider = AdmissionProbeProvider {
            uploads: AtomicU64::new(0),
            downloads: AtomicU64::new(0),
        };
        let _guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(&PROVIDER));
        PROVIDER.uploads.store(0, Ordering::Relaxed);
        PROVIDER.downloads.store(0, Ordering::Relaxed);
        assert_static_gpu_cpu_fallback_is_truthful("trace-contract-9");
        assert_eq!(PROVIDER.uploads.load(Ordering::Relaxed), 0);
        assert_eq!(PROVIDER.downloads.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn analysis_run_gpu_upload_failure_provider_is_not_reached() {
    let _fea_admission = fea_admission_support::guard();
    {
        static PROVIDER: AdmissionProbeProvider = AdmissionProbeProvider {
            uploads: AtomicU64::new(0),
            downloads: AtomicU64::new(0),
        };
        let _guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(&PROVIDER));
        PROVIDER.uploads.store(0, Ordering::Relaxed);
        PROVIDER.downloads.store(0, Ordering::Relaxed);
        assert_static_gpu_cpu_fallback_is_truthful("trace-contract-10");
        assert_eq!(PROVIDER.uploads.load(Ordering::Relaxed), 0);
        assert_eq!(PROVIDER.downloads.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn strict_policy_quality_reasons_propagate_to_results_contracts() {
    let _fea_admission = fea_admission_support::guard();
    {
        let _artifact_store_guard = study_artifact_env_guard();
        let model = material_conflict_contract_model();
        let run = analysis_run_linear_static_with_options(
            &model,
            ComputeBackend::Cpu,
            AnalysisRunOptions {
                deterministic_mode: true,
                precision_mode: PrecisionMode::Fp64,
                preconditioner_mode: PreconditionerMode::Auto,
                quality_policy: QualityPolicy::Strict,
                solver_mesh_artifact_path: None,
            },
            OperationContext::new(Some("trace-contract-11-run".to_string()), None),
        )
        .expect("run should succeed");

        assert_eq!(run.data.run_status, RunStatus::Degraded);
        assert!(!run.data.numerically_acceptable);
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
        assert!(!direct_results.data.numerically_acceptable);
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
        assert!(!by_id_results.data.numerically_acceptable);
        assert_eq!(by_id_results.data.provenance.quality_policy, "strict");
        assert!(by_id_results
            .data
            .quality_reasons
            .iter()
            .any(|reason| reason.code == QualityReasonCode::MaterialAssignmentConflict));
    }
}

#[test]
fn balanced_and_strict_reject_the_same_material_conflict() {
    let _fea_admission = fea_admission_support::guard();
    {
        let model = material_conflict_contract_model();

        let balanced = analysis_run_linear_static_with_options(
            &model,
            ComputeBackend::Cpu,
            AnalysisRunOptions {
                deterministic_mode: true,
                precision_mode: PrecisionMode::Fp64,
                preconditioner_mode: PreconditionerMode::Auto,
                quality_policy: QualityPolicy::Balanced,
                solver_mesh_artifact_path: None,
            },
            OperationContext::new(Some("trace-contract-12-balanced".to_string()), None),
        )
        .expect("balanced run should succeed");

        let strict = analysis_run_linear_static_with_options(
            &model,
            ComputeBackend::Cpu,
            AnalysisRunOptions {
                deterministic_mode: true,
                precision_mode: PrecisionMode::Fp64,
                preconditioner_mode: PreconditionerMode::Auto,
                quality_policy: QualityPolicy::Strict,
                solver_mesh_artifact_path: None,
            },
            OperationContext::new(Some("trace-contract-12-strict".to_string()), None),
        )
        .expect("strict run should succeed");

        assert_eq!(
            balanced.data.solver_convergence,
            strict.data.solver_convergence
        );
        assert_eq!(balanced.data.result_quality, strict.data.result_quality);
        assert_eq!(balanced.data.solver_convergence, QualityGate::Pass);
        assert_eq!(balanced.data.result_quality, QualityGate::Warn);
        assert_eq!(balanced.data.run.fields, strict.data.run.fields);
        assert_eq!(balanced.data.run.diagnostics, strict.data.run.diagnostics);
        assert_eq!(balanced.data.quality_reasons, strict.data.quality_reasons);
        assert!(balanced.data.provenance.fallback_events.is_empty());
        assert!(strict.data.provenance.fallback_events.is_empty());
        for result in [&balanced.data, &strict.data] {
            assert!(result
                .run
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code
                    == "ANALYSIS_MATERIAL_ASSIGNMENT_CONFLICT_INFERRED"));
        }
        assert_eq!(balanced.data.provenance.quality_policy, "balanced");
        assert_eq!(strict.data.provenance.quality_policy, "strict");
        assert!(balanced
            .data
            .quality_reasons
            .iter()
            .any(|reason| reason.code == QualityReasonCode::MaterialAssignmentConflict));
        assert!(strict
            .data
            .quality_reasons
            .iter()
            .any(|reason| reason.code == QualityReasonCode::MaterialAssignmentConflict));

        assert!(!balanced.data.numerically_acceptable);
        assert_eq!(balanced.data.run_status, RunStatus::Degraded);
        assert!(!strict.data.numerically_acceptable);
        assert_eq!(strict.data.run_status, RunStatus::Degraded);
    }
}

#[test]
fn balanced_and_strict_material_rejection_propagates_through_results_endpoints() {
    let _fea_admission = fea_admission_support::guard();
    {
        let _artifact_store_guard = study_artifact_env_guard();
        let model = material_conflict_contract_model();

        let balanced_run = analysis_run_linear_static_with_options(
            &model,
            ComputeBackend::Cpu,
            AnalysisRunOptions {
                deterministic_mode: true,
                precision_mode: PrecisionMode::Fp64,
                preconditioner_mode: PreconditionerMode::Auto,
                quality_policy: QualityPolicy::Balanced,
                solver_mesh_artifact_path: None,
            },
            OperationContext::new(Some("trace-contract-13-balanced-run".to_string()), None),
        )
        .expect("balanced run should succeed");
        let strict_run = analysis_run_linear_static_with_options(
            &model,
            ComputeBackend::Cpu,
            AnalysisRunOptions {
                deterministic_mode: true,
                precision_mode: PrecisionMode::Fp64,
                preconditioner_mode: PreconditionerMode::Auto,
                quality_policy: QualityPolicy::Strict,
                solver_mesh_artifact_path: None,
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

        assert_eq!(
            balanced_results.data.quality_reasons,
            balanced_run.data.quality_reasons
        );
        assert_eq!(balanced_results.data.run_status, RunStatus::Degraded);
        assert!(!balanced_results.data.numerically_acceptable);
        assert_eq!(balanced_results.data.provenance.quality_policy, "balanced");
        assert!(balanced_results
            .data
            .quality_reasons
            .iter()
            .any(|reason| reason.code == QualityReasonCode::MaterialAssignmentConflict));

        assert_eq!(
            strict_results.data.quality_reasons,
            strict_run.data.quality_reasons
        );
        assert_eq!(strict_results.data.run_status, RunStatus::Degraded);
        assert!(!strict_results.data.numerically_acceptable);
        assert_eq!(strict_results.data.provenance.quality_policy, "strict");
        assert!(strict_results
            .data
            .quality_reasons
            .iter()
            .any(|reason| reason.code == QualityReasonCode::MaterialAssignmentConflict));

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

        assert_eq!(
            balanced_by_id.data.quality_reasons,
            balanced_run.data.quality_reasons
        );
        assert_eq!(balanced_by_id.data.run_status, RunStatus::Degraded);
        assert!(!balanced_by_id.data.numerically_acceptable);
        assert_eq!(balanced_by_id.data.provenance.quality_policy, "balanced");
        assert!(balanced_by_id
            .data
            .quality_reasons
            .iter()
            .any(|reason| reason.code == QualityReasonCode::MaterialAssignmentConflict));

        assert_eq!(
            strict_by_id.data.quality_reasons,
            strict_run.data.quality_reasons
        );
        assert_eq!(strict_by_id.data.run_status, RunStatus::Degraded);
        assert!(!strict_by_id.data.numerically_acceptable);
        assert_eq!(strict_by_id.data.provenance.quality_policy, "strict");
        assert!(strict_by_id
            .data
            .quality_reasons
            .iter()
            .any(|reason| reason.code == QualityReasonCode::MaterialAssignmentConflict));
    }
}

#[test]
fn retired_electromagnetic_proxy_operation_contract_cannot_deserialize() {
    let _fea_admission = fea_admission_support::guard();
    let removed = serde_json::json!({
        "solver": "electromagnetic_potential",
        "options": {
            "deterministic_mode": true, "precision_mode": "fp64",
            "quality_policy": "strict", "residual_target": 1e-6,
            "harmonic_tolerance": 1e-7, "harmonic_max_iterations": 96,
            "sweep_enabled": true, "sweep_frequency_hz": [60.0]
        }
    });
    assert!(serde_json::from_value::<AnalysisSolverOptions>(removed).is_err());
}

#[test]
fn maxwell_operation_contract_selects_genuine_complex_defaults() {
    let _fea_admission = fea_admission_support::guard();
    let options =
        AnalysisSolverOptions::defaults_for(&ProcedureStep::electromagnetic_harmonic("field"))
            .unwrap();
    let wire = serde_json::to_value(&options).unwrap();
    assert_eq!(wire["solver"], "maxwell_harmonic");
    assert_eq!(
        serde_json::from_value::<AnalysisSolverOptions>(wire).unwrap(),
        options
    );
    // Authored CAD/worker execution and retained cold-root evidence are covered by
    // native analysis_process_conformance/maxwell_authored, not this scalar smoke test.
}
