use super::*;
use crate::execution::{CanonicalSemanticState, SequenceIdentity, StepIdentity};
use runmat_analysis_core::{AnalysisSequence, ProcedureStep, RestartPolicy};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

mod initial_fields;

fn workload() -> ThermalWorkload {
    let mut model =
        crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
    model.thermal = Some(runmat_analysis_core::ThermalDomain {
        reference_temperature: runmat_analysis_core::Temperature::new(293.15),
    });
    model.sequence =
        AnalysisSequence::from_steps("heat", vec![ProcedureStep::thermal_transient("heat_a")]);
    ThermalWorkload {
        initial_fields: Vec::new(),
        model,
        step_id: ProcedureStepId("heat_a".into()),
        backend: ComputeBackend::Cpu,
        procedure: ThermalProcedureControl::BackwardEuler { snapshot_count: 3 },
        residual_target: 1e-4,
    }
}

#[test]
fn portable_workload_runs_the_selected_step_and_retains_authored_lineage() {
    let mut workload = workload();
    workload
        .model
        .sequence
        .steps
        .push(ProcedureStep::thermal_transient("heat_b"));
    workload.step_id = ProcedureStepId("heat_b".into());
    let decoded =
        ThermalWorkload::from_canonical_bytes(&workload.canonical_bytes().unwrap()).unwrap();
    assert_eq!(decoded, workload);
    assert_eq!(decoded.identity().unwrap(), workload.identity().unwrap());
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let result = decoded.execute(&mesh).unwrap();
    result.validate_complete().unwrap();
    assert_eq!(result.initial_state.procedure_step_id, "heat_b");
    assert_eq!(
        result.producer,
        StepIdentity::from_solver_contract(
            SequenceIdentity::from_solver_contract(&workload.model.sequence).unwrap(),
            &workload.model.sequence.steps[1],
        )
        .unwrap()
    );
    // Uniform initial temperature with no thermal sources is an exact insulated solution.
    for field in &result.temperature_snapshots {
        assert!(field
            .as_host_f64()
            .unwrap()
            .iter()
            .all(|value| (*value - 293.15).abs() < 1e-9));
    }
    assert_eq!(result.temperature_snapshots.len(), 3);
    CanonicalSemanticState::from_thermal_result(&workload.model, &mesh, &result).unwrap();
    assert!(crate::run_thermal_with_options(
        &workload.model,
        ComputeBackend::Cpu,
        crate::ThermalSolveOptions {
            solver_mesh: Some(mesh),
            ..Default::default()
        }
    )
    .is_err());
}

#[test]
fn thermal_workload_identity_binds_controls_and_selected_step() {
    let original = workload();
    let mut changed = original.clone();
    changed.residual_target *= 0.5;
    assert_ne!(original.identity().unwrap(), changed.identity().unwrap());
    changed = original.clone();
    changed.procedure = ThermalProcedureControl::BackwardEuler { snapshot_count: 5 };
    assert_ne!(original.identity().unwrap(), changed.identity().unwrap());
    changed = original.clone();
    changed
        .model
        .sequence
        .steps
        .push(ProcedureStep::thermal_transient("heat_b"));
    let first = changed.identity().unwrap();
    changed.step_id = ProcedureStepId("heat_b".into());
    assert_ne!(first, changed.identity().unwrap());
}

#[test]
fn malformed_workloads_are_rejected_on_decode_and_before_execution() {
    let mut invalid = workload();
    invalid.step_id = ProcedureStepId("missing".into());
    assert!(invalid.canonical_bytes().is_err());
    let bytes = encode_contract(PREFIX, "thermal-workload", &invalid, LIMITS).unwrap();
    assert!(ThermalWorkload::from_canonical_bytes(&bytes).is_err());
    assert!(invalid
        .execute(&canonical_tetrahedron_solver_mesh(ElementOrder::Tet4))
        .is_err());
    invalid = workload();
    invalid.procedure = ThermalProcedureControl::Steady;
    assert!(invalid.validate().is_err());
    invalid = workload();
    invalid.procedure = ThermalProcedureControl::BackwardEuler { snapshot_count: 1 };
    assert!(invalid.validate().is_err());
    invalid = workload();
    invalid.residual_target = f64::NAN;
    assert!(invalid.validate().is_err());
}

#[test]
fn unbound_restart_is_not_silently_treated_as_fresh() {
    let mut workload = workload();
    workload.model.sequence.steps[0].restart = RestartPolicy::ResumeCheckpoint {
        checkpoint_id: "missing-checkpoint".into(),
        required_state_digest: "missing-state".into(),
    };
    assert!(workload.validate().is_err());
    let error = crate::run_thermal_with_options(
        &workload.model,
        ComputeBackend::Cpu,
        crate::ThermalSolveOptions {
            solver_mesh: Some(canonical_tetrahedron_solver_mesh(ElementOrder::Tet4)),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("restart"), "{error}");
}

#[test]
fn canonical_mesh_tampering_is_rejected_before_assembly() {
    let mut mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    mesh.topology.nodes[0].node_id += 100;
    assert!(workload().execute(&mesh).is_err());
}

#[test]
fn steady_portable_entry_preserves_uniform_temperature_on_both_tetrahedra() {
    use runmat_analysis_core::{
        AnalysisConstraint, PrescribedValue, RegionTarget, Temperature, ThermalConstraint,
        ThermalConstraintKind,
    };
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut workload = workload();
        workload.model.sequence =
            AnalysisSequence::from_steps("steady", vec![ProcedureStep::thermal_steady("heat_a")]);
        workload.procedure = ThermalProcedureControl::Steady;
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let projection = crate::topology::AnalysisProjection::derive(&mesh).unwrap();
        let face = projection.boundary_faces[0].provenance[0]
            .analysis_selection_id()
            .unwrap();
        workload
            .model
            .constraints
            .push(AnalysisConstraint::Thermal(ThermalConstraint {
                constraint_id: "prescribed_temperature".into(),
                kind: ThermalConstraintKind::Prescribed {
                    target: RegionTarget::boundary(
                        runmat_meshing_core::fixtures::fixture_boundary_selection(&(face)),
                    ),
                    value: PrescribedValue::Constant(Temperature::new(315.0)),
                },
            }));
        let decoded =
            ThermalWorkload::from_canonical_bytes(&workload.canonical_bytes().unwrap()).unwrap();
        let result = decoded.execute(&mesh).unwrap();
        result.validate_complete().unwrap();
        assert_eq!(result.temperature_snapshots.len(), 1);
        assert!(result.temperature_snapshots[0]
            .as_host_f64()
            .unwrap()
            .iter()
            .all(|value| (*value - 315.0).abs() < 1e-8));
        let direct = crate::run_thermal_with_options(
            &workload.model,
            ComputeBackend::Cpu,
            crate::ThermalSolveOptions {
                procedure: workload.procedure,
                residual_target: workload.residual_target,
                solver_mesh: Some(mesh),
                initial_fields: workload.initial_fields.clone(),
            },
        )
        .unwrap();
        assert_eq!(
            result.source_result_identity().unwrap(),
            direct.source_result_identity().unwrap()
        );
    }
}

#[test]
fn shared_step_admission_rejects_unbound_upstream_temperature() {
    use runmat_analysis_core::{
        InitialStateConflictPolicy, StateInput, StateKind, StateSource, StateUse,
    };
    let mut step = ProcedureStep::thermal_transient("consumer");
    let producer = ProcedureStep::thermal_transient("producer");
    step.state_inputs.push(StateInput {
        input_id: "initial_temperature".into(),
        required_kind: StateKind::Thermal,
        source: StateSource::Upstream {
            producer_step: producer.step_id,
            state_id: producer.state_outputs[0].state_id.clone(),
        },
        use_as: StateUse::PhysicalInitialState,
        conflict: InitialStateConflictPolicy::Reject,
    });
    let error = crate::pipeline::thermal::validate_step(
        &step,
        ThermalProcedureControl::BackwardEuler { snapshot_count: 3 },
    )
    .unwrap_err();
    assert!(error.to_string().contains("upstream state"), "{error}");
}
