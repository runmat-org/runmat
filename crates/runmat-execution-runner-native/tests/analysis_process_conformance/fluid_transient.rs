//! Actual shared task-pool blocks, cold fenced import and original pressure coordinates.
use super::fluid_support as support;
use super::*;
use runmat_analysis_core::*;
use runmat_analysis_execution::{
    import_fluid_transient_checkpoint, prepare_fluid_transient_task,
    validate_analysis_phase_accounting, AnalysisOutputPolicy, FluidTransientTaskPreparation,
    PlannedResourceUsageAttempt, PlannedResourceUsageGraph,
};
use runmat_analysis_fea::execution::{
    FluidTransientBlockWorkload, FluidTransientCheckpoint, FluidTransientWorkload,
};
use runmat_execution_runner_native::{
    execute_fluid_transient_block_locally, FluidTransientBlockExecutionPolicy,
};

pub(super) fn fixture() -> (
    FluidTransientWorkload,
    runmat_meshing_core::SolverMeshArtifact,
) {
    // A large absolute reference tests original coordinates; it is never the Newton unknown.
    let (steady, mesh) = support::workload(true, 1e12);
    let mut model = steady.model;
    model.sequence = AnalysisSequence::from_steps(
        "scheduled-fluid",
        vec![ProcedureStep::fluid_transient("advance")],
    );
    model.sequence.steps[0].independent_domain = IndependentDomain::TimeSeconds {
        origin: 0.,
        start: 0.,
        end: 0.3,
        interval_count: 3,
    };
    model.signals.push(PhysicalSignal::Velocity(Signal {
        signal_id: SignalId("u-history".into()),
        shape: SignalShape::Vector3,
        frame: Some(CoordinateFrame::Global),
        representation: SignalRepresentation::Tabulated(TabulatedSignal {
            domain: TabulatedSignalDomain::TimeSeconds,
            knots: vec![
                SignalKnot::continuous(
                    0.,
                    SignalOrdinate::Real(SignalValue::Vector3([0.001, 0., 0.].map(Velocity::new))),
                ),
                SignalKnot::continuous(
                    1.,
                    SignalOrdinate::Real(SignalValue::Vector3([0.011, 0., 0.].map(Velocity::new))),
                ),
            ],
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::Reject,
        }),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: Some(0.),
            max_relative_error: Some(0.),
            derivative_max_absolute_error: None,
            sample_count: 2,
        },
        provenance: SignalProvenance::Authored,
    }));
    for constraint in &mut model.constraints {
        let AnalysisConstraint::Fluid(fluid) = constraint else {
            unreachable!()
        };
        let target = match &fluid.kind {
            FluidConstraintKind::MovingWall { target, .. }
            | FluidConstraintKind::PrescribedVelocity { target, .. } => target.clone(),
            _ => unreachable!(),
        };
        fluid.kind = FluidConstraintKind::PrescribedVelocity {
            target,
            value: PrescribedValue::Signal(SignalId("u-history".into())),
        };
    }
    model
        .initial_conditions
        .push(InitialCondition::FluidVelocity(InitialConditionValue {
            initial_condition_id: "original-u".into(),
            target: InitialConditionTarget::WholeModel,
            frame: CoordinateFrame::Global,
            value: InitialValue::UniformVector3([
                Velocity::new(0.001),
                Velocity::new(0.),
                Velocity::new(0.),
            ]),
            applies_before_step_id: "advance".into(),
        }));
    (
        FluidTransientWorkload {
            model,
            step_id: ProcedureStepId("advance".into()),
            policy: steady.policy,
            geometry_policy: steady.geometry_policy,
            initial_fields: Vec::new(),
        },
        mesh,
    )
}

pub(super) async fn run() {
    let (workload, mesh) = fixture();
    let initial = workload.initial_checkpoint(&mesh).unwrap();
    let mut split = initial.clone();
    let mut full = None;
    let mut attempts = Vec::<PlannedResourceUsageAttempt>::new();
    for (ordinal, (predecessor, end, chunk)) in [(initial, 3, 8192), (split.clone(), 1, 1024)]
        .into_iter()
        .enumerate()
    {
        let (checkpoint, attempt) =
            run_block(&workload, &mesh, predecessor, end, chunk, ordinal).await;
        if ordinal == 0 {
            full = Some(checkpoint);
        } else {
            split = checkpoint;
            attempts.extend(attempt);
        }
    }
    let (checkpoint, attempt) = run_block(&workload, &mesh, split, 3, 8192, 2).await;
    attempts.extend(attempt);
    let full = full.unwrap();
    assert_eq!(checkpoint.next_interval(), 3);
    assert_eq!(checkpoint.next_output_ordinal(), 3);
    assert_eq!(
        checkpoint.intervals().collect::<Vec<_>>(),
        full.intervals().collect::<Vec<_>>()
    );
    assert_eq!(
        checkpoint.retained_results().last(),
        full.retained_results().last()
    );
    // Independent accelerating-flow and affine-pressure references at every original sample.
    let projection = runmat_analysis_fea::topology::AnalysisProjection::derive(&mesh).unwrap();
    let topology = runmat_analysis_fea::topology::FiniteVolumeTopology::derive(
        &projection,
        workload.geometry_policy,
    )
    .unwrap();
    for (ordinal, result) in checkpoint.retained_results() {
        let ordinal = *ordinal;
        let time = 0.3 * ((ordinal + 1) as f64 / 3.);
        assert!(result
            .velocity_m_per_s
            .canonical_values()
            .iter()
            .all(|u| (u[0] - (0.001 + 0.01 * time)).abs() < 1e-8
                && u[1].abs() < 1e-8
                && u[2].abs() < 1e-8));
        assert!(result
            .pressure_reference_pa
            .canonical_values()
            .iter()
            .all(|p| *p == 1e12));
        for (cell, fluctuation) in topology
            .cells
            .iter()
            .zip(result.pressure_fluctuation_pa.canonical_values())
        {
            assert!((fluctuation + 0.01 * (cell.centroid_m[0] - 1.)).abs() < 2e-7);
        }
        // Reported absolute pressure cannot reproduce the original low-order bits.
        assert!(result
            .pressure_pa
            .canonical_values()
            .iter()
            .zip(result.pressure_fluctuation_pa.canonical_values())
            .any(|(absolute, fluctuation)| absolute - 1e12 != *fluctuation));
    }
    let wall = attempts
        .iter()
        .map(|a| a.accounting.resources.usage.wall_millis)
        .sum();
    let graph = PlannedResourceUsageGraph::new(attempts, wall).unwrap();
    assert_eq!(graph.attempts.len(), 2);
    println!("scheduled original fluid blocks PASS: 3 terminated sessions; cold fences/source/accounting; original 1e12-Pa reference; split history exact");
}

pub(super) async fn run_block(
    workload: &FluidTransientWorkload,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
    predecessor: FluidTransientCheckpoint,
    end_interval: usize,
    chunk: usize,
    ordinal: usize,
) -> (FluidTransientCheckpoint, Vec<PlannedResourceUsageAttempt>) {
    let directory = tempfile::tempdir().unwrap();
    let mut config = NativeExecutionConfig::for_current_executable().unwrap();
    config.store_root = directory.path().join("scheduled-fluid");
    config.max_workers = 1;
    config.worker_cpu_millicores = 1000;
    config.worker_memory_bytes = workload.model.sequence.steps[0]
        .accuracy
        .resources
        .resident_byte_budget();
    config.worker_scratch_bytes = config.worker_memory_bytes;
    config
        .worker_capabilities
        .insert(Capability::Custom(format!(
            "runmat.analysis.host:{}",
            runmat_analysis_execution::ANALYSIS_HOST_ABI
        )));
    let session = NativeProgramSession::new(config).unwrap();
    let source = session.object_source();
    let request = FluidTransientBlockWorkload {
        workload: workload.clone(),
        predecessor,
        end_interval,
    };
    let task = prepare_fluid_transient_task(FluidTransientTaskPreparation {
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        authored_study_digest: digest("original-scheduled-fluid"),
        frozen_project_digest: None,
        program_revision: revision(),
        artifact_access: session.artifact_access().clone(),
        workload: request.clone(),
        mesh: mesh.clone(),
        output_policy: AnalysisOutputPolicy::new(chunk, ObjectInventoryLimits::default()).unwrap(),
        required_capabilities: Default::default(),
    })
    .unwrap();
    let done = execute_fluid_transient_block_locally(
        session,
        &task,
        FluidTransientBlockExecutionPolicy {
            retry: RetryPolicy::Never,
            deadline_unix_millis: None,
            priority: 0,
        },
    )
    .await
    .unwrap();
    // The API owns/drops the original session. Reads below use only its immutable object source.
    let cold = import_fluid_transient_checkpoint(&source, done.response.clone(), &task).unwrap();
    assert_eq!(cold.checkpoint, done.original.checkpoint);
    request.recover_output(&cold.checkpoint, mesh).unwrap();
    let phase = runmat_analysis_execution::AnalysisPhaseExecutionEvidence::new(
        done.execution.clone(),
        done.usage.wall_millis,
    )
    .unwrap();
    validate_analysis_phase_accounting(&task.host, &task.resource_plan, &done.response, &phase)
        .unwrap();
    for alter_bytes in [false, true] {
        let mut original = done.execution.clone();
        let resources = super::phase_execution_support::winning_accounting_mut(&mut original);
        if alter_bytes {
            resources.resources.artifact_bytes += 1;
        } else {
            resources.resources.allocation.memory_bytes += 1;
        }
        assert!(
            runmat_analysis_execution::AnalysisPhaseExecutionEvidence::new(
                original,
                done.usage.wall_millis
            )
            .is_err()
        );
    }
    let mut response = done.response.clone();
    if let ProgramExecutionResponse::ExternalizedSuccess { result_objects, .. } = &mut response {
        result_objects.push(result_objects[0].clone());
    }
    assert!(import_fluid_transient_checkpoint(&source, response, &task).is_err());
    // A valid output from another interval request is still foreign to this exact task.
    let foreign_end = if end_interval == 3 { 2 } else { 3 };
    let foreign = prepare_fluid_transient_task(FluidTransientTaskPreparation {
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        authored_study_digest: digest("original-scheduled-fluid"),
        frozen_project_digest: None,
        program_revision: revision(),
        artifact_access: task.host.artifact_access.clone(),
        workload: FluidTransientBlockWorkload {
            end_interval: foreign_end,
            ..request
        },
        mesh: mesh.clone(),
        output_policy: task.output_policy,
        required_capabilities: Default::default(),
    })
    .unwrap();
    assert!(import_fluid_transient_checkpoint(&source, done.response.clone(), &foreign).is_err());
    println!("scheduled fluid block {ordinal}: original ranges, admitted allocation, closed output import PASS");
    done.execution.validate().unwrap();
    assert_eq!(done.execution.attempts.len(), done.usage.attempts.len());
    assert!(
        runmat_analysis_fea::planning::resource::allocation_matches_plan(
            &done.execution.attempts[0].request.task.resources,
            &task.resource_plan.request,
        )
    );
    let cold_execution: runmat_execution_runner::TaskExecutionEvidence =
        serde_json::from_slice(&serde_json::to_vec(&done.execution).unwrap()).unwrap();
    cold_execution.validate().unwrap();
    assert_eq!(cold_execution, done.execution);
    let usage = PlannedResourceUsageGraph::from_execution(
        &task.resource_plan,
        &cold_execution,
        done.usage.wall_millis,
    )
    .unwrap();
    assert_eq!(usage, done.usage);
    (cold.checkpoint, done.usage.attempts)
}
