use super::*;

fn damped_step_state(mass: f64, damping: f64, stiffness: f64, force: f64, time_s: f64) -> [f64; 3] {
    let natural_frequency = (stiffness / mass).sqrt();
    let decay = damping / (2.0 * mass);
    let damped_frequency = (natural_frequency * natural_frequency - decay * decay).sqrt();
    let static_displacement = force / stiffness;
    let exponential = (-decay * time_s).exp();
    let phase = damped_frequency * time_s;
    let displacement = static_displacement
        * (1.0 - exponential * (phase.cos() + decay / damped_frequency * phase.sin()));
    let velocity = static_displacement * exponential * natural_frequency * natural_frequency
        / damped_frequency
        * phase.sin();
    let acceleration = (force - damping * velocity - stiffness * displacement) / mass;
    [displacement, velocity, acceleration]
}

fn physical_reference_coefficients(
    model: &runmat_analysis_core::AnalysisModel,
    mesh: &SolverMeshArtifact,
) -> (usize, f64, f64, f64, f64) {
    let step = &model.sequence.steps[0];
    let excitations = crate::excitation::realize_step_excitations(
        model,
        step,
        crate::excitation::ExcitationSamplePoint::time(0.0),
    )
    .expect("the physical transient excitation should realize");
    let summary = crate::assembly::assemble_structural_system(
        model,
        &excitations,
        Some(mesh.clone()),
        crate::assembly::StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .expect("the physical Tet4 reference should assemble");
    let free_dofs = summary
        .operator
        .constrained_dofs()
        .iter()
        .enumerate()
        .filter_map(|(dof, constrained)| (!constrained).then_some(dof))
        .collect::<Vec<_>>();
    assert_eq!(free_dofs.len(), 1);
    let free_dof = free_dofs[0];
    (
        free_dof,
        summary.operator.operators().mass().get(free_dof, free_dof),
        summary
            .operator
            .operators()
            .damping()
            .get(free_dof, free_dof),
        summary
            .operator
            .operators()
            .stiffness()
            .get(free_dof, free_dof),
        summary.operator.load()[free_dof],
    )
}

fn native_state_at(result: &crate::FeaTransientRunResult, snapshot: usize, dof: usize) -> [f64; 3] {
    [
        result.displacement_snapshots[snapshot]
            .as_host_f64()
            .expect("displacement should be host-backed")[dof],
        result.velocity_snapshots[snapshot]
            .as_host_f64()
            .expect("velocity should be host-backed")[dof],
        result.acceleration_snapshots[snapshot]
            .as_host_f64()
            .expect("acceleration should be host-backed")[dof],
    ]
}

fn normalized_state_error(
    actual: [f64; 3],
    exact: [f64; 3],
    stiffness: f64,
    mass: f64,
    force: f64,
) -> f64 {
    let natural_frequency = (stiffness / mass).sqrt();
    let displacement_scale = force / stiffness;
    ((actual[0] - exact[0]) / displacement_scale)
        .hypot((actual[1] - exact[1]) / (displacement_scale * natural_frequency))
        .hypot(
            (actual[2] - exact[2]) / (displacement_scale * natural_frequency * natural_frequency),
        )
}

fn support_motion_model() -> runmat_analysis_core::AnalysisModel {
    use runmat_analysis_core::*;

    let mut model = fixture_model(FixtureId::StructuralTransientTet4Reference);
    model.excitations.clear();
    model.excitation_cases.clear();
    model.signals.clear();
    let acceleration_id = SignalId("root-x-acceleration".into());
    let acceleration = |value: f64| {
        SignalOrdinate::Real(SignalValue::Vector3([
            Acceleration::new(value),
            Acceleration::new(value),
            Acceleration::new(value),
        ]))
    };
    model.signals.push(PhysicalSignal::Acceleration(Signal {
        signal_id: acceleration_id.clone(),
        shape: SignalShape::Vector3,
        frame: Some(CoordinateFrame::Global),
        representation: SignalRepresentation::Tabulated(TabulatedSignal {
            domain: TabulatedSignalDomain::TimeSeconds,
            knots: vec![
                SignalKnot::continuous(0.0, acceleration(1.0)),
                SignalKnot::continuous(0.5, acceleration(-1.0)),
                SignalKnot::continuous(1.0, acceleration(1.0)),
            ],
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::Reject,
        }),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: Some(0.0),
            max_relative_error: Some(0.0),
            derivative_max_absolute_error: None,
            sample_count: 3,
        },
        provenance: SignalProvenance::Authored,
    }));
    model.constraints[0] = AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: "moving_root_x".into(),
        frame: CoordinateFrame::Global,
        kind: StructuralConstraintKind::PrescribedMotion(PrescribedStructuralMotion {
            target: RegionTarget::boundary("root"),
            translations: vec![TranslationDofMotion {
                dof: StructuralTranslationDof::Ux,
                history: StructuralKinematicHistory::AccelerationDriven {
                    acceleration: AccelerationSignalId(acceleration_id),
                    initial_displacement: Displacement::new(0.0),
                    initial_velocity: Velocity::new(0.0),
                    reconstruction:
                        StructuralMotionReconstruction::PiecewiseLinearAccelerationExact,
                },
            }],
        }),
    });
    model
        .constraints
        .push(AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "fixed_root_yz".into(),
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                target: RegionTarget::boundary("root"),
                translations: vec![
                    TranslationDofValue {
                        dof: StructuralTranslationDof::Uy,
                        value: PrescribedValue::Constant(Displacement::new(0.0)),
                    },
                    TranslationDofValue {
                        dof: StructuralTranslationDof::Uz,
                        value: PrescribedValue::Constant(Displacement::new(0.0)),
                    },
                ],
                rotations: Vec::new(),
            }),
        }));
    model
}

fn transient_checkpoint_scope(
    model: &runmat_analysis_core::AnalysisModel,
    options: &TransientSolveOptions,
    schedule: crate::execution::StructuralTransientSchedule,
) -> crate::execution::StructuralTransientCheckpointScope {
    let step = &model.sequence.steps[0];
    let excitations = crate::excitation::realize_step_excitations(
        model,
        step,
        crate::excitation::ExcitationSamplePoint::time(0.0),
    )
    .unwrap();
    let summary = crate::assembly::assemble_structural_system(
        model,
        &excitations,
        options.solver_mesh.clone(),
        crate::assembly::StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .unwrap();
    let workload = transient_workload(model, options);
    crate::execution::StructuralTransientCheckpointScope::from_prepared_workload(
        crate::execution::StructuralTransientCheckpointScopeInput {
            workload: &workload,
            solver_mesh: options.solver_mesh.as_ref().unwrap(),
            initial_assembly: &summary,
            modal_state: None,
            schedule,
        },
    )
    .unwrap()
}

fn transient_workload(
    model: &runmat_analysis_core::AnalysisModel,
    options: &TransientSolveOptions,
) -> crate::execution::StructuralTransientWorkload {
    let step = &model.sequence.steps[0];
    crate::execution::StructuralTransientWorkload {
        schema_version: 1,
        model: model.clone(),
        step_id: step.step_id.clone(),
        backend: ComputeBackend::Cpu,
        formulation_requirement: options.formulation_requirement,
        solver: crate::execution::StructuralTransientSolverPlan::Direct,
        time_step_policy: options.time_step_policy,
        residual_target: options.residual_target,
    }
}

fn modal_support_motion_workloads(
    interval_count: usize,
) -> (
    crate::execution::StructuralTransientWorkload,
    crate::execution::StructuralTransientWorkload,
    SolverMeshArtifact,
    crate::execution::CanonicalSemanticState,
) {
    use runmat_analysis_core::*;
    use runmat_execution::Digest;

    let mut direct_model = support_motion_model();
    let AnalysisConstraint::Structural(constraint) = &mut direct_model.constraints[0] else {
        panic!("the modal comparison support should be structural")
    };
    let StructuralConstraintKind::PrescribedMotion(motion) = &mut constraint.kind else {
        panic!("the modal comparison requires prescribed support motion")
    };
    let StructuralKinematicHistory::AccelerationDriven {
        initial_displacement,
        initial_velocity,
        ..
    } = &mut motion.translations[0].history;
    *initial_displacement = Displacement::new(0.02);
    *initial_velocity = Velocity::new(-0.03);
    let mut model = direct_model.clone();
    let modal_step = ProcedureStep::structural_modal("modes");
    let mut transient_step =
        ProcedureStep::structural_modal_transient("response", "modes", "modal_basis");
    transient_step.independent_domain = IndependentDomain::TimeSeconds {
        origin: 0.0,
        start: 0.0,
        end: 1.0,
    };
    model.sequence.steps = vec![modal_step, transient_step];
    model.sequence.dependencies = vec![SequenceDependency {
        predecessor: ProcedureStepId("modes".into()),
        successor: ProcedureStepId("response".into()),
        reason: SequenceDependencyReason::StateTransfer,
    }];
    model.sequence.transitions = vec![StateTransition {
        transition_id: TransitionId("modal-basis-to-moving-response".into()),
        producer_step: ProcedureStepId("modes".into()),
        consumer_step: ProcedureStepId("response".into()),
        state_id: StateId("modal_basis".into()),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    }];
    let mesh = single_tetrahedron_solver_mesh_for_model(&model);
    let modal = crate::run_modal_with_options(
        &model,
        ComputeBackend::Cpu,
        ModalSolveOptions {
            truncation: crate::ModalTruncationPolicy::requested(1),
            solver_mesh: Some(mesh.clone()),
            ..experimental_modal_options()
        },
    )
    .expect("support motion should be homogeneous during modal extraction");
    let sequence = crate::execution::SequenceIdentity::from_solver_contract(&model.sequence)
        .expect("the modal support-motion sequence should have an identity");
    let producer =
        crate::execution::StepIdentity::from_solver_contract(sequence, &model.sequence.steps[0])
            .unwrap();
    let source_result = crate::execution::SolverResultIdentity::from_step_result(
        producer,
        modal.basis.identity().digest(),
    )
    .unwrap();
    let state = crate::execution::CanonicalSemanticState::new(
        crate::execution::CanonicalSemanticStateInput {
            producer,
            declaration: model.sequence.steps[0].state_outputs[0].clone(),
            topology_digest: Digest::from_bytes(*mesh.canonical_digest.bytes()),
            source_result,
            payload: crate::execution::CanonicalSemanticStatePayload::ModalBasis {
                basis: Box::new(modal.basis),
            },
        },
    )
    .unwrap();
    let transition = crate::execution::CanonicalStateTransition::bind_upstream(
        &model,
        &ProcedureStepId("response".into()),
        &state,
        StateUse::ModalBasis,
        Digest::from_bytes(*mesh.canonical_digest.bytes()),
    )
    .unwrap();
    let direct = crate::execution::StructuralTransientWorkload {
        schema_version: 1,
        model: direct_model,
        step_id: ProcedureStepId("transient".into()),
        backend: ComputeBackend::Cpu,
        formulation_requirement: FormulationRequirement::Experimental,
        solver: crate::execution::StructuralTransientSolverPlan::Direct,
        time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count },
        residual_target: 1.0e-11,
    };
    let reduced = crate::execution::StructuralTransientWorkload {
        schema_version: 1,
        model: model.clone(),
        step_id: ProcedureStepId("response".into()),
        backend: ComputeBackend::Cpu,
        formulation_requirement: FormulationRequirement::Experimental,
        solver: crate::execution::StructuralTransientSolverPlan::ModalReduced {
            basis_transition: Box::new(transition),
        },
        time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count },
        residual_target: 1.0e-11,
    };
    (direct, reduced, mesh, state)
}

fn execute_transient_workload(
    workload: &crate::execution::StructuralTransientWorkload,
    mesh: &SolverMeshArtifact,
    modal_state: Option<&crate::execution::CanonicalSemanticState>,
    intervals_per_block: usize,
) -> crate::FeaTransientRunResult {
    let scope =
        crate::pipeline::transient::prepare_transient_checkpoint_scope(workload, mesh, modal_state)
            .unwrap();
    let terminal = u64::try_from(scope.schedule.points.len() - 1).unwrap();
    let terminals = crate::execution::StructuralTransientBlockPlan::new(
        u64::try_from(intervals_per_block).unwrap(),
    )
    .unwrap()
    .terminal_schedule_ordinals(terminal)
    .unwrap();
    let mut predecessor = None;
    let mut segments = Vec::new();
    for terminal in terminals {
        let request = crate::execution::StructuralTransientBlockRequest::new(
            scope.clone(),
            predecessor.as_ref(),
            terminal,
        )
        .unwrap();
        let segment = crate::pipeline::transient::run_transient_block(
            crate::pipeline::transient::StructuralTransientRunBlockInput {
                workload,
                solver_mesh: mesh,
                request: &request,
                predecessor: predecessor.as_ref(),
                modal_state,
                observation_cursors: Vec::new(),
            },
        )
        .unwrap();
        predecessor = Some(segment.manifest().terminal_checkpoint.clone());
        segments.push(segment);
    }
    let request =
        crate::execution::StructuralTransientResultAssemblyRequest::new(&segments).unwrap();
    let result =
        crate::execution::StructuralTransientSolverResult::assemble(&request, &segments).unwrap();
    let numerical_receipt =
        crate::execution::StructuralTransientNumericalReceipt::new(&result, &request, &segments)
            .unwrap();
    crate::pipeline::transient::recover_completed_transient_run(
        crate::pipeline::transient::CompletedStructuralTransientRunInput {
            workload,
            solver_mesh: mesh,
            modal_state,
            result: &result,
            numerical_receipt: &numerical_receipt,
        },
    )
    .unwrap()
}

#[test]
fn modal_support_motion_matches_direct_physics_and_restarts_exactly() {
    let interval_count = 40;
    let (direct, modal, mesh, state) = modal_support_motion_workloads(interval_count);
    let direct = execute_transient_workload(&direct, &mesh, None, 13);
    let modal_one_block = execute_transient_workload(&modal, &mesh, Some(&state), interval_count);
    let modal_restarted = execute_transient_workload(&modal, &mesh, Some(&state), 13);

    assert_eq!(modal_restarted.time_points_s, modal_one_block.time_points_s);
    assert_eq!(
        modal_restarted.displacement_snapshots,
        modal_one_block.displacement_snapshots
    );
    assert_eq!(
        modal_restarted.velocity_snapshots,
        modal_one_block.velocity_snapshots
    );
    assert_eq!(
        modal_restarted.acceleration_snapshots,
        modal_one_block.acceleration_snapshots
    );
    assert_eq!(
        modal_restarted.reaction_force_snapshots,
        modal_one_block.reaction_force_snapshots
    );
    assert_eq!(
        modal_restarted.support_work_snapshots,
        modal_one_block.support_work_snapshots
    );
    let initial_displacement = modal_one_block.displacement_snapshots[0]
        .as_host_f64()
        .unwrap();
    let initial_velocity = modal_one_block.velocity_snapshots[0].as_host_f64().unwrap();
    for dof in [0, 3, 6] {
        assert_eq!(initial_displacement[dof], 0.02);
        assert_eq!(initial_velocity[dof], -0.03);
    }

    for (direct, modal) in direct
        .displacement_snapshots
        .iter()
        .chain(&direct.velocity_snapshots)
        .chain(&direct.acceleration_snapshots)
        .chain(&direct.reaction_force_snapshots)
        .zip(
            modal_one_block
                .displacement_snapshots
                .iter()
                .chain(&modal_one_block.velocity_snapshots)
                .chain(&modal_one_block.acceleration_snapshots)
                .chain(&modal_one_block.reaction_force_snapshots),
        )
    {
        let direct = direct.as_host_f64().unwrap();
        let modal = modal.as_host_f64().unwrap();
        assert_eq!(direct.len(), modal.len());
        let scale = direct
            .iter()
            .chain(modal)
            .map(|value| value.abs())
            .fold(1.0_f64, f64::max);
        let error = direct
            .iter()
            .zip(modal)
            .map(|(direct, modal)| (direct - modal).abs())
            .fold(0.0_f64, f64::max);
        assert!(error <= 1.0e-8 * scale, "direct/modal error {error}");
    }
    for (direct, modal) in [
        (
            &direct.strain_energy_snapshots,
            &modal_one_block.strain_energy_snapshots,
        ),
        (
            &direct.kinetic_energy_snapshots,
            &modal_one_block.kinetic_energy_snapshots,
        ),
        (
            &direct.external_work_snapshots,
            &modal_one_block.external_work_snapshots,
        ),
        (
            &direct.support_work_snapshots,
            &modal_one_block.support_work_snapshots,
        ),
        (
            &direct.damping_dissipation_snapshots,
            &modal_one_block.damping_dissipation_snapshots,
        ),
        (
            &direct.energy_balance_residual_snapshots,
            &modal_one_block.energy_balance_residual_snapshots,
        ),
    ] {
        assert_eq!(direct.len(), modal.len());
        for (direct, modal) in direct.iter().zip(modal) {
            let direct = direct.as_host_f64().unwrap()[0];
            let modal = modal.as_host_f64().unwrap()[0];
            let scale = direct.abs().max(modal.abs()).max(1.0);
            assert!(
                (direct - modal).abs() <= 1.0e-8 * scale,
                "direct/modal scalar-history error: direct={direct}, modal={modal}"
            );
        }
    }
}

#[test]
fn physical_tet4_prescribed_motion_keeps_absolute_constraint_kinematics() {
    let model = support_motion_model();
    let mesh = single_tetrahedron_solver_mesh_for_model(&model);
    let result = crate::run_transient_with_options(
        &model,
        ComputeBackend::Cpu,
        TransientSolveOptions {
            time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 40 },
            residual_target: 1.0e-10,
            solver_mesh: Some(mesh),
            ..experimental_transient_options()
        },
    )
    .expect("the physical Tet4 support-motion response should solve");
    let root_x_dof = 0;
    for (index, time_s) in result.time_points_s.iter().copied().enumerate() {
        let expected = if time_s <= 0.5 {
            time_s * time_s / 2.0 - 2.0 * time_s.powi(3) / 3.0
        } else {
            let dt = time_s - 0.5;
            1.0 / 24.0 - dt * dt / 2.0 + 2.0 * dt.powi(3) / 3.0
        };
        let actual = result.displacement_snapshots[index].as_host_f64().unwrap()[root_x_dof];
        assert!((actual - expected).abs() <= 2.0e-12, "t={time_s}");
    }
}

#[test]
fn prescribed_motion_blocks_restart_from_only_the_physical_checkpoint() {
    let model = support_motion_model();
    let mesh = single_tetrahedron_solver_mesh_for_model(&model);
    let options = TransientSolveOptions {
        time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 8 },
        residual_target: 1.0e-10,
        solver_mesh: Some(mesh.clone()),
        ..experimental_transient_options()
    };
    let uninterrupted =
        crate::run_transient_with_options(&model, ComputeBackend::Cpu, options.clone()).unwrap();
    let schedule = crate::solve::transient::canonical_transient_schedule(
        8,
        0.0,
        1.0,
        &crate::solve::transient::canonical_transient_step_boundaries(
            &[],
            &[0.0, 0.5, 1.0],
            0.0,
            1.0,
        )
        .unwrap(),
    )
    .unwrap();
    let scope = transient_checkpoint_scope(&model, &options, schedule);
    let workload = transient_workload(&model, &options);
    let first_request =
        crate::execution::StructuralTransientBlockRequest::new(scope.clone(), None, 4).unwrap();
    let first = crate::pipeline::transient::run_transient_block(
        crate::pipeline::transient::StructuralTransientRunBlockInput {
            workload: &workload,
            solver_mesh: &mesh,
            request: &first_request,
            predecessor: None,
            modal_state: None,
            observation_cursors: Vec::new(),
        },
    )
    .unwrap();
    let second_request = crate::execution::StructuralTransientBlockRequest::new(
        scope,
        Some(&first.manifest().terminal_checkpoint),
        8,
    )
    .unwrap();
    let second = crate::pipeline::transient::run_transient_block(
        crate::pipeline::transient::StructuralTransientRunBlockInput {
            workload: &workload,
            solver_mesh: &mesh,
            request: &second_request,
            predecessor: Some(&first.manifest().terminal_checkpoint),
            modal_state: None,
            observation_cursors: Vec::new(),
        },
    )
    .unwrap();
    let samples = first
        .samples()
        .iter()
        .chain(second.samples())
        .collect::<Vec<_>>();
    assert_eq!(samples.len() + 1, uninterrupted.time_points_s.len());
    for (index, sample) in samples.into_iter().enumerate() {
        let snapshot = index + 1;
        assert_eq!(
            sample.physical.state.displacement,
            uninterrupted.displacement_snapshots[snapshot]
                .as_host_f64()
                .unwrap()
        );
        assert_eq!(
            sample.physical.state.velocity,
            uninterrupted.velocity_snapshots[snapshot]
                .as_host_f64()
                .unwrap()
        );
        assert_eq!(
            sample.physical.state.acceleration,
            uninterrupted.acceleration_snapshots[snapshot]
                .as_host_f64()
                .unwrap()
        );
        assert_eq!(
            sample.physical.support_work_j,
            uninterrupted.support_work_snapshots[snapshot]
                .as_host_f64()
                .unwrap()[0]
        );
        assert_eq!(
            sample.physical.energy_balance_residual_j,
            uninterrupted.energy_balance_residual_snapshots[snapshot]
                .as_host_f64()
                .unwrap()[0]
        );
    }
    assert_eq!(
        second.manifest().terminal_checkpoint.state.clone(),
        crate::execution::StructuralTransientContinuationState {
            displacement: uninterrupted
                .displacement_snapshots
                .last()
                .unwrap()
                .as_host_f64()
                .unwrap()
                .to_vec(),
            velocity: uninterrupted
                .velocity_snapshots
                .last()
                .unwrap()
                .as_host_f64()
                .unwrap()
                .to_vec(),
            acceleration: uninterrupted
                .acceleration_snapshots
                .last()
                .unwrap()
                .as_host_f64()
                .unwrap()
                .to_vec(),
        }
    );
    let segments = vec![first, second];
    let request =
        crate::execution::StructuralTransientResultAssemblyRequest::new(&segments).unwrap();
    let solver_result =
        crate::execution::StructuralTransientSolverResult::assemble(&request, &segments).unwrap();
    let receipt = crate::execution::StructuralTransientNumericalReceipt::new(
        &solver_result,
        &request,
        &segments,
    )
    .unwrap();
    let recovered = crate::pipeline::transient::recover_completed_transient_run(
        crate::pipeline::transient::CompletedStructuralTransientRunInput {
            workload: &transient_workload(&model, &options),
            solver_mesh: options.solver_mesh.as_ref().unwrap(),
            modal_state: None,
            result: &solver_result,
            numerical_receipt: &receipt,
        },
    )
    .unwrap();
    assert_eq!(recovered, uninterrupted);
}

#[test]
fn prescribed_motion_uneven_blocks_preserve_discontinuity_and_final_result() {
    use runmat_analysis_core::*;

    let mut model = support_motion_model();
    let PhysicalSignal::Acceleration(signal) = &mut model.signals[0] else {
        panic!("support motion should use an acceleration signal")
    };
    let value = |x: f64| {
        SignalOrdinate::Real(SignalValue::Vector3([
            Acceleration::new(x),
            Acceleration::new(x),
            Acceleration::new(x),
        ]))
    };
    signal.representation = SignalRepresentation::Tabulated(TabulatedSignal {
        domain: TabulatedSignalDomain::TimeSeconds,
        knots: vec![
            SignalKnot::continuous(0.0, value(1.0)),
            SignalKnot::jump(0.5, value(-1.0), value(2.0), SignalPointContinuity::Right),
            SignalKnot::continuous(1.0, value(1.0)),
        ],
        interpolation: SignalInterpolation::Linear,
        extrapolation: SignalExtrapolation::Reject,
    });
    let mesh = single_tetrahedron_solver_mesh_for_model(&model);
    let options = TransientSolveOptions {
        time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 9 },
        residual_target: 1.0e-10,
        solver_mesh: Some(mesh.clone()),
        ..experimental_transient_options()
    };
    let workload = transient_workload(&model, &options);
    let uninterrupted =
        crate::run_transient_with_options(&model, ComputeBackend::Cpu, options.clone()).unwrap();
    let scope =
        crate::pipeline::transient::prepare_transient_checkpoint_scope(&workload, &mesh, None)
            .unwrap();
    let terminal = u64::try_from(scope.schedule.points.len() - 1).unwrap();
    let terminals = crate::execution::StructuralTransientBlockPlan::new(4)
        .unwrap()
        .terminal_schedule_ordinals(terminal)
        .unwrap();
    assert_eq!(terminals.len(), 3);
    assert!(terminals[2] - terminals[1] < terminals[1] - terminals[0]);

    let mut predecessor = None;
    let mut segments = Vec::new();
    for terminal in terminals {
        let request = crate::execution::StructuralTransientBlockRequest::new(
            scope.clone(),
            predecessor.as_ref(),
            terminal,
        )
        .unwrap();
        let segment = crate::pipeline::transient::run_transient_block(
            crate::pipeline::transient::StructuralTransientRunBlockInput {
                workload: &workload,
                solver_mesh: &mesh,
                request: &request,
                predecessor: predecessor.as_ref(),
                modal_state: None,
                observation_cursors: Vec::new(),
            },
        )
        .unwrap();
        predecessor = Some(segment.manifest().terminal_checkpoint.clone());
        segments.push(segment);
    }
    assert!(segments
        .iter()
        .flat_map(|segment| segment.samples())
        .any(|sample| {
            sample.physical.time_s == 0.5
                && scope.schedule.points[sample.physical.schedule_ordinal as usize].arrival
                    == crate::execution::StructuralTransientScheduleArrival::LeftLimit
        }));
    let request =
        crate::execution::StructuralTransientResultAssemblyRequest::new(&segments).unwrap();
    let result =
        crate::execution::StructuralTransientSolverResult::assemble(&request, &segments).unwrap();
    let numerical_receipt =
        crate::execution::StructuralTransientNumericalReceipt::new(&result, &request, &segments)
            .unwrap();
    let recovered = crate::pipeline::transient::recover_completed_transient_run(
        crate::pipeline::transient::CompletedStructuralTransientRunInput {
            workload: &workload,
            solver_mesh: &mesh,
            modal_state: None,
            result: &result,
            numerical_receipt: &numerical_receipt,
        },
    )
    .unwrap();
    assert_eq!(recovered, uninterrupted);
}

#[test]
fn governed_support_motion_fixture_uses_the_physical_constraint_row_space() {
    let model = fixture_model(FixtureId::StructuralTransientTet4SupportMotionReference);
    let mesh = single_tetrahedron_solver_mesh_for_model(&model);
    crate::run_transient_with_options(
        &model,
        ComputeBackend::Cpu,
        TransientSolveOptions {
            time_step_policy: StructuralTimeStepPolicy::Fixed {
                interval_count: 160,
            },
            residual_target: 1.0e-10,
            solver_mesh: Some(mesh),
            ..experimental_transient_options()
        },
    )
    .expect("the governed support-motion fixture should execute on its physical constraint rows");
}

#[test]
fn fully_prescribed_tet4_advances_the_complete_motion_schedule() {
    use runmat_analysis_core::*;

    let mut model = support_motion_model();
    let StructuralConstraintKind::PrescribedMotion(root_motion) = &model.constraints[0]
        .structural()
        .expect("the first constraint should be structural")
        .kind
    else {
        panic!("the first constraint should prescribe support motion")
    };
    let history = root_motion.translations[0].history.clone();
    model.constraints = [
        ("moving_root", RegionTarget::boundary("root")),
        ("moving_free_node", RegionTarget::point("node:4")),
    ]
    .into_iter()
    .map(|(constraint_id, target)| {
        AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: constraint_id.into(),
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::PrescribedMotion(PrescribedStructuralMotion {
                target,
                translations: [
                    StructuralTranslationDof::Ux,
                    StructuralTranslationDof::Uy,
                    StructuralTranslationDof::Uz,
                ]
                .into_iter()
                .map(|dof| TranslationDofMotion {
                    dof,
                    history: history.clone(),
                })
                .collect(),
            }),
        })
    })
    .collect();
    let mesh = single_tetrahedron_solver_mesh_for_model(&model);
    let options = TransientSolveOptions {
        time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 8 },
        residual_target: 1.0e-10,
        solver_mesh: Some(mesh.clone()),
        ..experimental_transient_options()
    };
    let result = crate::run_transient_with_options(&model, ComputeBackend::Cpu, options.clone())
        .expect("the fully prescribed physical Tet4 should traverse its complete schedule");

    assert_eq!(result.time_points_s.len(), 9);
    for (snapshot, time_s) in result.time_points_s.iter().copied().enumerate() {
        let expected = if time_s <= 0.5 {
            time_s * time_s / 2.0 - 2.0 * time_s.powi(3) / 3.0
        } else {
            let dt = time_s - 0.5;
            1.0 / 24.0 - dt * dt / 2.0 + 2.0 * dt.powi(3) / 3.0
        };
        let displacement = result.displacement_snapshots[snapshot]
            .as_host_f64()
            .expect("displacement should be host-backed");
        assert_eq!(displacement.len(), 12);
        assert!(displacement
            .iter()
            .all(|actual| (*actual - expected).abs() <= 2.0e-12));
    }
    assert_eq!(
        result.support_work_snapshots.len(),
        result.time_points_s.len()
    );

    let schedule = crate::solve::transient::canonical_transient_schedule(
        8,
        0.0,
        1.0,
        &crate::solve::transient::canonical_transient_step_boundaries(
            &[],
            &[0.0, 0.5, 1.0],
            0.0,
            1.0,
        )
        .unwrap(),
    )
    .unwrap();
    let scope = transient_checkpoint_scope(&model, &options, schedule);
    let workload = transient_workload(&model, &options);
    let first_request =
        crate::execution::StructuralTransientBlockRequest::new(scope.clone(), None, 4).unwrap();
    let first = crate::pipeline::transient::run_transient_block(
        crate::pipeline::transient::StructuralTransientRunBlockInput {
            workload: &workload,
            solver_mesh: &mesh,
            request: &first_request,
            predecessor: None,
            modal_state: None,
            observation_cursors: Vec::new(),
        },
    )
    .unwrap();
    let second_request = crate::execution::StructuralTransientBlockRequest::new(
        scope,
        Some(&first.manifest().terminal_checkpoint),
        8,
    )
    .unwrap();
    let second = crate::pipeline::transient::run_transient_block(
        crate::pipeline::transient::StructuralTransientRunBlockInput {
            workload: &workload,
            solver_mesh: &mesh,
            request: &second_request,
            predecessor: Some(&first.manifest().terminal_checkpoint),
            modal_state: None,
            observation_cursors: Vec::new(),
        },
    )
    .unwrap();
    for (index, sample) in first.samples().iter().chain(second.samples()).enumerate() {
        let snapshot = index + 1;
        assert_eq!(
            sample.physical.state.displacement,
            result.displacement_snapshots[snapshot]
                .as_host_f64()
                .unwrap()
        );
        assert_eq!(
            sample.physical.state.velocity,
            result.velocity_snapshots[snapshot].as_host_f64().unwrap()
        );
        assert_eq!(
            sample.physical.state.acceleration,
            result.acceleration_snapshots[snapshot]
                .as_host_f64()
                .unwrap()
        );
        assert_eq!(
            sample.physical.support_work_j,
            result.support_work_snapshots[snapshot]
                .as_host_f64()
                .unwrap()[0]
        );
    }
    assert_eq!(
        second.manifest().terminal_checkpoint.counters,
        crate::execution::StructuralTransientSolverCounters::default()
    );
}

#[test]
fn acceleration_point_continuity_changes_public_reaction_but_not_interval_work() {
    use runmat_analysis_core::*;

    let with_continuity = |continuity| {
        let mut model = support_motion_model();
        let PhysicalSignal::Acceleration(signal) = &mut model.signals[0] else {
            panic!("support motion should use an acceleration signal")
        };
        let value = |x: f64| {
            SignalOrdinate::Real(SignalValue::Vector3([
                Acceleration::new(x),
                Acceleration::new(x),
                Acceleration::new(x),
            ]))
        };
        signal.representation = SignalRepresentation::Tabulated(TabulatedSignal {
            domain: TabulatedSignalDomain::TimeSeconds,
            knots: vec![
                SignalKnot::continuous(0.0, value(1.0)),
                SignalKnot::jump(0.5, value(-1.0), value(2.0), continuity),
                SignalKnot::continuous(1.0, value(1.0)),
            ],
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::Reject,
        });
        model
    };
    let solve = |model: &AnalysisModel| {
        crate::run_transient_with_options(
            model,
            ComputeBackend::Cpu,
            TransientSolveOptions {
                time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 16 },
                residual_target: 1.0e-10,
                solver_mesh: Some(single_tetrahedron_solver_mesh_for_model(model)),
                ..experimental_transient_options()
            },
        )
        .expect("the discontinuous prescribed support motion should solve")
    };
    let left = solve(&with_continuity(SignalPointContinuity::Left));
    let right = solve(&with_continuity(SignalPointContinuity::Right));
    assert_eq!(left.time_points_s, right.time_points_s);
    let event = left
        .time_points_s
        .iter()
        .position(|time| time.to_bits() == 0.5_f64.to_bits())
        .expect("the acceleration jump should be an exact integration boundary");
    assert!((native_state_at(&left, event, 0)[2] + 1.0).abs() <= 1.0e-12);
    assert!((native_state_at(&right, event, 0)[2] - 2.0).abs() <= 1.0e-12);
    let scalar = |field: &runmat_analysis_core::AnalysisField| {
        field
            .as_host_f64()
            .expect("scalar history should be host-backed")[0]
    };
    for (left, right) in left
        .support_work_snapshots
        .iter()
        .zip(&right.support_work_snapshots)
    {
        assert!((scalar(left) - scalar(right)).abs() <= 1.0e-12);
    }
    let left_reaction = left.reaction_force_snapshots[event].as_host_f64().unwrap()[0];
    let right_reaction = right.reaction_force_snapshots[event].as_host_f64().unwrap()[0];
    assert!((left_reaction - right_reaction).abs() > 1.0e-6);
}

fn constant_acceleration_support_model(
    acceleration_m_per_s2: f64,
) -> runmat_analysis_core::AnalysisModel {
    use runmat_analysis_core::*;

    let mut model = support_motion_model();
    let PhysicalSignal::Acceleration(signal) = &mut model.signals[0] else {
        panic!("support motion should use an acceleration signal")
    };
    let value = SignalOrdinate::Real(SignalValue::Vector3([
        Acceleration::new(acceleration_m_per_s2),
        Acceleration::new(0.0),
        Acceleration::new(0.0),
    ]));
    signal.representation = SignalRepresentation::Tabulated(TabulatedSignal {
        domain: TabulatedSignalDomain::TimeSeconds,
        knots: vec![
            SignalKnot::continuous(0.0, value.clone()),
            SignalKnot::continuous(1.0, value),
        ],
        interpolation: SignalInterpolation::Linear,
        extrapolation: SignalExtrapolation::Reject,
    });
    signal.evidence.sample_count = 2;
    model.materials[0].damping = Some(MechanicalDampingModel::Undamped);
    model
}

fn exact_free_coordinate_under_constant_support_acceleration(
    model: &runmat_analysis_core::AnalysisModel,
    mesh: &SolverMeshArtifact,
    support_acceleration: f64,
    time_s: f64,
) -> [f64; 3] {
    let excitations = crate::excitation::realize_step_excitations(
        model,
        &model.sequence.steps[0],
        crate::excitation::ExcitationSamplePoint::time(0.0),
    )
    .unwrap();
    let summary = crate::assembly::assemble_structural_system(
        model,
        &excitations,
        Some(mesh.clone()),
        crate::assembly::StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .unwrap();
    let free = 9;
    let moving = [0, 3, 6];
    assert_eq!(
        summary.dof_count, 12,
        "the reference fixture must remain one Tet4"
    );
    assert_eq!(
        summary.structural_dof_layout.address(free),
        Some(crate::assembly::dofs::StructuralDofAddress {
            node_index: 3,
            kind: crate::assembly::dofs::StructuralDofKind::Ux,
        }),
        "the independently reduced coordinate must remain node 4 Ux"
    );
    for (dof, node_index) in moving.into_iter().zip(0..3) {
        assert_eq!(
            summary.structural_dof_layout.address(dof),
            Some(crate::assembly::dofs::StructuralDofAddress {
                node_index,
                kind: crate::assembly::dofs::StructuralDofKind::Ux,
            }),
            "each coupled column must remain a moving-root Ux coordinate"
        );
    }
    let operators = summary.operator.operators();
    let mass = operators.mass().get(free, free);
    let damping = operators.damping().get(free, free);
    let stiffness = operators.stiffness().get(free, free);
    let coupling = |operator: &crate::algebra::RealSparseMatrix| {
        moving
            .iter()
            .map(|column| operator.get(free, *column))
            .sum::<f64>()
    };
    let p0 = summary.physical_load[free] - coupling(operators.mass()) * support_acceleration;
    let p1 = -coupling(operators.damping()) * support_acceleration;
    let p2 = -0.5 * coupling(operators.stiffness()) * support_acceleration;
    let quadratic = p2 / stiffness;
    let linear = (p1 - 2.0 * damping * quadratic) / stiffness;
    let constant = (p0 - 2.0 * mass * quadratic - damping * linear) / stiffness;
    let decay = damping / (2.0 * mass);
    let frequency = (stiffness / mass - decay * decay).sqrt();
    let cosine = -constant;
    let sine = (decay * cosine - linear) / frequency;
    let phase = frequency * time_s;
    let exponential = (-decay * time_s).exp();
    let homogeneous = exponential * (cosine * phase.cos() + sine * phase.sin());
    let homogeneous_velocity = exponential
        * ((-decay * cosine + frequency * sine) * phase.cos()
            + (-decay * sine - frequency * cosine) * phase.sin());
    let displacement = quadratic * time_s * time_s + linear * time_s + constant + homogeneous;
    let velocity = 2.0 * quadratic * time_s + linear + homogeneous_velocity;
    let load = p0 + p1 * time_s + p2 * time_s * time_s;
    let acceleration = (load - damping * velocity - stiffness * displacement) / mass;
    [displacement, velocity, acceleration]
}

#[test]
fn physical_tet4_support_motion_converges_to_an_independent_closed_form() {
    let support_acceleration = 1.0;
    let model = constant_acceleration_support_model(support_acceleration);
    let mesh = single_tetrahedron_solver_mesh_for_model(&model);
    let exact = exact_free_coordinate_under_constant_support_acceleration(
        &model,
        &mesh,
        support_acceleration,
        1.0,
    );
    let excitations = crate::excitation::realize_step_excitations(
        &model,
        &model.sequence.steps[0],
        crate::excitation::ExcitationSamplePoint::time(0.0),
    )
    .unwrap();
    let summary = crate::assembly::assemble_structural_system(
        &model,
        &excitations,
        Some(mesh.clone()),
        crate::assembly::StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .unwrap();
    let mut exact_displacement = vec![0.0; 12];
    let mut exact_velocity = vec![0.0; 12];
    for dof in [0, 3, 6] {
        exact_displacement[dof] = 0.5 * support_acceleration;
        exact_velocity[dof] = support_acceleration;
    }
    exact_displacement[9] = exact[0];
    exact_velocity[9] = exact[1];
    let quadratic_energy = |operator: &crate::algebra::RealSparseMatrix, values: &[f64]| {
        0.5 * values
            .iter()
            .zip(crate::algebra::spmv(operator, values).unwrap())
            .map(|(left, right)| left * right)
            .sum::<f64>()
    };
    let exact_support_work =
        quadratic_energy(
            summary.operator.operators().stiffness(),
            &exact_displacement,
        ) + quadratic_energy(summary.operator.operators().mass(), &exact_velocity);
    let mut errors = Vec::new();
    let mut support_work_errors = Vec::new();
    for interval_count in [80, 160, 320] {
        let result = crate::run_transient_with_options(
            &model,
            ComputeBackend::Cpu,
            TransientSolveOptions {
                time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count },
                residual_target: 1.0e-11,
                solver_mesh: Some(mesh.clone()),
                ..experimental_transient_options()
            },
        )
        .unwrap();
        let actual = native_state_at(&result, result.time_points_s.len() - 1, 9);
        errors.push(
            (actual[0] - exact[0])
                .hypot(actual[1] - exact[1])
                .hypot(actual[2] - exact[2]),
        );
        let computed_support_work = result
            .support_work_snapshots
            .last()
            .unwrap()
            .as_host_f64()
            .unwrap()[0];
        support_work_errors.push((computed_support_work - exact_support_work).abs());
    }
    assert!(errors[0] / errors[1] > 3.5, "errors={errors:?}");
    assert!(errors[1] / errors[2] > 3.5, "errors={errors:?}");
    assert!(errors[2] < 2.0e-2, "errors={errors:?}");
    assert!(
        support_work_errors[0] / support_work_errors[1] > 3.5,
        "support_work_errors={support_work_errors:?}"
    );
    assert!(
        support_work_errors[1] / support_work_errors[2] > 3.5,
        "support_work_errors={support_work_errors:?}"
    );
    assert!(
        support_work_errors[2] / exact_support_work.abs() < 1.0e-3,
        "support_work_errors={support_work_errors:?} exact={exact_support_work}"
    );
}

#[test]
fn physical_tet4_finite_pulse_converges_across_a_force_discontinuity() {
    let mut model = fixture_model(FixtureId::StructuralTransientTet4Reference);
    let runmat_analysis_core::PhysicalSignal::Force(signal) = &mut model.signals[0] else {
        panic!("the physical reference must own a force signal")
    };
    let ordinate = |x: f64| {
        runmat_analysis_core::SignalOrdinate::Real(runmat_analysis_core::SignalValue::Vector3([
            runmat_analysis_core::Force::new(x),
            runmat_analysis_core::Force::new(0.0),
            runmat_analysis_core::Force::new(0.0),
        ]))
    };
    signal.representation = runmat_analysis_core::SignalRepresentation::Tabulated(
        runmat_analysis_core::TabulatedSignal {
            domain: runmat_analysis_core::TabulatedSignalDomain::TimeSeconds,
            knots: vec![
                runmat_analysis_core::SignalKnot::continuous(0.0, ordinate(1.0)),
                runmat_analysis_core::SignalKnot::jump(
                    0.25,
                    ordinate(1.0),
                    ordinate(0.0),
                    runmat_analysis_core::SignalPointContinuity::Right,
                ),
                runmat_analysis_core::SignalKnot::continuous(1.0, ordinate(0.0)),
            ],
            interpolation: runmat_analysis_core::SignalInterpolation::LeftEndpoint,
            extrapolation: runmat_analysis_core::SignalExtrapolation::Reject,
        },
    );
    signal.evidence.sample_count = 3;
    let mesh = single_tetrahedron_solver_mesh_for_model(&model);
    let (free_dof, mass, damping, stiffness, force) =
        physical_reference_coefficients(&model, &mesh);
    let positive_step = damped_step_state(mass, damping, stiffness, force, 1.0);
    let negative_step = damped_step_state(mass, damping, stiffness, force, 0.75);
    let exact = [
        positive_step[0] - negative_step[0],
        positive_step[1] - negative_step[1],
        positive_step[2] - negative_step[2],
    ];

    let errors = [40, 80, 160].map(|interval_count| {
        let result = crate::run_transient_with_options(
            &model,
            ComputeBackend::Cpu,
            TransientSolveOptions {
                time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count },
                residual_target: 1.0e-10,
                solver_mesh: Some(mesh.clone()),
                ..experimental_transient_options()
            },
        )
        .expect("the physical Tet4 finite-pulse response should solve");
        assert_eq!(result.time_points_s.len(), interval_count + 1);
        assert!(result.time_points_s.contains(&0.25));
        normalized_state_error(
            native_state_at(&result, result.time_points_s.len() - 1, free_dof),
            exact,
            stiffness,
            mass,
            force,
        )
    });

    assert!((errors[0] / errors[1]).log2() > 1.8, "errors={errors:?}");
    assert!((errors[1] / errors[2]).log2() > 1.8, "errors={errors:?}");
}

#[test]
fn physical_tet4_undamped_step_preserves_work_energy_balance() {
    let mut model = fixture_model(FixtureId::StructuralTransientTet4Reference);
    model.materials[0].damping = Some(runmat_analysis_core::MechanicalDampingModel::Undamped);
    let mesh = single_tetrahedron_solver_mesh_for_model(&model);
    let (free_dof, mass, damping, stiffness, force) =
        physical_reference_coefficients(&model, &mesh);
    assert_eq!(damping, 0.0);
    let exact = damped_step_state(mass, damping, stiffness, force, 1.0);
    let mut finest = None;
    let errors = [160, 320, 640].map(|interval_count| {
        let result = crate::run_transient_with_options(
            &model,
            ComputeBackend::Cpu,
            TransientSolveOptions {
                time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count },
                residual_target: 1.0e-10,
                solver_mesh: Some(mesh.clone()),
                ..experimental_transient_options()
            },
        )
        .expect("the undamped physical Tet4 response should solve");
        assert_eq!(result.time_points_s.len(), interval_count + 1);
        let last = result.time_points_s.len() - 1;
        let error = normalized_state_error(
            native_state_at(&result, last, free_dof),
            exact,
            stiffness,
            mass,
            force,
        );
        finest = Some(result);
        error
    });
    assert!((errors[0] / errors[1]).log2() > 1.8, "errors={errors:?}");
    assert!((errors[1] / errors[2]).log2() > 1.8, "errors={errors:?}");
    assert!(errors[2] < 3.0e-3, "errors={errors:?}");
    let result = finest.expect("the finest undamped reference run should be retained");
    assert!(result
        .damping_dissipation_snapshots
        .iter()
        .all(|field| { field.as_host_f64().is_some_and(|values| values == [0.0]) }));
    let maximum_relative_balance = result
        .relative_energy_balance_error_snapshots
        .iter()
        .map(|field| field.as_host_f64().unwrap()[0])
        .fold(0.0_f64, f64::max);
    assert!(
        maximum_relative_balance < 1.0e-9,
        "maximum relative balance={maximum_relative_balance}"
    );
}
