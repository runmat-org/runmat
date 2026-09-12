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
    model.materials[0].mechanical.damping = runmat_analysis_core::MechanicalDampingModel::Undamped;
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
