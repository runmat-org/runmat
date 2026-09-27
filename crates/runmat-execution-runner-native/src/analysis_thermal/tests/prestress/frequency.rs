use super::*;
use runmat_analysis_fea::execution::AnalysisNumericalObjectKind;
use runmat_execution_artifact::ObjectNamespace;
use runmat_meshing_core::CanonicalMeshingContract;

fn frequency_setup(order: ElementOrder) -> ThermalStudyPreparation {
    let mut setup = setup(order, false);
    let model = &mut setup.product.compiled.model;
    let mut consumer = ProcedureStep::structural_prestressed_harmonic(
        "consumer",
        "preload",
        "geometric_stiffness",
    );
    consumer.independent_domain = IndependentDomain::FrequencyHertz(vec![60.0]);
    model.sequence.steps[2] = consumer;
    model.materials[0].damping = Some(MechanicalDampingModel::Rayleigh {
        mass_coefficient: RayleighMassCoefficient::new(0.2),
        stiffness_coefficient: RayleighStiffnessCoefficient::new(2.0e-4),
    });
    let projection =
        runmat_analysis_fea::topology::AnalysisProjection::derive(&setup.mesh).unwrap();
    let selection = projection
        .boundary_faces
        .iter()
        .find(|face| {
            face.node_indices.iter().all(|&index| {
                (setup.mesh.topology.nodes[index]
                    .coordinates_m
                    .iter()
                    .sum::<f64>()
                    - 1.0)
                    .abs()
                    < 1.0e-12
            })
        })
        .unwrap()
        .provenance[0]
        .analysis_selection_id()
        .unwrap();
    let fixture = fixture_model(FixtureId::CantileverLinearStatic);
    model.excitations = fixture.excitations;
    model.signals = fixture.signals;
    model.excitation_cases = fixture.excitation_cases;
    let PhysicalSignal::Force(signal) = &mut model.signals[0] else {
        unreachable!()
    };
    signal.shape = SignalShape::Vector3;
    signal.representation = SignalRepresentation::Constant(SignalValue::Vector3([
        Force::new(0.0),
        Force::new(1000.0),
        Force::new(0.0),
    ]));
    let ExcitationAction::Force(probe) = &mut model.excitations[0] else {
        unreachable!()
    };
    probe.target = ExcitationTarget::Region(RegionTarget::boundary(selection));
    probe.distribution = ExcitationDistribution::PreserveResultant;
    probe.direction = None;
    probe.active_step_ids = vec!["consumer".into()];
    // Per-action step selection isolates the harmonic probe without unsupported
    // object-lifecycle controls in the closed structural execution envelope.
    setup.product.compiled.properties = compile_property_map(model, Some(&setup.mesh)).unwrap();
    setup.product.workload =
        plan_thermal_step(&setup.product.compiled, "heat", planning_policy()).unwrap();
    setup
}

#[test]
fn replayed_named_thermal_publication_drives_nonzero_complex_frequency_worker() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let directory = tempfile::tempdir().unwrap();
        let limits = NativeAnalysisHostLimits::default();
        let mut store =
            FilesystemObjectStore::open(directory.path(), limits.inventory.max_object_bytes)
                .unwrap();
        let completed = solve_heat(&mut store, frequency_setup(order));
        let source = seal(&store, &completed);
        let dependency = source
            .state_publication_dependency(&StateId("second-temperature".into()))
            .unwrap();
        assert!(
            StructuralPrestressedFrequencyStudyCoordinator::new(structural_preparation!(
                StructuralPrestressedFrequencyStudyPreparation,
                completed
            ))
            .is_err()
        );
        let mut study = StructuralPrestressedFrequencyStudyCoordinator::new_with_thermal_source(
            structural_preparation!(StructuralPrestressedFrequencyStudyPreparation, completed),
            source,
        )
        .unwrap();
        preload!(study, &mut store, dependency);
        let bound = study.bind_frequency().unwrap();
        assert_eq!(bound.state_task, study.state_task_identity().unwrap());
        execute_frequency(&mut store, &completed, bound);
    }
}

fn execute_frequency(
    store: &mut FilesystemObjectStore,
    completed: &CompletedThermalStudy,
    bound: PreparedStructuralPrestressedFrequency,
) {
    let limits = NativeAnalysisHostLimits::default();
    let heat = completed.preparation();
    let workload = prepare_structural_frequency_workload_objects(
        &bound.workload,
        limits.chunk_bytes,
        ObjectNamespace::InputValue,
        limits.inventory,
    )
    .unwrap();
    let mesh = prepare_analysis_objects(
        AnalysisNumericalObjectKind::Mesh,
        Digest::from_bytes(*heat.mesh.canonical_digest.bytes()),
        &heat.mesh.canonical_encode().unwrap(),
        limits.chunk_bytes,
        ObjectNamespace::InputValue,
        limits.inventory,
    )
    .unwrap();
    let prestress_state = prepare_analysis_objects(
        AnalysisNumericalObjectKind::SemanticState,
        bound.prestress_state.identity.digest(),
        &bound.prestress_state.canonical_bytes().unwrap(),
        limits.chunk_bytes,
        ObjectNamespace::InputValue,
        limits.inventory,
    )
    .unwrap();
    let modal_state = bound.modal_state.as_ref().map(|state| {
        prepare_analysis_objects(
            AnalysisNumericalObjectKind::SemanticState,
            state.identity.digest(),
            &state.canonical_bytes().unwrap(),
            limits.chunk_bytes,
            ObjectNamespace::InputValue,
            limits.inventory,
        )
        .unwrap()
    });
    let tasks = prepare_structural_frequency_fanout(StructuralFrequencyFanoutPreparation {
        workload,
        mesh,
        prestress_state: Some(prestress_state),
        modal_state,
        artifact_access: heat.artifact_access,
        program_revision: heat.program_revision,
        chunk_bytes: limits.chunk_bytes,
        inventory_limits: limits.inventory,
    })
    .unwrap();
    assert_eq!(tasks.len(), 1);
    let task = &tasks[0];
    let (response, _) = execute(store, &task.host, &task.program, &task.resource_plan, 7);
    let result =
        import_structural_frequency_entry(store, response, task, limits.inventory).unwrap();
    assert_eq!(result.result.request(), &task.request);
    let displacement = result.result.sample().displacement();
    assert!(displacement
        .real()
        .iter()
        .any(|value| value.abs() > 1.0e-15));
    assert!(displacement
        .imaginary()
        .iter()
        .any(|value| value.abs() > 1.0e-18));
}

mod reduced;
