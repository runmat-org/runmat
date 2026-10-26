//! Shared real thermal-prestress physics fixture for unit and process conformance tests.
use runmat_analysis_core::*;
use runmat_analysis_execution::{ThermalStudyPreparation, ThermalStudyScope};
use runmat_analysis_fea::{
    fixtures::{fixture_model, FixtureId},
    properties::compile_property_map,
    study::{plan_thermal_step, ThermalStudyPlanningPolicy},
    ComputeBackend,
};

fn planning_policy() -> ThermalStudyPlanningPolicy {
    ThermalStudyPlanningPolicy {
        backend: ComputeBackend::Cpu,
        element_heat_source: None,
    }
}

pub fn setup(mut setup: ThermalStudyPreparation, buckling: bool) -> ThermalStudyPreparation {
    let model = &mut setup.product.compiled.model;
    model.thermo_mechanical = Some(ThermoMechanicalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    model.materials[0]
        .thermal
        .as_mut()
        .unwrap()
        .expansion_coefficient_per_k = runmat_types::physical::ThermalExpansion::new(1.0e-5);
    for constraint in &mut model.constraints {
        if let AnalysisConstraint::Thermal(ThermalConstraint {
            kind: ThermalConstraintKind::Prescribed { value, .. },
            ..
        }) = constraint
        {
            *value = PrescribedValue::Constant(Temperature::new(303.15));
        }
    }
    // Restrained x expansion creates uniform compression; y/z remain physically free.
    for node in &setup.mesh.topology.nodes {
        let axes: &[usize] = if node.coordinates_m.canonical_value() == [0.0, 0.0, 0.0] {
            &[0, 1, 2]
        } else if node.coordinates_m.canonical_value() == [0.0, 1.0, 0.0] {
            &[0, 2]
        } else {
            &[0]
        };
        model
            .constraints
            .push(AnalysisConstraint::Structural(StructuralConstraint {
                constraint_id: format!("axial-{}", node.node_id),
                frame: CoordinateFrame::Global,
                kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                    target: RegionTarget::point(format!("node:{}", node.node_id)),
                    translations: axes
                        .iter()
                        .map(|&axis| TranslationDofValue {
                            dof: [
                                StructuralTranslationDof::Ux,
                                StructuralTranslationDof::Uy,
                                StructuralTranslationDof::Uz,
                            ][axis],
                            value: PrescribedValue::Constant(Displacement::new(0.0)),
                        })
                        .collect(),
                    rotations: Vec::new(),
                }),
            }));
    }
    let heat = model.sequence.steps[0].clone();
    let mut preload = ProcedureStep::coupled(
        "preload",
        CoupledPhysicsFamily::Thermomechanical,
        AnalysisProcedure::Static,
    );
    preload.state_outputs = ProcedureStep::structural_preload("template").state_outputs;
    preload.state_inputs.push(StateInput {
        input_id: "named-temperature".into(),
        required_kind: StateKind::Thermal,
        source: StateSource::Upstream {
            producer_step: heat.step_id.clone(),
            state_id: heat.state_outputs[1].state_id.clone(),
        },
        use_as: StateUse::Prestress,
        conflict: InitialStateConflictPolicy::Reject,
    });
    let mut consumer = if buckling {
        ProcedureStep::structural_buckling("consumer", "preload", "geometric_stiffness")
    } else {
        ProcedureStep::structural_prestressed_modal("consumer", "preload", "geometric_stiffness")
    };
    consumer.independent_domain = IndependentDomain::Eigenvalue { requested_count: 1 };
    model.sequence = AnalysisSequence::from_steps(
        "thermal-prestress-consumer",
        vec![heat.clone(), preload.clone(), consumer.clone()],
    );
    for dependency in &mut model.sequence.dependencies {
        dependency.reason = SequenceDependencyReason::StateTransfer;
    }
    model.sequence.transitions = vec![
        transition(
            "heat-preload",
            &heat,
            &preload,
            heat.state_outputs[1].state_id.clone(),
        ),
        transition(
            "preload-consumer",
            &preload,
            &consumer,
            StateId("geometric_stiffness".into()),
        ),
    ];
    setup.product.compiled.properties = compile_property_map(model, Some(&setup.mesh)).unwrap();
    setup.product.workload =
        plan_thermal_step(&setup.product.compiled, "heat", planning_policy()).unwrap();
    setup.scope = ThermalStudyScope::SelectedStep;
    setup
}

fn transition(
    id: &str,
    producer: &ProcedureStep,
    consumer: &ProcedureStep,
    state_id: StateId,
) -> StateTransition {
    StateTransition {
        transition_id: TransitionId(id.into()),
        producer_step: producer.step_id.clone(),
        consumer_step: consumer.step_id.clone(),
        state_id,
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    }
}

pub fn frequency_setup(base: ThermalStudyPreparation) -> ThermalStudyPreparation {
    let mut setup = setup(base, false);
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
                    .canonical_value()
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

pub fn reduced_setup(base: ThermalStudyPreparation) -> ThermalStudyPreparation {
    let mut setup = frequency_setup(base);
    let model = &mut setup.product.compiled.model;
    let preload = model.sequence.steps[1].clone();
    let mut modal =
        ProcedureStep::structural_prestressed_modal("modes", "preload", "geometric_stiffness");
    modal.independent_domain = IndependentDomain::Eigenvalue { requested_count: 1 };
    modal.activations = preload.activations.clone();
    let consumer = &mut model.sequence.steps[2];
    consumer.solution_space = SolutionSpace::ModalReduced;
    consumer.state_inputs.extend(
        ProcedureStep::structural_modal_harmonic("template", "modes", "modal_basis").state_inputs,
    );
    let consumer = consumer.clone();
    model.sequence.dependencies.push(SequenceDependency {
        predecessor: preload.step_id.clone(),
        successor: modal.step_id.clone(),
        reason: SequenceDependencyReason::StateTransfer,
    });
    model.sequence.dependencies.push(SequenceDependency {
        predecessor: modal.step_id.clone(),
        successor: consumer.step_id.clone(),
        reason: SequenceDependencyReason::StateTransfer,
    });
    model.sequence.transitions.push(transition(
        "preload-modes",
        &preload,
        &modal,
        StateId("geometric_stiffness".into()),
    ));
    model.sequence.transitions.push(transition(
        "modes-frequency",
        &modal,
        &consumer,
        StateId("modal_basis".into()),
    ));
    model.sequence.steps.insert(2, modal);
    setup.product.compiled.properties = compile_property_map(model, Some(&setup.mesh)).unwrap();
    setup.product.workload =
        plan_thermal_step(&setup.product.compiled, "heat", planning_policy()).unwrap();
    setup
}
