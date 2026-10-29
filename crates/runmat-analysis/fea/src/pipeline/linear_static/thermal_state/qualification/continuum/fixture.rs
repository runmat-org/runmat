use super::*;

pub(super) fn model(mesh: &SolverMeshArtifact) -> AnalysisModel {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut action = model.excitations[0].excitation().clone();
    action.target = ExcitationTarget::WholeModel;
    action.direction = None;
    action.distribution = ExcitationDistribution::IntensityPerVolume;
    action.active_step_ids = vec!["heat".into()];
    action.projection.preserve_moment = false;
    model.signals = vec![PhysicalSignal::HeatGenerationDensity(Signal {
        signal_id: action.signal_id.clone(),
        shape: SignalShape::Scalar,
        frame: None,
        representation: SignalRepresentation::Constant(SignalValue::Scalar(
            HeatGenerationDensity::new(GENERATION),
        )),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: Some(0.0),
            max_relative_error: Some(0.0),
            derivative_max_absolute_error: None,
            sample_count: 1,
        },
        provenance: SignalProvenance::Authored,
    })];
    model.excitation_cases = vec![ExcitationCase {
        case_id: ExcitationCaseId("heat-case".into()),
        action_ids: vec![action.excitation_id.clone()],
        activation: ExcitationCaseActivation::Always,
    }];
    model.excitations = vec![ExcitationAction::HeatGeneration(action)];
    model.boundaries.clear();
    model.constraints.clear();
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("slab"));
    let MaterialElasticityModel::IsotropicLinear(elasticity) =
        model.materials[0].elasticity.as_mut().unwrap()
    else {
        panic!("isotropic fixture")
    };
    elasticity.youngs_modulus_pa = runmat_types::physical::YoungsModulus::new(YOUNGS);
    elasticity.poisson_ratio = runmat_types::physical::PoissonRatio::new(0.0);
    let thermal = model.materials[0].thermal.as_mut().unwrap();
    thermal.conductivity_w_per_mk = runmat_types::physical::ThermalConductivity::new(CONDUCTIVITY);
    thermal.expansion_coefficient_per_k = runmat_types::physical::ThermalExpansion::new(ALPHA);
    thermal.reference_temperature_k = runmat_types::physical::Temperature::new(REFERENCE);
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(REFERENCE),
    });
    model.thermo_mechanical = Some(ThermoMechanicalDomain {
        reference_temperature: Temperature::new(REFERENCE),
    });
    for (id, region) in [
        ("left_temperature", "slab_left"),
        ("right_temperature", "slab_right"),
    ] {
        model
            .constraints
            .push(AnalysisConstraint::Thermal(ThermalConstraint {
                constraint_id: id.into(),
                kind: ThermalConstraintKind::Prescribed {
                    target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                        region,
                    )),
                    value: PrescribedValue::Constant(Temperature::new(REFERENCE)),
                },
            }));
    }
    for node in &mesh.topology.nodes {
        let translations = [
            StructuralTranslationDof::Ux,
            StructuralTranslationDof::Uy,
            StructuralTranslationDof::Uz,
        ]
        .into_iter()
        .filter(|dof| {
            node.coordinates_m.canonical_value()[0] == 0.0 || *dof != StructuralTranslationDof::Ux
        })
        .map(|dof| TranslationDofValue {
            dof,
            value: PrescribedValue::Constant(Displacement::new(0.0)),
        })
        .collect();
        model
            .constraints
            .push(AnalysisConstraint::Structural(StructuralConstraint {
                constraint_id: format!("lateral_pin_{}", node.node_id),
                frame: CoordinateFrame::Global,
                kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                    target: RegionTarget::point(format!("node:{}", node.node_id)),
                    translations,
                    rotations: Vec::new(),
                }),
            }));
    }
    let mut producer = ProcedureStep::thermal_steady("heat");
    // A nonempty step activation list is a closed object inventory. Naming only the
    // thermal constraints would suppress the source despite its active_step_ids.
    producer.activations.push(StepObjectActivation {
        object: StepObjectRef::Excitation(
            model.excitations[0].excitation().excitation_id.0.clone(),
        ),
        lifecycle: ActivationLifecycle::Activate,
    });
    let mut consumer = ProcedureStep::coupled(
        "expansion",
        CoupledPhysicsFamily::Thermomechanical,
        AnalysisProcedure::Static,
    );
    consumer.state_inputs.push(StateInput {
        input_id: "solved_temperature".into(),
        required_kind: StateKind::Thermal,
        source: StateSource::Upstream {
            producer_step: producer.step_id.clone(),
            state_id: producer.state_outputs[0].state_id.clone(),
        },
        use_as: StateUse::Prestress,
        conflict: InitialStateConflictPolicy::Reject,
    });
    for constraint in &model.constraints {
        let (id, activations) = match constraint {
            AnalysisConstraint::Thermal(c) => (&c.constraint_id, &mut producer.activations),
            AnalysisConstraint::Structural(c) => (&c.constraint_id, &mut consumer.activations),
            _ => unreachable!(),
        };
        activations.push(StepObjectActivation {
            object: StepObjectRef::Constraint(id.clone()),
            lifecycle: ActivationLifecycle::Activate,
        });
    }
    let transition = StateTransition {
        transition_id: TransitionId("same_mesh_temperature".into()),
        producer_step: producer.step_id.clone(),
        consumer_step: consumer.step_id.clone(),
        state_id: producer.state_outputs[0].state_id.clone(),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    };
    model.sequence =
        AnalysisSequence::from_steps("parabolic_heat_expansion", vec![producer, consumer]);
    model.sequence.dependencies[0].reason = SequenceDependencyReason::StateTransfer;
    model.sequence.transitions.push(transition);
    model
}
