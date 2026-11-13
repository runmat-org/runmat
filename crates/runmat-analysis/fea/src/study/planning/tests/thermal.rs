use super::*;
use runmat_analysis_core::*;

fn thermal_chain(consumer: ProcedureStep) -> CompiledAnalysisStudy {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.excitations.clear();
    model.excitation_cases.clear();
    model.signals.clear();
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    model.thermo_mechanical = Some(ThermoMechanicalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    let heat = ProcedureStep::thermal_steady("heat");
    let mut preload = ProcedureStep::structural_preload("preload");
    preload.family = PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical);
    preload.state_inputs.push(StateInput {
        input_id: "temperature".into(),
        required_kind: StateKind::Thermal,
        source: StateSource::Upstream {
            producer_step: heat.step_id.clone(),
            state_id: heat.state_outputs[0].state_id.clone(),
        },
        use_as: StateUse::Prestress,
        conflict: InitialStateConflictPolicy::Reject,
    });
    model.sequence = AnalysisSequence::from_steps(
        "thermal-prestress",
        vec![heat.clone(), preload.clone(), consumer.clone()],
    );
    for dependency in &mut model.sequence.dependencies {
        dependency.reason = SequenceDependencyReason::StateTransfer;
    }
    model.sequence.transitions = vec![
        StateTransition {
            transition_id: TransitionId("thermal-transfer".into()),
            producer_step: heat.step_id,
            consumer_step: preload.step_id.clone(),
            state_id: heat.state_outputs[0].state_id.clone(),
            method: StateTransitionMethod::Reuse,
            topology: TopologyTransitionPolicy::RequireIdentical,
            reference: ReferenceStatePolicy::Preserve,
        },
        StateTransition {
            transition_id: TransitionId("prestress-transfer".into()),
            producer_step: preload.step_id,
            consumer_step: consumer.step_id,
            state_id: StateId("geometric_stiffness".into()),
            method: StateTransitionMethod::Reuse,
            topology: TopologyTransitionPolicy::RequireIdentical,
            reference: ReferenceStatePolicy::Preserve,
        },
    ];
    compiled(model)
}

#[test]
fn thermal_prefixed_prestress_chains_keep_complete_inventory_and_model() {
    for consumer in [
        ProcedureStep::structural_buckling("consumer", "preload", "geometric_stiffness"),
        ProcedureStep::structural_prestressed_modal("consumer", "preload", "geometric_stiffness"),
        ProcedureStep::structural_prestressed_harmonic(
            "consumer",
            "preload",
            "geometric_stiffness",
        ),
    ] {
        let study = thermal_chain(consumer);
        let inventory = plan_structural_study(&study, policy()).unwrap();
        assert_eq!(inventory.authored_model().unwrap(), &study.model);
        let prefix = inventory.prestress_prefix().unwrap();
        assert_eq!(prefix.thermal.unwrap().step_id.0, "heat");
        assert_eq!(prefix.preload.step_id.0, "preload");
        assert!(prefix.preload.state_input.is_none());
        assert_eq!(
            prefix.consumer_step_ids,
            vec![ProcedureStepId("consumer".into())]
        );
        let restored =
            StructuralStudyWorkloads::from_canonical_bytes(&inventory.canonical_bytes().unwrap())
                .unwrap();
        assert_eq!(inventory, restored);
        let StructuralStudyWorkloads::HeterogeneousSequence { mut workloads } = inventory else {
            panic!("heterogeneous chain")
        };
        workloads.remove(&ProcedureStepId("heat".into()));
        assert!(
            StructuralStudyWorkloads::HeterogeneousSequence { workloads }
                .validate()
                .is_err()
        );
    }
}

#[test]
fn thermal_inventory_rejects_tampered_controls_and_missing_coupled_input() {
    let study = thermal_chain(ProcedureStep::structural_prestressed_modal(
        "consumer",
        "preload",
        "geometric_stiffness",
    ));
    let mut inventory = plan_structural_study(&study, policy()).unwrap();
    let StructuralStudyWorkloads::HeterogeneousSequence { workloads } = &mut inventory else {
        panic!("heterogeneous chain")
    };
    let StructuralStepWorkload::Thermal { workload } =
        workloads.get_mut(&ProcedureStepId("heat".into())).unwrap()
    else {
        panic!("thermal producer")
    };
    workload.residual_target *= 0.5;
    assert!(inventory.validate().is_err());
    let mut invalid = study;
    invalid.model.sequence.steps[1].state_inputs.clear();
    invalid.model.sequence.transitions.remove(0);
    assert!(plan_structural_study(&invalid, policy()).is_err());
}

#[test]
fn thermal_prefixed_modal_frequency_keeps_both_prestress_and_basis_lineage() {
    let mut study = thermal_chain(ProcedureStep::structural_prestressed_modal(
        "modes",
        "preload",
        "geometric_stiffness",
    ));
    let mut frequency = ProcedureStep::structural_prestressed_harmonic(
        "frequency",
        "preload",
        "geometric_stiffness",
    );
    frequency.solution_space = SolutionSpace::ModalReduced;
    frequency.state_inputs.extend(
        ProcedureStep::structural_modal_harmonic("basis-source", "modes", "modal_basis")
            .state_inputs,
    );
    study.model.sequence.steps.push(frequency);
    for (producer, state) in [("preload", "geometric_stiffness"), ("modes", "modal_basis")] {
        study.model.sequence.dependencies.push(SequenceDependency {
            predecessor: ProcedureStepId(producer.into()),
            successor: ProcedureStepId("frequency".into()),
            reason: SequenceDependencyReason::StateTransfer,
        });
        study.model.sequence.transitions.push(StateTransition {
            transition_id: TransitionId(format!("{producer}-frequency")),
            producer_step: ProcedureStepId(producer.into()),
            consumer_step: ProcedureStepId("frequency".into()),
            state_id: StateId(state.into()),
            method: StateTransitionMethod::Reuse,
            topology: TopologyTransitionPolicy::RequireIdentical,
            reference: ReferenceStatePolicy::Preserve,
        });
    }
    let inventory = plan_structural_study(&study, policy()).unwrap();
    assert_eq!(
        inventory.prestress_prefix().unwrap().consumer_step_ids,
        vec![
            ProcedureStepId("modes".into()),
            ProcedureStepId("frequency".into())
        ]
    );
    assert_eq!(inventory.authored_model().unwrap(), &study.model);
}

#[test]
fn thermal_only_and_thermal_static_without_prestress_consumer_are_not_promoted() {
    let mut study = thermal_chain(ProcedureStep::structural_buckling(
        "consumer",
        "preload",
        "geometric_stiffness",
    ));
    study.model.sequence.steps.pop();
    study.model.sequence.dependencies.pop();
    study.model.sequence.transitions.pop();
    assert!(plan_structural_study(&study, policy()).is_err());
    study.model.sequence.steps.pop();
    study.model.sequence.dependencies.clear();
    study.model.sequence.transitions.clear();
    assert!(plan_structural_study(&study, policy()).is_err());
}
