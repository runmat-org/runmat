use super::*;

#[test]
fn static_electrical_template_is_direct_and_publishes_static_state() {
    let step = ProcedureStep::electromagnetic_static("electric");
    assert_eq!(step.family, PhysicsFamily::Electromagnetic);
    assert_eq!(step.procedure, AnalysisProcedure::Static);
    assert_eq!(step.solution_space, SolutionSpace::Direct);
    assert_eq!(step.independent_domain, IndependentDomain::Static);
    assert_eq!(step.state_outputs.len(), 1);
    assert_eq!(step.state_outputs[0].kind, StateKind::ElectromagneticStatic);
    assert_ne!(
        step.state_outputs,
        ProcedureStep::electromagnetic_harmonic("electric").state_outputs
    );
    let sequence = AnalysisSequence::from_steps("electrical", vec![step.clone()]);
    crate::validate_sequence::validate_sequence_contract(&sequence).unwrap();
    assert_eq!(
        serde_json::from_slice::<ProcedureStep>(&serde_json::to_vec(&step).unwrap()).unwrap(),
        step
    );
}

// witness: FEA-SHARED-IDENTITY-001/distinct_identity_layers @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: Excitation selection changes solver identity and round-trips distinctly.
#[test]
fn excitation_selection_changes_solver_identity_and_roundtrips_distinctly() {
    let original = ProcedureStep::structural_static("response");
    let mut encoded_contracts = std::collections::BTreeSet::new();
    for selection in [
        ExcitationCaseSelection::All,
        ExcitationCaseSelection::Only(vec![]),
        ExcitationCaseSelection::Only(vec![crate::ExcitationCaseId("probe".into())]),
        ExcitationCaseSelection::Only(vec![crate::ExcitationCaseId("other".into())]),
    ] {
        let mut step = original.clone();
        step.excitation_case_selection = selection;
        assert!(
            encoded_contracts.insert(serde_json::to_vec(&step.solver_identity_contract()).unwrap())
        );
        let bytes = serde_json::to_vec(&step).unwrap();
        assert_eq!(
            serde_json::from_slice::<ProcedureStep>(&bytes).unwrap(),
            step
        );
    }
    // Authoring omission is lowered to All; a canonical contract must state its selection.
    let mut incomplete = serde_json::to_value(&original).unwrap();
    incomplete
        .as_object_mut()
        .unwrap()
        .remove("excitation_case_selection");
    assert!(serde_json::from_value::<ProcedureStep>(incomplete).is_err());
}

#[test]
fn thermomechanical_preload_consumes_exact_thermal_state_and_produces_structural_state() {
    let step = ProcedureStep::thermomechanical_preload("preload", "heat", "hot");
    assert_eq!(
        step.family,
        PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical)
    );
    assert_eq!(step.procedure, AnalysisProcedure::Static);
    assert_eq!(
        step.state_outputs,
        ProcedureStep::structural_preload("preload").state_outputs
    );
    assert_eq!(step.state_inputs.len(), 1);
    let input = &step.state_inputs[0];
    assert_eq!(input.required_kind, StateKind::Thermal);
    assert_eq!(input.use_as, StateUse::Prestress);
    assert_eq!(input.conflict, InitialStateConflictPolicy::Reject);
    assert_eq!(
        input.source,
        StateSource::Upstream {
            producer_step: ProcedureStepId("heat".into()),
            state_id: StateId("hot".into()),
        }
    );
    let changed = ProcedureStep::thermomechanical_preload("preload", "heat", "other");
    assert_ne!(
        serde_json::to_vec(&step).unwrap(),
        serde_json::to_vec(&changed).unwrap()
    );
    let mut heat = ProcedureStep::thermal_steady("heat");
    heat.state_outputs[0].state_id = StateId("hot".into());
    let mut sequence = AnalysisSequence::from_steps("coupled", vec![heat.clone(), step]);
    assert!(crate::validate_sequence::validate_sequence_contract(&sequence).is_err());
    sequence.transitions.push(StateTransition {
        transition_id: TransitionId("heat_to_preload_hot".into()),
        producer_step: ProcedureStepId("heat".into()),
        consumer_step: ProcedureStepId("preload".into()),
        state_id: StateId("hot".into()),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    });
    assert!(crate::validate_sequence::validate_sequence_contract(&sequence).is_err());
    sequence.dependencies[0].reason = SequenceDependencyReason::StateTransfer;
    crate::validate_sequence::validate_sequence_contract(&sequence).unwrap();
    let mut wrong_state = sequence.clone();
    wrong_state.steps[1] = changed;
    assert!(crate::validate_sequence::validate_sequence_contract(&wrong_state).is_err());
    heat.state_outputs[0].kind = StateKind::StructuralPrimary;
    let mut wrong_kind = sequence;
    wrong_kind.steps[0] = heat;
    assert!(crate::validate_sequence::validate_sequence_contract(&wrong_kind).is_err());
}

#[test]
fn thermal_template_names_its_transient_equation_and_domain() {
    let step = ProcedureStep::thermal_transient("thermal");
    assert_eq!(step.family, PhysicsFamily::Thermal);
    assert_eq!(step.procedure, AnalysisProcedure::TransientResponse);
    assert_eq!(
        step.independent_domain,
        IndependentDomain::TimeSeconds {
            origin: 0.0,
            start: 0.0,
            end: 1.0,
            interval_count: 10,
        }
    );
    assert_eq!(step.state_outputs[0].kind, StateKind::Thermal);
}

#[test]
fn thermal_steady_template_names_equilibrium_and_static_domain() {
    let step = ProcedureStep::thermal_steady("thermal_equilibrium");
    assert_eq!(step.family, PhysicsFamily::Thermal);
    assert_eq!(step.procedure, AnalysisProcedure::Static);
    assert_eq!(step.independent_domain, IndependentDomain::Static);
    assert_eq!(step.state_outputs[0].kind, StateKind::Thermal);
}

#[test]
fn electromagnetic_template_names_harmonic_complex_equation() {
    let step = ProcedureStep::electromagnetic_harmonic("em");
    assert_eq!(step.family, PhysicsFamily::Electromagnetic);
    assert_eq!(step.procedure, AnalysisProcedure::HarmonicResponse);
    assert_eq!(
        step.independent_domain,
        IndependentDomain::FrequencyHertz(vec![60.0])
    );
    assert_eq!(
        step.state_outputs[0].kind,
        StateKind::ElectromagneticComplex
    );
}

#[test]
fn structural_dynamic_templates_distinguish_direct_and_modal_solution_spaces() {
    let direct = ProcedureStep::structural_harmonic("direct");
    assert_eq!(direct.solution_space, SolutionSpace::Direct);
    assert_eq!(direct.procedure, AnalysisProcedure::HarmonicResponse);
    assert_eq!(direct.state_outputs[0].kind, StateKind::StructuralComplex);

    let transient = ProcedureStep::structural_modal_transient("modal_time", "modes", "modal_basis");
    assert_eq!(transient.solution_space, SolutionSpace::ModalReduced);
    assert_eq!(transient.procedure, AnalysisProcedure::TransientResponse);
    assert_eq!(transient.state_inputs.len(), 1);
    assert_eq!(transient.state_inputs[0].use_as, StateUse::ModalBasis);

    let harmonic =
        ProcedureStep::structural_modal_harmonic("modal_frequency", "modes", "modal_basis");
    assert_eq!(harmonic.solution_space, SolutionSpace::ModalReduced);
    assert_eq!(harmonic.procedure, AnalysisProcedure::HarmonicResponse);
    assert_eq!(
        harmonic.state_inputs[0].required_kind,
        StateKind::ModalBasis
    );
    assert_eq!(harmonic.state_outputs[0].kind, StateKind::StructuralComplex);
}
