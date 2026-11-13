use runmat_analysis_core::{
    ActivationLifecycle, ExcitationCaseActivation, ExcitationCaseId, ExcitationCaseSelection,
    StepObjectActivation, StepObjectRef,
};

use super::{active_actions, realize_step_excitations, ExcitationSamplePoint};
use crate::fixtures::{fixture_model, FixtureId};

#[test]
fn empty_case_selection_suppresses_actions_without_disabling_supports_or_materials() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let step = &mut model.sequence.steps[0];
    step.excitation_case_selection = ExcitationCaseSelection::Only(vec![]);
    assert!(step.object_is_active(&StepObjectRef::Constraint("support".into())));
    assert!(step.object_is_active(&StepObjectRef::MaterialParameter("elasticity".into())));
    assert_eq!(
        active_actions(&model, Some(&model.sequence.steps[0]))
            .unwrap()
            .count(),
        0
    );
    assert_eq!(
        realize_step_excitations(
            &model,
            &model.sequence.steps[0],
            ExcitationSamplePoint::Static
        )
        .unwrap()
        .iter()
        .count(),
        0
    );
}

#[test]
fn selection_realizes_shared_action_once_with_only_active_selected_case_ids() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let original = model.excitation_cases[0].clone();
    let mut unselected_action = model.excitations[0].clone();
    unselected_action.excitation_mut().excitation_id =
        runmat_analysis_core::ExcitationId("unselected_action".into());
    let mut unselected_case = original.clone();
    unselected_case.case_id = ExcitationCaseId("unselected".into());
    unselected_case.action_ids = vec![unselected_action.excitation().excitation_id.clone()];
    model.excitations.push(unselected_action);
    model.excitation_cases.push(unselected_case);
    for (id, activation) in [
        ("selected", ExcitationCaseActivation::Always),
        (
            "inactive",
            ExcitationCaseActivation::Steps(vec!["other".into()]),
        ),
    ] {
        let mut case = original.clone();
        case.case_id = ExcitationCaseId(id.into());
        case.activation = activation;
        model.excitation_cases.push(case);
    }
    model.sequence.steps[0].excitation_case_selection = ExcitationCaseSelection::Only(vec![
        ExcitationCaseId("selected".into()),
        ExcitationCaseId("inactive".into()),
    ]);
    let selected = active_actions(&model, Some(&model.sequence.steps[0]))
        .unwrap()
        .collect::<Vec<_>>();
    assert_eq!(selected.len(), 1);
    assert_eq!(
        selected[0].case_ids().cloned().collect::<Vec<_>>(),
        vec![ExcitationCaseId("selected".into())]
    );
    let realized = realize_step_excitations(
        &model,
        &model.sequence.steps[0],
        ExcitationSamplePoint::Static,
    )
    .unwrap();
    let entries = realized.iter().collect::<Vec<_>>();
    assert_eq!(entries.len(), 1);
    assert_eq!(
        entries[0].case_ids,
        vec![ExcitationCaseId("selected".into())]
    );
}

#[test]
fn selected_cases_intersect_action_step_ids_and_explicit_object_activations() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.sequence.steps[0].excitation_case_selection =
        ExcitationCaseSelection::Only(vec![model.excitation_cases[0].case_id.clone()]);
    let count = |model: &runmat_analysis_core::AnalysisModel| {
        active_actions(model, Some(&model.sequence.steps[0]))
            .unwrap()
            .count()
    };
    assert_eq!(count(&model), 1);
    model.excitations[0].excitation_mut().active_step_ids = vec!["other".into()];
    assert_eq!(count(&model), 0);
    model.excitations[0].excitation_mut().active_step_ids =
        vec![model.sequence.steps[0].step_id.0.clone()];
    assert_eq!(count(&model), 1);
    let support = StepObjectRef::Constraint("support".into());
    model.sequence.steps[0].activations = vec![StepObjectActivation {
        object: support.clone(),
        lifecycle: ActivationLifecycle::Activate,
    }];
    assert_eq!(count(&model), 0);
    assert!(model.sequence.steps[0].object_is_active(&support));
    model.sequence.steps[0]
        .activations
        .push(StepObjectActivation {
            object: StepObjectRef::Excitation(
                model.excitations[0].excitation().excitation_id.0.clone(),
            ),
            lifecycle: ActivationLifecycle::Activate,
        });
    assert_eq!(count(&model), 1);
}
