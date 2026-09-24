use super::*;
use crate::{
    assembly::{assemble_structural_system, StructuralFormulationRequirement},
    excitation::{realize_step_excitations, ExcitationSamplePoint},
    fixtures::{fixture_model, FixtureId},
};

#[test]
fn recovery_rejects_changed_shell_recovery_count_with_stable_dofs() {
    let model = fixture_model(FixtureId::StructuralShellPlateMomentReference);
    let excitations = realize_step_excitations(
        &model,
        &model.sequence.steps[0],
        ExcitationSamplePoint::Static,
    )
    .unwrap();
    let summary = assemble_structural_system(
        &model,
        &excitations,
        None,
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .unwrap();
    assert!(!summary.structural_shell_recovery.is_empty());
    let state = vec![0.0; summary.dof_count];
    let mut changed = summary.clone();
    changed.structural_shell_recovery.clear();
    assert!(matches!(
        validate_snapshot_shape(&summary, &changed, 1, &state, &state, &state, &state),
        Err(FeaRunError::Assembly(message)) if message.contains("changed structural recovery topology")
    ));
}
