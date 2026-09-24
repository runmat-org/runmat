use super::*;
use crate::{
    assembly::{
        assemble_structural_system, dofs::StructuralDofLayout, StructuralFormulationRequirement,
    },
    excitation::{realize_step_excitations, ExcitationSamplePoint},
    fixtures::{fixture_model, FixtureId},
    solve::transient::{solve_transient_system, StructuralTimeStepPolicy, TransientSolveOptions},
    ComputeBackend,
};

#[test]
fn recovery_rejects_changed_dof_ownership_and_malformed_state_vectors() {
    let model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
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
    let state = vec![0.0; summary.dof_count];
    assert!(validate_snapshot_shape(&summary, &summary, 1, &state, &state, &state, &state).is_ok());

    let mut changed = summary.clone();
    changed.structural_dof_layout =
        StructuralDofLayout::reduced_translational_rows(summary.dof_count);
    assert!(matches!(
        validate_snapshot_shape(&summary, &changed, 1, &state, &state, &state, &state),
        Err(FeaRunError::Assembly(message)) if message.contains("changed structural DOF ownership")
    ));
    for change in [
        |summary: &mut AssemblySummary| summary.structural_node_count += 1,
        |summary: &mut AssemblySummary| summary.structural_beam_recovery.clear(),
    ] {
        let mut changed = summary.clone();
        change(&mut changed);
        assert!(matches!(
            validate_snapshot_shape(&summary, &changed, 1, &state, &state, &state, &state),
            Err(FeaRunError::Assembly(message)) if message.contains("changed structural recovery topology")
        ));
    }
    assert!(summary.structural_connector_plan.is_some());
    let mut changed = summary.clone();
    changed.structural_connector_plan = None;
    assert!(matches!(
        validate_snapshot_shape(&summary, &changed, 1, &state, &state, &state, &state),
        Err(FeaRunError::Assembly(message)) if message.contains("changed structural recovery topology")
    ));
    let short = &state[..state.len() - 1];
    for (displacement, velocity, acceleration, reaction) in [
        (short, state.as_slice(), state.as_slice(), state.as_slice()),
        (state.as_slice(), short, state.as_slice(), state.as_slice()),
        (state.as_slice(), state.as_slice(), short, state.as_slice()),
        (state.as_slice(), state.as_slice(), state.as_slice(), short),
    ] {
        assert!(matches!(
            validate_snapshot_shape(
                &summary, &summary, 1, displacement, velocity, acceleration, reaction
            ),
            Err(FeaRunError::Assembly(message)) if message.contains("expected")
        ));
    }
}

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

#[test]
fn recovery_consumes_reporting_assemblies_in_accepted_order() {
    let model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
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
    let mut solved = solve_transient_system(
        &summary,
        TransientSolveOptions {
            time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 2 },
            ..TransientSolveOptions::default()
        },
        ComputeBackend::Cpu,
    )
    .unwrap();
    assert_eq!(solved.assembly_snapshots.len(), 3);
    let recovered = recover_transient_fields(&mut solved, None).unwrap();
    assert!(solved.assembly_snapshots.is_empty());
    assert_eq!(recovered.displacement_snapshots.len(), 3);
    assert_eq!(recovered.velocity_snapshots.len(), 3);
    assert_eq!(recovered.acceleration_snapshots.len(), 3);
    assert_eq!(recovered.reaction_force_snapshots.len(), 3);
    assert_eq!(recovered.connector_results.len(), 3);
    assert_eq!(recovered.electro_thermal_summaries.len(), 3);
}
