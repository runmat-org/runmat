use std::convert::Infallible;

use runmat_analysis_core::SignalSampleApproach;

use super::*;
use crate::{
    assembly::{assemble_structural_system, StructuralFormulationRequirement},
    excitation::{realize_step_excitations, ExcitationSamplePoint},
    fixtures::{fixture_model, FixtureId},
};

#[test]
fn retaining_intervals_for_motion_changes_only_ephemeral_history() {
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
    let options = TransientSolveOptions {
        time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 3 },
        ..TransientSolveOptions::default()
    };
    let time_domain = TransientTimeDomain {
        start_time_s: runmat_types::physical::Time::new(0.0),
        end_time_s: runmat_types::physical::Time::new(1.0),
        step_boundaries: &[],
    };
    let initial_displacement = vec![0.0; summary.dof_count];
    let initial_velocity = vec![0.0; summary.dof_count];
    let discarded = solve_transient_system_with_assembly(
        &summary,
        options.clone(),
        ComputeBackend::Cpu,
        time_domain,
        initial_displacement.clone(),
        initial_velocity.clone(),
        |_, _| Ok::<_, Infallible>(summary.clone()),
    )
    .unwrap();
    let mut retained = solve_transient_system_with_physical_mapping(
        TransientPhysicalMappingInput {
            summary: &summary,
            options,
            backend: ComputeBackend::Cpu,
            time_domain,
            initial_displacement,
            initial_velocity,
            interval_retention: TransientIntervalRetention::RetainForMotion,
        },
        |_, _| Ok::<_, Infallible>(summary.clone()),
        |_: f64, _: SignalSampleApproach, summary: &AssemblySummary, state: &SecondOrderState| {
            Ok::<_, Infallible>((summary.clone(), state.clone()))
        },
    )
    .unwrap();
    assert!(discarded.integration_intervals.is_empty());
    assert_eq!(retained.integration_intervals.len(), 3);
    retained.integration_intervals.clear();
    assert_eq!(retained, discarded);
}
