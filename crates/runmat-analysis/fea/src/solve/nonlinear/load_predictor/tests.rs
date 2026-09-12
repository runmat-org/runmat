use super::*;
use crate::{
    algebra::apply_k,
    assembly::{assemble_structural_system, StructuralFormulationRequirement},
    excitation::{realize_step_excitations, ExcitationSamplePoint},
    fixtures::{fixture_model, FixtureId},
};

#[test]
fn predictor_preserves_the_mass_regularized_load_increment_recurrence() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let step = &model.sequence.steps[0];
    let excitations = realize_step_excitations(&model, step, ExcitationSamplePoint::Static)
        .expect("fixture excitations should realize");
    let summary = assemble_structural_system(
        &model,
        &excitations,
        None,
        StructuralFormulationRequirement::ExperimentalSyntheticOperator,
    )
    .expect("fixture predictor system should assemble");
    let schedule = LoadIncrementSchedule::canonical(&[], &[0.3], 0.0, 1.0)
        .expect("required load factor should be admissible");
    let result = predict_load_increments(
        &summary,
        LoadPredictorOptions {
            increment_count: 2,
            max_linear_iters: 256,
            tolerance: 1.0e-10,
        },
        ComputeBackend::Cpu,
        0.0,
        1.0,
        &schedule,
    );

    assert_eq!(result.load_factors, vec![0.3, 0.8, 1.0]);
    assert_eq!(result.displacement_snapshots.len(), 4);
    assert_eq!(result.residual_norms.len(), 3);
    for step_index in 0..result.load_factors.len() {
        let left_factor = if step_index == 0 {
            0.0
        } else {
            result.load_factors[step_index - 1]
        };
        let right_factor = result.load_factors[step_index];
        let increment = right_factor - left_factor;
        let previous = &result.displacement_snapshots[step_index];
        let current = &result.displacement_snapshots[step_index + 1];
        let mass_current = apply_m(&summary.operator, current);
        let mass_previous = apply_m(&summary.operator, previous);
        let stiffness_current = apply_k(&summary.operator, current);
        let equation_scale = mass_previous
            .iter()
            .zip(summary.operator.load())
            .map(|(mass, load)| (mass + increment * load).powi(2))
            .sum::<f64>()
            .sqrt()
            .max(1.0);
        let free_residual = mass_current
            .iter()
            .zip(stiffness_current)
            .zip(mass_previous)
            .zip(summary.operator.load())
            .enumerate()
            .filter(|(dof, _)| !summary.operator.constrained_dofs()[*dof])
            .map(|(_, (((mass, stiffness), previous_mass), load))| {
                let residual = mass + increment * stiffness - previous_mass - increment * load;
                residual * residual
            })
            .sum::<f64>()
            .sqrt()
            / equation_scale;
        assert!(
            free_residual <= 1.0e-8,
            "predictor recurrence residual at increment {step_index} was {free_residual}"
        );
    }
    assert_eq!(
        result.solver_backend,
        LinearAlgebraBackendKind::CpuReference
    );
    assert_eq!(result.preconditioner, SolverPreconditioner::None);
    assert_eq!(result.solver_host_sync_count, 0);
    assert_eq!(result.device_apply_k_attempt_count, 0);
}
