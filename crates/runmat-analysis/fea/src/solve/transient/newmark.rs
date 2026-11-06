use runmat_simulation_usage::{LedgerError, OperationCount, UsageCategory, UsageContext};
use thiserror::Error;

use crate::algebra::{
    spmv, CanonicalRealOperators, ContributionDomain, ContributionId, ContributionProvenance,
    MatrixProperties, OperatorRole, RealSparseMatrix, RealSparseMatrixError,
};

use super::state::{SecondOrderState, SecondOrderStateError};

/// The average-acceleration Newmark method.
///
/// Fixing these parameters, rather than accepting an arbitrary pair, preserves the method's
/// second-order accuracy and unconditional stability for linear structural systems.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct AverageAccelerationNewmark;

impl AverageAccelerationNewmark {
    pub(super) const BETA: f64 = 0.25;
    pub(super) const GAMMA: f64 = 0.5;
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct NewmarkPredictor {
    displacement: Vec<f64>,
    velocity: Vec<f64>,
    time_step_s: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct NewmarkEffectiveStep {
    effective_operator: RealSparseMatrix,
    right_hand_side: Vec<f64>,
    predictor: NewmarkPredictor,
}

#[derive(Clone, Debug, Error, PartialEq)]
pub(super) enum NewmarkError {
    #[error("the Newmark time step must be finite and positive")]
    InvalidTimeStep,
    #[error("the structural operator, state, load, and acceleration dimensions must agree")]
    DimensionMismatch,
    #[error("the structural load or acceleration contains a nonfinite value")]
    NonFiniteInput,
    #[error("the effective Newmark operator could not be assembled: {0}")]
    EffectiveOperator(RealSparseMatrixError),
    #[error("the completed Newmark state is invalid: {0}")]
    InvalidState(SecondOrderStateError),
    #[error("simulation usage admission failed: {0}")]
    Usage(#[from] LedgerError),
}

pub(super) fn count(elements: usize, per_element: u64) -> Result<OperationCount, LedgerError> {
    OperationCount::from_operations(
        u64::try_from(elements).map_err(|_| LedgerError::CountOverflow)?,
    )
    .checked_mul(per_element)
    .map_err(LedgerError::from)
}

fn measured<R>(
    quote: impl FnOnce() -> Result<OperationCount, LedgerError>,
    run: impl FnOnce() -> Result<R, NewmarkError>,
) -> Result<R, NewmarkError> {
    let Some(context) = UsageContext::capture() else {
        return run();
    };
    drop(context);
    let work = quote()?;
    crate::algebra::capacity::measured_sync(UsageCategory::Solve, work, run)
}

pub(super) fn spmv_work(matrix: &RealSparseMatrix) -> Result<OperationCount, LedgerError> {
    crate::algebra::capacity::real_spmv_work(matrix)
}

/// Three scaled sparse terms form one support-based reference sum. Each stored
/// term has one product, and each union column adds its later terms. Backend
/// zero-product skips do not change the logical reference count.
fn effective_operator_work(
    mass: &RealSparseMatrix,
    damping: &RealSparseMatrix,
    stiffness: &RealSparseMatrix,
) -> Result<OperationCount, LedgerError> {
    let matrices = [mass, damping, stiffness];
    let mut work = OperationCount::from_operations(2); // γ·dt and β·dt²
    for row in 0..mass.rows() {
        let mut cursors = matrices.map(|matrix| matrix.pattern().row_range(row).start);
        let ends = matrices.map(|matrix| matrix.pattern().row_range(row).end);
        loop {
            let column = (0..matrices.len())
                .filter(|&index| cursors[index] < ends[index])
                .map(|index| matrices[index].pattern().column(cursors[index]))
                .min();
            let Some(column) = column else { break };
            let mut entries = 0u64;
            for index in 0..matrices.len() {
                if cursors[index] < ends[index]
                    && matrices[index].pattern().column(cursors[index]) == column
                {
                    cursors[index] += 1;
                    entries += 1;
                }
            }
            work = work
                .checked_add(OperationCount::direct_real_dot(entries)?)
                .map_err(LedgerError::from)?;
        }
    }
    Ok(work)
}

impl NewmarkEffectiveStep {
    pub(super) fn effective_operator(&self) -> &RealSparseMatrix {
        &self.effective_operator
    }

    pub(super) fn right_hand_side(&self) -> &[f64] {
        &self.right_hand_side
    }

    /// Apply an equilibrium correction to each retained coordinate. Rebuilding
    /// displacement from a rounded acceleration can erase a correction smaller
    /// than that acceleration's ulp after predictor cancellation.
    pub(super) fn correct(
        &self,
        state: SecondOrderState,
        acceleration_correction: &[f64],
    ) -> Result<SecondOrderState, NewmarkError> {
        if state.dof_count() != self.predictor.displacement.len()
            || acceleration_correction.len() != state.dof_count()
        {
            return Err(NewmarkError::DimensionMismatch);
        }
        if acceleration_correction
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(NewmarkError::NonFiniteInput);
        }
        let (displacement, velocity, acceleration) = measured(
            || {
                count(acceleration_correction.len(), 5)?
                    .checked_add(OperationCount::from_operations(3))
                    .map_err(LedgerError::from)
            },
            || {
                let (mut displacement, mut velocity, mut acceleration) = state.into_parts();
                let displacement_scale =
                    AverageAccelerationNewmark::BETA * self.predictor.time_step_s.powi(2);
                let velocity_scale = AverageAccelerationNewmark::GAMMA * self.predictor.time_step_s;
                for (index, correction) in acceleration_correction.iter().copied().enumerate() {
                    displacement[index] =
                        correction.mul_add(displacement_scale, displacement[index]);
                    velocity[index] = correction.mul_add(velocity_scale, velocity[index]);
                    acceleration[index] += correction;
                }
                Ok((displacement, velocity, acceleration))
            },
        )?;
        SecondOrderState::try_new(displacement, velocity, acceleration)
            .map_err(NewmarkError::InvalidState)
    }

    pub(super) fn complete(
        &self,
        acceleration: Vec<f64>,
    ) -> Result<SecondOrderState, NewmarkError> {
        if acceleration.len() != self.predictor.displacement.len() {
            return Err(NewmarkError::DimensionMismatch);
        }
        if acceleration.iter().any(|value| !value.is_finite()) {
            return Err(NewmarkError::NonFiniteInput);
        }
        let (displacement, velocity) = measured(
            || {
                count(acceleration.len(), 4)?
                    .checked_add(OperationCount::from_operations(3))
                    .map_err(LedgerError::from)
            },
            || {
                let displacement_scale =
                    AverageAccelerationNewmark::BETA * self.predictor.time_step_s.powi(2);
                let velocity_scale = AverageAccelerationNewmark::GAMMA * self.predictor.time_step_s;
                let displacement = self
                    .predictor
                    .displacement
                    .iter()
                    .zip(&acceleration)
                    .map(|(predicted, acceleration)| predicted + displacement_scale * acceleration)
                    .collect();
                let velocity = self
                    .predictor
                    .velocity
                    .iter()
                    .zip(&acceleration)
                    .map(|(predicted, acceleration)| predicted + velocity_scale * acceleration)
                    .collect();
                Ok((displacement, velocity))
            },
        )?;
        SecondOrderState::try_new(displacement, velocity, acceleration)
            .map_err(NewmarkError::InvalidState)
    }
}

pub(super) fn prepare_average_acceleration_step(
    operators: &CanonicalRealOperators,
    state: &SecondOrderState,
    time_step_s: f64,
    load: &[f64],
) -> Result<NewmarkEffectiveStep, NewmarkError> {
    validate_inputs(operators, state, time_step_s, load)?;

    let (displacement, velocity, time_step_square) = measured(
        || {
            count(state.dof_count(), 6)?
                .checked_add(OperationCount::from_operations(5))
                .map_err(LedgerError::from)
        },
        || {
            let time_step_square = time_step_s.powi(2);
            let displacement_acceleration_scale =
                time_step_square * (0.5 - AverageAccelerationNewmark::BETA);
            let velocity_acceleration_scale =
                time_step_s * (1.0 - AverageAccelerationNewmark::GAMMA);
            let displacement = state
                .displacement()
                .iter()
                .zip(state.velocity())
                .zip(state.acceleration())
                .map(|((displacement, velocity), acceleration)| {
                    displacement
                        + time_step_s * velocity
                        + displacement_acceleration_scale * acceleration
                })
                .collect::<Vec<_>>();
            let velocity = state
                .velocity()
                .iter()
                .zip(state.acceleration())
                .map(|(velocity, acceleration)| {
                    velocity + velocity_acceleration_scale * acceleration
                })
                .collect::<Vec<_>>();
            Ok((displacement, velocity, time_step_square))
        },
    )?;

    let damping_force = measured(
        || spmv_work(operators.damping()),
        || spmv(operators.damping(), &velocity).map_err(|_| NewmarkError::DimensionMismatch),
    )?;
    let elastic_force = measured(
        || spmv_work(operators.stiffness()),
        || spmv(operators.stiffness(), &displacement).map_err(|_| NewmarkError::DimensionMismatch),
    )?;
    let right_hand_side = measured(
        || count(load.len(), 2),
        || {
            Ok(load
                .iter()
                .zip(damping_force)
                .zip(elastic_force)
                .map(|((load, damping), elastic)| load - damping - elastic)
                .collect::<Vec<_>>())
        },
    )?;

    let effective_operator = measured(
        || effective_operator_work(operators.mass(), operators.damping(), operators.stiffness()),
        || {
            RealSparseMatrix::scaled_sum(
                &[
                    (1.0, operators.mass()),
                    (
                        AverageAccelerationNewmark::GAMMA * time_step_s,
                        operators.damping(),
                    ),
                    (
                        AverageAccelerationNewmark::BETA * time_step_square,
                        operators.stiffness(),
                    ),
                ],
                OperatorRole::EffectiveTangent,
                MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
            )
            .map_err(NewmarkError::EffectiveOperator)
        },
    )?
    .with_contribution(ContributionProvenance {
        domain: ContributionDomain::TimeIntegration,
        contribution_id: ContributionId("average_acceleration_newmark".into()),
    });

    Ok(NewmarkEffectiveStep {
        effective_operator,
        right_hand_side,
        predictor: NewmarkPredictor {
            displacement,
            velocity,
            time_step_s,
        },
    })
}

fn validate_inputs(
    operators: &CanonicalRealOperators,
    state: &SecondOrderState,
    time_step_s: f64,
    load: &[f64],
) -> Result<(), NewmarkError> {
    if !time_step_s.is_finite() || time_step_s <= 0.0 {
        return Err(NewmarkError::InvalidTimeStep);
    }
    if operators.dof_count() != state.dof_count() || load.len() != state.dof_count() {
        return Err(NewmarkError::DimensionMismatch);
    }
    if load.iter().any(|value| !value.is_finite()) {
        return Err(NewmarkError::NonFiniteInput);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{MatrixProperties, OperatorRole};
    use crate::contracts::StructuralDynamicBalanceSnapshot;
    use runmat_simulation_usage::UsageLedger;
    use std::sync::Arc;

    fn scalar_operators(mass: f64, damping: f64, stiffness: f64) -> CanonicalRealOperators {
        CanonicalRealOperators::try_new(
            RealSparseMatrix::diagonal(
                vec![stiffness],
                OperatorRole::Stiffness,
                MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
            )
            .unwrap(),
            RealSparseMatrix::diagonal(
                vec![mass],
                OperatorRole::Mass,
                MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
            )
            .unwrap(),
            RealSparseMatrix::diagonal(
                vec![damping],
                OperatorRole::Damping,
                MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
            )
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn scalar_newmark_batches_refuse_before_effects_and_conserve_completed_work() {
        let operators = scalar_operators(2.0, 3.0, 5.0);
        let state = SecondOrderState::try_new(vec![1.0], vec![2.0], vec![3.0]).unwrap();
        let original = state.clone();
        // Predictor 5+6, two one-entry CSR dots, RHS 2, and scaled sum
        // 2 scalar scales + 3 products + 2 additions: 22. Completion is 7;
        // correction is 8. No enclosing solve adds these batches again.
        let ledger = Arc::new(
            UsageLedger::new(OperationCount::from_operations(37), OperationCount::ZERO).unwrap(),
        );
        let context = UsageContext::new(Arc::clone(&ledger));
        context.run_sync(|| {
            assert!(matches!(
                prepare_average_acceleration_step(&operators, &state, 0.1, &[7.0]),
                Err(NewmarkError::Usage(LedgerError::MoreGrantRequired { .. }))
            ));
            assert_eq!(state, original);
            assert_eq!(
                ledger.snapshot().unwrap().completed.solve,
                OperationCount::ZERO
            );

            ledger
                .update_grant(OperationCount::from_operations(37))
                .unwrap();
            let step = prepare_average_acceleration_step(&operators, &state, 0.1, &[7.0]).unwrap();
            let completed = step.complete(vec![1.0]).unwrap();
            step.correct(completed, &[0.1]).unwrap();
            assert_eq!(
                ledger.snapshot().unwrap().completed.solve,
                OperationCount::from_operations(37)
            );
        });
    }

    #[test]
    fn average_acceleration_coefficients_build_the_acceleration_effective_equation() {
        let operators = scalar_operators(2.0, 3.0, 5.0);
        let state = SecondOrderState::try_new(vec![1.0], vec![2.0], vec![3.0]).unwrap();
        let step = prepare_average_acceleration_step(&operators, &state, 0.1, &[7.0]).unwrap();

        assert!((step.predictor.displacement[0] - 1.2075).abs() <= 1.0e-14);
        assert!((step.predictor.velocity[0] - 2.15).abs() <= 1.0e-14);
        assert!((step.effective_operator().get(0, 0) - 2.1625).abs() <= 1.0e-14);
        assert!((step.right_hand_side()[0] + 5.4875).abs() <= 1.0e-14);

        let acceleration = step.right_hand_side()[0] / step.effective_operator().get(0, 0);
        let completed = step.complete(vec![acceleration]).unwrap();
        let balance =
            StructuralDynamicBalanceSnapshot::evaluate(&operators, &completed, &[7.0]).unwrap();
        assert!(balance.equilibrium_residual[0].abs() <= 2.0e-14);
        assert_eq!(balance.equilibrium_residual.len(), 1);
    }

    #[test]
    fn equilibrium_correction_survives_predictor_cancellation_below_acceleration_ulp() {
        let stiffness = 1.0e12;
        let operators = scalar_operators(1.0, 0.0, stiffness);
        let initial = SecondOrderState::try_new(vec![0.0], vec![0.0], vec![1.0]).unwrap();
        let step = prepare_average_acceleration_step(&operators, &initial, 0.5, &[1.0]).unwrap();
        let acceleration = step.right_hand_side()[0] / step.effective_operator().get(0, 0);
        let state = step.complete(vec![acceleration]).unwrap();
        let before =
            StructuralDynamicBalanceSnapshot::evaluate(&operators, &state, &[1.0]).unwrap();
        assert!(before.equilibrium_residual[0].abs() > 1.0e-8);
        let correction = -before.equilibrium_residual[0] / step.effective_operator().get(0, 0);
        assert_eq!(acceleration + correction, acceleration);

        let corrected = step.correct(state, &[correction]).unwrap();
        let after =
            StructuralDynamicBalanceSnapshot::evaluate(&operators, &corrected, &[1.0]).unwrap();
        assert!(after.equilibrium_residual[0].abs() <= 16.0 * f64::EPSILON);
        // Independent displacement-form scalar Newmark equation:
        // (K + 16 M) u_1 = f_1 + M a_0 for this initial state and dt.
        let expected_displacement = 2.0 / (stiffness + 16.0);
        assert!(
            (corrected.displacement()[0] / expected_displacement - 1.0).abs() <= 8.0 * f64::EPSILON
        );
    }

    #[test]
    fn invalid_time_step_and_input_dimensions_are_rejected() {
        let operators = scalar_operators(1.0, 0.0, 1.0);
        let state = SecondOrderState::try_new(vec![0.0], vec![0.0], vec![0.0]).unwrap();
        assert_eq!(
            prepare_average_acceleration_step(&operators, &state, 0.0, &[0.0]),
            Err(NewmarkError::InvalidTimeStep)
        );
        assert_eq!(
            prepare_average_acceleration_step(&operators, &state, 0.1, &[]),
            Err(NewmarkError::DimensionMismatch)
        );
        assert_eq!(
            StructuralDynamicBalanceSnapshot::evaluate(&operators, &state, &[f64::INFINITY]),
            Err(super::super::balance::DynamicBalanceError::NonFinite)
        );
    }

    // witness: FEA-DYNAMICS-RESPONSE-001/second_order_equilibrium_excitation @
    //   linear_structural_dynamics/mesh_independent/numerical_kernel; coverage: partial;
    //   boundary: Undamped free vibration under Newmark converges at second order; damped and forced
    //   cases are other tests.
    #[test]
    fn undamped_free_vibration_converges_at_second_order() {
        let omega = 2.0_f64;
        let coarse = integrate_free_vibration(omega, 10);
        let medium = integrate_free_vibration(omega, 20);
        let fine = integrate_free_vibration(omega, 40);
        let exact = [omega.cos(), -omega * omega.sin()];
        let error = |state: &SecondOrderState| {
            (state.displacement()[0] - exact[0]).hypot(state.velocity()[0] - exact[1])
        };
        let coarse_error = error(&coarse);
        let medium_error = error(&medium);
        let fine_error = error(&fine);

        assert!(coarse_error / medium_error > 3.8);
        assert!(medium_error / fine_error > 3.8);
        assert!(fine_error < 2.0e-3);
    }

    fn integrate_free_vibration(omega: f64, step_count: usize) -> SecondOrderState {
        let operators = scalar_operators(1.0, 0.0, omega * omega);
        let time_step_s = 1.0 / step_count as f64;
        let mut state =
            SecondOrderState::try_new(vec![1.0], vec![0.0], vec![-omega * omega]).unwrap();
        for _ in 0..step_count {
            let step =
                prepare_average_acceleration_step(&operators, &state, time_step_s, &[0.0]).unwrap();
            let acceleration = step.right_hand_side()[0] / step.effective_operator().get(0, 0);
            state = step.complete(vec![acceleration]).unwrap();
        }
        state
    }
}
