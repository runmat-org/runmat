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
}

impl NewmarkEffectiveStep {
    pub(super) fn effective_operator(&self) -> &RealSparseMatrix {
        &self.effective_operator
    }

    pub(super) fn right_hand_side(&self) -> &[f64] {
        &self.right_hand_side
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
        let displacement = self
            .predictor
            .displacement
            .iter()
            .zip(&acceleration)
            .map(|(predicted, acceleration)| {
                predicted
                    + AverageAccelerationNewmark::BETA
                        * self.predictor.time_step_s.powi(2)
                        * acceleration
            })
            .collect();
        let velocity = self
            .predictor
            .velocity
            .iter()
            .zip(&acceleration)
            .map(|(predicted, acceleration)| {
                predicted
                    + AverageAccelerationNewmark::GAMMA * self.predictor.time_step_s * acceleration
            })
            .collect();
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

    let displacement = state
        .displacement()
        .iter()
        .zip(state.velocity())
        .zip(state.acceleration())
        .map(|((displacement, velocity), acceleration)| {
            displacement
                + time_step_s * velocity
                + time_step_s.powi(2) * (0.5 - AverageAccelerationNewmark::BETA) * acceleration
        })
        .collect::<Vec<_>>();
    let velocity = state
        .velocity()
        .iter()
        .zip(state.acceleration())
        .map(|(velocity, acceleration)| {
            velocity + time_step_s * (1.0 - AverageAccelerationNewmark::GAMMA) * acceleration
        })
        .collect::<Vec<_>>();

    let damping_force =
        spmv(operators.damping(), &velocity).map_err(|_| NewmarkError::DimensionMismatch)?;
    let elastic_force =
        spmv(operators.stiffness(), &displacement).map_err(|_| NewmarkError::DimensionMismatch)?;
    let right_hand_side = load
        .iter()
        .zip(damping_force)
        .zip(elastic_force)
        .map(|((load, damping), elastic)| load - damping - elastic)
        .collect::<Vec<_>>();

    let effective_operator = RealSparseMatrix::scaled_sum(
        &[
            (1.0, operators.mass()),
            (
                AverageAccelerationNewmark::GAMMA * time_step_s,
                operators.damping(),
            ),
            (
                AverageAccelerationNewmark::BETA * time_step_s.powi(2),
                operators.stiffness(),
            ),
        ],
        OperatorRole::EffectiveTangent,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )
    .map_err(NewmarkError::EffectiveOperator)?
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
