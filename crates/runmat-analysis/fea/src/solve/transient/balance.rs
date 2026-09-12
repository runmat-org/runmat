use thiserror::Error;

use crate::{
    algebra::{spmv, CanonicalRealOperators},
    assembly::AssemblySummary,
    contracts::StructuralDynamicBalanceSnapshot,
};

use super::{constraint_state::physical_load, state::SecondOrderState};

#[derive(Clone, Debug, PartialEq)]
pub(super) struct TransientBalanceHistory {
    initial_mechanical_energy_j: f64,
    balances: Vec<StructuralDynamicBalanceSnapshot>,
    reaction_snapshots: Vec<Vec<f64>>,
    strain_energy_j: Vec<f64>,
    kinetic_energy_j: Vec<f64>,
    external_work_j: Vec<f64>,
    damping_dissipation_j: Vec<f64>,
    energy_balance_residual_j: Vec<f64>,
    relative_energy_balance_error: Vec<f64>,
}

#[derive(Clone, Debug, Error, PartialEq)]
pub(super) enum DynamicBalanceError {
    #[error("the structural dynamic-balance operator, state, and load dimensions must agree")]
    DimensionMismatch,
    #[error("the structural dynamic-balance input or result contains a nonfinite value")]
    NonFinite,
    #[error("the structural damping work is negative ({increment_j} J)")]
    NegativeDampingDissipation { increment_j: f64 },
}

impl StructuralDynamicBalanceSnapshot {
    pub(super) fn evaluate(
        operators: &CanonicalRealOperators,
        state: &SecondOrderState,
        load: &[f64],
    ) -> Result<Self, DynamicBalanceError> {
        if operators.dof_count() != state.dof_count() || load.len() != state.dof_count() {
            return Err(DynamicBalanceError::DimensionMismatch);
        }
        if load.iter().any(|value| !value.is_finite()) {
            return Err(DynamicBalanceError::NonFinite);
        }
        let inertia_force = spmv(operators.mass(), state.acceleration())
            .map_err(|_| DynamicBalanceError::DimensionMismatch)?;
        let damping_force = spmv(operators.damping(), state.velocity())
            .map_err(|_| DynamicBalanceError::DimensionMismatch)?;
        let elastic_force = spmv(operators.stiffness(), state.displacement())
            .map_err(|_| DynamicBalanceError::DimensionMismatch)?;
        let equilibrium_residual = inertia_force
            .iter()
            .zip(&damping_force)
            .zip(&elastic_force)
            .zip(load)
            .map(|(((inertia, damping), elastic), load)| inertia + damping + elastic - load)
            .collect::<Vec<_>>();
        if inertia_force
            .iter()
            .chain(&damping_force)
            .chain(&elastic_force)
            .chain(&equilibrium_residual)
            .any(|value| !value.is_finite())
        {
            return Err(DynamicBalanceError::NonFinite);
        }
        let snapshot = Self {
            applied_load: load.to_vec(),
            inertia_force,
            damping_force,
            elastic_force,
            equilibrium_residual,
        };
        snapshot
            .validate()
            .map_err(|_| DynamicBalanceError::NonFinite)?;
        Ok(snapshot)
    }

    pub(super) fn imbalance(&self) -> Vec<f64> {
        self.equilibrium_residual
            .iter()
            .map(|value| -*value)
            .collect()
    }
}

impl TransientBalanceHistory {
    pub(super) fn start(
        summary: &AssemblySummary,
        state: &SecondOrderState,
    ) -> Result<Self, DynamicBalanceError> {
        let load = physical_load(summary);
        let balance =
            StructuralDynamicBalanceSnapshot::evaluate(summary.operator.operators(), state, &load)?;
        let strain = strain_energy(summary.operator.operators(), state)?;
        let kinetic = kinetic_energy(summary.operator.operators(), state)?;
        let initial_mechanical_energy_j = strain + kinetic;
        Ok(Self {
            initial_mechanical_energy_j,
            balances: vec![balance.clone()],
            reaction_snapshots: vec![constrained_reaction(summary, &balance)],
            strain_energy_j: vec![strain],
            kinetic_energy_j: vec![kinetic],
            external_work_j: vec![0.0],
            damping_dissipation_j: vec![0.0],
            energy_balance_residual_j: vec![0.0],
            relative_energy_balance_error: vec![0.0],
        })
    }

    pub(super) fn push_step(
        &mut self,
        previous_summary: &AssemblySummary,
        previous_state: &SecondOrderState,
        current_summary: &AssemblySummary,
        current_state: &SecondOrderState,
        time_step_s: f64,
    ) -> Result<(), DynamicBalanceError> {
        if !time_step_s.is_finite() || time_step_s <= 0.0 {
            return Err(DynamicBalanceError::NonFinite);
        }
        let previous_load = physical_load(previous_summary);
        let current_load = physical_load(current_summary);
        let balance = StructuralDynamicBalanceSnapshot::evaluate(
            current_summary.operator.operators(),
            current_state,
            &current_load,
        )?;
        let displacement_increment = current_state
            .displacement()
            .iter()
            .zip(previous_state.displacement())
            .map(|(current, previous)| current - previous)
            .collect::<Vec<_>>();
        let average_load = current_load
            .iter()
            .zip(previous_load)
            .map(|(current, previous)| 0.5 * (current + previous))
            .collect::<Vec<_>>();
        let average_velocity = current_state
            .velocity()
            .iter()
            .zip(previous_state.velocity())
            .map(|(current, previous)| 0.5 * (current + previous))
            .collect::<Vec<_>>();
        let average_damping_force = spmv(
            current_summary.operator.operators().damping(),
            &average_velocity,
        )
        .map_err(|_| DynamicBalanceError::DimensionMismatch)?;
        let external_work_increment = dot(&average_load, &displacement_increment)?;
        let damping_dissipation_increment =
            time_step_s * dot(&average_velocity, &average_damping_force)?;
        let damping_scale = time_step_s
            * norm(&average_velocity)
            * norm(&average_damping_force).max(f64::MIN_POSITIVE);
        if damping_dissipation_increment < -1.0e-12 * damping_scale {
            return Err(DynamicBalanceError::NegativeDampingDissipation {
                increment_j: damping_dissipation_increment,
            });
        }
        let damping_dissipation_increment = damping_dissipation_increment.max(0.0);
        let external_work =
            self.external_work_j.last().copied().unwrap_or(0.0) + external_work_increment;
        let damping_dissipation = self.damping_dissipation_j.last().copied().unwrap_or(0.0)
            + damping_dissipation_increment;
        let strain = strain_energy(current_summary.operator.operators(), current_state)?;
        let kinetic = kinetic_energy(current_summary.operator.operators(), current_state)?;
        let mechanical = strain + kinetic;
        let residual =
            mechanical - self.initial_mechanical_energy_j - external_work + damping_dissipation;
        let scale = mechanical
            .abs()
            .max(self.initial_mechanical_energy_j.abs())
            .max(external_work.abs())
            .max(damping_dissipation.abs());
        let relative = if scale == 0.0 {
            0.0
        } else {
            residual.abs() / scale
        };
        if [
            external_work,
            damping_dissipation,
            strain,
            kinetic,
            residual,
            relative,
        ]
        .iter()
        .any(|value| !value.is_finite())
        {
            return Err(DynamicBalanceError::NonFinite);
        }
        self.reaction_snapshots
            .push(constrained_reaction(current_summary, &balance));
        self.balances.push(balance);
        self.strain_energy_j.push(strain);
        self.kinetic_energy_j.push(kinetic);
        self.external_work_j.push(external_work);
        self.damping_dissipation_j.push(damping_dissipation);
        self.energy_balance_residual_j.push(residual);
        self.relative_energy_balance_error.push(relative);
        Ok(())
    }

    pub(super) fn refresh_reaction(
        &mut self,
        summary: &AssemblySummary,
        state: &SecondOrderState,
    ) -> Result<(), DynamicBalanceError> {
        let load = physical_load(summary);
        let balance =
            StructuralDynamicBalanceSnapshot::evaluate(summary.operator.operators(), state, &load)?;
        *self
            .reaction_snapshots
            .last_mut()
            .ok_or(DynamicBalanceError::DimensionMismatch)? =
            constrained_reaction(summary, &balance);
        *self
            .balances
            .last_mut()
            .ok_or(DynamicBalanceError::DimensionMismatch)? = balance;
        Ok(())
    }

    pub(super) fn balances(&self) -> &[StructuralDynamicBalanceSnapshot] {
        &self.balances
    }

    pub(super) fn reaction_snapshots(&self) -> &[Vec<f64>] {
        &self.reaction_snapshots
    }

    pub(super) fn strain_energy_j(&self) -> &[f64] {
        &self.strain_energy_j
    }

    pub(super) fn kinetic_energy_j(&self) -> &[f64] {
        &self.kinetic_energy_j
    }

    pub(super) fn external_work_j(&self) -> &[f64] {
        &self.external_work_j
    }

    pub(super) fn damping_dissipation_j(&self) -> &[f64] {
        &self.damping_dissipation_j
    }

    pub(super) fn energy_balance_residual_j(&self) -> &[f64] {
        &self.energy_balance_residual_j
    }

    pub(super) fn relative_energy_balance_error(&self) -> &[f64] {
        &self.relative_energy_balance_error
    }
}

fn constrained_reaction(
    summary: &AssemblySummary,
    balance: &StructuralDynamicBalanceSnapshot,
) -> Vec<f64> {
    let mut owned = summary.operator.constrained_dofs().to_vec();
    if let Some(matrix) = summary.constraint_matrix.as_ref() {
        for dof in &matrix.column_indices {
            if let Some(value) = owned.get_mut(*dof) {
                *value = true;
            }
        }
    }
    balance
        .equilibrium_residual
        .iter()
        .zip(owned)
        .map(|(residual, constrained)| if constrained { *residual } else { 0.0 })
        .collect()
}

fn strain_energy(
    operators: &CanonicalRealOperators,
    state: &SecondOrderState,
) -> Result<f64, DynamicBalanceError> {
    let force = spmv(operators.stiffness(), state.displacement())
        .map_err(|_| DynamicBalanceError::DimensionMismatch)?;
    Ok(0.5 * dot(state.displacement(), &force)?)
}

fn kinetic_energy(
    operators: &CanonicalRealOperators,
    state: &SecondOrderState,
) -> Result<f64, DynamicBalanceError> {
    let momentum = spmv(operators.mass(), state.velocity())
        .map_err(|_| DynamicBalanceError::DimensionMismatch)?;
    Ok(0.5 * dot(state.velocity(), &momentum)?)
}

fn dot(left: &[f64], right: &[f64]) -> Result<f64, DynamicBalanceError> {
    if left.len() != right.len() {
        return Err(DynamicBalanceError::DimensionMismatch);
    }
    let value = left
        .iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum::<f64>();
    value
        .is_finite()
        .then_some(value)
        .ok_or(DynamicBalanceError::NonFinite)
}

fn norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{MatrixProperties, OperatorRole, RealSparseMatrix};

    #[test]
    fn dynamic_balance_retains_each_physical_component_and_exact_residual_sum() {
        let operators = CanonicalRealOperators::try_new(
            RealSparseMatrix::diagonal(
                vec![10.0, 20.0],
                OperatorRole::Stiffness,
                MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
            )
            .unwrap(),
            RealSparseMatrix::diagonal(
                vec![2.0, 4.0],
                OperatorRole::Mass,
                MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
            )
            .unwrap(),
            RealSparseMatrix::diagonal(
                vec![0.5, 1.0],
                OperatorRole::Damping,
                MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
            )
            .unwrap(),
        )
        .unwrap();
        let state =
            SecondOrderState::try_new(vec![0.2, -0.1], vec![3.0, 2.0], vec![-4.0, 5.0]).unwrap();

        let balance =
            StructuralDynamicBalanceSnapshot::evaluate(&operators, &state, &[7.0, 11.0]).unwrap();

        assert_eq!(balance.applied_load, vec![7.0, 11.0]);
        assert_eq!(balance.inertia_force, vec![-8.0, 20.0]);
        assert_eq!(balance.damping_force, vec![1.5, 2.0]);
        assert_eq!(balance.elastic_force, vec![2.0, -2.0]);
        assert_eq!(balance.equilibrium_residual, vec![-11.5, 9.0]);
        for dof in 0..2 {
            assert_eq!(
                balance.equilibrium_residual[dof],
                balance.inertia_force[dof]
                    + balance.damping_force[dof]
                    + balance.elastic_force[dof]
                    - balance.applied_load[dof]
            );
        }
    }
}
