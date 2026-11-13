use runmat_simulation_usage::{LedgerError, OperationCount, UsageCategory, UsageContext};
use thiserror::Error;

use crate::{
    algebra::{spmv, CanonicalRealOperators, RealSparseMatrix},
    assembly::AssemblySummary,
    contracts::StructuralDynamicBalanceSnapshot,
    execution::StructuralTransientBalanceAccumulator,
    solve::reaction::{recover_structural_reaction, StructuralReactionError},
};

use super::{
    constraint_state::physical_load,
    newmark::{count, spmv_work},
    state::SecondOrderState,
    TransientSolveError,
};

pub(super) struct TransientBalanceStep<'a> {
    pub(super) integration_start_summary: &'a AssemblySummary,
    pub(super) integration_start_state: &'a SecondOrderState,
    pub(super) integration_end_summary: &'a AssemblySummary,
    pub(super) integration_end_state: &'a SecondOrderState,
    pub(super) published_summary: &'a AssemblySummary,
    pub(super) published_state: &'a SecondOrderState,
    pub(super) time_step_s: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct TransientBalanceHistory {
    initial_mechanical_energy_j: runmat_types::physical::Energy,
    balances: Vec<StructuralDynamicBalanceSnapshot>,
    reaction_snapshots: Vec<Vec<f64>>,
    strain_energy_j:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Energy, Vec<f64>>,
    kinetic_energy_j:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Energy, Vec<f64>>,
    external_work_history:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Energy, Vec<f64>>,
    support_work_history:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Energy, Vec<f64>>,
    damping_dissipation_history:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Energy, Vec<f64>>,
    energy_balance_residual_history:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Energy, Vec<f64>>,
    relative_energy_balance_error: Vec<f64>,
}

#[derive(Clone, Debug, Error, PartialEq)]
pub(super) enum DynamicBalanceError {
    #[error("transient balance reconstruction is invalid: {0}")]
    InvalidHistory(String),
    #[error("the structural dynamic-balance operator, state, and load dimensions must agree")]
    DimensionMismatch,
    #[error("the structural dynamic-balance input or result contains a nonfinite value")]
    NonFinite,
    #[error("the structural damping work is negative ({increment_j} J)")]
    NegativeDampingDissipation { increment_j: f64 },
    #[error("the {quantity} quadratic energy is materially negative ({value_j} J)")]
    NegativeQuadraticEnergy {
        quantity: &'static str,
        value_j: f64,
    },
    #[error("constraint-row reactions could not be recovered from the dynamic residual")]
    ReactionRecovery,
    #[error("simulation usage admission failed: {0}")]
    Usage(#[from] LedgerError),
}

impl DynamicBalanceError {
    pub(super) fn into_transient_solve<E>(self) -> TransientSolveError<E> {
        match self {
            Self::Usage(error) => TransientSolveError::Usage(error),
            other => TransientSolveError::DynamicBalance {
                detail: other.to_string(),
            },
        }
    }
}

fn measured<R>(
    quote: impl FnOnce() -> Result<OperationCount, LedgerError>,
    run: impl FnOnce() -> Result<R, DynamicBalanceError>,
) -> Result<R, DynamicBalanceError> {
    let Some(context) = UsageContext::capture() else {
        return run();
    };
    drop(context);
    crate::algebra::capacity::measured_sync(UsageCategory::Solve, quote()?, run)
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
        let inertia_force = measured(
            || spmv_work(operators.mass()),
            || {
                spmv(operators.mass(), state.acceleration())
                    .map_err(|_| DynamicBalanceError::DimensionMismatch)
            },
        )?;
        let damping_force = measured(
            || spmv_work(operators.damping()),
            || {
                spmv(operators.damping(), state.velocity())
                    .map_err(|_| DynamicBalanceError::DimensionMismatch)
            },
        )?;
        let elastic_force = measured(
            || spmv_work(operators.stiffness()),
            || {
                spmv(operators.stiffness(), state.displacement())
                    .map_err(|_| DynamicBalanceError::DimensionMismatch)
            },
        )?;
        let equilibrium_residual = measured(
            || count(load.len(), 3),
            || {
                Ok(inertia_force
                    .iter()
                    .zip(&damping_force)
                    .zip(&elastic_force)
                    .zip(load)
                    .map(|(((inertia, damping), elastic), load)| inertia + damping + elastic - load)
                    .collect::<Vec<_>>())
            },
        )?;
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
        let initial_mechanical_energy_j = measured(
            || Ok(OperationCount::from_operations(1)),
            || Ok(strain + kinetic),
        )?;
        Ok(Self {
            initial_mechanical_energy_j: runmat_types::physical::Energy::new(
                initial_mechanical_energy_j,
            ),
            balances: vec![balance.clone()],
            reaction_snapshots: vec![constrained_reaction(summary, &balance)?],
            strain_energy_j: runmat_types::physical::Quantity::new(vec![strain]),
            kinetic_energy_j: runmat_types::physical::Quantity::new(vec![kinetic]),
            external_work_history: runmat_types::physical::Quantity::new(vec![0.0]),
            support_work_history: runmat_types::physical::Quantity::new(vec![0.0]),
            damping_dissipation_history: runmat_types::physical::Quantity::new(vec![0.0]),
            energy_balance_residual_history: runmat_types::physical::Quantity::new(vec![0.0]),
            relative_energy_balance_error: vec![0.0],
        })
    }

    pub(super) fn resume(
        summary: &AssemblySummary,
        state: &SecondOrderState,
        accumulator: StructuralTransientBalanceAccumulator,
    ) -> Result<Self, DynamicBalanceError> {
        if [
            accumulator.initial_mechanical_energy_j.canonical_value(),
            accumulator.applied_load_work_j.canonical_value(),
            accumulator.support_work_j.canonical_value(),
            accumulator.damping_dissipation_j.canonical_value(),
        ]
        .iter()
        .any(|value| !value.is_finite())
            || accumulator.initial_mechanical_energy_j.canonical_value() < 0.0
            || accumulator.damping_dissipation_j.canonical_value() < 0.0
        {
            return Err(DynamicBalanceError::NonFinite);
        }
        let load = physical_load(summary);
        let balance =
            StructuralDynamicBalanceSnapshot::evaluate(summary.operator.operators(), state, &load)?;
        let strain = strain_energy(summary.operator.operators(), state)?;
        let kinetic = kinetic_energy(summary.operator.operators(), state)?;
        let (mechanical, residual) = measured(
            || Ok(OperationCount::from_operations(5)),
            || {
                let mechanical = strain + kinetic;
                Ok((
                    mechanical,
                    mechanical
                        - accumulator.initial_mechanical_energy_j.canonical_value()
                        - accumulator.applied_load_work_j.canonical_value()
                        - accumulator.support_work_j.canonical_value()
                        + accumulator.damping_dissipation_j.canonical_value(),
                ))
            },
        )?;
        let scale = mechanical
            .abs()
            .max(
                accumulator
                    .initial_mechanical_energy_j
                    .canonical_value()
                    .abs(),
            )
            .max(accumulator.applied_load_work_j.canonical_value().abs())
            .max(accumulator.support_work_j.canonical_value().abs())
            .max(accumulator.damping_dissipation_j.canonical_value().abs());
        let relative = if scale == 0.0 {
            0.0
        } else {
            measured(
                || Ok(OperationCount::from_operations(1)),
                || Ok(residual.abs() / scale),
            )?
        };
        if !residual.is_finite() || !relative.is_finite() {
            return Err(DynamicBalanceError::NonFinite);
        }
        Ok(Self {
            initial_mechanical_energy_j: runmat_types::physical::Energy::new(
                accumulator.initial_mechanical_energy_j.canonical_value(),
            ),
            reaction_snapshots: vec![constrained_reaction(summary, &balance)?],
            balances: vec![balance],
            strain_energy_j: runmat_types::physical::Quantity::new(vec![strain]),
            kinetic_energy_j: runmat_types::physical::Quantity::new(vec![kinetic]),
            external_work_history: runmat_types::physical::Quantity::new(vec![accumulator
                .applied_load_work_j
                .canonical_value()]),
            support_work_history: runmat_types::physical::Quantity::new(vec![accumulator
                .support_work_j
                .canonical_value()]),
            damping_dissipation_history: runmat_types::physical::Quantity::new(vec![accumulator
                .damping_dissipation_j
                .canonical_value()]),
            energy_balance_residual_history: runmat_types::physical::Quantity::new(vec![residual]),
            relative_energy_balance_error: vec![relative],
        })
    }

    pub(super) fn accumulator(&self) -> StructuralTransientBalanceAccumulator {
        StructuralTransientBalanceAccumulator {
            initial_mechanical_energy_j: runmat_types::physical::Energy::new(
                self.initial_mechanical_energy_j.canonical_value(),
            ),
            applied_load_work_j: runmat_types::physical::Energy::new(
                self.external_work_history
                    .canonical_values()
                    .last()
                    .copied()
                    .unwrap_or(0.0),
            ),
            support_work_j: runmat_types::physical::Energy::new(
                self.support_work_history
                    .canonical_values()
                    .last()
                    .copied()
                    .unwrap_or(0.0),
            ),
            damping_dissipation_j: runmat_types::physical::Energy::new(
                self.damping_dissipation_history
                    .canonical_values()
                    .last()
                    .copied()
                    .unwrap_or(0.0),
            ),
        }
    }

    pub(super) fn push_step(
        &mut self,
        step: TransientBalanceStep<'_>,
    ) -> Result<(), DynamicBalanceError> {
        if !step.time_step_s.is_finite() || step.time_step_s <= 0.0 {
            return Err(DynamicBalanceError::NonFinite);
        }
        let previous_load = physical_load(step.integration_start_summary);
        let current_load = physical_load(step.integration_end_summary);
        let balance = StructuralDynamicBalanceSnapshot::evaluate(
            step.published_summary.operator.operators(),
            step.published_state,
            &physical_load(step.published_summary),
        )?;
        let displacement_increment = measured(
            || count(step.integration_end_state.displacement().len(), 1),
            || {
                Ok(step
                    .integration_end_state
                    .displacement()
                    .iter()
                    .zip(step.integration_start_state.displacement())
                    .map(|(current, previous)| current - previous)
                    .collect::<Vec<_>>())
            },
        )?;
        let average_load = measured(
            || count(current_load.len(), 2),
            || {
                Ok(current_load
                    .iter()
                    .zip(&previous_load)
                    .map(|(current, previous)| 0.5 * (current + previous))
                    .collect::<Vec<_>>())
            },
        )?;
        let average_velocity = measured(
            || count(step.integration_end_state.velocity().len(), 2),
            || {
                Ok(step
                    .integration_end_state
                    .velocity()
                    .iter()
                    .zip(step.integration_start_state.velocity())
                    .map(|(current, previous)| 0.5 * (current + previous))
                    .collect::<Vec<_>>())
            },
        )?;
        let average_damping_force = measured(
            || spmv_work(step.integration_end_summary.operator.operators().damping()),
            || {
                spmv(
                    step.integration_end_summary.operator.operators().damping(),
                    &average_velocity,
                )
                .map_err(|_| DynamicBalanceError::DimensionMismatch)
            },
        )?;
        let external_work_increment = dot(&average_load, &displacement_increment)?;
        let previous_balance = StructuralDynamicBalanceSnapshot::evaluate(
            step.integration_start_summary.operator.operators(),
            step.integration_start_state,
            &previous_load,
        )?;
        let previous_reaction =
            constrained_reaction(step.integration_start_summary, &previous_balance)?;
        let integration_end_balance = StructuralDynamicBalanceSnapshot::evaluate(
            step.integration_end_summary.operator.operators(),
            step.integration_end_state,
            &current_load,
        )?;
        let integration_end_reaction =
            constrained_reaction(step.integration_end_summary, &integration_end_balance)?;
        let average_reaction = measured(
            || count(previous_reaction.len(), 2),
            || {
                Ok(previous_reaction
                    .iter()
                    .zip(&integration_end_reaction)
                    .map(|(previous, current)| 0.5 * (previous + current))
                    .collect::<Vec<_>>())
            },
        )?;
        let reaction_work = dot(&average_reaction, &displacement_increment)?;
        let kinematics_work = prescribed_kinematics_work_correction(&step, &average_velocity)?;
        let support_work_increment = measured(
            || Ok(OperationCount::from_operations(1)),
            || Ok(reaction_work + kinematics_work),
        )?;
        let damping_dot = dot(&average_velocity, &average_damping_force)?;
        let damping_dissipation_increment = measured(
            || Ok(OperationCount::from_operations(1)),
            || Ok(step.time_step_s * damping_dot),
        )?;
        let velocity_norm = norm(&average_velocity)?;
        let damping_norm = norm(&average_damping_force)?;
        let damping_scale = measured(
            || Ok(OperationCount::from_operations(2)),
            || Ok(step.time_step_s * velocity_norm * damping_norm.max(f64::MIN_POSITIVE)),
        )?;
        if damping_dissipation_increment < -1.0e-12 * damping_scale {
            return Err(DynamicBalanceError::NegativeDampingDissipation {
                increment_j: damping_dissipation_increment,
            });
        }
        let damping_dissipation_increment = damping_dissipation_increment.max(0.0);
        let (external_work, support_work, damping_dissipation) = measured(
            || Ok(OperationCount::from_operations(3)),
            || {
                Ok((
                    self.external_work_history
                        .canonical_values()
                        .last()
                        .copied()
                        .unwrap_or(0.0)
                        + external_work_increment,
                    self.support_work_history
                        .canonical_values()
                        .last()
                        .copied()
                        .unwrap_or(0.0)
                        + support_work_increment,
                    self.damping_dissipation_history
                        .canonical_values()
                        .last()
                        .copied()
                        .unwrap_or(0.0)
                        + damping_dissipation_increment,
                ))
            },
        )?;
        let strain = strain_energy(
            step.published_summary.operator.operators(),
            step.published_state,
        )?;
        let kinetic = kinetic_energy(
            step.published_summary.operator.operators(),
            step.published_state,
        )?;
        let (mechanical, residual) = measured(
            || Ok(OperationCount::from_operations(5)),
            || {
                let mechanical = strain + kinetic;
                Ok((
                    mechanical,
                    mechanical
                        - self.initial_mechanical_energy_j.canonical_value()
                        - external_work
                        - support_work
                        + damping_dissipation,
                ))
            },
        )?;
        let scale = mechanical
            .abs()
            .max(self.initial_mechanical_energy_j.canonical_value().abs())
            .max(external_work.abs())
            .max(support_work.abs())
            .max(damping_dissipation.abs());
        let relative = if scale == 0.0 {
            0.0
        } else {
            measured(
                || Ok(OperationCount::from_operations(1)),
                || Ok(residual.abs() / scale),
            )?
        };
        if [
            external_work,
            support_work,
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
            .push(constrained_reaction(step.published_summary, &balance)?);
        self.balances.push(balance);
        self.strain_energy_j
            .try_push(runmat_types::physical::Energy::new(strain))
            .map_err(|_| DynamicBalanceError::NonFinite)?;
        self.kinetic_energy_j
            .try_push(runmat_types::physical::Energy::new(kinetic))
            .map_err(|_| DynamicBalanceError::NonFinite)?;
        self.external_work_history
            .try_push(runmat_types::physical::Energy::new(external_work))
            .map_err(|_| DynamicBalanceError::NonFinite)?;
        self.support_work_history
            .try_push(runmat_types::physical::Energy::new(support_work))
            .map_err(|_| DynamicBalanceError::NonFinite)?;
        self.damping_dissipation_history
            .try_push(runmat_types::physical::Energy::new(damping_dissipation))
            .map_err(|_| DynamicBalanceError::NonFinite)?;
        self.energy_balance_residual_history
            .try_push(runmat_types::physical::Energy::new(residual))
            .map_err(|_| DynamicBalanceError::NonFinite)?;
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
            constrained_reaction(summary, &balance)?;
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
        self.strain_energy_j.canonical_values()
    }

    pub(super) fn kinetic_energy_j(&self) -> &[f64] {
        self.kinetic_energy_j.canonical_values()
    }

    pub(super) fn external_work_j(&self) -> &[f64] {
        self.external_work_history.canonical_values()
    }

    pub(super) fn support_work_j(&self) -> &[f64] {
        self.support_work_history.canonical_values()
    }

    pub(super) fn damping_dissipation_j(&self) -> &[f64] {
        self.damping_dissipation_history.canonical_values()
    }

    pub(super) fn energy_balance_residual_j(&self) -> &[f64] {
        self.energy_balance_residual_history.canonical_values()
    }

    pub(super) fn relative_energy_balance_error(&self) -> &[f64] {
        &self.relative_energy_balance_error
    }
}

/// Support work that the trapezoidal reaction quadrature `avg(R)·Δu` misses when constrained
/// coordinates follow prescribed kinematics that are not trapezoid-consistent.
///
/// The average-acceleration integrator advances every integrated coordinate with
/// `Δu = Δt·avg(v)` and `Δv = Δt·avg(a)`. Under those relations the step's change in
/// `½vᵀMv + ½uᵀKu` equals `avg(F + R)·Δu − Δt·avg(v)ᵀC·avg(v)` exactly. A prescribed support
/// motion is instead reconstructed exactly from its signal (for example the cubic displacement
/// of a piecewise-linear acceleration), so on its constrained columns the kinematic defects
/// `δu = Δu − Δt·avg(v)` and `δv = Δv − Δt·avg(a)` are nonzero. The energy the moving support
/// delivers through that defect is `avg(v)ᵀM·δv − avg(Ma + Cv)ᵀ·δu`; it vanishes for fixed and
/// trapezoid-consistent supports and converges to zero at O(Δt²) over a fixed interval.
fn prescribed_kinematics_work_correction(
    step: &TransientBalanceStep<'_>,
    average_velocity: &[f64],
) -> Result<f64, DynamicBalanceError> {
    let Some(constraints) = step.integration_end_summary.constraint_matrix.as_ref() else {
        return Ok(0.0);
    };
    let start = step.integration_start_state;
    let end = step.integration_end_state;
    let dof_count = end.dof_count();
    let mut displacement_defect = vec![0.0; dof_count];
    let mut velocity_defect = vec![0.0; dof_count];
    measured(
        || count(constraints.column_indices.len(), 8),
        || {
            for dof in constraints.column_indices.iter().copied() {
                if dof >= dof_count {
                    return Err(DynamicBalanceError::DimensionMismatch);
                }
                displacement_defect[dof] = end.displacement()[dof]
                    - start.displacement()[dof]
                    - step.time_step_s * average_velocity[dof];
                velocity_defect[dof] = end.velocity()[dof]
                    - start.velocity()[dof]
                    - step.time_step_s
                        * 0.5
                        * (end.acceleration()[dof] + start.acceleration()[dof]);
            }
            Ok(())
        },
    )?;
    if displacement_defect
        .iter()
        .chain(&velocity_defect)
        .all(|value| *value == 0.0)
    {
        return Ok(0.0);
    }
    let operators = step.integration_end_summary.operator.operators();
    let average_acceleration = measured(
        || count(end.acceleration().len(), 2),
        || {
            Ok(end
                .acceleration()
                .iter()
                .zip(start.acceleration())
                .map(|(current, previous)| 0.5 * (current + previous))
                .collect::<Vec<_>>())
        },
    )?;
    let inertia = measured(
        || spmv_work(operators.mass()),
        || {
            spmv(operators.mass(), &average_acceleration)
                .map_err(|_| DynamicBalanceError::DimensionMismatch)
        },
    )?;
    let damping = measured(
        || spmv_work(operators.damping()),
        || {
            spmv(operators.damping(), average_velocity)
                .map_err(|_| DynamicBalanceError::DimensionMismatch)
        },
    )?;
    let mass_velocity_defect = measured(
        || spmv_work(operators.mass()),
        || {
            spmv(operators.mass(), &velocity_defect)
                .map_err(|_| DynamicBalanceError::DimensionMismatch)
        },
    )?;
    let inertia_and_damping = measured(
        || count(inertia.len(), 1),
        || {
            Ok(inertia
                .iter()
                .zip(&damping)
                .map(|(inertia, damping)| inertia + damping)
                .collect::<Vec<_>>())
        },
    )?;
    let kinetic_defect = dot(average_velocity, &mass_velocity_defect)?;
    let force_defect = dot(&inertia_and_damping, &displacement_defect)?;
    measured(
        || Ok(OperationCount::from_operations(1)),
        || Ok(kinetic_defect - force_defect),
    )
}

fn constrained_reaction(
    summary: &AssemblySummary,
    balance: &StructuralDynamicBalanceSnapshot,
) -> Result<Vec<f64>, DynamicBalanceError> {
    recover_structural_reaction(summary, &balance.equilibrium_residual).map_err(|error| match error
    {
        StructuralReactionError::Accounting(error) => DynamicBalanceError::Usage(error),
        _ => DynamicBalanceError::ReactionRecovery,
    })
}

fn strain_energy(
    operators: &CanonicalRealOperators,
    state: &SecondOrderState,
) -> Result<f64, DynamicBalanceError> {
    let force = measured(
        || spmv_work(operators.stiffness()),
        || {
            spmv(operators.stiffness(), state.displacement())
                .map_err(|_| DynamicBalanceError::DimensionMismatch)
        },
    )?;
    nonnegative_quadratic_energy(
        "strain",
        operators.stiffness(),
        state.displacement(),
        &force,
    )
}

fn kinetic_energy(
    operators: &CanonicalRealOperators,
    state: &SecondOrderState,
) -> Result<f64, DynamicBalanceError> {
    let momentum = measured(
        || spmv_work(operators.mass()),
        || {
            spmv(operators.mass(), state.velocity())
                .map_err(|_| DynamicBalanceError::DimensionMismatch)
        },
    )?;
    nonnegative_quadratic_energy("kinetic", operators.mass(), state.velocity(), &momentum)
}

fn nonnegative_quadratic_energy(
    quantity: &'static str,
    operator: &RealSparseMatrix,
    state: &[f64],
    conjugate: &[f64],
) -> Result<f64, DynamicBalanceError> {
    if state.len() != conjugate.len() || operator.rows() != state.len() {
        return Err(DynamicBalanceError::DimensionMismatch);
    }
    let quadratic_dot = dot(state, conjugate)?;
    let value_j = measured(
        || Ok(OperationCount::from_operations(1)),
        || Ok(0.5 * quadratic_dot),
    )?;
    let absolute_sum_j = measured(
        || count(operator.values().len(), 4),
        || {
            let mut absolute_sum_j = 0.0;
            for (row, row_state) in state.iter().copied().enumerate() {
                for entry in operator.pattern().row_range(row) {
                    let column = operator.pattern().column(entry);
                    absolute_sum_j +=
                        0.5 * (row_state * operator.values()[entry] * state[column]).abs();
                }
            }
            Ok(absolute_sum_j)
        },
    )?;
    if !absolute_sum_j.is_finite() {
        return Err(DynamicBalanceError::NonFinite);
    }
    let roundoff_bound_j = measured(
        || Ok(OperationCount::from_operations(2)),
        || Ok(64.0 * f64::EPSILON * absolute_sum_j),
    )?;
    if value_j < -roundoff_bound_j {
        return Err(DynamicBalanceError::NegativeQuadraticEnergy { quantity, value_j });
    }
    Ok(value_j.max(0.0))
}

fn dot(left: &[f64], right: &[f64]) -> Result<f64, DynamicBalanceError> {
    if left.len() != right.len() {
        return Err(DynamicBalanceError::DimensionMismatch);
    }
    let value = measured(
        || {
            OperationCount::direct_real_dot(
                u64::try_from(left.len()).map_err(|_| LedgerError::CountOverflow)?,
            )
            .map_err(LedgerError::from)
        },
        || {
            Ok(left
                .iter()
                .zip(right)
                .map(|(left, right)| left * right)
                .sum::<f64>())
        },
    )?;
    value
        .is_finite()
        .then_some(value)
        .ok_or(DynamicBalanceError::NonFinite)
}

fn norm(values: &[f64]) -> Result<f64, DynamicBalanceError> {
    measured(
        || count(values.len(), 2),
        || Ok(values.iter().map(|value| value * value).sum::<f64>().sqrt()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{MatrixProperties, OperatorRole, RealSparseMatrix};
    use runmat_simulation_usage::UsageLedger;
    use std::sync::Arc;

    #[test]
    fn dynamic_balance_batches_refuse_before_effects_and_keep_typed_usage() {
        let operators = CanonicalRealOperators::try_new(
            RealSparseMatrix::diagonal(
                vec![10.0],
                OperatorRole::Stiffness,
                MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
            )
            .unwrap(),
            RealSparseMatrix::diagonal(
                vec![2.0],
                OperatorRole::Mass,
                MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
            )
            .unwrap(),
            RealSparseMatrix::diagonal(
                vec![0.5],
                OperatorRole::Damping,
                MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
            )
            .unwrap(),
        )
        .unwrap();
        let state = SecondOrderState::try_new(vec![0.2], vec![3.0], vec![-4.0]).unwrap();
        // Three one-entry CSR dots and three residual additions/subtractions.
        // Strain energy later adds its own CSR dot, quadratic dot, half scale,
        // absolute-entry reduction and roundoff bound: nine more operations.
        let ledger = Arc::new(
            UsageLedger::new(OperationCount::from_operations(15), OperationCount::ZERO).unwrap(),
        );
        UsageContext::new(Arc::clone(&ledger)).run_sync(|| {
            assert!(matches!(
                StructuralDynamicBalanceSnapshot::evaluate(&operators, &state, &[7.0]),
                Err(DynamicBalanceError::Usage(
                    LedgerError::MoreGrantRequired { .. }
                ))
            ));
            assert_eq!(
                ledger.snapshot().unwrap().completed.solve,
                OperationCount::ZERO
            );
            ledger
                .update_grant(OperationCount::from_operations(6))
                .unwrap();
            StructuralDynamicBalanceSnapshot::evaluate(&operators, &state, &[7.0]).unwrap();
            assert_eq!(
                ledger.snapshot().unwrap().completed.solve,
                OperationCount::from_operations(6)
            );
            assert!(matches!(
                strain_energy(&operators, &state),
                Err(DynamicBalanceError::Usage(
                    LedgerError::MoreGrantRequired { .. }
                ))
            ));
            ledger
                .update_grant(OperationCount::from_operations(15))
                .unwrap();
            strain_energy(&operators, &state).unwrap();
            assert_eq!(
                ledger.snapshot().unwrap().completed.solve,
                OperationCount::from_operations(15)
            );
        });
    }

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

    #[test]
    fn materially_negative_quadratic_energy_is_rejected() {
        let operator = RealSparseMatrix::diagonal(
            vec![-1.0],
            OperatorRole::Stiffness,
            MatrixProperties::GENERAL,
        )
        .unwrap();

        assert_eq!(
            nonnegative_quadratic_energy("strain", &operator, &[1.0], &[-1.0]),
            Err(DynamicBalanceError::NegativeQuadraticEnergy {
                quantity: "strain",
                value_j: -0.5,
            })
        );
    }
}
