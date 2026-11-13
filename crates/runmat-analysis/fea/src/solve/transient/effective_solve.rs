use runmat_simulation_usage::{LedgerError, OperationCount, UsageCategory, UsageContext};
use thiserror::Error;

use crate::{
    algebra::{
        spmv, LinearPreparationReuse, MatrixProperties, OperatorRole, RealSparseMatrix,
        RealSystemError, ScopedSpmvError,
    },
    assembly::AssemblySummary,
    constraints::{AffineConstraintReduction, ConstraintAlgebraError, ResolvedConstraintMatrix},
    contracts::StructuralDynamicBalanceSnapshot,
    solve::{
        affine::{affine_rhs_work, affine_scatter_work},
        backend::{cpu_reference::CpuReferenceBackend, kind::LinearAlgebraBackendKind},
        linear::{solve_structural_system_reusing, LinearSolveResult},
        preconditioner::SpdPreconditionerKind,
    },
};

use super::{
    balance::DynamicBalanceError,
    constraint_state::{
        canonicalize_homogeneous_solution, expand_homogeneous_solution, homogeneous_constraints,
        physical_load, project_vector, validate_constant_constraint_kinematics,
    },
    newmark::{count, prepare_average_acceleration_step, spmv_work, NewmarkError},
    state::{SecondOrderState, SecondOrderStateError},
};

mod system;
use system::{effective_system, reduced_effective_system, EffectiveLinearSystem};

const MAXIMUM_EQUILIBRIUM_REFINEMENTS: usize = 4;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct ProjectedDynamicResidual {
    relative_norm: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct EffectiveDynamicsSolveResult {
    state: SecondOrderState,
    linear_solves: Vec<LinearSolveResult>,
    projected_residual: ProjectedDynamicResidual,
}

struct EquilibriumRefinement<'a> {
    summary: &'a AssemblySummary,
    physical_load: &'a [f64],
    effective_operator: RealSparseMatrix,
    backend: LinearAlgebraBackendKind,
    target: f64,
    preparation: &'a mut LinearPreparationReuse,
}

struct NormalizedCorrectionEquation<'a> {
    summary: EffectiveLinearSystem<'a>,
    solution_scale: f64,
    expansion: Option<std::sync::Arc<AffineConstraintReduction>>,
}

#[derive(Debug, Error)]
pub(super) enum EffectiveDynamicsSolveError {
    #[error("the supplied structural dynamics state is invalid: {0}")]
    InvalidState(SecondOrderStateError),
    #[error(
        "the structural dynamics {quantity} state violates its constant kinematic constraints by {maximum_violation}"
    )]
    InconsistentConstraintKinematics {
        quantity: &'static str,
        maximum_violation: f64,
    },
    #[error("the canonical structural constraint state is inconsistent: {0}")]
    InvalidConstraintState(ConstraintAlgebraError),
    #[error("the effective structural dynamics system is invalid: {0}")]
    InvalidEffectiveSystem(RealSystemError),
    #[error("the average-acceleration Newmark step is invalid: {0}")]
    Newmark(NewmarkError),
    #[error("the structural dynamic balance is invalid: {0}")]
    DynamicBalance(DynamicBalanceError),
    #[error("the effective structural dynamics solve did not converge (residual={residual_norm})")]
    LinearSolveFailed { residual_norm: f64 },
    #[error("the effective structural dynamics sparse solve failed: {0}")]
    SparseSolve(ScopedSpmvError),
    #[error("simulation usage admission failed: {0}")]
    Usage(#[from] LedgerError),
    #[error("the effective structural dynamics solve returned an invalid state vector")]
    InvalidSolution,
    #[error("the physical equilibrium target must be finite and positive")]
    InvalidEquilibriumTarget,
    #[error(
        "the effective structural dynamics solve did not satisfy physical equilibrium after bounded refinement ({residual_norm} > {target})"
    )]
    EquilibriumRefinementFailed { residual_norm: f64, target: f64 },
}

pub(super) fn measured<R>(
    quote: impl FnOnce() -> Result<OperationCount, LedgerError>,
    run: impl FnOnce() -> Result<R, EffectiveDynamicsSolveError>,
) -> Result<R, EffectiveDynamicsSolveError> {
    let Some(context) = UsageContext::capture() else {
        return run();
    };
    drop(context);
    crate::algebra::capacity::measured_sync(UsageCategory::Solve, quote()?, run)
}

pub(super) fn measured_affine<R>(
    quote: impl FnOnce() -> Result<OperationCount, ScopedSpmvError>,
    run: impl FnOnce() -> Result<R, EffectiveDynamicsSolveError>,
) -> Result<R, EffectiveDynamicsSolveError> {
    let Some(context) = UsageContext::capture() else {
        return run();
    };
    drop(context);
    let work = quote().map_err(EffectiveDynamicsSolveError::SparseSolve)?;
    crate::algebra::capacity::measured_sync(UsageCategory::Solve, work, run)
}

fn reduce_sparse_system_measured(
    reduction: &AffineConstraintReduction,
    operator: &RealSparseMatrix,
    right_hand_side: &[f64],
) -> Result<(RealSparseMatrix, Vec<f64>), EffectiveDynamicsSolveError> {
    measured_affine(
        || {
            affine_rhs_work(reduction, operator)?
                .checked_add(affine_scatter_work(reduction, operator)?)
                .map_err(|error| ScopedSpmvError::Usage(error.into()))
        },
        || {
            reduction
                .reduce_sparse_system(operator, right_hand_side)
                .map_err(EffectiveDynamicsSolveError::InvalidConstraintState)
        },
    )
}

impl EffectiveDynamicsSolveResult {
    pub(super) fn state(&self) -> &SecondOrderState {
        &self.state
    }

    pub(super) fn linear_solves(&self) -> impl Iterator<Item = &LinearSolveResult> {
        self.linear_solves.iter()
    }

    pub(super) fn projected_residual(&self) -> &ProjectedDynamicResidual {
        &self.projected_residual
    }
}

impl ProjectedDynamicResidual {
    pub(super) fn relative_norm(&self) -> f64 {
        self.relative_norm
    }
}

/// Solves the physical equilibrium equation at the initial instant,
/// `M a_0 = f_0 - C v_0 - K u_0`.
///
/// Prescribed and relation constraints are homogenized for acceleration. The authored
/// displacement offset remains present in `u_0`; it is never interpreted as a force.
pub(super) fn solve_initial_acceleration_reusing(
    summary: &AssemblySummary,
    displacement: Vec<f64>,
    velocity: Vec<f64>,
    backend: LinearAlgebraBackendKind,
    equilibrium_target: f64,
    preparation: &mut LinearPreparationReuse,
) -> Result<EffectiveDynamicsSolveResult, EffectiveDynamicsSolveError> {
    validate_equilibrium_target(equilibrium_target)?;
    let placeholder =
        SecondOrderState::try_new(displacement, velocity, vec![0.0; summary.dof_count])
            .map_err(EffectiveDynamicsSolveError::InvalidState)?;
    validate_constant_constraint_kinematics(summary, &placeholder, false)?;
    let physical_load = physical_load(summary);
    let damping_force = measured(
        || spmv_work(summary.operator.operators().damping()),
        || {
            spmv(
                summary.operator.operators().damping(),
                placeholder.velocity(),
            )
            .map_err(|_| EffectiveDynamicsSolveError::InvalidSolution)
        },
    )?;
    let elastic_force = measured(
        || spmv_work(summary.operator.operators().stiffness()),
        || {
            spmv(
                summary.operator.operators().stiffness(),
                placeholder.displacement(),
            )
            .map_err(|_| EffectiveDynamicsSolveError::InvalidSolution)
        },
    )?;
    let right_hand_side = measured(
        || count(physical_load.len(), 2),
        || {
            Ok(physical_load
                .iter()
                .zip(damping_force)
                .zip(elastic_force)
                .map(|((load, damping), elastic)| load - damping - elastic)
                .collect::<Vec<_>>())
        },
    )?;
    let mass_operator = measured(
        || count(summary.operator.operators().mass().values().len(), 1),
        || {
            RealSparseMatrix::scaled_sum(
                &[(1.0, summary.operator.operators().mass())],
                OperatorRole::EffectiveTangent,
                MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
            )
            .map_err(|_| EffectiveDynamicsSolveError::InvalidSolution)
        },
    )?;
    let effective_summary = effective_system(summary, mass_operator.clone(), right_hand_side)?;
    let linear_solve = solve_effective_system(
        &effective_summary,
        backend,
        None,
        SpdPreconditionerKind::Auto,
        preparation,
    )?;
    let mut acceleration = linear_solve.solution.clone();
    canonicalize_homogeneous_solution(&effective_summary.view(), &mut acceleration)?;
    let state = SecondOrderState::try_new(
        placeholder.displacement().to_vec(),
        placeholder.velocity().to_vec(),
        acceleration,
    )
    .map_err(EffectiveDynamicsSolveError::InvalidState)?;
    EquilibriumRefinement {
        summary,
        physical_load: &physical_load,
        effective_operator: mass_operator,
        backend,
        target: equilibrium_target,
        preparation,
    }
    .finish(state, vec![linear_solve], |state, correction| {
        let (displacement, velocity, mut acceleration) = state.into_parts();
        add_scaled_correction(&mut acceleration, correction, 1.0)?;
        SecondOrderState::try_new(displacement, velocity, acceleration)
            .map_err(EffectiveDynamicsSolveError::InvalidState)
    })
}

/// Advances one average-acceleration Newmark step against the operators and load in `summary`.
pub(super) fn solve_average_acceleration_step_reusing(
    summary: &AssemblySummary,
    state: &SecondOrderState,
    time_step_s: f64,
    backend: LinearAlgebraBackendKind,
    equilibrium_target: f64,
    preparation: &mut LinearPreparationReuse,
) -> Result<EffectiveDynamicsSolveResult, EffectiveDynamicsSolveError> {
    validate_equilibrium_target(equilibrium_target)?;
    validate_constant_constraint_kinematics(summary, state, true)?;
    let physical_load = physical_load(summary);
    let step = prepare_average_acceleration_step(
        summary.operator.operators(),
        state,
        time_step_s,
        &physical_load,
    )
    .map_err(EffectiveDynamicsSolveError::Newmark)?;
    let effective_summary = effective_system(
        summary,
        step.effective_operator().clone(),
        step.right_hand_side().to_vec(),
    )?;
    let linear_solve = solve_effective_system(
        &effective_summary,
        backend,
        Some(state.acceleration()),
        SpdPreconditionerKind::Auto,
        preparation,
    )?;
    let mut acceleration = linear_solve.solution.clone();
    canonicalize_homogeneous_solution(&effective_summary.view(), &mut acceleration)?;
    let state = step
        .complete(acceleration)
        .map_err(EffectiveDynamicsSolveError::Newmark)?;
    EquilibriumRefinement {
        summary,
        physical_load: &physical_load,
        effective_operator: step.effective_operator().clone(),
        backend,
        target: equilibrium_target,
        preparation,
    }
    .finish(state, vec![linear_solve], |state, correction| {
        step.correct(state, correction)
            .map_err(EffectiveDynamicsSolveError::Newmark)
    })
}

impl EquilibriumRefinement<'_> {
    fn finish(
        self,
        mut state: SecondOrderState,
        mut linear_solves: Vec<LinearSolveResult>,
        correct_state: impl Fn(
            SecondOrderState,
            &[f64],
        ) -> Result<SecondOrderState, EffectiveDynamicsSolveError>,
    ) -> Result<EffectiveDynamicsSolveResult, EffectiveDynamicsSolveError> {
        for refinement_count in 0..=MAXIMUM_EQUILIBRIUM_REFINEMENTS {
            validate_constant_constraint_kinematics(self.summary, &state, true)?;
            let full_balance = StructuralDynamicBalanceSnapshot::evaluate(
                self.summary.operator.operators(),
                &state,
                self.physical_load,
            )
            .map_err(EffectiveDynamicsSolveError::DynamicBalance)?;
            let projected_residual = project_residual(self.summary, &full_balance)?;
            if projected_residual.relative_norm <= self.target {
                return Ok(EffectiveDynamicsSolveResult {
                    state,
                    linear_solves,
                    projected_residual,
                });
            }
            if refinement_count == MAXIMUM_EQUILIBRIUM_REFINEMENTS {
                return Err(EffectiveDynamicsSolveError::EquilibriumRefinementFailed {
                    residual_norm: projected_residual.relative_norm,
                    target: self.target,
                });
            }

            // The physical imbalance is the right-hand side for an acceleration correction: changing
            // acceleration by da changes displacement by beta*dt^2*da and velocity by gamma*dt*da,
            // exactly the terms already represented by the effective operator.
            let imbalance = measured(
                || count(full_balance.equilibrium_residual.len(), 1),
                || Ok(full_balance.imbalance()),
            )?;
            let correction_equation =
                normalize_correction_equation(self.summary, &self.effective_operator, &imbalance)?;
            let correction = solve_effective_system(
                &correction_equation.summary,
                self.backend,
                None,
                // The correction operator is SPD. ILU0 does not guarantee an SPD
                // preconditioner and can break CG even when the operator is valid.
                SpdPreconditionerKind::Auto,
                self.preparation,
            )?;
            let mut correction_values = match correction_equation.expansion.as_ref() {
                Some(reduction) => expand_homogeneous_solution(reduction, &correction.solution)?,
                None => correction.solution.clone(),
            };
            canonicalize_homogeneous_solution(&self.summary.into(), &mut correction_values)?;
            let scale = correction_equation.solution_scale;
            if !scale.is_finite() || scale <= 0.0 {
                return Err(EffectiveDynamicsSolveError::InvalidSolution);
            }
            let correction_count = correction_values.len();
            measured(
                || count(correction_count, 1),
                || {
                    for value in &mut correction_values {
                        *value *= scale;
                        if !value.is_finite() {
                            return Err(EffectiveDynamicsSolveError::InvalidSolution);
                        }
                    }
                    Ok(())
                },
            )?;
            state = correct_state(state, &correction_values)?;
            linear_solves.push(correction);
        }
        unreachable!("the bounded equilibrium-refinement loop always returns")
    }
}

fn validate_equilibrium_target(target: f64) -> Result<(), EffectiveDynamicsSolveError> {
    if !target.is_finite() || target <= 0.0 {
        return Err(EffectiveDynamicsSolveError::InvalidEquilibriumTarget);
    }
    Ok(())
}

fn normalize_correction_rhs(
    right_hand_side: &[f64],
) -> Result<(Vec<f64>, f64), EffectiveDynamicsSolveError> {
    let scale = right_hand_side
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if !scale.is_finite() || scale <= 0.0 {
        return Err(EffectiveDynamicsSolveError::InvalidSolution);
    }
    measured(
        || count(right_hand_side.len(), 1),
        || {
            Ok((
                right_hand_side.iter().map(|value| value / scale).collect(),
                scale,
            ))
        },
    )
}

fn normalize_correction_equation<'a>(
    summary: &'a AssemblySummary,
    operator: &RealSparseMatrix,
    right_hand_side: &[f64],
) -> Result<NormalizedCorrectionEquation<'a>, EffectiveDynamicsSolveError> {
    let (homogeneous_matrix, _) = homogeneous_constraints(summary)?;
    let reduction = homogeneous_matrix
        .as_ref()
        .map(ResolvedConstraintMatrix::affine_reduction)
        .transpose()
        .map_err(EffectiveDynamicsSolveError::InvalidConstraintState)?
        .or_else(|| {
            summary
                .operator
                .constrained_dofs()
                .iter()
                .any(|value| *value)
                .then(|| {
                    AffineConstraintReduction::from_constrained_dofs(
                        summary.operator.constrained_dofs(),
                    )
                })
        })
        .map(std::sync::Arc::new);
    let mut constrained_rhs = right_hand_side.to_vec();
    for (value, constrained) in constrained_rhs
        .iter_mut()
        .zip(summary.operator.constrained_dofs())
    {
        if *constrained {
            *value = 0.0;
        }
    }
    let (active_operator, active_rhs) = match reduction.as_ref() {
        Some(reduction) => reduce_sparse_system_measured(reduction, operator, &constrained_rhs)?,
        None => (operator.clone(), constrained_rhs),
    };
    let (operator_scale, solution_scale) = active_correction_scales(
        &active_operator,
        &active_rhs,
        &vec![false; active_operator.rows()],
        None,
    )?;
    let normalized_operator = measured(
        || {
            count(active_operator.values().len(), 1)?
                .checked_add(OperationCount::from_operations(1))
                .map_err(LedgerError::from)
        },
        || {
            RealSparseMatrix::scaled_sum(
                &[(operator_scale.recip(), &active_operator)],
                OperatorRole::EffectiveTangent,
                MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
            )
            .map_err(|_| EffectiveDynamicsSolveError::InvalidSolution)
        },
    )?;
    let normalized_rhs = measured(
        || count(active_rhs.len(), 2),
        || {
            Ok(active_rhs
                .iter()
                .map(|value| value / operator_scale / solution_scale)
                .collect::<Vec<_>>())
        },
    )?;
    let normalized_summary = if let Some(reduction) = reduction.as_ref() {
        reduced_effective_system(
            summary,
            std::sync::Arc::clone(reduction),
            normalized_operator,
            normalized_rhs,
        )?
    } else {
        effective_system(summary, normalized_operator, normalized_rhs)?
    };
    Ok(NormalizedCorrectionEquation {
        summary: normalized_summary,
        solution_scale,
        expansion: reduction,
    })
}

fn active_correction_scales(
    operator: &RealSparseMatrix,
    right_hand_side: &[f64],
    directly_constrained: &[bool],
    reduction: Option<&crate::constraints::AffineConstraintReduction>,
) -> Result<(f64, f64), EffectiveDynamicsSolveError> {
    if right_hand_side.len() != operator.rows() || directly_constrained.len() != operator.rows() {
        return Err(EffectiveDynamicsSolveError::InvalidSolution);
    }
    let (operator_scale, active_rhs) = if let Some(reduction) = reduction {
        let (reduced_operator, reduced_rhs) =
            reduce_sparse_system_measured(reduction, operator, right_hand_side)?;
        (maximum_absolute(reduced_operator.values()), reduced_rhs)
    } else {
        let mut operator_scale = 0.0_f64;
        for row in 0..operator.rows() {
            if directly_constrained[row] {
                continue;
            }
            for entry in operator.pattern().row_range(row) {
                let column = operator.pattern().column(entry);
                if !directly_constrained[column] {
                    operator_scale = operator_scale.max(operator.values()[entry].abs());
                }
            }
        }
        let active_rhs = right_hand_side
            .iter()
            .zip(directly_constrained)
            .filter_map(|(value, constrained)| (!constrained).then_some(*value))
            .collect();
        (operator_scale, active_rhs)
    };
    if !operator_scale.is_finite() || operator_scale <= 0.0 {
        return Err(EffectiveDynamicsSolveError::InvalidSolution);
    }
    let operator_scaled_rhs = measured(
        || count(active_rhs.len(), 1),
        || {
            Ok(active_rhs
                .iter()
                .map(|value| value / operator_scale)
                .collect::<Vec<_>>())
        },
    )?;
    let (_, solution_scale) = normalize_correction_rhs(&operator_scaled_rhs)?;
    Ok((operator_scale, solution_scale))
}

fn maximum_absolute(values: &[f64]) -> f64 {
    values
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
}

fn add_scaled_correction(
    values: &mut [f64],
    correction: &[f64],
    scale: f64,
) -> Result<(), EffectiveDynamicsSolveError> {
    if values.len() != correction.len() || !scale.is_finite() || scale <= 0.0 {
        return Err(EffectiveDynamicsSolveError::InvalidSolution);
    }
    let value_count = values.len();
    measured(
        || count(value_count, 2),
        || {
            for (value, correction) in values.iter_mut().zip(correction) {
                *value += scale * correction;
                if !value.is_finite() {
                    return Err(EffectiveDynamicsSolveError::InvalidSolution);
                }
            }
            Ok(())
        },
    )
}

fn solve_effective_system(
    summary: &EffectiveLinearSystem<'_>,
    backend: LinearAlgebraBackendKind,
    initial_guess: Option<&[f64]>,
    preconditioner: SpdPreconditionerKind,
    preparation: &mut LinearPreparationReuse,
) -> Result<LinearSolveResult, EffectiveDynamicsSolveError> {
    let result = solve_structural_system_reusing(
        &summary.view(),
        preconditioner,
        backend,
        &CpuReferenceBackend,
        initial_guess,
        preparation,
    )
    .map_err(EffectiveDynamicsSolveError::SparseSolve)?;
    if !result.converged {
        return Err(EffectiveDynamicsSolveError::LinearSolveFailed {
            residual_norm: result.residual_norm,
        });
    }
    if result.solution.len() != summary.view().dof_count
        || result.solution.iter().any(|value| !value.is_finite())
    {
        return Err(EffectiveDynamicsSolveError::InvalidSolution);
    }
    Ok(result)
}

fn project_residual(
    summary: &AssemblySummary,
    balance: &StructuralDynamicBalanceSnapshot,
) -> Result<ProjectedDynamicResidual, EffectiveDynamicsSolveError> {
    let imbalance = project_vector(summary, &balance.equilibrium_residual)?;
    let projected_load = project_vector(summary, &balance.applied_load)?;
    let projected_inertia = project_vector(summary, &balance.inertia_force)?;
    let projected_damping = project_vector(summary, &balance.damping_force)?;
    let projected_elastic = project_vector(summary, &balance.elastic_force)?;
    let absolute_norm = measured(|| count(imbalance.len(), 2), || Ok(norm(&imbalance)))?;
    let load_norm = measured(
        || count(projected_load.len(), 2),
        || Ok(norm(&projected_load)),
    )?;
    let inertia_norm = measured(
        || count(projected_inertia.len(), 2),
        || Ok(norm(&projected_inertia)),
    )?;
    let damping_norm = measured(
        || count(projected_damping.len(), 2),
        || Ok(norm(&projected_damping)),
    )?;
    let elastic_norm = measured(
        || count(projected_elastic.len(), 2),
        || Ok(norm(&projected_elastic)),
    )?;
    let force_scale = measured(
        || Ok(OperationCount::from_operations(2)),
        || Ok(load_norm.max(inertia_norm + damping_norm + elastic_norm)),
    )?;
    if imbalance.iter().any(|value| !value.is_finite())
        || !absolute_norm.is_finite()
        || !force_scale.is_finite()
    {
        return Err(EffectiveDynamicsSolveError::InvalidSolution);
    }
    if force_scale == 0.0 {
        return if absolute_norm == 0.0 {
            Ok(ProjectedDynamicResidual { relative_norm: 0.0 })
        } else {
            Err(EffectiveDynamicsSolveError::InvalidSolution)
        };
    }
    measured(
        || Ok(OperationCount::from_operations(1)),
        || {
            Ok(ProjectedDynamicResidual {
                relative_norm: absolute_norm / force_scale,
            })
        },
    )
}

fn norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(super) fn solve_initial_acceleration(
    summary: &AssemblySummary,
    displacement: Vec<f64>,
    velocity: Vec<f64>,
    backend: LinearAlgebraBackendKind,
    equilibrium_target: f64,
) -> Result<EffectiveDynamicsSolveResult, EffectiveDynamicsSolveError> {
    solve_initial_acceleration_reusing(
        summary,
        displacement,
        velocity,
        backend,
        equilibrium_target,
        &mut LinearPreparationReuse::default(),
    )
}
