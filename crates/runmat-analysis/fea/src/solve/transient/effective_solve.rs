use thiserror::Error;

use crate::{
    algebra::{
        spmv, AssembledRealSystem, CanonicalRealOperators, MatrixProperties, OperatorRole,
        RealSparseMatrix, RealSystemError,
    },
    assembly::AssemblySummary,
    constraints::{AffineConstraintReduction, ConstraintAlgebraError, ResolvedConstraintMatrix},
    contracts::StructuralDynamicBalanceSnapshot,
    solve::{
        backend::{cpu_reference::CpuReferenceBackend, kind::LinearAlgebraBackendKind},
        linear::{solve_linear_system, LinearSolveResult},
        preconditioner::SpdPreconditionerKind,
    },
};

use super::{
    balance::DynamicBalanceError,
    constraint_state::{
        canonicalize_homogeneous_solution, homogeneous_constraints, physical_load, project_vector,
        validate_constant_constraint_kinematics,
    },
    newmark::{prepare_average_acceleration_step, NewmarkError},
    state::{SecondOrderState, SecondOrderStateError},
};

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
}

struct NormalizedCorrectionEquation {
    summary: AssemblySummary,
    solution_scale: f64,
    expansion: Option<AffineConstraintReduction>,
}

#[derive(Clone, Debug, Error, PartialEq)]
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
    #[error("the effective structural dynamics solve returned an invalid state vector")]
    InvalidSolution,
    #[error("the physical equilibrium target must be finite and positive")]
    InvalidEquilibriumTarget,
    #[error(
        "the effective structural dynamics solve did not satisfy physical equilibrium after bounded refinement ({residual_norm} > {target})"
    )]
    EquilibriumRefinementFailed { residual_norm: f64, target: f64 },
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
pub(super) fn solve_initial_acceleration(
    summary: &AssemblySummary,
    displacement: Vec<f64>,
    velocity: Vec<f64>,
    backend: LinearAlgebraBackendKind,
    equilibrium_target: f64,
) -> Result<EffectiveDynamicsSolveResult, EffectiveDynamicsSolveError> {
    validate_equilibrium_target(equilibrium_target)?;
    let placeholder =
        SecondOrderState::try_new(displacement, velocity, vec![0.0; summary.dof_count])
            .map_err(EffectiveDynamicsSolveError::InvalidState)?;
    validate_constant_constraint_kinematics(summary, &placeholder, false)?;
    let physical_load = physical_load(summary);
    let damping_force = spmv(
        summary.operator.operators().damping(),
        placeholder.velocity(),
    )
    .map_err(|_| EffectiveDynamicsSolveError::InvalidSolution)?;
    let elastic_force = spmv(
        summary.operator.operators().stiffness(),
        placeholder.displacement(),
    )
    .map_err(|_| EffectiveDynamicsSolveError::InvalidSolution)?;
    let right_hand_side = physical_load
        .iter()
        .zip(damping_force)
        .zip(elastic_force)
        .map(|((load, damping), elastic)| load - damping - elastic)
        .collect::<Vec<_>>();
    let mass_operator = RealSparseMatrix::scaled_sum(
        &[(1.0, summary.operator.operators().mass())],
        OperatorRole::EffectiveTangent,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )
    .map_err(|_| EffectiveDynamicsSolveError::InvalidSolution)?;
    let effective_summary = effective_summary(summary, mass_operator.clone(), right_hand_side)?;
    let linear_solve = solve_effective_system(
        &effective_summary,
        backend,
        None,
        SpdPreconditionerKind::Jacobi,
    )?;
    let mut acceleration = linear_solve.solution.clone();
    canonicalize_homogeneous_solution(&effective_summary, &mut acceleration)?;
    EquilibriumRefinement {
        summary,
        physical_load: &physical_load,
        effective_operator: mass_operator,
        backend,
        target: equilibrium_target,
    }
    .finish(acceleration, vec![linear_solve], |acceleration| {
        SecondOrderState::try_new(
            placeholder.displacement().to_vec(),
            placeholder.velocity().to_vec(),
            acceleration.to_vec(),
        )
        .map_err(EffectiveDynamicsSolveError::InvalidState)
    })
}

/// Advances one average-acceleration Newmark step against the operators and load in `summary`.
pub(super) fn solve_average_acceleration_step(
    summary: &AssemblySummary,
    state: &SecondOrderState,
    time_step_s: f64,
    backend: LinearAlgebraBackendKind,
    equilibrium_target: f64,
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
    let effective_summary = effective_summary(
        summary,
        step.effective_operator().clone(),
        step.right_hand_side().to_vec(),
    )?;
    let linear_solve = solve_effective_system(
        &effective_summary,
        backend,
        Some(state.acceleration()),
        SpdPreconditionerKind::Jacobi,
    )?;
    let mut acceleration = linear_solve.solution.clone();
    canonicalize_homogeneous_solution(&effective_summary, &mut acceleration)?;
    EquilibriumRefinement {
        summary,
        physical_load: &physical_load,
        effective_operator: step.effective_operator().clone(),
        backend,
        target: equilibrium_target,
    }
    .finish(acceleration, vec![linear_solve], |acceleration| {
        step.complete(acceleration.to_vec())
            .map_err(EffectiveDynamicsSolveError::Newmark)
    })
}

impl EquilibriumRefinement<'_> {
    fn finish(
        &self,
        mut acceleration: Vec<f64>,
        mut linear_solves: Vec<LinearSolveResult>,
        build_state: impl Fn(&[f64]) -> Result<SecondOrderState, EffectiveDynamicsSolveError>,
    ) -> Result<EffectiveDynamicsSolveResult, EffectiveDynamicsSolveError> {
        for refinement_count in 0..=MAXIMUM_EQUILIBRIUM_REFINEMENTS {
            let state = build_state(&acceleration)?;
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
            let correction_equation = normalize_correction_equation(
                self.summary,
                &self.effective_operator,
                &full_balance.imbalance(),
            )?;
            let correction = solve_effective_system(
                &correction_equation.summary,
                self.backend,
                None,
                // The correction operator is SPD. ILU0 does not guarantee an SPD
                // preconditioner and can break CG even when the operator is valid.
                SpdPreconditionerKind::Jacobi,
            )?;
            let mut correction_values = match correction_equation.expansion.as_ref() {
                Some(reduction) => reduction
                    .expand_linear(&correction.solution)
                    .map_err(EffectiveDynamicsSolveError::InvalidConstraintState)?,
                None => correction.solution.clone(),
            };
            canonicalize_homogeneous_solution(self.summary, &mut correction_values)?;
            add_scaled_correction(
                &mut acceleration,
                &correction_values,
                correction_equation.solution_scale,
            )?;
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
    Ok((
        right_hand_side.iter().map(|value| value / scale).collect(),
        scale,
    ))
}

fn normalize_correction_equation(
    summary: &AssemblySummary,
    operator: &RealSparseMatrix,
    right_hand_side: &[f64],
) -> Result<NormalizedCorrectionEquation, EffectiveDynamicsSolveError> {
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
        });
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
        Some(reduction) => reduction
            .reduce_sparse_system(operator, &constrained_rhs)
            .map_err(EffectiveDynamicsSolveError::InvalidConstraintState)?,
        None => (operator.clone(), constrained_rhs),
    };
    let (operator_scale, solution_scale) = active_correction_scales(
        &active_operator,
        &active_rhs,
        &vec![false; active_operator.rows()],
        None,
    )?;
    let normalized_operator = RealSparseMatrix::scaled_sum(
        &[(operator_scale.recip(), &active_operator)],
        OperatorRole::EffectiveTangent,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )
    .map_err(|_| EffectiveDynamicsSolveError::InvalidSolution)?;
    let normalized_rhs = active_rhs
        .iter()
        .map(|value| value / operator_scale / solution_scale)
        .collect::<Vec<_>>();
    let normalized_summary = if let Some(reduction) = reduction.as_ref() {
        reduced_effective_summary(summary, reduction, normalized_operator, normalized_rhs)?
    } else {
        effective_summary(summary, normalized_operator, normalized_rhs)?
    };
    Ok(NormalizedCorrectionEquation {
        summary: normalized_summary,
        solution_scale,
        expansion: reduction,
    })
}

fn reduced_effective_summary(
    source: &AssemblySummary,
    reduction: &AffineConstraintReduction,
    effective_operator: RealSparseMatrix,
    right_hand_side: Vec<f64>,
) -> Result<AssemblySummary, EffectiveDynamicsSolveError> {
    let mass = reduction
        .reduce_sparse_operator(source.operator.operators().mass())
        .map_err(EffectiveDynamicsSolveError::InvalidConstraintState)?;
    let damping = reduction
        .reduce_sparse_operator(source.operator.operators().damping())
        .map_err(EffectiveDynamicsSolveError::InvalidConstraintState)?;
    let operators = CanonicalRealOperators::try_new(effective_operator, mass, damping)
        .map_err(EffectiveDynamicsSolveError::InvalidEffectiveSystem)?;
    let operator = AssembledRealSystem::try_new(
        operators,
        vec![false; reduction.reduced_dof_count],
        right_hand_side,
    )
    .map_err(EffectiveDynamicsSolveError::InvalidEffectiveSystem)?;
    let mut reduced = source.clone();
    reduced.dof_count = reduction.reduced_dof_count;
    reduced.operator = operator;
    reduced.constraint_matrix = None;
    reduced.affine_constraint_reduction = None;
    Ok(reduced)
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
        let (reduced_operator, reduced_rhs) = reduction
            .reduce_sparse_system(operator, right_hand_side)
            .map_err(EffectiveDynamicsSolveError::InvalidConstraintState)?;
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
    let operator_scaled_rhs = active_rhs
        .iter()
        .map(|value| value / operator_scale)
        .collect::<Vec<_>>();
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
    for (value, correction) in values.iter_mut().zip(correction) {
        *value += scale * correction;
        if !value.is_finite() {
            return Err(EffectiveDynamicsSolveError::InvalidSolution);
        }
    }
    Ok(())
}

fn solve_effective_system(
    summary: &AssemblySummary,
    backend: LinearAlgebraBackendKind,
    initial_guess: Option<&[f64]>,
    preconditioner: SpdPreconditionerKind,
) -> Result<LinearSolveResult, EffectiveDynamicsSolveError> {
    let result = solve_linear_system(
        summary,
        preconditioner,
        backend,
        &CpuReferenceBackend,
        initial_guess,
    );
    if !result.converged {
        return Err(EffectiveDynamicsSolveError::LinearSolveFailed {
            residual_norm: result.residual_norm,
        });
    }
    if result.solution.len() != summary.dof_count
        || result.solution.iter().any(|value| !value.is_finite())
    {
        return Err(EffectiveDynamicsSolveError::InvalidSolution);
    }
    Ok(result)
}

fn effective_summary(
    source: &AssemblySummary,
    effective_operator: RealSparseMatrix,
    right_hand_side: Vec<f64>,
) -> Result<AssemblySummary, EffectiveDynamicsSolveError> {
    let (constraint_matrix, affine_constraint_reduction) = homogeneous_constraints(source)?;
    let mut constrained_load = right_hand_side;
    for (dof, constrained) in source
        .operator
        .constrained_dofs()
        .iter()
        .copied()
        .enumerate()
    {
        if constrained {
            constrained_load[dof] = 0.0;
        }
    }
    let operators = CanonicalRealOperators::try_new(
        effective_operator,
        source.operator.operators().mass().clone(),
        source.operator.operators().damping().clone(),
    )
    .map_err(EffectiveDynamicsSolveError::InvalidEffectiveSystem)?;
    let operator = AssembledRealSystem::try_new(
        operators,
        source.operator.constrained_dofs().to_vec(),
        constrained_load,
    )
    .map_err(EffectiveDynamicsSolveError::InvalidEffectiveSystem)?;
    let mut effective = source.clone();
    effective.operator = operator;
    effective.constraint_matrix = constraint_matrix;
    effective.affine_constraint_reduction = affine_constraint_reduction;
    Ok(effective)
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
    let absolute_norm = norm(&imbalance);
    let force_scale = norm(&projected_load)
        .max(norm(&projected_inertia) + norm(&projected_damping) + norm(&projected_elastic))
        .max(1.0);
    if imbalance.iter().any(|value| !value.is_finite())
        || !absolute_norm.is_finite()
        || !force_scale.is_finite()
    {
        return Err(EffectiveDynamicsSolveError::InvalidSolution);
    }
    Ok(ProjectedDynamicResidual {
        relative_norm: absolute_norm / force_scale,
    })
}

fn norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests;
