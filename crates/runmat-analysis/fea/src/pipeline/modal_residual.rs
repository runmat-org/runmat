use runmat_analysis_core::{CoordinateFrame, SpectrumDirection};
use serde::{Deserialize, Serialize};

use crate::{
    algebra::{
        apply_k, apply_m, AssembledRealSystem, CanonicalRealOperators, MatrixProperties,
        OperatorRole, RealSparseMatrix,
    },
    assembly::AssemblySummary,
    constraints::AffineConstraintReduction,
    contracts::FeaRunError,
    solve::{
        preconditioner::{build_spd_preconditioner, SpdPreconditionerKind},
        spd_linear::{solve_spd_system_cg, SpdCgSolveOptions},
    },
};

use super::modal_basis::{directional_influence_vector, ModalBasisMode};

const STATIC_TOLERANCE: f64 = 1.0e-10;
pub(crate) const RESIDUAL_MASS_ORTHOGONALIZATION_PASSES: usize = 2;

pub(crate) fn residual_static_iteration_limit(dimension: usize) -> usize {
    dimension.saturating_mul(20).max(1_000)
}

pub(crate) fn canonical_residual_vector_directions(
    directions: &[SpectrumDirection],
) -> Result<Vec<[f64; 3]>, String> {
    if directions.len() > 3 {
        return Err("modal extraction supports at most three residual-vector directions".into());
    }
    if directions
        .iter()
        .any(|direction| direction.frame != CoordinateFrame::Global)
    {
        return Err("modal residual vectors currently require global directions".into());
    }
    canonical_directions(
        &directions
            .iter()
            .map(|direction| direction.vector)
            .collect::<Vec<_>>(),
    )
    .map_err(|error| error.to_string())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModalResidualVector {
    direction: [f64; 3],
    frequency_hz: f64,
    modal_mass: f64,
    modal_stiffness: f64,
    participation_factor: f64,
    effective_mass: f64,
    static_equilibrium_residual_ratio: f64,
    maximum_mass_orthogonality_ratio: f64,
    dof_shape: Vec<f64>,
}

impl ModalResidualVector {
    pub const fn direction(&self) -> [f64; 3] {
        self.direction
    }
    pub const fn frequency_hz(&self) -> f64 {
        self.frequency_hz
    }
    pub const fn modal_mass(&self) -> f64 {
        self.modal_mass
    }
    pub const fn modal_stiffness(&self) -> f64 {
        self.modal_stiffness
    }
    pub const fn participation_factor(&self) -> f64 {
        self.participation_factor
    }
    pub const fn effective_mass(&self) -> f64 {
        self.effective_mass
    }
    pub const fn static_equilibrium_residual_ratio(&self) -> f64 {
        self.static_equilibrium_residual_ratio
    }
    pub const fn maximum_mass_orthogonality_ratio(&self) -> f64 {
        self.maximum_mass_orthogonality_ratio
    }
    pub fn dof_shape(&self) -> &[f64] {
        &self.dof_shape
    }
}

pub(super) fn build_directional_residual_vectors(
    summary: &AssemblySummary,
    modes: &[ModalBasisMode],
    directions: &[[f64; 3]],
) -> Result<Vec<ModalResidualVector>, FeaRunError> {
    let directions = canonical_directions(directions)?;
    let reduction = homogeneous_reduction(summary)?;
    let stiffness = reduction
        .reduce_sparse_operator(summary.operator.operators().stiffness())
        .map_err(|error| FeaRunError::Solve(error.to_string()))?;
    let dimension = stiffness.rows();
    let system = AssembledRealSystem::try_new(
        CanonicalRealOperators::try_new(
            stiffness,
            RealSparseMatrix::diagonal(
                vec![1.0; dimension],
                OperatorRole::Mass,
                MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
            )
            .map_err(|error| FeaRunError::Solve(error.to_string()))?,
            RealSparseMatrix::diagonal(
                vec![0.0; dimension],
                OperatorRole::Damping,
                MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
            )
            .map_err(|error| FeaRunError::Solve(error.to_string()))?,
        )
        .map_err(|error| FeaRunError::Solve(error.to_string()))?,
        vec![false; dimension],
        vec![0.0; dimension],
    )
    .map_err(|error| FeaRunError::Solve(error.to_string()))?;
    let prepared = system
        .prepare_linear()
        .map_err(|error| FeaRunError::Solve(error.to_string()))?;
    let preconditioner = build_spd_preconditioner(&prepared, SpdPreconditionerKind::Ilu0)
        .or_else(|_| build_spd_preconditioner(&prepared, SpdPreconditionerKind::Jacobi))
        .map_err(|error| FeaRunError::Solve(error.to_string()))?;

    let mut residuals: Vec<ModalResidualVector> = Vec::with_capacity(directions.len());
    for direction in directions {
        let influence = directional_influence_vector(summary, direction);
        let load = apply_m(&summary.operator, &influence);
        let reduced_load = reduction
            .transpose_apply(&load)
            .map_err(|error| FeaRunError::Solve(error.to_string()))?;
        if norm(&reduced_load) <= f64::EPSILON {
            return Err(FeaRunError::Solve(
                "residual-vector direction has no free structural mass".into(),
            ));
        }
        let attempt = solve_spd_system_cg(
            &system,
            &reduced_load,
            SpdCgSolveOptions {
                max_iters: residual_static_iteration_limit(dimension),
                tol: STATIC_TOLERANCE,
                prepared_accelerator_system: None,
                cpu_preconditioner: Some(preconditioner.as_ref()),
                initial_guess: None,
            },
        );
        if !attempt.converged {
            return Err(FeaRunError::Solve(
                "residual-vector static solve did not converge".into(),
            ));
        }
        let equilibrium_residual = subtract(&apply_k(&system, &attempt.vector), &reduced_load)?;
        let equilibrium_ratio =
            norm(&equilibrium_residual) / norm(&reduced_load).max(f64::MIN_POSITIVE);
        let static_shape = reduction
            .expand_linear(&attempt.vector)
            .map_err(|error| FeaRunError::Solve(error.to_string()))?;
        let mut shape = static_shape;
        for mode in modes {
            let coefficient = dot(mode.dof_shape(), &load) / mode.modal_stiffness();
            subtract_scaled(&mut shape, mode.dof_shape(), coefficient)?;
        }
        for _ in 0..RESIDUAL_MASS_ORTHOGONALIZATION_PASSES {
            mass_orthogonalize(summary, &mut shape, modes, &residuals)?;
        }
        let mass_applied = apply_m(&summary.operator, &shape);
        let modal_mass = dot(&shape, &mass_applied);
        if !modal_mass.is_finite() || modal_mass <= f64::EPSILON {
            return Err(FeaRunError::Solve(
                "residual vector is linearly dependent on the retained modal basis".into(),
            ));
        }
        let scale = modal_mass.sqrt().recip();
        shape.iter_mut().for_each(|value| *value *= scale);
        let modal_mass = dot(&shape, &apply_m(&summary.operator, &shape));
        let modal_stiffness = dot(&shape, &apply_k(&summary.operator, &shape));
        if !modal_stiffness.is_finite() || modal_stiffness <= 0.0 {
            return Err(FeaRunError::Solve(
                "residual vector has no positive Rayleigh frequency".into(),
            ));
        }
        let maximum_mass_orthogonality_ratio = modes
            .iter()
            .map(|mode| dot(mode.dof_shape(), &apply_m(&summary.operator, &shape)).abs())
            .chain(residuals.iter().map(|residual| {
                dot(residual.dof_shape(), &apply_m(&summary.operator, &shape)).abs()
            }))
            .fold(0.0_f64, f64::max);
        let participation_factor = dot(&shape, &load) / modal_mass;
        let frequency_hz = (modal_stiffness / modal_mass).sqrt() / (2.0 * std::f64::consts::PI);
        if modes
            .last()
            .is_some_and(|mode| frequency_hz <= mode.frequency_hz())
        {
            return Err(FeaRunError::Solve(
                "residual-vector frequency does not lie above the retained eigenspace; extract all significant low modes first"
                    .into(),
            ));
        }
        residuals.push(ModalResidualVector {
            direction,
            frequency_hz,
            modal_mass,
            modal_stiffness,
            participation_factor,
            effective_mass: participation_factor * participation_factor * modal_mass,
            static_equilibrium_residual_ratio: equilibrium_ratio,
            maximum_mass_orthogonality_ratio,
            dof_shape: shape,
        });
    }
    Ok(residuals)
}

fn homogeneous_reduction(
    summary: &AssemblySummary,
) -> Result<AffineConstraintReduction, FeaRunError> {
    if let Some(matrix) = &summary.constraint_matrix {
        if matrix.right_hand_side.iter().any(|value| *value != 0.0) {
            return Err(FeaRunError::Solve(
                "residual vectors require homogeneous modal constraints".into(),
            ));
        }
        matrix
            .affine_reduction()
            .map_err(|error| FeaRunError::Solve(error.to_string()))
    } else {
        Ok(AffineConstraintReduction::from_constrained_dofs(
            summary.operator.constrained_dofs(),
        ))
    }
}

fn canonical_directions(directions: &[[f64; 3]]) -> Result<Vec<[f64; 3]>, FeaRunError> {
    let mut canonical = Vec::with_capacity(directions.len());
    for direction in directions {
        let norm = direction
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if !norm.is_finite() || norm <= f64::EPSILON {
            return Err(FeaRunError::Solve(
                "residual-vector direction must be finite and nonzero".into(),
            ));
        }
        let direction = direction.map(|value| value / norm);
        if canonical
            .iter()
            .any(|existing| directions_close(*existing, direction))
        {
            return Err(FeaRunError::Solve(
                "residual-vector directions must be unique".into(),
            ));
        }
        canonical.push(direction);
    }
    Ok(canonical)
}

fn directions_close(left: [f64; 3], right: [f64; 3]) -> bool {
    left.into_iter().zip(right).all(|(left, right)| {
        (left - right).abs() <= 128.0 * f64::EPSILON * left.abs().max(right.abs()).max(1.0)
    })
}

fn mass_orthogonalize(
    summary: &AssemblySummary,
    shape: &mut [f64],
    modes: &[ModalBasisMode],
    residuals: &[ModalResidualVector],
) -> Result<(), FeaRunError> {
    for vector in modes
        .iter()
        .map(ModalBasisMode::dof_shape)
        .chain(residuals.iter().map(ModalResidualVector::dof_shape))
    {
        let denominator = dot(vector, &apply_m(&summary.operator, vector));
        let coefficient = dot(vector, &apply_m(&summary.operator, shape)) / denominator;
        subtract_scaled(shape, vector, coefficient)?;
    }
    Ok(())
}

fn subtract(left: &[f64], right: &[f64]) -> Result<Vec<f64>, FeaRunError> {
    if left.len() != right.len() {
        return Err(FeaRunError::Solve(
            "residual-vector dimensions are inconsistent".into(),
        ));
    }
    Ok(left
        .iter()
        .zip(right)
        .map(|(left, right)| left - right)
        .collect())
}

fn subtract_scaled(target: &mut [f64], vector: &[f64], scale: f64) -> Result<(), FeaRunError> {
    if target.len() != vector.len() || !scale.is_finite() {
        return Err(FeaRunError::Solve(
            "residual-vector projection is invalid".into(),
        ));
    }
    for (target, vector) in target.iter_mut().zip(vector) {
        *target -= scale * vector;
    }
    Ok(())
}

fn norm(vector: &[f64]) -> f64 {
    dot(vector, vector).sqrt()
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}
