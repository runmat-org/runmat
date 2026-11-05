use runmat_analysis_core::{CoordinateFrame, SpectrumDirection};
use serde::{Deserialize, Serialize};

use crate::{
    algebra::{apply_k, apply_m},
    assembly::AssemblySummary,
    contracts::FeaRunError,
    solve::preconditioner::SpdPreconditionerKind,
};

use super::{
    homogeneous_static::HomogeneousStaticSolver,
    modal_basis::{directional_influence_vector, ModalBasisMode},
};

pub(crate) const RESIDUAL_MASS_ORTHOGONALIZATION_PASSES: usize = 2;

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
    static_correction_preconditioner: SpdPreconditionerKind,
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
    pub const fn static_correction_preconditioner(&self) -> SpdPreconditionerKind {
        self.static_correction_preconditioner
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
    let static_solver = HomogeneousStaticSolver::new(summary)?;

    let mut residuals: Vec<ModalResidualVector> = Vec::with_capacity(directions.len());
    for direction in directions {
        let influence = directional_influence_vector(summary, direction);
        let load = apply_m(&summary.operator, &influence);
        let solution = static_solver.solve(&load)?;
        let equilibrium_ratio = solution.equilibrium_residual_ratio;
        let mut shape = solution.displacement;
        let initial_mass = dot(&shape, &apply_m(&summary.operator, &shape));
        for mode in modes {
            let coefficient = dot(mode.dof_shape(), &load) / mode.modal_stiffness();
            subtract_scaled(&mut shape, mode.dof_shape(), coefficient)?;
        }
        for _ in 0..RESIDUAL_MASS_ORTHOGONALIZATION_PASSES {
            mass_orthogonalize(summary, &mut shape, modes, &residuals)?;
        }
        let mass_applied = apply_m(&summary.operator, &shape);
        let modal_mass = dot(&shape, &mass_applied);
        // Dependence is relative to the static response itself: an absolute bound on a modal
        // mass in kg m^2 rejects every physically small (stiff, SI-unit) residual vector.
        if !modal_mass.is_finite()
            || !initial_mass.is_finite()
            || modal_mass.sqrt() <= STATIC_CORRECTION_DEPENDENCE_RATIO * initial_mass.sqrt()
        {
            return Err(FeaRunError::Solve(
                "residual vector is linearly dependent on the retained modal basis".into(),
            ));
        }
        let reference = super::modal_basis::ModalBasisCoordinates::canonical_reference_mass();
        let scale = runmat_types::physical::Mass::new(modal_mass)
            .ratio_to(reference)
            .canonical_value()
            .sqrt()
            .recip();
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
            static_correction_preconditioner: static_solver.preconditioner_kind(),
            maximum_mass_orthogonality_ratio,
            dof_shape: shape,
        });
    }
    Ok(residuals)
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

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

/// Relative mass norm below which a static correction already lies in the retained space.
const STATIC_CORRECTION_DEPENDENCE_RATIO: f64 = 1.0e-8;

/// Static corrections for the applied loads of a modal-reduced frequency sweep: for each nonzero
/// load `f`, the static response `K⁻¹ f` with every retained shape removed in the mass metric,
/// scaled to unit reference modal mass. With these in the reduced space the quasi-static part of
/// the response is exact, so the full-space balance error is left to the dynamic truncation.
/// A load whose static response the retained space already spans adds no vector.
pub(crate) fn applied_load_static_corrections(
    summary: &AssemblySummary,
    retained: &[&[f64]],
    loads: &[&[f64]],
) -> Result<Vec<Vec<f64>>, FeaRunError> {
    if loads
        .iter()
        .all(|load| load.iter().all(|value| *value == 0.0))
    {
        return Ok(Vec::new());
    }
    let static_solver = HomogeneousStaticSolver::new(summary)?;
    let reference = super::modal_basis::ModalBasisCoordinates::canonical_reference_mass();
    let mut corrections: Vec<Vec<f64>> = Vec::with_capacity(loads.len());
    for load in loads {
        if load.iter().all(|value| *value == 0.0) {
            continue;
        }
        let mut shape = static_solver.solve(load)?.displacement;
        let initial_mass = dot(&shape, &apply_m(&summary.operator, &shape));
        for _ in 0..RESIDUAL_MASS_ORTHOGONALIZATION_PASSES {
            for vector in retained
                .iter()
                .copied()
                .chain(corrections.iter().map(Vec::as_slice))
            {
                let denominator = dot(vector, &apply_m(&summary.operator, vector));
                let coefficient = dot(vector, &apply_m(&summary.operator, &shape)) / denominator;
                subtract_scaled(&mut shape, vector, coefficient)?;
            }
        }
        let modal_mass = dot(&shape, &apply_m(&summary.operator, &shape));
        if !modal_mass.is_finite() || !initial_mass.is_finite() || initial_mass <= 0.0 {
            return Err(FeaRunError::Solve(
                "applied-load static correction is not finite".into(),
            ));
        }
        if modal_mass.sqrt() <= STATIC_CORRECTION_DEPENDENCE_RATIO * initial_mass.sqrt() {
            continue;
        }
        let scale = runmat_types::physical::Mass::new(modal_mass)
            .ratio_to(reference)
            .canonical_value()
            .sqrt()
            .recip();
        shape.iter_mut().for_each(|value| *value *= scale);
        corrections.push(shape);
    }
    Ok(corrections)
}
