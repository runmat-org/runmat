use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    algebra::{spmv, PreparedModalSystem},
    assembly::AssemblySummary,
    solve::krylov::{restarted_gmres, GmresPolicy},
};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DirectHarmonicPolicy {
    pub relative_tolerance: f64,
    pub restart: usize,
    pub maximum_iterations: usize,
}

impl Default for DirectHarmonicPolicy {
    fn default() -> Self {
        Self {
            relative_tolerance: 1.0e-8,
            restart: 32,
            maximum_iterations: 2_048,
        }
    }
}

impl DirectHarmonicPolicy {
    pub fn validate(self) -> Result<(), DirectHarmonicError> {
        if !self.relative_tolerance.is_finite()
            || self.relative_tolerance <= 0.0
            || self.restart == 0
            || self.maximum_iterations == 0
        {
            return Err(DirectHarmonicError::NonFinite);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DirectHarmonicResponse {
    pub angular_frequency_rad_s: f64,
    pub displacement_real: Vec<f64>,
    pub displacement_imaginary: Vec<f64>,
    pub iterations: u32,
    pub residual_norm: f64,
    pub relative_residual: f64,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum DirectHarmonicError {
    #[error("direct harmonic inputs have inconsistent dimensions")]
    DimensionMismatch,
    #[error("direct harmonic inputs contain a nonfinite value or invalid solver policy")]
    NonFinite,
    #[error("direct harmonic constraints must be homogeneous")]
    NonHomogeneousConstraint,
    #[error("direct harmonic constraint reduction failed: {0}")]
    ConstraintReduction(String),
    #[error("direct harmonic Krylov solve broke down")]
    Breakdown,
    #[error("direct harmonic Krylov solve was cancelled")]
    Cancelled,
    #[error("direct harmonic Krylov solve did not converge")]
    NotConverged,
}

/// Solve the full sparse complex structural system through a matrix-free real block operator.
pub fn solve_direct_harmonic(
    summary: &AssemblySummary,
    imaginary_load: &[f64],
    angular_frequency_rad_s: f64,
    policy: DirectHarmonicPolicy,
) -> Result<DirectHarmonicResponse, DirectHarmonicError> {
    if imaginary_load.len() != summary.dof_count {
        return Err(DirectHarmonicError::DimensionMismatch);
    }
    if !angular_frequency_rad_s.is_finite()
        || angular_frequency_rad_s <= 0.0
        || imaginary_load.iter().any(|value| !value.is_finite())
    {
        return Err(DirectHarmonicError::NonFinite);
    }
    policy.validate()?;
    let prepared = prepare(summary)?;
    let reduced_real = prepared
        .reduction()
        .transpose_apply(summary.operator.load())
        .map_err(|error| DirectHarmonicError::ConstraintReduction(error.to_string()))?;
    let reduced_imaginary = prepared
        .reduction()
        .transpose_apply(imaginary_load)
        .map_err(|error| DirectHarmonicError::ConstraintReduction(error.to_string()))?;
    let reduced_size = prepared.operators().dof_count();
    if reduced_size == 0 {
        return Ok(DirectHarmonicResponse {
            angular_frequency_rad_s,
            displacement_real: vec![0.0; summary.dof_count],
            displacement_imaginary: vec![0.0; summary.dof_count],
            iterations: 0,
            residual_norm: 0.0,
            relative_residual: 0.0,
        });
    }
    let mut right_hand_side = reduced_real;
    right_hand_side.extend(reduced_imaginary);
    let apply = |value: &[f64]| {
        Ok::<_, std::convert::Infallible>(apply_dynamic_block(
            &prepared,
            angular_frequency_rad_s,
            value,
        ))
    };
    let result = restarted_gmres(
        &right_hand_side,
        &vec![0.0; reduced_size * 2],
        GmresPolicy {
            relative_tolerance: policy.relative_tolerance,
            restart: policy.restart,
            maximum_iterations: policy.maximum_iterations,
        },
        apply,
        dot,
        crate::is_fea_cancelled,
    )
    .expect("direct harmonic operator application is infallible")
    .ok_or(DirectHarmonicError::Breakdown)?;
    if result.cancelled {
        return Err(DirectHarmonicError::Cancelled);
    }
    if result.breakdown {
        return Err(DirectHarmonicError::Breakdown);
    }
    if !result.converged {
        return Err(DirectHarmonicError::NotConverged);
    }
    let displacement_real = prepared
        .expand(&result.solution[..reduced_size])
        .map_err(|error| DirectHarmonicError::ConstraintReduction(error.to_string()))?;
    let displacement_imaginary = prepared
        .expand(&result.solution[reduced_size..])
        .map_err(|error| DirectHarmonicError::ConstraintReduction(error.to_string()))?;
    let load_norm = dot(&right_hand_side, &right_hand_side).sqrt().max(1.0);
    Ok(DirectHarmonicResponse {
        angular_frequency_rad_s,
        displacement_real,
        displacement_imaginary,
        iterations: result.iterations,
        residual_norm: result.residual_norm,
        relative_residual: result.residual_norm / load_norm,
    })
}

fn prepare(summary: &AssemblySummary) -> Result<PreparedModalSystem, DirectHarmonicError> {
    if summary
        .operator
        .constrained_dofs()
        .iter()
        .zip(summary.operator.load())
        .any(|(constrained, value)| *constrained && *value != 0.0)
    {
        return Err(DirectHarmonicError::NonHomogeneousConstraint);
    }
    match summary.constraint_matrix.as_ref() {
        Some(constraints) => {
            if constraints
                .right_hand_side
                .iter()
                .any(|value| *value != 0.0)
            {
                return Err(DirectHarmonicError::NonHomogeneousConstraint);
            }
            let reduction = constraints
                .affine_reduction()
                .map_err(|error| DirectHarmonicError::ConstraintReduction(error.to_string()))?;
            summary
                .operator
                .prepare_modal_with_reduction(reduction)
                .map_err(|error| DirectHarmonicError::ConstraintReduction(error.to_string()))
        }
        None => summary
            .operator
            .prepare_modal()
            .map_err(|error| DirectHarmonicError::ConstraintReduction(error.to_string())),
    }
}

fn apply_dynamic_block(
    prepared: &PreparedModalSystem,
    angular_frequency_rad_s: f64,
    value: &[f64],
) -> Vec<f64> {
    let size = prepared.operators().dof_count();
    if value.len() != size * 2 {
        return Vec::new();
    }
    let real = &value[..size];
    let imaginary = &value[size..];
    let apply_part = |input: &[f64]| {
        let stiffness = spmv(prepared.operators().stiffness(), input).unwrap_or_default();
        let mass = spmv(prepared.operators().mass(), input).unwrap_or_default();
        stiffness
            .into_iter()
            .zip(mass)
            .map(|(stiffness, mass)| stiffness - angular_frequency_rad_s.powi(2) * mass)
            .collect::<Vec<_>>()
    };
    let real_dynamic = apply_part(real);
    let imaginary_dynamic = apply_part(imaginary);
    let real_damping = spmv(prepared.operators().damping(), real).unwrap_or_default();
    let imaginary_damping = spmv(prepared.operators().damping(), imaginary).unwrap_or_default();
    real_dynamic
        .iter()
        .zip(&imaginary_damping)
        .map(|(dynamic, damping)| dynamic - angular_frequency_rad_s * damping)
        .chain(
            imaginary_dynamic
                .iter()
                .zip(&real_damping)
                .map(|(dynamic, damping)| dynamic + angular_frequency_rad_s * damping),
        )
        .collect()
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::FixtureId;

    #[test]
    fn sparse_direct_response_closes_the_assembled_complex_equilibrium() {
        let model = crate::fixtures::fixture_model(FixtureId::CantileverLinearStatic);
        let step = &model.sequence.steps[0];
        let excitations = crate::excitation::realize_step_excitations(
            &model,
            step,
            crate::excitation::ExcitationSamplePoint::Static,
        )
        .unwrap();
        let summary = crate::assembly::assemble_structural_system(
            &model,
            &excitations,
            None,
            crate::assembly::StructuralFormulationRequirement::ExperimentalSyntheticOperator,
        )
        .unwrap();
        let response = solve_direct_harmonic(
            &summary,
            &vec![0.0; summary.dof_count],
            0.5,
            DirectHarmonicPolicy {
                relative_tolerance: 1.0e-10,
                ..DirectHarmonicPolicy::default()
            },
        )
        .unwrap();
        assert_eq!(response.displacement_real.len(), summary.dof_count);
        assert!(response.relative_residual <= 1.0e-10);
        assert!(response
            .displacement_real
            .iter()
            .chain(&response.displacement_imaginary)
            .all(|value| value.is_finite()));
    }
}
