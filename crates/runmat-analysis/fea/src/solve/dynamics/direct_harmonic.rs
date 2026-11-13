use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{algebra::PreparedModalSystem, assembly::AssemblySummary};

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

#[derive(Clone, Debug, Error, PartialEq)]
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
    #[error("direct harmonic right preconditioning failed: {0}")]
    Preconditioning(String),
    #[error("direct harmonic numerical usage: {0}")]
    Usage(runmat_simulation_usage::LedgerError),
    #[error("direct harmonic Krylov solve did not converge: iterations={iterations}/{maximum_iterations}, restart={restart}, reduced_dofs={reduced_dof_count}, residual_norm={residual_norm}, relative_residual={relative_residual}")]
    NotConverged {
        iterations: u32,
        maximum_iterations: usize,
        restart: usize,
        reduced_dof_count: usize,
        residual_norm: f64,
        relative_residual: f64,
    },
}

/// Solve the full sparse complex structural system `(K − ω²M + iωC) u = f` by sparse LU.
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
    if reduced_real
        .iter()
        .chain(&reduced_imaginary)
        .all(|value| *value == 0.0)
    {
        return Ok(DirectHarmonicResponse {
            angular_frequency_rad_s,
            displacement_real: vec![0.0; summary.dof_count],
            displacement_imaginary: vec![0.0; summary.dof_count],
            iterations: 0,
            residual_norm: 0.0,
            relative_residual: 0.0,
        });
    }
    // Complex dynamic stiffness K − ω²M + iωC on the reduced space, solved by sparse LU (the
    // matrix is complex symmetric, not Hermitian) with iterative refinement.
    use runmat_sparse_solvers::{
        c64,
        direct::{DirectOptions, DirectSolver, FactorKind},
    };
    let omega = angular_frequency_rad_s;
    let operators = prepared.operators();
    let dynamic = crate::solve::shared_sparse::complex_combination(&[
        (c64::new(1.0, 0.0), operators.stiffness()),
        (c64::new(-omega * omega, 0.0), operators.mass()),
        (c64::new(0.0, omega), operators.damping()),
    ])
    .map_err(|error| match error {
        runmat_sparse_solvers::SolverError::Usage(error) => DirectHarmonicError::Usage(error),
        other => DirectHarmonicError::Preconditioning(other.to_string()),
    })?;
    let rhs: Vec<c64> = reduced_real
        .iter()
        .zip(&reduced_imaginary)
        .map(|(&re, &im)| c64::new(re, im))
        .collect();
    crate::progress::check_cancelled("fea.direct_harmonic")
        .map_err(|_| DirectHarmonicError::Cancelled)?;
    let mut solution = vec![c64::new(0.0, 0.0); reduced_size];
    let solved = crate::solve::shared_sparse::with_plan(
        crate::planning::resource::LocalExecutionPlan::serial(),
        || -> Result<_, runmat_sparse_solvers::SolverError> {
            let solver = DirectSolver::<c64>::new(
                dynamic.as_ref().into(),
                DirectOptions {
                    refinement_steps: 3,
                    ..DirectOptions::new(FactorKind::Lu)
                },
                crate::solve::shared_sparse::control(),
            )?;
            solver.solve_refined(dynamic.as_ref().into(), &rhs, &mut solution, 1)
        },
    );
    match solved {
        Ok(_) => {}
        Err(runmat_sparse_solvers::SolverError::Cancelled) => {
            return Err(DirectHarmonicError::Cancelled)
        }
        Err(runmat_sparse_solvers::SolverError::Singular { .. }) => {
            return Err(DirectHarmonicError::Breakdown)
        }
        Err(runmat_sparse_solvers::SolverError::Usage(error)) => {
            return Err(DirectHarmonicError::Usage(error))
        }
        Err(error) => return Err(DirectHarmonicError::Preconditioning(error.to_string())),
    }
    let mut action = vec![c64::new(0.0, 0.0); reduced_size];
    dynamic.as_ref().spmv(&solution, &mut action);
    let residual_norm = rhs
        .iter()
        .zip(&action)
        .map(|(b, a)| (b - a).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let load_norm = rhs.iter().map(|b| b.norm_sqr()).sum::<f64>().sqrt();
    let relative_residual = if load_norm > 0.0 {
        residual_norm / load_norm
    } else {
        0.0
    };
    if !relative_residual.is_finite() || relative_residual > policy.relative_tolerance {
        return Err(DirectHarmonicError::NotConverged {
            iterations: 1,
            maximum_iterations: policy.maximum_iterations,
            restart: policy.restart,
            reduced_dof_count: reduced_size,
            residual_norm,
            relative_residual,
        });
    }
    let real: Vec<f64> = solution.iter().map(|value| value.re).collect();
    let imaginary: Vec<f64> = solution.iter().map(|value| value.im).collect();
    let displacement_real = prepared
        .expand(&real)
        .map_err(|error| DirectHarmonicError::ConstraintReduction(error.to_string()))?;
    let displacement_imaginary = prepared
        .expand(&imaginary)
        .map_err(|error| DirectHarmonicError::ConstraintReduction(error.to_string()))?;
    Ok(DirectHarmonicResponse {
        angular_frequency_rad_s,
        displacement_real,
        displacement_imaginary,
        iterations: 1,
        residual_norm,
        relative_residual,
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
            crate::assembly::StructuralFormulationRequirement::SyntheticOperator,
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

    #[test]
    fn small_harmonic_load_retains_relative_residual_and_linear_response() {
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
            crate::assembly::StructuralFormulationRequirement::SyntheticOperator,
        )
        .unwrap();
        let policy = DirectHarmonicPolicy {
            relative_tolerance: 1.0e-10,
            ..DirectHarmonicPolicy::default()
        };
        let reference =
            solve_direct_harmonic(&summary, &vec![0.0; summary.dof_count], 0.5, policy).unwrap();
        let scale = 1.0e-20;
        let mut small = summary.clone();
        small.operator = small
            .operator
            .with_load(
                summary
                    .operator
                    .load()
                    .iter()
                    .map(|value| value * scale)
                    .collect(),
            )
            .unwrap();
        let response =
            solve_direct_harmonic(&small, &vec![0.0; small.dof_count], 0.5, policy).unwrap();
        assert!(response.iterations > 0);
        assert!(response.relative_residual <= policy.relative_tolerance);
        for (actual, expected) in response
            .displacement_real
            .iter()
            .zip(&reference.displacement_real)
        {
            assert!((actual / scale - expected).abs() <= 1.0e-8 * expected.abs().max(1.0));
        }

        let mut zero = summary.clone();
        zero.operator = zero.operator.with_load(vec![0.0; zero.dof_count]).unwrap();
        let response =
            solve_direct_harmonic(&zero, &vec![0.0; zero.dof_count], 0.5, policy).unwrap();
        assert_eq!(response.iterations, 0);
        assert_eq!(response.relative_residual, 0.0);
    }
}
