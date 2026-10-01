//! Exact affine reduction, fixed shared right LU, and physical warm corrections.
use super::*;
use crate::{
    algebra::{spmv, RealSparseMatrix},
    constraints::{build_phase_reduction, AffineConstraintReduction},
    contracts::SolverPreconditioner,
    solve::krylov::right_ilu::{SparseRightIlu, SparseRightIluError},
};

mod operator;
#[cfg(test)]
mod tests;

pub(super) struct PreparedReduced {
    reduction: AffineConstraintReduction,
    operator: RealSparseMatrix,
    rhs: Vec<f64>,
    residual_reference: f64,
    preconditioner: Option<SparseRightIlu>,
    size: usize,
}

impl PreparedReduced {
    /// Prepared once per unchanged numerical operator, reused by power refinement.
    pub(super) fn new(
        assembly: &AcousticAssembly,
        policy: AcousticHarmonicPolicy,
    ) -> Result<Self, AcousticHarmonicError> {
        check_cancelled()?;
        let reduction = match &assembly.reduction {
            Some(reduction) => reduction.clone(),
            None => {
                build_phase_reduction(assembly.projection.nodes.len(), &[], &assembly.prescribed)
                    .map_err(|error| AcousticHarmonicError::Boundary(error.to_string()))?
            }
        };
        let full_operator = operator::materialize(assembly)?;
        let source = assembly
            .source
            .iter()
            .map(|value| value.real)
            .chain(assembly.source.iter().map(|value| value.imaginary))
            .collect::<Vec<_>>();
        let operator = reduction
            .reduce_sparse_operator(&full_operator)
            .map_err(|error| AcousticHarmonicError::Operator(error.to_string()))?;
        let (mut lifting, lifting_imaginary) = assembly.apply(
            &reduction.offset[..assembly.projection.nodes.len()],
            &reduction.offset[assembly.projection.nodes.len()..],
        );
        lifting.extend(lifting_imaginary);
        let shifted = source
            .iter()
            .zip(&lifting)
            .map(|(source, lifting)| source - lifting)
            .collect::<Vec<_>>();
        let rhs = reduction
            .transpose_apply(&shifted)
            .map_err(|error| AcousticHarmonicError::Operator(error.to_string()))?;
        let mut residual_reference = norm(&rhs);
        if residual_reference == 0.0 {
            // Exact source/lifting cancellation still has a physical scale.
            // Do not compare a dimensional residual with a relative tolerance.
            for value in [&source, &lifting] {
                let restricted = reduction
                    .transpose_apply(value)
                    .map_err(|error| AcousticHarmonicError::Operator(error.to_string()))?;
                let restricted_norm = norm(&restricted);
                if !restricted_norm.is_finite() {
                    return Err(AcousticHarmonicError::Operator(
                        "nonfinite physical residual reference".into(),
                    ));
                }
                residual_reference = residual_reference.max(restricted_norm);
            }
        }
        if !residual_reference.is_finite() {
            return Err(AcousticHarmonicError::Operator(
                "nonfinite physical residual reference".into(),
            ));
        }
        check_cancelled()?;
        drop(full_operator);
        drop(source);
        drop(lifting);
        drop(shifted);
        let preconditioner = match policy.preconditioner {
            SolverPreconditioner::None => None,
            SolverPreconditioner::Ilu0 if reduction.reduced_dof_count == 0 => None,
            SolverPreconditioner::Ilu0 => Some(
                SparseRightIlu::factor(&operator, crate::is_fea_cancelled).map_err(ilu_error)?,
            ),
            _ => {
                return Err(AcousticHarmonicError::Unsupported(
                    "pressure preconditioner".into(),
                ))
            }
        };
        Ok(Self {
            reduction,
            operator,
            rhs,
            residual_reference,
            preconditioner,
            size: assembly.projection.nodes.len(),
        })
    }

    pub(super) fn solve(
        &self,
        assembly: &AcousticAssembly,
        policy: AcousticHarmonicPolicy,
        initial: Option<&[f64]>,
    ) -> Result<ReducedSolution, AcousticHarmonicError> {
        check_cancelled()?;
        let mut solution = initial
            .map(<[f64]>::to_vec)
            .unwrap_or_else(|| vec![0.0; self.rhs.len()]);
        if solution.len() != self.rhs.len() || solution.iter().any(|value| !value.is_finite()) {
            return Err(AcousticHarmonicError::InvalidInput(
                "physical warm-start shape or values".into(),
            ));
        }
        let mut iterations = 0;
        let mut reference = self.residual_reference;
        if !self.rhs.is_empty() {
            let residual = self.residual(&solution)?;
            let residual_norm = norm(&residual);
            if !residual_norm.is_finite() {
                return Err(AcousticHarmonicError::Operator(
                    "nonfinite original residual".into(),
                ));
            }
            // Truly homogeneous warm starts use the shared GMRES initial
            // residual convention, fixed for target, acceptance and reporting.
            if reference == 0.0 {
                reference = residual_norm;
            }
            let target = policy.relative_tolerance * reference;
            if !target.is_finite() || (reference > 0.0 && target == 0.0) {
                return Err(AcousticHarmonicError::NotConverged);
            }
            if residual_norm > target {
                // A physical warm start is never mistaken for preconditioned y.
                // Solve A M^-1 y = b-Ax0, then recover x=x0+M^-1 y.
                let zero = vec![0.0; residual.len()];
                let result = restarted_gmres(
                    &residual,
                    &zero,
                    GmresPolicy {
                        relative_tolerance: target / residual_norm,
                        restart: policy.restart,
                        maximum_iterations: policy.maximum_iterations,
                    },
                    |value| {
                        spmv(&self.operator, &self.right_apply(value)?)
                            .map_err(|error| AcousticHarmonicError::Operator(error.to_string()))
                    },
                    |a, b| Ok(a.iter().zip(b).map(|(a, b)| a * b).sum()),
                    crate::is_fea_cancelled,
                )?
                .ok_or(AcousticHarmonicError::Breakdown)?;
                if result.cancelled {
                    return Err(AcousticHarmonicError::Cancelled);
                }
                if result.breakdown {
                    return Err(AcousticHarmonicError::Breakdown);
                }
                if !result.converged {
                    return Err(AcousticHarmonicError::NotConverged);
                }
                for (value, correction) in
                    solution.iter_mut().zip(self.right_apply(&result.solution)?)
                {
                    *value += correction;
                }
                iterations = result.iterations;
            }
            // Original reduced operator acceptance, never a factor/scaled residual.
            // The separate power gate still evaluates original K/M/B.
            let final_norm = norm(&self.residual(&solution)?);
            if !final_norm.is_finite() || final_norm > target {
                return Err(AcousticHarmonicError::NotConverged);
            }
        }
        let mut full = self
            .reduction
            .expand(&solution)
            .map_err(|error| AcousticHarmonicError::Operator(error.to_string()))?;
        let (mut applied, applied_imaginary) =
            assembly.apply(&full[..self.size], &full[self.size..]);
        applied.extend(applied_imaginary);
        let residual = assembly
            .source
            .iter()
            .map(|value| value.real)
            .chain(assembly.source.iter().map(|value| value.imaginary))
            .zip(applied)
            .map(|(source, applied)| source - applied)
            .collect::<Vec<_>>();
        let residual = norm(
            &self
                .reduction
                .transpose_apply(&residual)
                .map_err(|error| AcousticHarmonicError::Operator(error.to_string()))?,
        );
        let relative_residual = if reference == 0.0 {
            if residual != 0.0 {
                return Err(AcousticHarmonicError::NotConverged);
            }
            0.0
        } else {
            residual / reference
        };
        if !relative_residual.is_finite() || relative_residual > policy.relative_tolerance {
            return Err(AcousticHarmonicError::NotConverged);
        }
        let imaginary = full.split_off(self.size);
        check_cancelled()?;
        Ok(ReducedSolution {
            real: full,
            imaginary,
            coordinates: solution,
            iterations,
            relative_residual,
        })
    }

    fn right_apply(&self, value: &[f64]) -> Result<Vec<f64>, AcousticHarmonicError> {
        match &self.preconditioner {
            Some(preconditioner) => preconditioner
                .apply(value, crate::is_fea_cancelled)
                .map_err(ilu_error),
            None => Ok(value.to_vec()),
        }
    }

    fn residual(&self, solution: &[f64]) -> Result<Vec<f64>, AcousticHarmonicError> {
        Ok(self
            .rhs
            .iter()
            .zip(
                spmv(&self.operator, solution)
                    .map_err(|error| AcousticHarmonicError::Operator(error.to_string()))?,
            )
            .map(|(source, applied)| source - applied)
            .collect())
    }
}

fn norm(value: &[f64]) -> f64 {
    value.iter().fold(0.0, |sum, value| sum.hypot(*value))
}

fn ilu_error(error: SparseRightIluError) -> AcousticHarmonicError {
    match error {
        SparseRightIluError::Cancelled => AcousticHarmonicError::Cancelled,
        _ => AcousticHarmonicError::Operator(error.to_string()),
    }
}

fn check_cancelled() -> Result<(), AcousticHarmonicError> {
    if crate::is_fea_cancelled() {
        Err(AcousticHarmonicError::Cancelled)
    } else {
        Ok(())
    }
}

#[cfg(test)]
pub(super) fn solve(
    assembly: &AcousticAssembly,
    policy: AcousticHarmonicPolicy,
    initial: Option<&[f64]>,
) -> Result<ReducedSolution, AcousticHarmonicError> {
    PreparedReduced::new(assembly, policy)?.solve(assembly, policy, initial)
}

pub(super) struct ReducedSolution {
    pub real: Vec<f64>,
    pub imaginary: Vec<f64>,
    pub coordinates: Vec<f64>,
    pub iterations: u32,
    pub relative_residual: f64,
}
