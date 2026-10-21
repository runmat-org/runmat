//! Exact affine reduction, then one certified sparse LU of the reduced real block operator.
use super::*;
use crate::{
    algebra::{spmv, RealSparseMatrix},
    constraints::{build_phase_reduction, AffineConstraintReduction},
    solve::certificate::{certify_nonsingular, CertificateError, CertifiedFactorization},
};

mod operator;
#[cfg(test)]
mod tests;

pub(super) struct PreparedReduced {
    reduction: AffineConstraintReduction,
    operator: RealSparseMatrix,
    rhs: Vec<f64>,
    residual_reference: f64,
    factorization: Option<CertifiedFactorization>,
    certificate: AcousticSolveCertificate,
    size: usize,
}

impl PreparedReduced {
    /// Certifies and factors the original reduced operator once; the solve reuses the factor.
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
        // Source-independent: the factorization that solves also certifies nonsingularity.
        let (factorization, certificate) = certify(&operator, policy.contraction_limit)?;
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
        Ok(Self {
            reduction,
            operator,
            rhs,
            residual_reference,
            factorization,
            certificate,
            size: assembly.projection.nodes.len(),
        })
    }

    pub(super) fn solve(
        &self,
        assembly: &AcousticAssembly,
        policy: AcousticHarmonicPolicy,
    ) -> Result<ReducedSolution, AcousticHarmonicError> {
        check_cancelled()?;
        let mut solution = vec![0.0; self.rhs.len()];
        let mut refinement_steps = 0;
        let mut reference = self.residual_reference;
        if let Some(factorization) = &self.factorization {
            let arrays = crate::solve::shared_sparse::CsrArrays::of(&self.operator);
            let report = factorization
                .factor
                .solve_refined(arrays.view().into(), &self.rhs, &mut solution, 1)
                .map_err(|error| match error {
                    runmat_sparse_solvers::SolverError::Cancelled => {
                        AcousticHarmonicError::Cancelled
                    }
                    other => AcousticHarmonicError::Operator(other.to_string()),
                })?;
            refinement_steps = u32::try_from(report.refinement_steps)
                .map_err(|_| AcousticHarmonicError::NotConverged)?;
            // Acceptance re-evaluates the original reduced operator, never the factor.
            let final_norm = norm(&self.residual(&solution)?);
            if reference == 0.0 {
                reference = final_norm;
            }
            if !final_norm.is_finite() || final_norm > policy.relative_tolerance * reference {
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
            iterations: refinement_steps,
            relative_residual,
            certificate: self.certificate,
        })
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
) -> Result<ReducedSolution, AcousticHarmonicError> {
    PreparedReduced::new(assembly, policy)?.solve(assembly, policy)
}

pub(super) struct ReducedSolution {
    pub real: Vec<f64>,
    pub imaginary: Vec<f64>,
    /// Iterative refinement steps of the certified direct solve.
    pub iterations: u32,
    pub relative_residual: f64,
    pub certificate: AcousticSolveCertificate,
}

/// Certify the reduced operator, or record that no free unknown remains.
fn certify(
    operator: &RealSparseMatrix,
    limit: f64,
) -> Result<(Option<CertifiedFactorization>, AcousticSolveCertificate), AcousticHarmonicError> {
    if operator.rows() == 0 {
        let operator_identity = operator
            .numerical_identity()
            .map_err(|error| AcousticHarmonicError::Operator(error.to_string()))?;
        return Ok((
            None,
            AcousticSolveCertificate::no_free_unknowns(operator_identity, limit),
        ));
    }
    let certified = certify_nonsingular(operator, limit).map_err(|error| match error {
        CertificateError::Cancelled => AcousticHarmonicError::Cancelled,
        other => AcousticHarmonicError::Certificate(other),
    })?;
    let certificate = AcousticSolveCertificate::from(&certified.certificate);
    Ok((Some(certified), certificate))
}
