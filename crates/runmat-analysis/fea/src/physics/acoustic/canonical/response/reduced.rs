//! Exact affine reduction, then one certified sparse LU of the reduced real block operator.
use super::super::quantity_layout::{
    equation_descriptor, ReducedPressureLayout, ReducedPressureOperator,
};
use super::*;
use crate::{
    algebra::{spmv, RealSparseMatrix},
    constraints::{build_phase_reduction, AffineConstraintReduction},
    solve::certificate::{certify_nonsingular, CertificateError, CertifiedFactorization},
};
use runmat_types::physical::{Quantity, QuantityLinearOperator, UniformQuantityLayout};

mod operator;
#[cfg(test)]
mod tests;

pub(super) struct PreparedReduced {
    reduction: std::sync::Arc<AffineConstraintReduction>,
    operator: ReducedPressureOperator,
    rhs: Quantity<runmat_types::physical::DynamicDimension, Vec<f64>>,
    residual_reference: Quantity<runmat_types::physical::DynamicDimension>,
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
            None => build_phase_reduction(
                assembly.projection.nodes.len(),
                &[],
                assembly.prescribed.canonical_values(),
            )
            .map_err(|error| AcousticHarmonicError::Boundary(error.to_string()))?,
        };
        let full_operator = operator::materialize(assembly)?;
        let source = assembly
            .source
            .canonical_values()
            .iter()
            .map(|value| value.real)
            .chain(
                assembly
                    .source
                    .canonical_values()
                    .iter()
                    .map(|value| value.imaginary),
            )
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

        let layout = ReducedPressureLayout::try_new(reduction, assembly.projection.nodes.len())?;
        let operator = QuantityLinearOperator::try_new(
            layout.clone(),
            UniformQuantityLayout::from_descriptor(rhs.len(), equation_descriptor(2)),
            operator,
        )
        .map_err(|e| AcousticHarmonicError::Operator(e.to_string()))?;
        let rhs = Quantity::try_from_descriptor(equation_descriptor(2), rhs)
            .map_err(|e| AcousticHarmonicError::Operator(e.to_string()))?;
        let residual_reference =
            Quantity::try_from_descriptor(equation_descriptor(2), residual_reference)
                .map_err(|e| AcousticHarmonicError::Operator(e.to_string()))?;
        Ok(Self {
            reduction: layout.reduction.clone(),
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
        let mut solution = vec![0.0; self.rhs.canonical_values().len()];
        let mut refinement_steps = 0;
        let mut reference = self.residual_reference.canonical_value();
        if let Some(factorization) = &self.factorization {
            let report = self
                .operator
                .with_canonical_storage(|operator| {
                    let arrays = crate::solve::shared_sparse::CsrArrays::of(operator);
                    factorization.factor.solve_refined(
                        arrays.view().into(),
                        self.rhs.canonical_values(),
                        &mut solution,
                        1,
                    )
                })
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
            .canonical_values()
            .iter()
            .map(|value| value.real)
            .chain(
                assembly
                    .source
                    .canonical_values()
                    .iter()
                    .map(|value| value.imaginary),
            )
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
            .canonical_values()
            .iter()
            .zip(
                self.operator
                    .with_canonical_storage(|matrix| spmv(matrix, solution))
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
