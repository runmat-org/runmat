//! Exact affine reduction, fixed shared right LU, and physical warm corrections.
use super::super::quantity_layout::{
    equation_descriptor, ReducedPressureLayout, ReducedPressureOperator,
};
use super::*;
use crate::{
    algebra::spmv,
    constraints::{build_phase_reduction, AffineConstraintReduction},
    contracts::SolverPreconditioner,
    solve::krylov::right_ilu::{SparseRightIlu, SparseRightIluError},
};
use runmat_types::physical::{
    kind, Quantity, QuantityField, QuantityLinearOperator, UniformQuantityLayout,
};

mod operator;
#[cfg(test)]
mod tests;

pub(super) struct PreparedReduced {
    reduction: std::sync::Arc<AffineConstraintReduction>,
    operator: ReducedPressureOperator,
    rhs: Quantity<runmat_types::physical::DynamicDimension, Vec<f64>>,
    residual_reference: Quantity<runmat_types::physical::DynamicDimension>,
    preconditioner: Option<SparseRightIlu>,
    size: usize,
    uniqueness: Option<crate::solve::uniqueness::SparseUniquenessRecord>,
}

impl PreparedReduced {
    /// Prepared once per unchanged numerical operator, reused by power refinement.
    #[cfg(test)]
    pub(super) fn new(
        assembly: &AcousticAssembly,
        policy: AcousticHarmonicPolicy,
    ) -> Result<Self, AcousticHarmonicError> {
        Self::new_admitted(assembly, policy, None)
    }

    pub(super) fn new_admitted(
        assembly: &AcousticAssembly,
        policy: AcousticHarmonicPolicy,
        admitted: Option<crate::solve::uniqueness::SparseUniquenessEnvelope>,
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
        // Source-independent, once per unchanged original reduced operator,
        // before physical RHS acceptance or physical preconditioner construction.
        let uniqueness = super::super::uniqueness::certify(&operator, policy.uniqueness, admitted)?;
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
            preconditioner,
            size: assembly.projection.nodes.len(),
            uniqueness,
        })
    }

    pub(super) fn solve(
        &self,
        assembly: &AcousticAssembly,
        policy: AcousticHarmonicPolicy,
        initial: Option<&QuantityField<ReducedPressureLayout, Vec<f64>>>,
    ) -> Result<ReducedSolution, AcousticHarmonicError> {
        check_cancelled()?;
        if initial.is_some_and(|field| field.layout() != self.operator.input_layout()) {
            return Err(AcousticHarmonicError::InvalidInput(
                "pressure warm start belongs to a different phase basis".into(),
            ));
        }
        let mut solution = initial
            .map(|field| field.with_canonical_storage(<[f64]>::to_vec))
            .unwrap_or_else(|| vec![0.0; self.rhs.canonical_values().len()]);
        if solution.len() != self.rhs.canonical_values().len()
            || solution.iter().any(|value| !value.is_finite())
        {
            return Err(AcousticHarmonicError::InvalidInput(
                "physical warm-start shape or values".into(),
            ));
        }
        let mut iterations = 0;
        let mut reference = self.residual_reference.canonical_value();
        if !self.rhs.canonical_values().is_empty() {
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
                        {
                            let applied = self.right_apply(value)?;
                            self.operator
                                .with_canonical_storage(|matrix| spmv(matrix, &applied))
                        }
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
            real: Quantity::try_new(full)
                .map_err(|e| AcousticHarmonicError::Operator(e.to_string()))?,
            imaginary: Quantity::try_new(imaginary)
                .map_err(|e| AcousticHarmonicError::Operator(e.to_string()))?,
            coordinates: QuantityField::try_new(self.operator.input_layout().clone(), solution)
                .map_err(|e| {
                    AcousticHarmonicError::Operator(format!("invalid pressure coordinates: {e:?}"))
                })?,
            iterations,
            relative_residual,
            uniqueness: self.uniqueness,
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
    initial: Option<&QuantityField<ReducedPressureLayout, Vec<f64>>>,
) -> Result<ReducedSolution, AcousticHarmonicError> {
    PreparedReduced::new(assembly, policy)?.solve(assembly, policy, initial)
}

pub(super) struct ReducedSolution {
    pub real: Quantity<kind::Pressure, Vec<f64>>,
    pub imaginary: Quantity<kind::Pressure, Vec<f64>>,
    pub coordinates: QuantityField<ReducedPressureLayout, Vec<f64>>,
    pub iterations: u32,
    pub relative_residual: f64,
    pub uniqueness: Option<crate::solve::uniqueness::SparseUniquenessRecord>,
}
