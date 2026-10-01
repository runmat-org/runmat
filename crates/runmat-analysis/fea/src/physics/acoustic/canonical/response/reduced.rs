//! Shared real-block affine reduction of complex Helmholtz pressure.

use super::*;
use crate::constraints::build_phase_reduction;

#[cfg(test)]
mod tests;

pub(super) fn solve(
    assembly: &AcousticAssembly,
    policy: AcousticHarmonicPolicy,
    initial: Option<&[f64]>,
) -> Result<ReducedSolution, AcousticHarmonicError> {
    let owned;
    let reduction = match &assembly.reduction {
        Some(reduction) => reduction,
        None => {
            owned =
                build_phase_reduction(assembly.projection.nodes.len(), &[], &assembly.prescribed)
                    .map_err(|error| AcousticHarmonicError::Boundary(error.to_string()))?;
            &owned
        }
    };
    let size = assembly.projection.nodes.len();
    let lifting = apply(assembly, &reduction.offset);
    let source = assembly
        .source
        .iter()
        .map(|value| value.real)
        .chain(assembly.source.iter().map(|value| value.imaginary))
        .zip(lifting)
        .map(|(source, lifting)| source - lifting)
        .collect::<Vec<_>>();
    let rhs = reduction
        .transpose_apply(&source)
        .map_err(|error| AcousticHarmonicError::Operator(error.to_string()))?;
    let (solution, iterations, relative_residual) = if reduction.reduced_dof_count == 0 {
        (Vec::new(), 0, 0.0)
    } else {
        let zero = vec![0.0; rhs.len()];
        let result = restarted_gmres(
            &rhs,
            initial.unwrap_or(&zero),
            GmresPolicy {
                relative_tolerance: policy.relative_tolerance,
                restart: policy.restart,
                maximum_iterations: policy.maximum_iterations,
            },
            |value| -> Result<Vec<f64>, AcousticHarmonicError> {
                let expanded = reduction
                    .expand_linear(value)
                    .map_err(|error| AcousticHarmonicError::Operator(error.to_string()))?;
                reduction
                    .transpose_apply(&apply(assembly, &expanded))
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
        let norm = rhs.iter().map(|value| value * value).sum::<f64>().sqrt();
        let residual = if norm == 0.0 {
            result.residual_norm
        } else {
            result.residual_norm / norm
        };
        (result.solution, result.iterations, residual)
    };
    let mut full = reduction
        .expand(&solution)
        .map_err(|error| AcousticHarmonicError::Operator(error.to_string()))?;
    let imaginary = full.split_off(size);
    Ok(ReducedSolution {
        real: full,
        imaginary,
        coordinates: solution,
        iterations,
        relative_residual,
    })
}

pub(super) struct ReducedSolution {
    pub real: Vec<f64>,
    pub imaginary: Vec<f64>,
    pub coordinates: Vec<f64>,
    pub iterations: u32,
    pub relative_residual: f64,
}

fn apply(assembly: &AcousticAssembly, full: &[f64]) -> Vec<f64> {
    let size = assembly.projection.nodes.len();
    let (mut real, imaginary) = assembly.apply(&full[..size], &full[size..]);
    real.extend(imaginary);
    real
}
