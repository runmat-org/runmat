use anyhow::{ensure, Result};

use super::{CancellationMonitor, Controlled, SparseMatrix, SparsePreconditioner};

pub(super) fn multiply(matrix: &SparseMatrix, input: &[f64]) -> Result<Vec<f64>> {
    multiply_impl(matrix, input, None)?.map_or_else(
        || unreachable!("uncancelled sparse multiplication cannot cancel"),
        Ok,
    )
}

pub(super) fn multiply_controlled(
    matrix: &SparseMatrix,
    input: &[f64],
    cancellation: &mut CancellationMonitor<'_>,
) -> Result<Controlled<Vec<f64>>> {
    Ok(match multiply_impl(matrix, input, Some(cancellation))? {
        Some(output) => Controlled::Completed(output),
        None => Controlled::Cancelled,
    })
}

fn multiply_impl(
    matrix: &SparseMatrix,
    input: &[f64],
    mut cancellation: Option<&mut CancellationMonitor<'_>>,
) -> Result<Option<Vec<f64>>> {
    ensure!(
        input.len() == matrix.pattern.descriptor.columns as usize,
        "sparse SpMV input length does not match matrix columns"
    );
    let mut output = vec![0.0; matrix.pattern.descriptor.rows as usize];
    for (row, output_value) in output.iter_mut().enumerate() {
        let start = matrix.pattern.row_offsets[row] as usize;
        let end = matrix.pattern.row_offsets[row + 1] as usize;
        if start == end
            && cancellation
                .as_deref_mut()
                .is_some_and(CancellationMonitor::complete_work_unit)
        {
            return Ok(None);
        }
        for index in start..end {
            *output_value +=
                matrix.values[index] * input[matrix.pattern.column_indices[index] as usize];
            if cancellation
                .as_deref_mut()
                .is_some_and(CancellationMonitor::complete_work_unit)
            {
                return Ok(None);
            }
        }
    }
    Ok(Some(output))
}

pub(super) fn apply_jacobi(
    preconditioner: &SparsePreconditioner,
    input: &[f64],
) -> Result<Vec<f64>> {
    ensure_jacobi_shape(preconditioner, input)?;
    Ok(input
        .iter()
        .zip(&preconditioner.inverse_diagonal)
        .map(|(value, inverse)| value * inverse)
        .collect())
}

pub(super) fn apply_jacobi_controlled(
    preconditioner: &SparsePreconditioner,
    input: &[f64],
    cancellation: &mut CancellationMonitor<'_>,
) -> Result<Controlled<Vec<f64>>> {
    ensure_jacobi_shape(preconditioner, input)?;
    let mut output = Vec::with_capacity(input.len());
    for (value, inverse) in input.iter().zip(&preconditioner.inverse_diagonal) {
        output.push(value * inverse);
        if cancellation.complete_work_unit() {
            return Ok(Controlled::Cancelled);
        }
    }
    Ok(Controlled::Completed(output))
}

fn ensure_jacobi_shape(preconditioner: &SparsePreconditioner, input: &[f64]) -> Result<()> {
    ensure!(
        input.len() == preconditioner.inverse_diagonal.len(),
        "Jacobi input length does not match its matrix"
    );
    Ok(())
}
