//! Exact borrowed Hermitian producer proof, with no coefficient snapshot.
use super::{
    CooperativeOperatorError, LinearOperator, OperatorWork, SourceScalar, TypedLinearOperator,
};
use crate::context::{host_allocation::HostAllocationReceipt, RuntimeContext};
use num_complex::Complex64;
async fn tick(work: &mut Option<OperatorWork<'_>>) -> Result<(), CooperativeOperatorError> {
    if let Some(work) = work.as_mut() {
        work.tick().await?;
    }
    Ok(())
}
async fn check<S: SourceScalar>(
    source: &TypedLinearOperator<'_, S>,
    work: &mut Option<OperatorWork<'_>>,
) -> Result<bool, CooperativeOperatorError> {
    if source.rows() != source.cols() {
        return Ok(false);
    }
    let n = source.rows();
    match source {
        TypedLinearOperator::Scalar(value) => {
            let value = value.complex();
            tick(work).await?;
            if !value.re.is_finite() || value.im != 0.0 {
                return Ok(false);
            }
        }
        TypedLinearOperator::Dense(matrix) => {
            let values = matrix.as_slice();
            for column in 0..n {
                for row in 0..=column {
                    let value = values[column * n + row].complex();
                    let reverse = values[row * n + column].complex().conj();
                    tick(work).await?;
                    if !value.re.is_finite() || !value.im.is_finite() || value != reverse {
                        return Ok(false);
                    }
                }
            }
        }
        TypedLinearOperator::Sparse(matrix) => {
            for column in 0..n {
                let (rows, values) = matrix.column(column);
                for (&row, &value) in rows.iter().zip(values) {
                    let value = value.complex();
                    let (reverse_rows, reverse_values) = matrix.column(row);
                    let (mut low, mut high) = (0, reverse_rows.len());
                    while low < high {
                        let middle = low + (high - low) / 2;
                        if reverse_rows[middle] < column {
                            low = middle + 1;
                        } else {
                            high = middle;
                        }
                        tick(work).await?;
                    }
                    let reverse = if low < reverse_rows.len() && reverse_rows[low] == column {
                        reverse_values[low].complex().conj()
                    } else {
                        Complex64::new(0.0, 0.0)
                    };
                    tick(work).await?;
                    if !value.re.is_finite() || !value.im.is_finite() || value != reverse {
                        return Ok(false);
                    }
                }
                tick(work).await?;
            }
        }
    }
    Ok(true)
}
impl LinearOperator<'_> {
    pub(crate) async fn is_hermitian(
        &self,
        runtime: Option<&RuntimeContext>,
    ) -> Result<bool, CooperativeOperatorError> {
        HostAllocationReceipt::policy(runtime)?;
        let mut work = runtime.map(OperatorWork::new).transpose()?;
        let result = match self {
            Self::RealF32(op) => check(op, &mut work).await?,
            Self::RealF64(op) => check(op, &mut work).await?,
            Self::ComplexF32(op) => check(op, &mut work).await?,
            Self::ComplexF64(op) => check(op, &mut work).await?,
        };
        if let Some(work) = work.as_mut() {
            work.flush().await?;
        }
        HostAllocationReceipt::policy(runtime)?;
        Ok(result)
    }
}
