//! Borrowed four-class products with an explicit caller-selected conversion.
//! Conversion visits only stored CSC coefficients (or actual dense entries),
//! never materializes another matrix, and retains native target arithmetic.
use super::OperatorWork;
use super::{
    scalar::LinearOperatorScalar, CooperativeOperatorError, LinearOperator, LinearOperatorError,
    TypedLinearOperator,
};
use super::{BorrowedWork, ProductWork, UnmanagedWork};
use crate::context::{host_allocation::HostAllocationReceipt, RuntimeContext};
use crate::execution::CpuCooperationBudget;
use num_complex::Complex64;
use runmat_value::ComplexElement;

pub(in crate::builtins::math::linalg) trait SourceScalar:
    Copy
{
    fn complex(self) -> Complex64;
}
#[derive(Clone, Copy)]
pub(crate) enum ProductOrientation {
    Normal,
    Adjoint,
}
impl SourceScalar for f32 {
    fn complex(self) -> Complex64 {
        Complex64::new(self as f64, 0.0)
    }
}
impl SourceScalar for f64 {
    fn complex(self) -> Complex64 {
        Complex64::new(self, 0.0)
    }
}
impl SourceScalar for ComplexElement<f32> {
    fn complex(self) -> Complex64 {
        Complex64::new(self.0 as f64, self.1 as f64)
    }
}
impl SourceScalar for ComplexElement<f64> {
    fn complex(self) -> Complex64 {
        Complex64::new(self.0, self.1)
    }
}

fn accumulate<T: LinearOperatorScalar>(
    value: Complex64,
    row: usize,
    column: usize,
    input: &[T],
    output: &mut [T],
    convert: fn(Complex64) -> Option<T>,
    orientation: ProductOrientation,
) -> Result<(), CooperativeOperatorError> {
    let (value, destination, source) = match orientation {
        ProductOrientation::Normal => (value, row, column),
        ProductOrientation::Adjoint => (value.conj(), column, row),
    };
    let coefficient = convert(value).ok_or(CooperativeOperatorError::Operator(
        LinearOperatorError::UnsupportedClass,
    ))?;
    T::add_product(&mut output[destination], coefficient, input[source]);
    Ok(())
}
async fn product<S: SourceScalar, T: LinearOperatorScalar>(
    source: &TypedLinearOperator<'_, S>,
    input: &[T],
    output: &mut [T],
    convert: fn(Complex64) -> Option<T>,
    runtime: Option<&RuntimeContext>,
    orientation: ProductOrientation,
    budget: Option<&mut CpuCooperationBudget>,
) -> Result<(), CooperativeOperatorError> {
    let (rows, columns) = match orientation {
        ProductOrientation::Normal => (source.rows(), source.cols()),
        ProductOrientation::Adjoint => (source.cols(), source.rows()),
    };
    if input.len() != columns || output.len() != rows {
        return Err(CooperativeOperatorError::Operator(
            LinearOperatorError::DimensionMismatch,
        ));
    }
    HostAllocationReceipt::policy(runtime)?;
    match (runtime, budget) {
        (Some(runtime), Some(budget)) => {
            if !budget
                .cancellation_token()
                .same_generation(&runtime.cancellation())
            {
                return Err(
                    crate::context::host_allocation::HostAllocationError::StaleAuthority.into(),
                );
            }
            budget
                .checkpoint(0)
                .await
                .map_err(CooperativeOperatorError::Execution)?;
            HostAllocationReceipt::policy(Some(runtime))?;
            product_values(
                source,
                input,
                output,
                convert,
                orientation,
                &mut BorrowedWork::new(runtime, budget)?,
            )
            .await?;
        }
        (Some(runtime), None) => {
            let mut work = OperatorWork::new(runtime)?;
            product_values(source, input, output, convert, orientation, &mut work).await?;
        }
        (None, None) => {
            product_values(
                source,
                input,
                output,
                convert,
                orientation,
                &mut UnmanagedWork,
            )
            .await?;
        }
        (None, Some(_)) => {
            return Err(crate::context::host_allocation::HostAllocationError::StaleAuthority.into())
        }
    };
    HostAllocationReceipt::policy(runtime)?;
    Ok(())
}
async fn product_values<S: SourceScalar, T: LinearOperatorScalar>(
    source: &TypedLinearOperator<'_, S>,
    input: &[T],
    output: &mut [T],
    convert: fn(Complex64) -> Option<T>,
    orientation: ProductOrientation,
    work: &mut impl ProductWork,
) -> Result<(), CooperativeOperatorError> {
    for value in output.iter_mut() {
        *value = T::zero();
        work.tick().await?;
    }
    for column in 0..source.cols() {
        work.tick().await?;
        match source {
            TypedLinearOperator::Scalar(value) => {
                accumulate(value.complex(), 0, 0, input, output, convert, orientation)?
            }
            TypedLinearOperator::Dense(matrix) => {
                for (row, &value) in matrix.as_slice()
                    [column * source.rows()..(column + 1) * source.rows()]
                    .iter()
                    .enumerate()
                {
                    accumulate(
                        value.complex(),
                        row,
                        column,
                        input,
                        output,
                        convert,
                        orientation,
                    )?;
                    work.tick().await?;
                }
            }
            TypedLinearOperator::Sparse(matrix) => {
                let (rows, values) = matrix.column(column);
                for (&row, &value) in rows.iter().zip(values) {
                    accumulate(
                        value.complex(),
                        row,
                        column,
                        input,
                        output,
                        convert,
                        orientation,
                    )?;
                    work.tick().await?;
                }
            }
        }
    }
    work.flush().await?;
    Ok(())
}
impl LinearOperator<'_> {
    /// Borrow the invocation's existing budget and original numeric backing.
    /// This entry has no complex-to-real coercion or new budget authority.
    pub(crate) async fn apply_real_with_budget(
        &self,
        input: &[f64],
        output: &mut [f64],
        runtime: Option<&RuntimeContext>,
        budget: Option<&mut CpuCooperationBudget>,
        orientation: ProductOrientation,
    ) -> Result<(), CooperativeOperatorError> {
        fn real(value: Complex64) -> Option<f64> {
            (value.im == 0.).then_some(value.re)
        }
        match self {
            Self::RealF32(op) => {
                product(op, input, output, real, runtime, orientation, budget).await
            }
            Self::RealF64(op) => {
                product(op, input, output, real, runtime, orientation, budget).await
            }
            _ => Err(CooperativeOperatorError::Operator(
                LinearOperatorError::UnsupportedClass,
            )),
        }
    }

    pub(in crate::builtins::math::linalg) async fn apply_converted<T: LinearOperatorScalar>(
        &self,
        input: &[T],
        output: &mut [T],
        convert: fn(Complex64) -> Option<T>,
        runtime: Option<&RuntimeContext>,
    ) -> Result<(), CooperativeOperatorError> {
        self.apply_oriented_converted(input, output, convert, runtime, ProductOrientation::Normal)
            .await
    }
    pub(in crate::builtins::math::linalg) async fn apply_oriented_converted<
        T: LinearOperatorScalar,
    >(
        &self,
        input: &[T],
        output: &mut [T],
        convert: fn(Complex64) -> Option<T>,
        runtime: Option<&RuntimeContext>,
        orientation: ProductOrientation,
    ) -> Result<(), CooperativeOperatorError> {
        match self {
            Self::RealF32(op) => {
                product(op, input, output, convert, runtime, orientation, None).await
            }
            Self::RealF64(op) => {
                product(op, input, output, convert, runtime, orientation, None).await
            }
            Self::ComplexF32(op) => {
                product(op, input, output, convert, runtime, orientation, None).await
            }
            Self::ComplexF64(op) => {
                product(op, input, output, convert, runtime, orientation, None).await
            }
        }
    }
}
