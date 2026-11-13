use super::{CscOperator, SparseOperatorError};
use crate::builtins::math::linalg::ops::linear_operator::scalar::LinearOperatorScalar;
use crate::context::host_allocation::HostAllocationReceipt;
use crate::context::RuntimeContext;
use runmat_value::ComplexElement;

#[derive(Clone, Copy)]
enum Orientation {
    Normal,
    Transpose,
    Adjoint,
}

fn apply<T: LinearOperatorScalar>(
    operator: &CscOperator<'_, T>,
    input: &[T],
    output: &mut [T],
    context: Option<&RuntimeContext>,
    orientation: Orientation,
) -> Result<(), SparseOperatorError> {
    let (input_len, output_len) = match orientation {
        Orientation::Normal => (operator.cols, operator.rows),
        Orientation::Transpose | Orientation::Adjoint => (operator.rows, operator.cols),
    };
    if input.len() != input_len || output.len() != output_len {
        return Err(SparseOperatorError::DimensionMismatch);
    }
    HostAllocationReceipt::policy(context)?;
    for chunk in output.chunks_mut(1024) {
        HostAllocationReceipt::policy(context)?;
        chunk.fill(T::zero());
    }
    for col in 0..operator.cols {
        if col % 256 == 0 {
            HostAllocationReceipt::policy(context)?;
        }
        for index in operator.col_ptrs[col]..operator.col_ptrs[col + 1] {
            if index % 1024 == 0 {
                HostAllocationReceipt::policy(context)?;
            }
            let row = operator.row_indices[index];
            let coefficient = operator.values[index];
            match orientation {
                Orientation::Normal => T::add_product(&mut output[row], coefficient, input[col]),
                Orientation::Transpose => T::add_product(&mut output[col], coefficient, input[row]),
                Orientation::Adjoint => {
                    T::add_product(&mut output[col], coefficient.conjugate(), input[row]);
                }
            }
        }
    }
    HostAllocationReceipt::policy(context)?;
    Ok(())
}

macro_rules! product_method {
    ($scalar:ty) => {
        impl CscOperator<'_, $scalar> {
            /// Compute `A*x` into an already-owned result vector. No payload or
            /// metadata allocation occurs, and cancellation leaves only this
            /// caller-owned scratch partially written.
            pub fn apply_into(
                &self,
                input: &[$scalar],
                output: &mut [$scalar],
                context: &RuntimeContext,
            ) -> Result<(), SparseOperatorError> {
                apply(self, input, output, Some(context), Orientation::Normal)
            }

            /// Explicitly unbounded compatibility path. A managed caller must
            /// use `apply_into` with its selected RuntimeContext.
            pub fn apply_into_unbounded(
                &self,
                input: &[$scalar],
                output: &mut [$scalar],
            ) -> Result<(), SparseOperatorError> {
                apply(self, input, output, None, Orientation::Normal)
            }

            pub fn apply_transpose_into(
                &self,
                input: &[$scalar],
                output: &mut [$scalar],
                context: &RuntimeContext,
            ) -> Result<(), SparseOperatorError> {
                apply(self, input, output, Some(context), Orientation::Transpose)
            }

            pub fn apply_adjoint_into(
                &self,
                input: &[$scalar],
                output: &mut [$scalar],
                context: &RuntimeContext,
            ) -> Result<(), SparseOperatorError> {
                apply(self, input, output, Some(context), Orientation::Adjoint)
            }

            pub fn apply_transpose_into_unbounded(
                &self,
                input: &[$scalar],
                output: &mut [$scalar],
            ) -> Result<(), SparseOperatorError> {
                apply(self, input, output, None, Orientation::Transpose)
            }

            pub fn apply_adjoint_into_unbounded(
                &self,
                input: &[$scalar],
                output: &mut [$scalar],
            ) -> Result<(), SparseOperatorError> {
                apply(self, input, output, None, Orientation::Adjoint)
            }
        }
    };
}

product_method!(f32);
product_method!(f64);
product_method!(ComplexElement<f32>);
product_method!(ComplexElement<f64>);
