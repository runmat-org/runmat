use super::scalar::LinearOperatorScalar;
use super::LinearOperatorError;
use crate::context::host_allocation::HostAllocationReceipt;
use crate::context::RuntimeContext;
use runmat_value::{ComplexElement, ComplexTensor, Tensor};

/// Borrowed canonical column-major dense host storage. The constructor checks
/// the complete shape once; a product only mutates caller-owned output.
#[derive(Clone, Copy)]
pub struct DenseOperator<'a, T> {
    rows: usize,
    cols: usize,
    values: &'a [T],
}

impl<'a, T> DenseOperator<'a, T> {
    /// Borrow an existing column-major payload without allocating metadata.
    pub fn from_column_major(
        values: &'a [T],
        rows: usize,
        cols: usize,
    ) -> Result<Self, LinearOperatorError> {
        if rows
            .checked_mul(cols)
            .ok_or(LinearOperatorError::Overflow)?
            != values.len()
        {
            return Err(LinearOperatorError::DimensionMismatch);
        }
        Ok(Self { rows, cols, values })
    }

    pub fn as_slice(&self) -> &'a [T] {
        self.values
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }
}

pub enum DenseLinearOperator<'a> {
    RealF32(DenseOperator<'a, f32>),
    RealF64(DenseOperator<'a, f64>),
    ComplexF32(DenseOperator<'a, ComplexElement<f32>>),
    ComplexF64(DenseOperator<'a, ComplexElement<f64>>),
}

impl<'a> DenseLinearOperator<'a> {
    pub fn borrow_real_with_context(
        tensor: &'a Tensor,
        context: &RuntimeContext,
    ) -> Result<Self, LinearOperatorError> {
        Self::borrow_real(tensor, Some(context))
    }

    pub fn borrow_real_unbounded(tensor: &'a Tensor) -> Result<Self, LinearOperatorError> {
        Self::borrow_real(tensor, None)
    }

    pub fn borrow_complex_with_context(
        tensor: &'a ComplexTensor,
        context: &RuntimeContext,
    ) -> Result<Self, LinearOperatorError> {
        Self::borrow_complex(tensor, Some(context))
    }

    pub fn borrow_complex_unbounded(
        tensor: &'a ComplexTensor,
    ) -> Result<Self, LinearOperatorError> {
        Self::borrow_complex(tensor, None)
    }

    fn borrow_real(
        tensor: &'a Tensor,
        context: Option<&RuntimeContext>,
    ) -> Result<Self, LinearOperatorError> {
        HostAllocationReceipt::policy(context)?;
        let (rows, cols) = validate_shape(tensor.shape(), tensor.len())?;
        let result = if let Some(values) = tensor.as_f32_slice() {
            Self::RealF32(DenseOperator { rows, cols, values })
        } else if let Some(values) = tensor.as_f64_slice() {
            Self::RealF64(DenseOperator { rows, cols, values })
        } else {
            return Err(LinearOperatorError::UnsupportedClass);
        };
        HostAllocationReceipt::policy(context)?;
        Ok(result)
    }

    fn borrow_complex(
        tensor: &'a ComplexTensor,
        context: Option<&RuntimeContext>,
    ) -> Result<Self, LinearOperatorError> {
        HostAllocationReceipt::policy(context)?;
        let (rows, cols) = validate_shape(tensor.shape(), tensor.len())?;
        let result = if let Some(values) = tensor.as_f32_slice() {
            Self::ComplexF32(DenseOperator { rows, cols, values })
        } else if let Some(values) = tensor.as_f64_slice() {
            Self::ComplexF64(DenseOperator { rows, cols, values })
        } else {
            return Err(LinearOperatorError::UnsupportedClass);
        };
        HostAllocationReceipt::policy(context)?;
        Ok(result)
    }

    pub fn rows(&self) -> usize {
        match self {
            Self::RealF32(view) => view.rows(),
            Self::RealF64(view) => view.rows(),
            Self::ComplexF32(view) => view.rows(),
            Self::ComplexF64(view) => view.rows(),
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::RealF32(view) => view.cols(),
            Self::RealF64(view) => view.cols(),
            Self::ComplexF32(view) => view.cols(),
            Self::ComplexF64(view) => view.cols(),
        }
    }
}

pub(in crate::builtins::math::linalg) fn validate_shape(
    shape: &[usize],
    elements: usize,
) -> Result<(usize, usize), LinearOperatorError> {
    if shape.iter().skip(2).any(|&extent| extent != 1) {
        return Err(LinearOperatorError::DimensionMismatch);
    }
    let rows = shape.first().copied().unwrap_or(1);
    let cols = shape.get(1).copied().unwrap_or(1);
    let expected = rows
        .checked_mul(cols)
        .ok_or(LinearOperatorError::Overflow)?;
    if expected != elements {
        return Err(LinearOperatorError::DimensionMismatch);
    }
    Ok((rows, cols))
}

#[derive(Clone, Copy)]
enum Orientation {
    Normal,
    Transpose,
    Adjoint,
}

fn apply<T: LinearOperatorScalar>(
    operator: &DenseOperator<'_, T>,
    input: &[T],
    output: &mut [T],
    context: Option<&RuntimeContext>,
    orientation: Orientation,
) -> Result<(), LinearOperatorError> {
    let (input_len, output_len) = match orientation {
        Orientation::Normal => (operator.cols, operator.rows),
        Orientation::Transpose | Orientation::Adjoint => (operator.rows, operator.cols),
    };
    if input.len() != input_len || output.len() != output_len {
        return Err(LinearOperatorError::DimensionMismatch);
    }
    HostAllocationReceipt::policy(context)?;
    for chunk in output.chunks_mut(1024) {
        HostAllocationReceipt::policy(context)?;
        chunk.fill(T::zero());
    }
    for (col, column) in operator.values.chunks(operator.rows.max(1)).enumerate() {
        if col % 256 == 0 {
            HostAllocationReceipt::policy(context)?;
        }
        for (row, &coefficient) in column.iter().enumerate() {
            if (col * operator.rows + row).is_multiple_of(1024) {
                HostAllocationReceipt::policy(context)?;
            }
            match orientation {
                Orientation::Normal => {
                    T::add_product(&mut output[row], coefficient, input[col]);
                }
                Orientation::Transpose => {
                    T::add_product(&mut output[col], coefficient, input[row]);
                }
                Orientation::Adjoint => {
                    T::add_product(&mut output[col], coefficient.conjugate(), input[row]);
                }
            }
        }
    }
    HostAllocationReceipt::policy(context)?;
    Ok(())
}

macro_rules! product_methods {
    ($scalar:ty) => {
        impl DenseOperator<'_, $scalar> {
            pub fn apply_into(
                &self,
                input: &[$scalar],
                output: &mut [$scalar],
                context: &RuntimeContext,
            ) -> Result<(), LinearOperatorError> {
                apply(self, input, output, Some(context), Orientation::Normal)
            }

            pub fn apply_transpose_into(
                &self,
                input: &[$scalar],
                output: &mut [$scalar],
                context: &RuntimeContext,
            ) -> Result<(), LinearOperatorError> {
                apply(self, input, output, Some(context), Orientation::Transpose)
            }

            pub fn apply_adjoint_into(
                &self,
                input: &[$scalar],
                output: &mut [$scalar],
                context: &RuntimeContext,
            ) -> Result<(), LinearOperatorError> {
                apply(self, input, output, Some(context), Orientation::Adjoint)
            }

            pub fn apply_into_unbounded(
                &self,
                input: &[$scalar],
                output: &mut [$scalar],
            ) -> Result<(), LinearOperatorError> {
                apply(self, input, output, None, Orientation::Normal)
            }

            pub fn apply_transpose_into_unbounded(
                &self,
                input: &[$scalar],
                output: &mut [$scalar],
            ) -> Result<(), LinearOperatorError> {
                apply(self, input, output, None, Orientation::Transpose)
            }

            pub fn apply_adjoint_into_unbounded(
                &self,
                input: &[$scalar],
                output: &mut [$scalar],
            ) -> Result<(), LinearOperatorError> {
                apply(self, input, output, None, Orientation::Adjoint)
            }
        }
    };
}

product_methods!(f32);
product_methods!(f64);
product_methods!(ComplexElement<f32>);
product_methods!(ComplexElement<f64>);
