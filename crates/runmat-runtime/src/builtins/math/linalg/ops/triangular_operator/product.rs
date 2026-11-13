use super::scalar::TriangularScalar;
use super::{Backing, Diagonal, Triangle, TriangularOperator, TriangularOperatorError};
use crate::context::host_allocation::HostAllocationReceipt;
use crate::context::RuntimeContext;
use runmat_value::ComplexElement;

#[derive(Clone, Copy)]
enum Orientation {
    Normal,
    #[cfg(test)]
    Transpose,
    #[cfg(test)]
    Adjoint,
}

fn forward(triangle: Triangle, orientation: Orientation) -> bool {
    match (triangle, orientation) {
        (Triangle::Lower, Orientation::Normal) => true,
        #[cfg(test)]
        (Triangle::Upper, Orientation::Transpose | Orientation::Adjoint) => true,
        _ => false,
    }
}

fn adjoint(orientation: Orientation) -> bool {
    match orientation {
        #[cfg(test)]
        Orientation::Adjoint => true,
        _ => false,
    }
}

pub(super) fn diagonal_divide<T: TriangularScalar>(
    value: T,
    diagonal: T,
    mode: Diagonal,
) -> Result<T, TriangularOperatorError> {
    let result = match mode {
        #[cfg(test)]
        Diagonal::Unit => value,
        Diagonal::Explicit => value
            .divide(diagonal)
            .ok_or(TriangularOperatorError::NumericalBreakdown)?,
    };
    result
        .is_finite()
        .then_some(result)
        .ok_or(TriangularOperatorError::NumericalBreakdown)
}

fn dense_solve<T: TriangularScalar>(
    operator: &TriangularOperator<'_, T>,
    coefficients: &[T],
    input: &[T],
    output: &mut [T],
    context: Option<&RuntimeContext>,
    orientation: Orientation,
) -> Result<(), TriangularOperatorError> {
    let n = operator.order;
    let forward = forward(operator.triangle, orientation);
    let mut examined = 0usize;
    for step in 0..n {
        HostAllocationReceipt::policy(context)?;
        let row = if forward { step } else { n - 1 - step };
        let mut value = input[row];
        for previous in 0..step {
            examined += 1;
            if examined.is_multiple_of(1024) {
                HostAllocationReceipt::policy(context)?;
            }
            let col = if forward { previous } else { n - 1 - previous };
            let coefficient = match orientation {
                Orientation::Normal => coefficients[col * n + row],
                #[cfg(test)]
                Orientation::Transpose => coefficients[row * n + col],
                #[cfg(test)]
                Orientation::Adjoint => coefficients[row * n + col].conjugate(),
            };
            if coefficient.is_zero() {
                continue;
            }
            T::subtract_product(&mut value, coefficient, output[col]);
            if !value.is_finite() {
                return Err(TriangularOperatorError::NumericalBreakdown);
            }
        }
        let diagonal = coefficients[row * n + row];
        let diagonal = if adjoint(orientation) {
            diagonal.conjugate()
        } else {
            diagonal
        };
        output[row] = diagonal_divide(value, diagonal, operator.diagonal)?;
    }
    HostAllocationReceipt::policy(context)?;
    Ok(())
}

fn csc_solve<T: TriangularScalar>(
    operator: &TriangularOperator<'_, T>,
    col_ptrs: &[usize],
    row_indices: &[usize],
    coefficients: &[T],
    input: &[T],
    output: &mut [T],
    context: Option<&RuntimeContext>,
    orientation: Orientation,
) -> Result<(), TriangularOperatorError> {
    let n = operator.order;
    let forward = forward(operator.triangle, orientation);
    if matches!(orientation, Orientation::Normal) {
        for (source, destination) in input.chunks(1024).zip(output.chunks_mut(1024)) {
            HostAllocationReceipt::policy(context)?;
            destination.copy_from_slice(source);
        }
    }
    for step in 0..n {
        HostAllocationReceipt::policy(context)?;
        let col = if forward { step } else { n - 1 - step };
        let mut diagonal = T::one();
        if !matches!(orientation, Orientation::Normal) {
            output[col] = input[col];
        }
        for index in col_ptrs[col]..col_ptrs[col + 1] {
            if index.is_multiple_of(1024) {
                HostAllocationReceipt::policy(context)?;
            }
            let row = row_indices[index];
            let coefficient = coefficients[index];
            if row == col {
                diagonal = coefficient;
            } else if !matches!(orientation, Orientation::Normal) && !coefficient.is_zero() {
                let coefficient = if adjoint(orientation) {
                    coefficient.conjugate()
                } else {
                    coefficient
                };
                let previous = output[row];
                T::subtract_product(&mut output[col], coefficient, previous);
                if !output[col].is_finite() {
                    return Err(TriangularOperatorError::NumericalBreakdown);
                }
            }
        }
        if adjoint(orientation) {
            diagonal = diagonal.conjugate();
        }
        output[col] = diagonal_divide(output[col], diagonal, operator.diagonal)?;
        if matches!(orientation, Orientation::Normal) {
            let solved = output[col];
            for index in col_ptrs[col]..col_ptrs[col + 1] {
                if index.is_multiple_of(1024) {
                    HostAllocationReceipt::policy(context)?;
                }
                let row = row_indices[index];
                if row == col || coefficients[index].is_zero() {
                    continue;
                }
                T::subtract_product(&mut output[row], coefficients[index], solved);
                if !output[row].is_finite() {
                    return Err(TriangularOperatorError::NumericalBreakdown);
                }
            }
        }
    }
    HostAllocationReceipt::policy(context)?;
    Ok(())
}

fn apply<T: TriangularScalar>(
    operator: &TriangularOperator<'_, T>,
    input: &[T],
    output: &mut [T],
    context: Option<&RuntimeContext>,
    orientation: Orientation,
) -> Result<(), TriangularOperatorError> {
    if input.len() != operator.order || output.len() != operator.order {
        return Err(TriangularOperatorError::DimensionMismatch);
    }
    for chunk in input.chunks(1024) {
        HostAllocationReceipt::policy(context)?;
        if chunk.iter().any(|&value| !value.is_finite()) {
            return Err(TriangularOperatorError::NumericalBreakdown);
        }
    }
    match operator.backing {
        Backing::Dense(values) => {
            dense_solve(operator, values, input, output, context, orientation)
        }
        Backing::Csc {
            col_ptrs,
            row_indices,
            values,
        } => csc_solve(
            operator,
            col_ptrs,
            row_indices,
            values,
            input,
            output,
            context,
            orientation,
        ),
    }
}

macro_rules! product_methods {
    ($scalar:ty) => {
        impl TriangularOperator<'_, $scalar> {
            pub fn apply_into(
                &self,
                input: &[$scalar],
                output: &mut [$scalar],
                context: &RuntimeContext,
            ) -> Result<(), TriangularOperatorError> {
                apply(self, input, output, Some(context), Orientation::Normal)
            }
        }
    };
}

product_methods!(f32);
product_methods!(f64);
product_methods!(ComplexElement<f32>);
product_methods!(ComplexElement<f64>);

#[cfg(test)]
pub(super) fn apply_into_unbounded<T: TriangularScalar>(
    operator: &TriangularOperator<'_, T>,
    input: &[T],
    output: &mut [T],
) -> Result<(), TriangularOperatorError> {
    apply(operator, input, output, None, Orientation::Normal)
}

#[cfg(test)]
pub(super) fn apply_transpose_into_unbounded<T: TriangularScalar>(
    operator: &TriangularOperator<'_, T>,
    input: &[T],
    output: &mut [T],
) -> Result<(), TriangularOperatorError> {
    apply(operator, input, output, None, Orientation::Transpose)
}

#[cfg(test)]
pub(super) fn apply_adjoint_into_unbounded<T: TriangularScalar>(
    operator: &TriangularOperator<'_, T>,
    input: &[T],
    output: &mut [T],
) -> Result<(), TriangularOperatorError> {
    apply(operator, input, output, None, Orientation::Adjoint)
}
