//! Borrowed native ingress and bounded structural inspection. Dense inputs
//! are never cloned/materialized; sparse transforms visit CSC storage only.
use super::super::managed_storage::Work;
use super::{RetainedFactorError as Error, RetainedFactorScalar};
use crate::context::{
    host_allocation::{HostAllocationPlan, HostAllocationReceipt},
    RuntimeContext,
};
use runmat_value::{ComplexElement, SparseCscValidation, SparseTensor, Value};
use std::num::NonZeroUsize;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::builtins::math::linalg) enum SelectedTriangle {
    Upper,
    Lower,
}
#[derive(Clone, Copy, Debug)]
pub(in crate::builtins::math::linalg) struct NativeMatrixFacts {
    pub diagonal: bool,
    pub upper: bool,
    pub lower: bool,
    pub real: bool,
    pub hermitian: bool,
}

fn dimensions(value: &Value) -> Result<(usize, usize), Error> {
    let shape = match value {
        Value::SparseTensor(matrix) => return Ok((matrix.rows, matrix.cols)),
        Value::Tensor(matrix) => matrix.shape(),
        Value::ComplexTensor(matrix) => matrix.shape(),
        _ => return Err(Error::UnsupportedClass),
    };
    if shape.iter().skip(2).any(|&extent| extent != 1) {
        return Err(Error::Invalid("coefficient input must be a matrix".into()));
    }
    Ok((
        shape.first().copied().unwrap_or(1),
        shape.get(1).copied().unwrap_or(1),
    ))
}

macro_rules! dispatch {
    ($value:expr, $function:ident, $($argument:expr),*) => {
        match $value {
            Value::Tensor(matrix) if matrix.as_f32_slice().is_some() => $function::<f32>($value, $($argument),*).await,
            Value::Tensor(matrix) if matrix.as_f64_slice().is_some() => $function::<f64>($value, $($argument),*).await,
            Value::ComplexTensor(matrix) if matrix.as_f32_slice().is_some() => $function::<ComplexElement<f32>>($value, $($argument),*).await,
            Value::ComplexTensor(matrix) if matrix.as_f64_slice().is_some() => $function::<ComplexElement<f64>>($value, $($argument),*).await,
            Value::SparseTensor(matrix) if matrix.as_f32_slice().is_some() => $function::<f32>($value, $($argument),*).await,
            Value::SparseTensor(matrix) if matrix.as_f64_slice().is_some() => $function::<f64>($value, $($argument),*).await,
            Value::SparseTensor(matrix) if matrix.as_complex_f32_slice().is_some() => $function::<ComplexElement<f32>>($value, $($argument),*).await,
            Value::SparseTensor(matrix) if matrix.as_complex_f64_slice().is_some() => $function::<ComplexElement<f64>>($value, $($argument),*).await,
            _ => Err(Error::UnsupportedClass),
        }
    };
}

pub(super) async fn validate_sparse(
    matrix: &SparseTensor,
    work: &mut Work<'_>,
) -> Result<(), Error> {
    let mut validation = SparseCscValidation::new(
        matrix.rows,
        matrix.cols,
        matrix.col_ptrs.clone(),
        matrix.row_indices.clone(),
        matrix.nnz(),
    )
    .map_err(Error::Invalid)?;
    loop {
        let count = work.capacity();
        let done = validation
            .advance(NonZeroUsize::new(count).expect("bounded sparse input work"))
            .map_err(Error::Invalid)?;
        work.charge(count).await?;
        if done {
            return Ok(());
        }
    }
}

pub(in crate::builtins::math::linalg) async fn inspect_native_matrix(
    value: &Value,
    check_hermitian: bool,
    runtime: &RuntimeContext,
) -> Result<NativeMatrixFacts, Error> {
    dispatch!(value, inspect, check_hermitian, runtime)
}

fn record(
    facts: &mut NativeMatrixFacts,
    row: usize,
    column: usize,
    value: (f64, f64),
    opposite: (f64, f64),
    check_hermitian: bool,
) {
    facts.real &= value.1 == 0.0;
    if value != (0.0, 0.0) {
        facts.diagonal &= row == column;
        facts.upper &= row <= column;
        facts.lower &= row >= column;
    }
    if check_hermitian {
        facts.hermitian &= value == (opposite.0, -opposite.1);
    }
}

async fn inspect<T: RetainedFactorScalar>(
    value: &Value,
    check_hermitian: bool,
    runtime: &RuntimeContext,
) -> Result<NativeMatrixFacts, Error> {
    let (rows, columns) = dimensions(value)?;
    let mut facts = NativeMatrixFacts {
        diagonal: true,
        upper: true,
        lower: true,
        real: true,
        hermitian: rows == columns,
    };
    let receipt = HostAllocationReceipt::admit(Some(runtime), HostAllocationPlan::new(0, 0)?)?;
    let mut work = Work::for_policy(runtime, receipt.as_ref())?;
    work.flush().await?;
    if let Value::SparseTensor(matrix) = value {
        validate_sparse(matrix, &mut work).await?;
        for column in 0..columns {
            for index in matrix.col_ptrs[column]..matrix.col_ptrs[column + 1] {
                let row = matrix.row_indices[index];
                let mut opposite = T::default();
                if check_hermitian && rows == columns {
                    let (mut low, mut high) = (matrix.col_ptrs[row], matrix.col_ptrs[row + 1]);
                    while low < high {
                        let mid = low + (high - low) / 2;
                        match matrix.row_indices[mid].cmp(&column) {
                            std::cmp::Ordering::Less => low = mid + 1,
                            std::cmp::Ordering::Greater => high = mid,
                            std::cmp::Ordering::Equal => {
                                opposite = T::read(matrix, mid);
                                break;
                            }
                        }
                        work.tick().await?;
                    }
                }
                record(
                    &mut facts,
                    row,
                    column,
                    T::read(matrix, index).parts(),
                    opposite.parts(),
                    check_hermitian,
                );
                work.tick().await?;
            }
            work.tick().await?;
        }
    } else {
        let input = T::dense(value).ok_or(Error::UnsupportedClass)?;
        if rows.checked_mul(columns) != Some(input.len()) {
            return Err(Error::Invalid(
                "dense native coefficient extent mismatch".into(),
            ));
        }
        for (index, &entry) in input.iter().enumerate() {
            let (row, column) = (index % rows, index / rows);
            let opposite = if check_hermitian && rows == columns {
                input[column + row * rows].parts()
            } else {
                (0.0, 0.0)
            };
            record(
                &mut facts,
                row,
                column,
                entry.parts(),
                opposite,
                check_hermitian,
            );
            work.tick().await?;
        }
    }
    work.flush().await?;
    Ok(facts)
}
