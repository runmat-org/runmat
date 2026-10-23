//! Packed native dense-factor geometry, independent of sparse CSC topology.
use super::*;
use crate::value::{DenseValue, DurableDenseOrthogonalFactor};

pub(super) fn geometry(value: &DurableNumericFactor) -> Result<(), ContractError> {
    match value {
        DurableNumericFactor::DenseLu { packed, rows } => {
            let (m, n) = matrix(packed)?;
            if m != n || !permutation_extent(rows, m) {
                return Err(invalid("numerical factor", "dense LU geometry differs"));
            }
        }
        DurableNumericFactor::DenseCholesky { lower } => {
            let (rows, columns) = matrix(lower)?;
            if rows != columns {
                return Err(invalid(
                    "numerical factor",
                    "dense Cholesky geometry differs",
                ));
            }
        }
        DurableNumericFactor::DenseQr(first) => orthogonal(first)?,
        DurableNumericFactor::DenseCod { first, completion } => {
            orthogonal(first)?;
            let (_, columns) = matrix(&first.packed)?;
            if let Some(second) = completion {
                orthogonal(second)?;
                let (rows, cols) = matrix(&second.packed)?;
                if first.rank == 0
                    || first.rank >= columns
                    || rows != columns
                    || cols != first.rank
                    || second.rank != first.rank
                    || second.packed.element_type != first.packed.element_type
                {
                    return Err(invalid(
                        "numerical factor",
                        "dense COD completion geometry differs",
                    ));
                }
            } else if first.rank != 0 && first.rank != columns {
                return Err(invalid(
                    "numerical factor",
                    "rank-deficient dense COD has no completion",
                ));
            }
        }
        _ => unreachable!("only dense factors reach dense geometry"),
    }
    Ok(())
}

fn matrix(value: &DenseValue) -> Result<(u64, u64), ContractError> {
    let [rows, columns, rest @ ..] = value.shape.as_slice() else {
        return Err(invalid("numerical factor", "dense factor must be a matrix"));
    };
    if rest.iter().any(|&dimension| dimension != 1)
        || *rows == 0
        || *columns == 0
        || !matches!(
            value.element_type,
            ElementType::F32 | ElementType::F64 | ElementType::ComplexF32 | ElementType::ComplexF64
        )
    {
        return Err(invalid(
            "numerical factor",
            "dense factor class or dimensions differ",
        ));
    }
    Ok((*rows, *columns))
}

fn orthogonal(value: &DurableDenseOrthogonalFactor) -> Result<(), ContractError> {
    let (rows, columns) = matrix(&value.packed)?;
    let (block_size, coefficient_columns) = matrix(&value.coefficients)?;
    if block_size == 0
        || block_size > rows.min(columns)
        || coefficient_columns != rows.min(columns)
        || value.coefficients.element_type != value.packed.element_type
        || value.rank > rows.min(columns)
        || !permutation_extent(&value.permutation, columns)
    {
        return Err(invalid(
            "numerical factor",
            "dense orthogonal geometry differs",
        ));
    }
    Ok(())
}

pub(in crate::value::validation::cursor::durable::numeric_factor) fn block(
    value: &DurableNumericFactor,
    index: usize,
) -> Option<&DenseValue> {
    match value {
        DurableNumericFactor::DenseLu { packed, .. } => (index == 0).then_some(packed),
        DurableNumericFactor::DenseCholesky { lower } => (index == 0).then_some(lower),
        DurableNumericFactor::DenseQr(first) => orthogonal_block(first, index),
        DurableNumericFactor::DenseCod { first, completion } => {
            if index < 2 {
                orthogonal_block(first, index)
            } else {
                completion
                    .as_ref()
                    .and_then(|second| orthogonal_block(second, index - 2))
            }
        }
        _ => None,
    }
}

fn orthogonal_block(value: &DurableDenseOrthogonalFactor, index: usize) -> Option<&DenseValue> {
    match index {
        0 => Some(&value.packed),
        1 => Some(&value.coefficients),
        _ => None,
    }
}
