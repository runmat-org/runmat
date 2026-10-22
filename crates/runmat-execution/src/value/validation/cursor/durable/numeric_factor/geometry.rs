//! Closed factor geometry and typed block/map lookup shared by wire validation.
use super::super::*;
use crate::value::{DurableEmptyFactorKind, DurableFactorPermutation, DurableOrthogonalFactor};
#[path = "geometry/dense.rs"]
mod dense;
pub(super) use dense::block as factor_dense_block;

pub(super) fn geometry(value: &DurableNumericFactor) -> Result<(), ContractError> {
    if matches!(
        value,
        DurableNumericFactor::DenseLu { .. }
            | DurableNumericFactor::DenseCholesky { .. }
            | DurableNumericFactor::DenseQr(_)
            | DurableNumericFactor::DenseCod { .. }
    ) {
        return dense::geometry(value);
    }
    if let DurableNumericFactor::Empty {
        kind,
        rows,
        columns,
        class,
    } = value
    {
        if *rows != 0 && *columns != 0 {
            return Err(invalid(
                "numerical factor",
                "empty factor has nonempty geometry",
            ));
        }
        if !matches!(
            class,
            ElementType::F32 | ElementType::F64 | ElementType::ComplexF32 | ElementType::ComplexF64
        ) {
            return Err(invalid(
                "numerical factor",
                "empty factor class is not floating",
            ));
        }
        if matches!(
            kind,
            DurableEmptyFactorKind::Lu | DurableEmptyFactorKind::Cholesky
        ) && rows != columns
        {
            return Err(invalid(
                "numerical factor",
                "empty square factor geometry differs",
            ));
        }
        return Ok(());
    }
    let first = factor_block(value, 0)
        .ok_or_else(|| invalid("numerical factor", "missing factor block"))?;
    let class = first.element_type;
    for index in 0..4 {
        if let Some(block) = factor_block(value, index) {
            if block.element_type != class {
                return Err(invalid(
                    "numerical factor",
                    "factor blocks have different classes",
                ));
            }
        }
    }
    match value {
        DurableNumericFactor::Empty { .. } => unreachable!("handled empty factor"),
        DurableNumericFactor::DenseLu { .. }
        | DurableNumericFactor::DenseCholesky { .. }
        | DurableNumericFactor::DenseQr(_)
        | DurableNumericFactor::DenseCod { .. } => unreachable!("handled dense factor"),
        DurableNumericFactor::Lu {
            lower,
            upper,
            rows,
            columns,
            row_scale,
        } => {
            if lower.rows != lower.columns
                || upper.rows != lower.rows
                || upper.columns != lower.columns
                || !permutation_extent(rows, lower.rows)
                || !permutation_extent(columns, lower.rows)
                || row_scale.as_ref().is_some_and(|scale| {
                    scale.shape.as_slice() != [lower.rows, 1]
                        || Some(scale.element_type) != real_class(class)
                })
            {
                return Err(invalid(
                    "numerical factor",
                    "LU geometry or row scale differs",
                ));
            }
        }
        DurableNumericFactor::Cholesky { lower, permutation } => {
            if lower.rows != lower.columns || !permutation_extent(permutation, lower.rows) {
                return Err(invalid("numerical factor", "Cholesky geometry differs"));
            }
        }
        DurableNumericFactor::Qr(first) => orthogonal_geometry(first)?,
        DurableNumericFactor::Cod { first, completion } => {
            orthogonal_geometry(first)?;
            if let Some(second) = completion {
                orthogonal_geometry(second)?;
                if first.rank == 0
                    || first.rank >= first.r.columns
                    || second.r.rows != first.r.columns
                    || second.r.columns != first.rank
                    || second.rank != first.rank
                {
                    return Err(invalid(
                        "numerical factor",
                        "COD completion geometry differs",
                    ));
                }
            } else if first.rank != 0 && first.rank != first.r.columns {
                return Err(invalid(
                    "numerical factor",
                    "rank-deficient COD has no completion",
                ));
            }
        }
    }
    Ok(())
}

fn orthogonal_geometry(value: &DurableOrthogonalFactor) -> Result<(), ContractError> {
    let extent = value.r.rows.min(value.r.columns);
    if value.rank > extent
        || value.reflectors.rows != value.r.rows
        || value.reflectors.columns != extent
        || !permutation_extent(&value.permutation, value.r.columns)
    {
        return Err(invalid(
            "numerical factor",
            "orthogonal factor geometry differs",
        ));
    }
    Ok(())
}

fn permutation_extent(value: &DurableFactorPermutation, count: u64) -> bool {
    u64::try_from(value.forward.len()) == Ok(count)
        && u64::try_from(value.inverse.len()) == Ok(count)
}

fn real_class(class: ElementType) -> Option<ElementType> {
    match class {
        ElementType::F32 | ElementType::ComplexF32 => Some(ElementType::F32),
        ElementType::F64 | ElementType::ComplexF64 => Some(ElementType::F64),
        _ => None,
    }
}

pub(super) fn factor_block(value: &DurableNumericFactor, index: usize) -> Option<&SparseValue> {
    match value {
        DurableNumericFactor::Empty { .. } => None,
        DurableNumericFactor::DenseLu { .. }
        | DurableNumericFactor::DenseCholesky { .. }
        | DurableNumericFactor::DenseQr(_)
        | DurableNumericFactor::DenseCod { .. } => None,
        DurableNumericFactor::Lu { lower, upper, .. } => match index {
            0 => Some(lower),
            1 => Some(upper),
            _ => None,
        },
        DurableNumericFactor::Cholesky { lower, .. } => (index == 0).then_some(lower),
        DurableNumericFactor::Qr(first) => orthogonal_block(first, index),
        DurableNumericFactor::Cod { first, completion } => {
            if index < 2 {
                orthogonal_block(first, index)
            } else {
                completion
                    .as_ref()
                    .and_then(|next| orthogonal_block(next, index - 2))
            }
        }
    }
}

fn orthogonal_block(value: &DurableOrthogonalFactor, index: usize) -> Option<&SparseValue> {
    match index {
        0 => Some(&value.r),
        1 => Some(&value.reflectors),
        _ => None,
    }
}

pub(super) fn factor_permutation(
    value: &DurableNumericFactor,
    index: usize,
) -> Option<(&DurableFactorPermutation, u64)> {
    match value {
        DurableNumericFactor::Empty { .. } => None,
        DurableNumericFactor::DenseLu { packed, rows } if index == 0 => {
            Some((rows, packed.shape[0]))
        }
        DurableNumericFactor::DenseCholesky { .. } => None,
        DurableNumericFactor::DenseQr(first) | DurableNumericFactor::DenseCod { first, .. }
            if index == 0 =>
        {
            Some((&first.permutation, first.packed.shape[1]))
        }
        DurableNumericFactor::DenseCod {
            completion: Some(second),
            ..
        } if index == 1 => Some((&second.permutation, second.packed.shape[1])),
        DurableNumericFactor::Lu {
            lower,
            rows,
            columns,
            ..
        } => match index {
            0 => Some((rows, lower.rows)),
            1 => Some((columns, lower.rows)),
            _ => None,
        },
        DurableNumericFactor::Cholesky { lower, permutation } => {
            (index == 0).then_some((permutation, lower.rows))
        }
        DurableNumericFactor::Qr(first) if index == 0 => {
            Some((&first.permutation, first.r.columns))
        }
        DurableNumericFactor::Cod { first, .. } if index == 0 => {
            Some((&first.permutation, first.r.columns))
        }
        DurableNumericFactor::Cod {
            completion: Some(second),
            ..
        } if index == 1 => Some((&second.permutation, second.r.columns)),
        _ => None,
    }
}

pub(super) fn orthogonal_for_block(
    value: &DurableNumericFactor,
    index: usize,
) -> Option<(&DurableOrthogonalFactor, bool)> {
    match value {
        DurableNumericFactor::Empty { .. } => None,
        DurableNumericFactor::DenseLu { .. }
        | DurableNumericFactor::DenseCholesky { .. }
        | DurableNumericFactor::DenseQr(_)
        | DurableNumericFactor::DenseCod { .. } => None,
        DurableNumericFactor::Qr(first) if index < 2 => Some((first, index == 1)),
        DurableNumericFactor::Cod { first, .. } if index < 2 => Some((first, index == 1)),
        DurableNumericFactor::Cod {
            completion: Some(second),
            ..
        } if index < 4 => Some((second, index == 3)),
        _ => None,
    }
}
