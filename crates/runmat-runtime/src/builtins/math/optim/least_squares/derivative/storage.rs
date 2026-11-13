//! Original typed coefficient scans and physical-column copies.
use super::*;
pub(super) async fn finite<T: Real>(
    view: crate::builtins::math::linalg::ops::linear_operator::TypedLinearOperator<'_, T>,
    c: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    use crate::builtins::math::linalg::ops::linear_operator::TypedLinearOperator as V;
    let check = |value: T| {
        if value.real().is_finite() {
            Ok(())
        } else {
            Err(invalid("Jacobian must contain finite coefficients"))
        }
    };
    match view {
        V::Scalar(value) => check(value)?,
        V::Dense(view) => {
            for chunk in view.as_slice().chunks(1024) {
                for &value in chunk {
                    check(value)?;
                }
                c.completed(chunk.len() as u64).await?;
            }
        }
        V::Sparse(view) => {
            for column in 0..view.cols() {
                let (_, values) = view.column(column);
                for chunk in values.chunks(1024) {
                    for &value in chunk {
                        check(value)?;
                    }
                    c.completed(chunk.len() as u64).await?;
                }
                c.completed(1).await?;
            }
        }
    }
    Ok(())
}
pub(super) async fn copy<T: Real>(
    view: crate::builtins::math::linalg::ops::linear_operator::TypedLinearOperator<'_, T>,
    output: &mut [f64],
    c: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    use crate::builtins::math::linalg::ops::linear_operator::TypedLinearOperator as V;
    let columns = view.cols();
    match view {
        V::Scalar(value) => output[0] = value.real(),
        V::Dense(view) => {
            for column in 0..columns {
                for row in 0..view.rows() {
                    output[row * columns + column] =
                        view.as_slice()[column * view.rows() + row].real();
                    c.completed(1).await?;
                }
            }
        }
        V::Sparse(view) => {
            for column in 0..columns {
                let (rows, values) = view.column(column);
                for (&row, &value) in rows.iter().zip(values) {
                    output[row * columns + column] = value.real();
                    c.completed(1).await?;
                }
                c.completed(1).await?;
            }
        }
    }
    Ok(())
}

pub(super) fn append(
    row: usize,
    coefficient: f64,
    scale: f64,
    offset: usize,
    indices: &mut Vec<usize>,
    values: &mut Vec<f64>,
) -> BuiltinResult<bool> {
    let value = coefficient * scale;
    if !value.is_finite() {
        return Ok(false);
    }
    if value != 0. {
        indices.push(
            offset
                .checked_add(row)
                .ok_or_else(|| invalid("Jacobian row index overflow"))?,
        );
        values.push(value);
    }
    Ok(true)
}
pub(super) async fn append_typed<T: Real>(
    view: crate::builtins::math::linalg::ops::linear_operator::TypedLinearOperator<'_, T>,
    axis: usize,
    scale: f64,
    offset: usize,
    indices: &mut Vec<usize>,
    values: &mut Vec<f64>,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Option<()>> {
    use crate::builtins::math::linalg::ops::linear_operator::TypedLinearOperator as V;
    match view {
        V::Scalar(value) => {
            if !append(0, value.real(), scale, offset, indices, values)? {
                return Ok(None);
            }
        }
        V::Dense(view) => {
            for row in 0..view.rows() {
                if !append(
                    row,
                    view.as_slice()[axis * view.rows() + row].real(),
                    scale,
                    offset,
                    indices,
                    values,
                )? {
                    return Ok(None);
                }
                c.completed(1).await?;
            }
        }
        V::Sparse(view) => {
            let (rows, source) = view.column(axis);
            for (&row, &value) in rows.iter().zip(source) {
                if !append(row, value.real(), scale, offset, indices, values)? {
                    return Ok(None);
                }
                c.completed(1).await?;
            }
        }
    }
    c.completed(1).await?;
    Ok(Some(()))
}
