//! Exact chain rules for source-admitted affine arithmetic.
use super::*;
use crate::builtins::common::broadcast::{broadcast_index, compute_strides};
use runmat_types::OperatorKind as Op;
fn constant(
    constants: &Evaluated,
    root: GcHandle,
    row: usize,
    policy: Policy,
) -> BuiltinResult<Rational> {
    constants
        .value(root, row, policy)?
        .ok_or_else(|| policy.invalid("active affine multiplier has an unassigned constant slot"))
}
pub(super) async fn propagate(
    state: &mut State<'_>,
    constants: &Evaluated,
    left: GcHandle,
    right: GcHandle,
    operator: Op,
    shape: &[usize],
    weights: Weights,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    let policy = state.policy;
    let a = state.shape(left)?;
    let b = state.shape(right)?;
    if operator == Op::MatrixMultiply
        && evaluation::element_count(a, policy)? != 1
        && evaluation::element_count(b, policy)? != 1
    {
        return matrix(state, constants, left, right, weights, cooperation).await;
    }
    let left_shape = a.to_vec();
    let right_shape = b.to_vec();
    let a_stride = compute_strides(&left_shape);
    let b_stride = compute_strides(&right_shape);
    for (row, value) in weights {
        let a = broadcast_index(row, shape, &left_shape, &a_stride);
        let b = broadcast_index(row, shape, &right_shape, &b_stride);
        match operator {
            Op::Add | Op::Subtract => {
                if state.active_at(left, a)? {
                    state.add(left, a, value.clone(), cooperation).await?;
                }
                if state.active_at(right, b)? {
                    state
                        .add(
                            right,
                            b,
                            if operator == Op::Subtract {
                                -value
                            } else {
                                value
                            },
                            cooperation,
                        )
                        .await?;
                }
            }
            Op::MatrixMultiply | Op::ElementwiseMultiply => {
                if state.active_at(left, a)? {
                    state
                        .add(
                            left,
                            a,
                            value
                                .mul(&constant(constants, right, b, policy)?, policy, cooperation)
                                .await?,
                            cooperation,
                        )
                        .await?;
                }
                if state.active_at(right, b)? {
                    state
                        .add(
                            right,
                            b,
                            value
                                .mul(&constant(constants, left, a, policy)?, policy, cooperation)
                                .await?,
                            cooperation,
                        )
                        .await?;
                }
            }
            Op::Mrdivide | Op::ElementwiseDivide => {
                let divisor = constant(constants, right, b, policy)?;
                if divisor.is_zero() {
                    return Err(policy.invalid("reverse affine divisor is zero"));
                }
                state
                    .add(
                        left,
                        a,
                        value.div(&divisor, policy, cooperation).await?,
                        cooperation,
                    )
                    .await?;
            }
            Op::Mldivide | Op::ElementwiseLeftDivide => {
                let divisor = constant(constants, left, a, policy)?;
                if divisor.is_zero() {
                    return Err(policy.invalid("reverse affine divisor is zero"));
                }
                state
                    .add(
                        right,
                        b,
                        value.div(&divisor, policy, cooperation).await?,
                        cooperation,
                    )
                    .await?;
            }
            _ => return Err(policy.invalid("unadmitted reverse arithmetic operation")),
        }
        cooperation.completed(1).await?;
    }
    Ok(())
}
async fn matrix(
    state: &mut State<'_>,
    constants: &Evaluated,
    left: GcHandle,
    right: GcHandle,
    weights: Weights,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    let policy = state.policy;
    let m = state.shape(left)?.first().copied().unwrap_or(1);
    let k = state.shape(left)?.get(1).copied().unwrap_or(1);
    if state.active(right)? && !state.active(left)? {
        if let Some(matrix) = constants.values.get(&left).and_then(Data::sparse) {
            if let Some(plan) = &mut state.plan {
                let rows = plan.rows(left, matrix, policy, cooperation).await?;
                for (output, value) in weights {
                    let row = output % m;
                    let column = output / m;
                    for entry in rows.row(row) {
                        let coefficient =
                            constant(constants, left, entry.row + entry.column * m, policy)?;
                        state
                            .add(
                                right,
                                entry.column + column * k,
                                value.mul(&coefficient, policy, cooperation).await?,
                                cooperation,
                            )
                            .await?;
                        cooperation.completed(1).await?;
                    }
                    cooperation.completed(1).await?;
                }
                return Ok(());
            }
            let n = state.shape(right)?.get(1).copied().unwrap_or(1);
            for axis in 0..k {
                for entry in matrix.col_ptrs[axis]..matrix.col_ptrs[axis + 1] {
                    let row = matrix.row_indices[entry];
                    let coefficient = constant(constants, left, row + axis * m, policy)?;
                    for column in 0..n {
                        if let Some(value) = weights.get(&(row + column * m)) {
                            state
                                .add(
                                    right,
                                    axis + column * k,
                                    value.mul(&coefficient, policy, cooperation).await?,
                                    cooperation,
                                )
                                .await?;
                        }
                        cooperation.completed(1).await?;
                    }
                }
                cooperation.completed(1).await?;
            }
            return Ok(());
        }
    }
    if state.active(left)? && !state.active(right)? {
        if let Some(matrix) = constants.values.get(&right).and_then(Data::sparse) {
            for (output, value) in weights {
                let row = output % m;
                let column = output / m;
                for entry in matrix.col_ptrs[column]..matrix.col_ptrs[column + 1] {
                    let axis = matrix.row_indices[entry];
                    let coefficient = constant(constants, right, axis + column * k, policy)?;
                    state
                        .add(
                            left,
                            row + axis * m,
                            value.mul(&coefficient, policy, cooperation).await?,
                            cooperation,
                        )
                        .await?;
                    cooperation.completed(1).await?;
                }
                cooperation.completed(1).await?;
            }
            return Ok(());
        }
    }

    for (output, value) in weights {
        let row = output % m;
        let column = output / m;
        for axis in 0..k {
            let a = row + axis * m;
            let b = axis + column * k;
            if state.active_at(left, a)? {
                state
                    .add(
                        left,
                        a,
                        value
                            .mul(&constant(constants, right, b, policy)?, policy, cooperation)
                            .await?,
                        cooperation,
                    )
                    .await?;
            }
            if state.active_at(right, b)? {
                state
                    .add(
                        right,
                        b,
                        value
                            .mul(&constant(constants, left, a, policy)?, policy, cooperation)
                            .await?,
                        cooperation,
                    )
                    .await?;
            }
            cooperation.completed(1).await?;
        }
    }
    Ok(())
}
