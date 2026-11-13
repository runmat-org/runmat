//! Sparse constant factors use their actual original CSC entries. This avoids
//! scanning virtual zero coefficients or constructing a dense source matrix.
use super::*;
pub(super) async fn right(
    input: &Array,
    matrix: &runmat_value::SparseTensor,
    shape: SharedShape,
    m: usize,
    n: usize,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Array> {
    let count = evaluation::element_count(&shape, policy)?;
    let mut rows = runmat_value::exact_capacity_vec(count).map_err(|e| policy.storage(e))?;
    let zero = Polynomial::zero();
    for column in 0..n {
        for row in 0..m {
            let mut sum = zero.clone();
            for entry in matrix.col_ptrs[column]..matrix.col_ptrs[column + 1] {
                let axis = matrix.row_indices[entry];
                let factor = matrix.get(axis, column).ok_or_else(|| {
                    policy.invalid("quadratic sparse factor lost real double backing")
                })?;
                if factor != 0. {
                    let value = input.at(row + axis * m, policy)?.ok_or_else(|| {
                        policy.invalid("quadratic sparse product lost assigned input")
                    })?;
                    let weight = Weight::constant(factor, policy)?;
                    sum = sum
                        .add(value.scaled(&weight, policy, c).await?, false, policy, c)
                        .await?;
                }
                c.completed(1).await?;
            }
            rows.push(Some(sum));
            c.completed(1).await?;
        }
    }
    Array::rows(shape, rows, policy)
}
pub(super) async fn left(
    matrix: &runmat_value::SparseTensor,
    input: &Array,
    shape: SharedShape,
    m: usize,
    n: usize,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Array> {
    let count = evaluation::element_count(&shape, policy)?;
    let mut rows = runmat_value::exact_capacity_vec(count).map_err(|e| policy.storage(e))?;
    let zero = Polynomial::zero();
    for _ in 0..count {
        rows.push(Some(zero.clone()));
        c.completed(1).await?;
    }
    for column in 0..n {
        for axis in 0..matrix.cols {
            let value = input
                .at(axis + column * matrix.cols, policy)?
                .ok_or_else(|| policy.invalid("quadratic sparse product lost assigned input"))?;
            for entry in matrix.col_ptrs[axis]..matrix.col_ptrs[axis + 1] {
                let row = matrix.row_indices[entry];
                let factor = matrix.get(row, axis).ok_or_else(|| {
                    policy.invalid("quadratic sparse factor lost real double backing")
                })?;
                if factor != 0. {
                    let target = row + column * m;
                    let sum = rows[target].take().ok_or_else(|| {
                        policy.invalid("quadratic sparse product lost output state")
                    })?;
                    let weight = Weight::constant(factor, policy)?;
                    rows[target] = Some(
                        sum.add(value.scaled(&weight, policy, c).await?, false, policy, c)
                            .await?,
                    );
                }
                c.completed(1).await?;
            }
            c.completed(1).await?;
        }
    }
    Array::rows(shape, rows, policy)
}
