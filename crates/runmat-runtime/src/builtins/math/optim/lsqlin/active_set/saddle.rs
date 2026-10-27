//! Equality-constrained residual KKT targets through the existing QR/COD owner.
//! The saddle contains C and C', never C'C, and permits dependent active rows.
use super::*;
use crate::builtins::math::linalg::ops::sparse_factor::least_squares;
pub(super) async fn solve(
    model: &Model,
    equality: &Matrix,
    rhs: &[f64],
    c: &mut ScalarCooperation,
) -> BuiltinResult<Option<working_set::Step>> {
    if equality.n != model.c.n || equality.m != rhs.len() {
        return Err(invalid("active equality target geometry disagrees"));
    }
    let m = model.c.m;
    let n = model.c.n;
    let k = rhs.len();
    let variables = m
        .checked_add(n)
        .and_then(|x| x.checked_add(k))
        .ok_or_else(|| invalid("active residual saddle extent overflow"))?;
    let nnz = model
        .c
        .nzval
        .len()
        .checked_add(equality.nzval.len())
        .and_then(|x| x.checked_mul(2))
        .and_then(|x| x.checked_add(m))
        .ok_or_else(|| invalid("active residual saddle storage overflow"))?;
    let index = CscRows::prepare(
        n,
        &equality.colptr,
        &equality.rowval,
        allocation,
        reserve,
        c,
    )
    .await?;
    let mut pointers = capacity(
        variables
            .checked_add(1)
            .ok_or_else(|| invalid("active saddle pointer overflow"))?,
    )?;
    let mut indices = capacity(nnz)?;
    let mut values = capacity(nnz)?;
    let mut target = zeros(variables, c).await?;
    let root = model.units.sqrt();
    for row in 0..m {
        target[row] = root * model.weights[row].sqrt() * model.d[row];
        if !target[row].is_finite() {
            return Ok(None);
        }
        c.completed(1).await?;
    }
    for (to, &value) in target[m + n..].iter_mut().zip(rhs) {
        *to = value;
        c.completed(1).await?;
    }
    pointers.push(0);
    for row in 0..m {
        indices.push(row);
        values.push(-1.);
        for entry in model.index.row(row) {
            let value = root * model.weights[row].sqrt() * model.c.nzval[entry.entry];
            if !value.is_finite() {
                return Ok(None);
            }
            indices.push(m + entry.column);
            values.push(value);
            c.completed(1).await?;
        }
        pointers.push(indices.len());
        c.completed(1).await?;
    }
    for column in 0..n {
        for entry in model.c.colptr[column]..model.c.colptr[column + 1] {
            let row = model.c.rowval[entry];
            let value = root * model.weights[row].sqrt() * model.c.nzval[entry];
            if !value.is_finite() {
                return Ok(None);
            }
            indices.push(row);
            values.push(value);
            c.completed(1).await?;
        }
        for entry in equality.colptr[column]..equality.colptr[column + 1] {
            indices.push(m + n + equality.rowval[entry]);
            values.push(equality.nzval[entry]);
            c.completed(1).await?;
        }
        pointers.push(indices.len());
        c.completed(1).await?;
    }
    for row in 0..k {
        for entry in index.row(row) {
            indices.push(m + entry.column);
            values.push(equality.nzval[entry.entry]);
            c.completed(1).await?;
        }
        pointers.push(indices.len());
        c.completed(1).await?;
    }
    drop(index);
    let solved = least_squares::solve_borrowed(
        least_squares::BorrowedProblem {
            rows: variables,
            columns: variables,
            pointers: &pointers,
            indices: &indices,
            values: &values,
            rhs: &target,
        },
        c.runtime(),
        runmat_builtins::LSQLIN_CATALOG_ENTRY.identity,
    )
    .await?;
    c.completed(0).await?;
    let Some(solved) = solved else {
        return Ok(None);
    };
    if solved.values.len() != variables {
        return Err(invalid("active residual saddle solution extent changed"));
    }
    let mut direction = capacity(n)?;
    let mut multipliers = capacity(k)?;
    for chunk in solved.values[m..m + n].chunks(1024) {
        direction.extend_from_slice(chunk);
        c.completed(chunk.len() as u64).await?;
    }
    for chunk in solved.values[m + n..].chunks(1024) {
        multipliers.extend_from_slice(chunk);
        c.completed(chunk.len() as u64).await?;
    }
    Ok(Some(working_set::Step {
        direction,
        multipliers,
    }))
}
