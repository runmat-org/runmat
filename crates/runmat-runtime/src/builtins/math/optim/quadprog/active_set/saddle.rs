//! Bordered original H/E system through the existing sparse QR/COD owner.
//! Singular residuals are proposals only; the target and terminal source owner
//! must independently validate equality tangency, curvature and descent.
use super::*;
use crate::builtins::math::linalg::ops::sparse_factor::least_squares;

pub(super) struct Proposal {
    pub point: working_set::Step,
    pub residual: Vec<f64>,
}
pub(super) async fn solve(
    model: &Model,
    equality: &Matrix,
    rhs: &[f64],
    c: &mut ScalarCooperation,
) -> BuiltinResult<Option<Proposal>> {
    let n = model.f.len();
    if equality.n != n || equality.m != rhs.len() {
        return Err(invalid("active equality target geometry disagrees"));
    }
    let dimension = n
        .checked_add(rhs.len())
        .ok_or_else(|| invalid("active KKT extent overflow"))?;
    let capacity_hint = model
        .p
        .nzval
        .len()
        .checked_add(equality.nzval.len())
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| invalid("active KKT storage overflow"))?;
    let hrows =
        CscRows::prepare(n, &model.p.colptr, &model.p.rowval, allocation, reserve, c).await?;
    let erows = CscRows::prepare(
        n,
        &equality.colptr,
        &equality.rowval,
        allocation,
        reserve,
        c,
    )
    .await?;
    let mut pointers = capacity(
        dimension
            .checked_add(1)
            .ok_or_else(|| invalid("active KKT pointers overflow"))?,
    )?;
    let mut indices = capacity(capacity_hint)?;
    let mut values = capacity(capacity_hint)?;
    let mut target = capacity(dimension)?;
    for &value in &model.f {
        target.push(-value);
        c.completed(1).await?;
    }
    for &value in rhs {
        target.push(value);
        c.completed(1).await?;
    }
    pointers.push(0);
    for column in 0..n {
        for entry in model.p.colptr[column]..model.p.colptr[column + 1] {
            indices.push(model.p.rowval[entry]);
            values.push(model.p.nzval[entry]);
            c.completed(1).await?;
        }
        // Existing upper entries above are already sorted. Mirrored entries
        // strictly below the diagonal follow, without a duplicated diagonal.
        for entry in hrows.row(column) {
            if entry.column != column {
                indices.push(entry.column);
                values.push(model.p.nzval[entry.entry]);
            }
            c.completed(1).await?;
        }
        for entry in equality.colptr[column]..equality.colptr[column + 1] {
            indices.push(n + equality.rowval[entry]);
            values.push(equality.nzval[entry]);
            c.completed(1).await?;
        }
        pointers.push(indices.len());
    }
    for row in 0..rhs.len() {
        for entry in erows.row(row) {
            indices.push(entry.column);
            values.push(equality.nzval[entry.entry]);
            c.completed(1).await?;
        }
        pointers.push(indices.len());
    }
    drop((hrows, erows));
    let Some(mut solved) = least_squares::solve_borrowed(
        least_squares::BorrowedProblem {
            rows: dimension,
            columns: dimension,
            pointers: &pointers,
            indices: &indices,
            values: &values,
            rhs: &target,
        },
        c.runtime(),
        runmat_builtins::QUADPROG_CATALOG_ENTRY.identity,
    )
    .await?
    else {
        return Ok(None);
    };
    if solved.values.len() != dimension {
        return Err(invalid("active KKT returned a different coordinate extent"));
    }
    // Preserve RHS-K*u in the same already-owned RHS buffer. No K'K, inverse
    // or rank assumption is introduced for dependent/inconsistent equations.
    for column in 0..dimension {
        for entry in pointers[column]..pointers[column + 1] {
            target[indices[entry]] -= values[entry] * solved.values[column];
            c.completed(1).await?;
        }
    }
    if target.iter().any(|value| !value.is_finite())
        || solved.values.iter().any(|value| !value.is_finite())
    {
        return Ok(None);
    }
    let mut multipliers = capacity(rhs.len())?;
    for &value in &solved.values[n..] {
        multipliers.push(value);
        c.completed(1).await?;
    }
    solved.values.truncate(n);
    target.truncate(n);
    Ok(Some(Proposal {
        point: working_set::Step {
            direction: solved.values,
            multipliers,
        },
        residual: target,
    }))
}
