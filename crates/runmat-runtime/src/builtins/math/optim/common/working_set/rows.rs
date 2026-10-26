//! Sparse original-row projection and original feasibility checks.
use super::{workspace, Rows};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
use clarabel::algebra::CscMatrix;
use runmat_builtins::BuiltinCatalogIdentity;

pub(crate) async fn product(
    matrix: &CscMatrix<f64>,
    x: &[f64],
    owner: BuiltinCatalogIdentity,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Vec<f64>>> {
    let mut output = workspace::capacity(matrix.m, owner)?;
    for _ in 0..matrix.m {
        output.push(0.0);
        cooperation.completed(1).await?;
    }
    for (column, &x) in x.iter().enumerate() {
        for slot in matrix.colptr[column]..matrix.colptr[column + 1] {
            output[matrix.rowval[slot]] += matrix.nzval[slot] * x;
            cooperation.completed(1).await?;
        }
        cooperation.completed(1).await?;
    }
    Ok(output
        .iter()
        .all(|value| value.is_finite())
        .then_some(output))
}
pub(crate) async fn feasible(
    model: Rows<'_>,
    x: &[f64],
    tolerance: f64,
    owner: BuiltinCatalogIdentity,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    let Some(activity) = product(model.a, x, owner, cooperation).await? else {
        return Ok(false);
    };
    for (row, (&activity, &rhs)) in activity.iter().zip(model.b).enumerate() {
        let residual = activity - rhs;
        if !residual.is_finite()
            || if row < model.equalities {
                residual.abs() > tolerance
            } else {
                residual > tolerance
            }
        {
            return Ok(false);
        }
        cooperation.completed(1).await?;
    }
    Ok(true)
}

/// Ascending original row IDs preserve CSC order and exact dual projection.
pub(crate) async fn select(
    model: Rows<'_>,
    active: &[bool],
    owner: BuiltinCatalogIdentity,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<(CscMatrix<f64>, Vec<f64>, Vec<usize>)> {
    let mut map = workspace::capacity(active.len(), owner)?;
    let mut original = workspace::capacity(active.len(), owner)?;
    let mut rhs = workspace::capacity(active.len(), owner)?;
    for (row, &selected) in active.iter().enumerate() {
        map.push(if selected { Some(rhs.len()) } else { None });
        if selected {
            original.push(row);
            rhs.push(model.b[row]);
        }
        cooperation.completed(1).await?;
    }
    let mut pointers = workspace::capacity(model.a.colptr.len(), owner)?;
    let mut indices = workspace::capacity(model.a.rowval.len(), owner)?;
    let mut values = workspace::capacity(model.a.nzval.len(), owner)?;
    pointers.push(0);
    for column in 0..model.a.n {
        for source in model.a.colptr[column]..model.a.colptr[column + 1] {
            if let Some(row) = map[model.a.rowval[source]] {
                indices.push(row);
                values.push(model.a.nzval[source]);
            }
            cooperation.completed(1).await?;
        }
        pointers.push(indices.len());
        cooperation.completed(1).await?;
    }
    Ok((
        CscMatrix::new(rhs.len(), model.a.n, pointers, indices, values),
        rhs,
        original,
    ))
}
