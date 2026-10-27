//! Sparse residual lifting of B+Jg'*(z/s)*Jg. No Gram matrix is formed.
use super::super::super::{model::Linearization, workspace};
use super::super::state::{Layout, Residuals, Row, State};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
use clarabel::{algebra::CscMatrix, solver::SupportedConeT};
use runmat_builtins::BuiltinCatalogIdentity;
pub(super) struct Lifted {
    pub p: CscMatrix<f64>,
    pub a: CscMatrix<f64>,
    pub b: Vec<f64>,
    pub f: Vec<f64>,
    pub cones: Vec<SupportedConeT<f64>>,
    pub equality_count: usize,
}
pub(super) async fn build(
    hessian: &CscMatrix<f64>,
    original: &Linearization,
    layout: &Layout,
    state: &State,
    residuals: &Residuals,
    dual: &[f64],
    owner: BuiltinCatalogIdentity,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Lifted>> {
    let n = hessian.n;
    let m = layout.inequalities;
    let Some(variables) = n.checked_add(m) else {
        return Ok(None);
    };
    let Some(eq) = layout.equality_count() else {
        return Ok(None);
    };
    let Some(rows) = eq.checked_add(m) else {
        return Ok(None);
    };
    let Some(columns) = variables.checked_add(1) else {
        return Ok(None);
    };
    let mut root = workspace::capacity(m, owner)?;
    let mut correction = workspace::capacity(m, owner)?;
    for index in 0..m {
        let ratio = state.z[index] / state.s[index];
        let value = residuals.centering[index] / state.s[index] - ratio * residuals.primal[index];
        let weight = ratio.sqrt();
        if !ratio.is_finite()
            || ratio <= 0.0
            || !weight.is_finite()
            || weight <= 0.0
            || !value.is_finite()
        {
            return Ok(None);
        }
        root.push(weight);
        correction.push(value);
        cooperation.completed(1).await?;
    }
    let mut f = workspace::capacity(variables, owner)?;
    let mut rhs = workspace::capacity(rows, owner)?;
    for &value in &residuals.equality {
        rhs.push(-value);
        cooperation.completed(1).await?;
    }
    for _ in 0..layout.fixed.len() + m {
        rhs.push(0.0);
        cooperation.completed(1).await?;
    }
    let mut aptr = workspace::capacity(columns, owner)?;
    let mut arows = Vec::new();
    let mut avals = Vec::new();
    aptr.push(0);
    for column in 0..n {
        let mut q = dual[column];
        let mut compensated = 0.0;
        for source in original.a.colptr[column]..original.a.colptr[column + 1] {
            let row = original.a.rowval[source];
            let coefficient = original.a.nzval[source];
            match layout.rows[row] {
                Row::Equality(index) => {
                    workspace::push(&mut arows, index, owner)?;
                    workspace::push(&mut avals, coefficient, owner)?;
                }
                Row::Inequality(index) => {
                    let term = coefficient * correction[index];
                    let adjusted = term - compensated;
                    let next = q - adjusted;
                    compensated = (q - next) - adjusted;
                    q = next;
                    if !term.is_finite() || !q.is_finite() {
                        return Ok(None);
                    }
                }
                Row::Fixed => {}
            }
            cooperation.completed(1).await?;
        }
        if let Ok(index) = layout.fixed.binary_search(&column) {
            workspace::push(&mut arows, layout.original_equalities + index, owner)?;
            workspace::push(&mut avals, 1.0, owner)?;
        }
        for source in original.a.colptr[column]..original.a.colptr[column + 1] {
            let row = original.a.rowval[source];
            if let Row::Inequality(index) = layout.rows[row] {
                let value = original.a.nzval[source] * root[index];
                if !value.is_finite() {
                    return Ok(None);
                }
                workspace::push(&mut arows, eq + index, owner)?;
                workspace::push(&mut avals, value, owner)?;
            }
            cooperation.completed(1).await?;
        }
        f.push(q);
        aptr.push(arows.len());
        cooperation.completed(1).await?;
    }
    for index in 0..m {
        workspace::push(&mut arows, eq + index, owner)?;
        workspace::push(&mut avals, -1.0, owner)?;
        aptr.push(arows.len());
        f.push(0.0);
        cooperation.completed(1).await?;
    }
    let mut pptr = workspace::capacity(columns, owner)?;
    for &value in &hessian.colptr {
        pptr.push(value);
        cooperation.completed(1).await?;
    }
    let mut prows = workspace::capacity(hessian.rowval.len(), owner)?;
    let mut pvalues = workspace::capacity(hessian.nzval.len(), owner)?;
    for (&row, &value) in hessian.rowval.iter().zip(&hessian.nzval) {
        if !value.is_finite() {
            return Ok(None);
        }
        prows.push(row);
        pvalues.push(value);
        cooperation.completed(1).await?;
    }
    for index in 0..m {
        workspace::push(&mut prows, n + index, owner)?;
        workspace::push(&mut pvalues, 1.0, owner)?;
        pptr.push(prows.len());
        cooperation.completed(1).await?;
    }
    let cones = if rows == 0 {
        Vec::new()
    } else {
        vec![SupportedConeT::ZeroConeT(rows)]
    };
    Ok(Some(Lifted {
        p: CscMatrix::new(variables, variables, pptr, prows, pvalues),
        a: CscMatrix::new(rows, variables, aptr, arows, avals),
        b: rhs,
        f,
        cones,
        equality_count: eq,
    }))
}
