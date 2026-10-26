//! Actual primal-dual Newton equations, not an SQP step or a terminal proof.
mod model;
mod validation;
use super::super::{model::Linearization, quadratic, workspace};
use super::state::{zeros, Layout, Row, State};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
use clarabel::algebra::CscMatrix;
use runmat_builtins::BuiltinCatalogIdentity;
pub(super) struct Direction {
    pub x: Vec<f64>,
    pub s: Vec<f64>,
    pub z: Vec<f64>,
    pub y: Vec<f64>,
}
pub(super) async fn solve(
    hessian: &CscMatrix<f64>,
    gradient: &[f64],
    original: &Linearization,
    layout: &Layout,
    state: &State,
    multipliers: &[f64],
    accuracy: f64,
    owner: BuiltinCatalogIdentity,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Direction>> {
    let Some(residuals) = state
        .residuals(layout, original, owner, cooperation)
        .await?
    else {
        return Ok(None);
    };
    let Some(dual) =
        super::super::validation::lagrangian(gradient, original, multipliers, owner, cooperation)
            .await?
    else {
        return Ok(None);
    };
    let Some(lifted) = model::build(
        hessian,
        original,
        layout,
        state,
        &residuals,
        &dual,
        owner,
        cooperation,
    )
    .await?
    else {
        return Ok(None);
    };
    let constraints = quadratic::Constraints {
        a: &lifted.a,
        b: &lifted.b,
        equalities: lifted.b.len(),
        cones: &lifted.cones,
    };
    let quadratic::Result::Step(mut proposed) = quadratic::solve(
        &lifted.p,
        &lifted.f,
        &constraints,
        accuracy,
        owner,
        cooperation,
    )
    .await?
    else {
        return Ok(None);
    };
    // Drop lifted numerical workspace at this method's boundary; no second
    // backend is started and no local certificate is promoted to nonlinear status.
    proposed.direction.truncate(gradient.len());
    for &variable in &layout.fixed {
        proposed.direction[variable] = 0.0;
        cooperation.completed(1).await?;
    }
    let mut activity = zeros(layout.inequalities, owner, cooperation).await?;
    for column in 0..original.a.n {
        for source in original.a.colptr[column]..original.a.colptr[column + 1] {
            if let Row::Inequality(index) = layout.rows[original.a.rowval[source]] {
                activity[index] += original.a.nzval[source] * proposed.direction[column];
                if !activity[index].is_finite() {
                    return Ok(None);
                }
            }
            cooperation.completed(1).await?;
        }
        cooperation.completed(1).await?;
    }
    let mut ds = workspace::capacity(layout.inequalities, owner)?;
    let mut dz = workspace::capacity(layout.inequalities, owner)?;
    for index in 0..layout.inequalities {
        let slack = -residuals.primal[index] - activity[index];
        let dual = -(residuals.centering[index] + state.z[index] * slack) / state.s[index];
        if !slack.is_finite() || !dual.is_finite() {
            return Ok(None);
        }
        ds.push(slack);
        dz.push(dual);
        cooperation.completed(1).await?;
    }
    if proposed.multipliers.len() != lifted.b.len() {
        return Ok(None);
    }
    let mut dy = workspace::capacity(layout.original_equalities, owner)?;
    for &value in &proposed.multipliers[..layout.original_equalities] {
        dy.push(value);
        cooperation.completed(1).await?;
    }
    let direction = Direction {
        x: proposed.direction,
        s: ds,
        z: dz,
        y: dy,
    };
    if !validation::check(
        hessian,
        original,
        layout,
        state,
        &residuals,
        &dual,
        &direction,
        &proposed.multipliers[layout.original_equalities..lifted.equality_count],
        accuracy,
        owner,
        cooperation,
    )
    .await?
    {
        return Ok(None);
    }
    Ok(Some(direction))
}
