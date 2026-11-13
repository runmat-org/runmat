//! Independent checks of the original primal-dual Newton blocks, not the
//! lifted kernel's transformed residuals and never nonlinear terminal authority.
use super::super::super::model::Linearization;
use super::super::state::Residuals;
use super::{zeros, Direction, Layout, Row, State};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
use clarabel::algebra::CscMatrix;
use runmat_builtins::BuiltinCatalogIdentity;
fn add(sum: &mut f64, magnitude: &mut f64, term: f64) -> bool {
    *sum += term;
    *magnitude += term.abs();
    sum.is_finite() && magnitude.is_finite() && term.is_finite()
}
fn accepted(value: f64, magnitude: f64, accuracy: f64) -> bool {
    value.is_finite() && magnitude.is_finite() && value.abs() <= accuracy * (1.0 + magnitude)
}
pub(super) async fn check(
    hessian: &CscMatrix<f64>,
    original: &Linearization,
    layout: &Layout,
    state: &State,
    residuals: &Residuals,
    dual: &[f64],
    direction: &Direction,
    fixed_dual: &[f64],
    accuracy: f64,
    owner: BuiltinCatalogIdentity,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    if !accuracy.is_finite()
        || accuracy < 0.0
        || direction.x.len() != hessian.n
        || dual.len() != hessian.n
        || direction.y.len() != layout.original_equalities
        || fixed_dual.len() != layout.fixed.len()
        || direction.s.len() != layout.inequalities
        || direction.z.len() != layout.inequalities
    {
        return Ok(false);
    }
    let mut stationarity = zeros(hessian.n, owner, cooperation).await?;
    let mut scale = zeros(hessian.n, owner, cooperation).await?;
    for index in 0..dual.len() {
        stationarity[index] = dual[index];
        scale[index] = dual[index].abs();
        cooperation.completed(1).await?;
    }
    let mut equality = zeros(layout.original_equalities, owner, cooperation).await?;
    let mut equality_scale = zeros(layout.original_equalities, owner, cooperation).await?;
    for index in 0..equality.len() {
        equality[index] = residuals.equality[index];
        equality_scale[index] = equality[index].abs();
        cooperation.completed(1).await?;
    }
    let mut primal = zeros(layout.inequalities, owner, cooperation).await?;
    let mut primal_scale = zeros(layout.inequalities, owner, cooperation).await?;
    for index in 0..primal.len() {
        primal[index] = residuals.primal[index];
        primal_scale[index] = primal[index].abs();
        cooperation.completed(1).await?;
    }
    for column in 0..hessian.n {
        for source in hessian.colptr[column]..hessian.colptr[column + 1] {
            let row = hessian.rowval[source];
            let coefficient = hessian.nzval[source];
            if !add(
                &mut stationarity[row],
                &mut scale[row],
                coefficient * direction.x[column],
            ) {
                return Ok(false);
            }
            if row != column
                && !add(
                    &mut stationarity[column],
                    &mut scale[column],
                    coefficient * direction.x[row],
                )
            {
                return Ok(false);
            }
            cooperation.completed(1).await?;
        }
        for source in original.a.colptr[column]..original.a.colptr[column + 1] {
            let row = original.a.rowval[source];
            let coefficient = original.a.nzval[source];
            let dual_step = match layout.rows[row] {
                Row::Equality(index) => {
                    if !add(
                        &mut equality[index],
                        &mut equality_scale[index],
                        coefficient * direction.x[column],
                    ) {
                        return Ok(false);
                    }
                    direction.y[index]
                }
                Row::Inequality(index) => {
                    if !add(
                        &mut primal[index],
                        &mut primal_scale[index],
                        coefficient * direction.x[column],
                    ) {
                        return Ok(false);
                    }
                    direction.z[index]
                }
                Row::Fixed => {
                    cooperation.completed(1).await?;
                    continue;
                }
            };
            if !add(
                &mut stationarity[column],
                &mut scale[column],
                coefficient * dual_step,
            ) {
                return Ok(false);
            }
            cooperation.completed(1).await?;
        }
        cooperation.completed(1).await?;
    }
    for (&variable, &dual) in layout.fixed.iter().zip(fixed_dual) {
        if direction.x[variable] != 0.0
            || !add(&mut stationarity[variable], &mut scale[variable], dual)
        {
            return Ok(false);
        }
        cooperation.completed(1).await?;
    }
    for (&value, &magnitude) in stationarity.iter().zip(&scale) {
        if !accepted(value, magnitude, accuracy) {
            return Ok(false);
        }
        cooperation.completed(1).await?;
    }
    for (&value, &magnitude) in equality.iter().zip(&equality_scale) {
        if !accepted(value, magnitude, accuracy) {
            return Ok(false);
        }
        cooperation.completed(1).await?;
    }
    for index in 0..primal.len() {
        if !add(
            &mut primal[index],
            &mut primal_scale[index],
            direction.s[index],
        ) || !accepted(primal[index], primal_scale[index], accuracy)
        {
            return Ok(false);
        }
        let first = state.z[index] * direction.s[index];
        let second = state.s[index] * direction.z[index];
        let mut center = residuals.centering[index];
        let mut magnitude = center.abs();
        if !add(&mut center, &mut magnitude, first)
            || !add(&mut center, &mut magnitude, second)
            || !accepted(center, magnitude, accuracy)
        {
            return Ok(false);
        }
        cooperation.completed(1).await?;
    }
    Ok(true)
}
