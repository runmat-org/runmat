//! Independent ordinary numerical KKT checks on the original QP, not kernel status.
use super::*;
use model::Model;
pub(super) struct Checked {
    pub objective: f64,
    pub violation: f64,
    pub optimality: f64,
    pub gap: f64,
    pub valid: bool,
}
pub(super) async fn primal(
    model: &Model,
    x: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Checked>> {
    let Some(point) = point(model, x, cooperation).await? else {
        return Ok(None);
    };
    Ok(Some(Checked {
        objective: point.objective,
        violation: point.violation,
        optimality: f64::NAN,
        gap: f64::NAN,
        valid: false,
    }))
}
struct Point {
    objective: f64,
    violation: f64,
    activity: Vec<f64>,
    px: Vec<f64>,
    quadratic: f64,
}
async fn point(
    model: &Model,
    x: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Point>> {
    if x.len() != model.f.len() || x.iter().any(|v| !v.is_finite()) {
        return Ok(None);
    }
    let mut px = vec![0.0; x.len()];
    for col in 0..x.len() {
        for entry in model.p.colptr[col]..model.p.colptr[col + 1] {
            let row = model.p.rowval[entry];
            let value = model.p.nzval[entry];
            px[row] += value * x[col];
            if row != col {
                px[col] += value * x[row];
            }
            if entry % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        cooperation.completed(1).await?;
    }
    let mut activity = vec![0.0; model.b.len()];
    for col in 0..x.len() {
        for entry in model.a.colptr[col]..model.a.colptr[col + 1] {
            activity[model.a.rowval[entry]] += model.a.nzval[entry] * x[col];
            if entry % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        cooperation.completed(1).await?;
    }
    let mut violation = 0.0_f64;
    for (row, (&value, &rhs)) in activity.iter().zip(&model.b).enumerate() {
        if !value.is_finite() {
            return Ok(None);
        }
        let error = value - rhs;
        if !error.is_finite() {
            return Ok(None);
        }
        violation = violation.max(if row < model.eq {
            error.abs()
        } else {
            error.max(0.0)
        });
        if row % 1024 == 1023 {
            cooperation.completed(1024).await?;
        }
    }
    let Some(quadratic) = dot(x, &px, cooperation).await? else {
        return Ok(None);
    };
    let Some(linear) = dot(x, &model.f, cooperation).await? else {
        return Ok(None);
    };
    let objective = quadratic * 0.5 + linear;
    if !objective.is_finite() {
        return Ok(None);
    }
    Ok(Some(Point {
        objective,
        violation,
        activity,
        px,
        quadratic,
    }))
}
pub(super) async fn optimum(
    model: &Model,
    x: &[f64],
    z: &[f64],
    controls: &options::Controls,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Checked>> {
    if z.len() != model.b.len() || z.iter().any(|v| !v.is_finite()) {
        return Ok(None);
    }
    let Some(point) = point(model, x, cooperation).await? else {
        return Ok(None);
    };
    let mut residual = 0.0_f64;
    for col in 0..x.len() {
        let mut sum = model.f[col] + point.px[col];
        let mut magnitude = model.f[col].abs() + point.px[col].abs();
        for entry in model.a.colptr[col]..model.a.colptr[col + 1] {
            let term = model.a.nzval[entry] * z[model.a.rowval[entry]];
            sum += term;
            magnitude += term.abs();
            if entry % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        if !sum.is_finite() || !magnitude.is_finite() {
            return Ok(None);
        }
        residual = residual.max(sum.abs() / (1.0 + magnitude));
        cooperation.completed(1).await?;
    }
    let Some(rhs) = dot(&model.b, z, cooperation).await? else {
        return Ok(None);
    };
    let Some(linear) = dot(&model.f, x, cooperation).await? else {
        return Ok(None);
    };
    let gap = (point.quadratic + linear + rhs).abs();
    let scale = 1.0 + point.quadratic.abs() + linear.abs() + rhs.abs();
    if !gap.is_finite() || !scale.is_finite() {
        return Ok(None);
    }
    let mut complementarity = 0.0_f64;
    let mut cone_error = 0.0_f64;
    for row in model.eq..model.b.len() {
        cone_error = cone_error.max((-z[row]).max(0.0) / (1.0 + z[row].abs()));
        let product = z[row] * (model.b[row] - point.activity[row]);
        if !product.is_finite() {
            return Ok(None);
        }
        complementarity = complementarity.max(product.abs() / scale);
        if row % 1024 == 1023 {
            cooperation.completed(1024).await?;
        }
    }
    let valid = point.violation <= controls.feasibility
        && residual <= controls.optimality
        && cone_error <= controls.optimality
        && gap / scale <= controls.optimality
        && complementarity <= controls.optimality;
    Ok(Some(Checked {
        objective: point.objective,
        violation: point.violation,
        optimality: residual,
        gap,
        valid,
    }))
}
async fn dot(
    a: &[f64],
    b: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<f64>> {
    let mut sum = 0.0;
    for (index, (&a, &b)) in a.iter().zip(b).enumerate() {
        sum += a * b;
        if !sum.is_finite() {
            return Ok(None);
        }
        if index % 1024 == 1023 {
            cooperation.completed(1024).await?;
        }
    }
    Ok(Some(sum))
}
