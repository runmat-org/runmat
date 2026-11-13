//! Independent original-row primal/dual/SOC checks, never backend status alone.
use super::*;
use clarabel::solver::SupportedConeT;
use model::ConicModel;
pub(super) struct Point {
    pub objective: f64,
    pub violation: f64,
    pub normalized: f64,
    activity: Vec<f64>,
}
pub(super) struct Optimality {
    pub point: Point,
    pub dual: f64,
    pub gap: f64,
    pub valid: bool,
}

pub(super) async fn primal(
    model: &ConicModel,
    x: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Point>> {
    if x.len() != model.f.len() {
        return Ok(None);
    }
    for chunk in x.chunks(1024) {
        if chunk.iter().any(|v| !v.is_finite()) {
            return Ok(None);
        }
        cooperation.completed(chunk.len() as u64).await?;
    }
    let mut activity = vec![0.0_f64; model.b.len()];
    let mut magnitude = vec![0.0_f64; model.b.len()];
    for col in 0..model.f.len() {
        for first in (model.a.colptr[col]..model.a.colptr[col + 1]).step_by(1024) {
            let end = first.saturating_add(1024).min(model.a.colptr[col + 1]);
            for entry in first..end {
                let row = model.a.rowval[entry];
                let value = model.a.nzval[entry] * x[col];
                activity[row] += value;
                magnitude[row] += value.abs();
                if !activity[row].is_finite() || !magnitude[row].is_finite() {
                    return Ok(None);
                }
            }
            cooperation.completed((end - first) as u64).await?;
        }
        cooperation.completed(1).await?;
    }
    let mut violation = 0.0_f64;
    let mut normalized = 0.0_f64;
    let mut offset = 0;
    for cone in &model.cones {
        let Some(size) = conic::dimension(cone) else {
            return Ok(None);
        };
        match cone {
            SupportedConeT::ZeroConeT(_) | SupportedConeT::NonnegativeConeT(_) => {
                for row in offset..offset + size {
                    let value = activity[row] - model.b[row];
                    let error = if matches!(cone, SupportedConeT::ZeroConeT(_)) {
                        value.abs()
                    } else {
                        value.max(0.0)
                    };
                    let scale = model.row_scale[row] + model.b[row].abs() + magnitude[row];
                    if !scale.is_finite() {
                        return Ok(None);
                    }
                    violation = violation.max(error);
                    normalized = normalized.max(if scale == 0.0 { error } else { error / scale });
                    if row % 1024 == 1023 {
                        cooperation.completed(1024).await?;
                    }
                }
            }
            SupportedConeT::SecondOrderConeT(_) => {
                let head = model.b[offset] - activity[offset];
                let mut norm = 0.0_f64;
                let mut scale = 0.0_f64;
                for row in offset..offset + size {
                    if row > offset {
                        norm = norm.hypot(model.b[row] - activity[row]);
                    }
                    scale += model.row_scale[row] + model.b[row].abs() + magnitude[row];
                    if row % 1024 == 1023 {
                        cooperation.completed(1024).await?;
                    }
                }
                let error = (norm - head).max(0.0);
                if !error.is_finite() || !scale.is_finite() {
                    return Ok(None);
                }
                violation = violation.max(error);
                normalized = normalized.max(if scale == 0.0 { error } else { error / scale });
            }
            _ => return Ok(None),
        }
        offset += size;
        cooperation.completed(1).await?;
    }
    let Some(objective) = dot(&model.f, x, cooperation).await? else {
        return Ok(None);
    };
    if !violation.is_finite() || !normalized.is_finite() {
        return Ok(None);
    }
    Ok(Some(Point {
        objective,
        violation,
        normalized,
        activity,
    }))
}

pub(super) async fn optimum(
    model: &ConicModel,
    x: &[f64],
    z: &[f64],
    tol: f64,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Optimality>> {
    if z.len() != model.b.len() {
        return Ok(None);
    }
    for chunk in z.chunks(1024) {
        if chunk.iter().any(|v| !v.is_finite()) {
            return Ok(None);
        }
        cooperation.completed(chunk.len() as u64).await?;
    }
    let Some(point) = primal(model, x, cooperation).await? else {
        return Ok(None);
    };
    let mut dual = 0.0_f64;
    for col in 0..model.f.len() {
        let mut sum = model.f[col];
        let mut magnitude = sum.abs();
        for first in (model.a.colptr[col]..model.a.colptr[col + 1]).step_by(1024) {
            let end = first.saturating_add(1024).min(model.a.colptr[col + 1]);
            for entry in first..end {
                let term = model.a.nzval[entry] * z[model.a.rowval[entry]];
                sum += term;
                magnitude += term.abs();
            }
            cooperation.completed((end - first) as u64).await?;
        }
        let scale = model.objective_scale + magnitude;
        if !sum.is_finite() || !scale.is_finite() {
            return Ok(None);
        }
        dual = dual.max(sum.abs() / scale);
        cooperation.completed(1).await?;
    }
    let Some(rhs_dual) = dot(&model.b, z, cooperation).await? else {
        return Ok(None);
    };
    let gap = (point.objective + rhs_dual).abs();
    let scale = model.objective_scale + point.objective.abs() + rhs_dual.abs();
    if !gap.is_finite() || !scale.is_finite() {
        return Ok(None);
    }
    let mut complementarity = 0.0_f64;
    let mut offset = 0;
    for cone in &model.cones {
        let Some(size) = conic::dimension(cone) else {
            return Ok(None);
        };
        match cone {
            SupportedConeT::ZeroConeT(_) => {}
            SupportedConeT::NonnegativeConeT(_) => {
                for row in offset..offset + size {
                    dual = dual.max((-z[row]).max(0.0) / (model.objective_scale + z[row].abs()));
                    let product = z[row] * (model.b[row] - point.activity[row]);
                    if !product.is_finite() {
                        return Ok(None);
                    }
                    complementarity = complementarity.max(product.abs() / scale);
                    if row % 1024 == 1023 {
                        cooperation.completed(1024).await?;
                    }
                }
            }
            SupportedConeT::SecondOrderConeT(_) => {
                let mut norm = 0.0_f64;
                let mut product = 0.0_f64;
                for row in offset..offset + size {
                    if row > offset {
                        norm = norm.hypot(z[row]);
                    }
                    product += z[row] * (model.b[row] - point.activity[row]);
                    if row % 1024 == 1023 {
                        cooperation.completed(1024).await?;
                    }
                }
                let dual_scale = model.objective_scale + z[offset].abs() + norm;
                if !norm.is_finite() || !product.is_finite() || !dual_scale.is_finite() {
                    return Ok(None);
                }
                dual = dual.max((norm - z[offset]).max(0.0) / dual_scale);
                complementarity = complementarity.max(product.abs() / scale);
            }
            _ => return Ok(None),
        }
        offset += size;
        cooperation.completed(1).await?;
    }
    Ok(Some(Optimality {
        valid: point.normalized <= tol
            && dual <= tol
            && gap / scale <= tol
            && complementarity <= tol,
        point,
        dual,
        gap,
    }))
}
async fn dot(
    left: &[f64],
    right: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<f64>> {
    let mut sum = 0.0_f64;
    for first in (0..left.len()).step_by(1024) {
        let end = first.saturating_add(1024).min(left.len());
        for i in first..end {
            sum += left[i] * right[i];
        }
        if !sum.is_finite() {
            return Ok(None);
        }
        cooperation.completed((end - first) as u64).await?;
    }
    Ok(Some(sum))
}
