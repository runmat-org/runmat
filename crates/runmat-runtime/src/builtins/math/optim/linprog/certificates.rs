//! Independent checks use original, unscaled rows and costs, not backend status alone.
use super::*;
use model::ConicModel;

pub(super) struct Point {
    pub objective: f64,
    pub violation: f64,
    pub normalized: f64,
    activity: Vec<f64>,
}
pub(super) struct Kkt {
    pub point: Point,
    pub dual_residual: f64,
    pub raw_stationarity: f64,
    pub gap: f64,
    pub valid: bool,
}

pub(super) async fn primal(
    model: &ConicModel,
    x: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Point>> {
    if x.len() != model.f.len() || finite_norm(x, cooperation).await?.is_none() {
        return Ok(None);
    }
    let Some((activity, magnitude)) = products(model, x, cooperation).await? else {
        return Ok(None);
    };
    let mut violation: f64 = 0.0;
    let mut normalized: f64 = 0.0;
    for row in 0..model.b.len() {
        let error = activity[row] - model.b[row];
        let error = if row < model.equality_rows {
            error.abs()
        } else {
            error.max(0.0)
        };
        violation = violation.max(error);
        let scale = model.row_scale[row] + model.b[row].abs() + magnitude[row];
        if !scale.is_finite() {
            return Ok(None);
        }
        normalized = normalized.max(if scale == 0.0 { error } else { error / scale });
        if row % 1024 == 1023 {
            cooperation.completed(1024).await?;
        }
    }
    let Some((objective, _)) = dot_checked(&model.f, x, cooperation).await? else {
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

pub(super) async fn solution(
    model: &ConicModel,
    x: &[f64],
    z: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Kkt>> {
    if z.len() != model.b.len() || finite_norm(z, cooperation).await?.is_none() {
        return Ok(None);
    }
    let Some(point) = primal(model, x, cooperation).await? else {
        return Ok(None);
    };
    let mut dual_residual: f64 = 0.0;
    let mut raw_stationarity: f64 = 0.0;
    for col in 0..model.f.len() {
        let mut value = model.f[col];
        let mut magnitude = model.f[col].abs();
        for first in (model.a.colptr[col]..model.a.colptr[col + 1]).step_by(1024) {
            let end = first.saturating_add(1024).min(model.a.colptr[col + 1]);
            for entry in first..end {
                let product = model.a.nzval[entry] * z[model.a.rowval[entry]];
                value += product;
                magnitude += product.abs();
            }
            cooperation.completed((end - first) as u64).await?;
        }
        if !value.is_finite() || !magnitude.is_finite() {
            return Ok(None);
        }
        raw_stationarity = raw_stationarity.max(value.abs());
        let scale = model.objective_scale + magnitude;
        if !scale.is_finite() {
            return Ok(None);
        }
        dual_residual = dual_residual.max(value.abs() / scale);
        cooperation.completed(1).await?;
    }
    let Some((b_dot_z, _)) = dot_checked(&model.b, z, cooperation).await? else {
        return Ok(None);
    };
    let gap = (point.objective + b_dot_z).abs();
    let scale = model.objective_scale + point.objective.abs() + b_dot_z.abs();
    let mut complementarity: f64 = 0.0;
    for row in model.equality_rows..model.b.len() {
        dual_residual =
            dual_residual.max((-z[row]).max(0.0) / (model.objective_scale + z[row].abs()));
        complementarity =
            complementarity.max((z[row] * (model.b[row] - point.activity[row])).abs() / scale);
        if row % 1024 == 1023 {
            cooperation.completed(1024).await?;
        }
    }
    if !gap.is_finite() || !scale.is_finite() || !complementarity.is_finite() {
        return Ok(None);
    }
    let valid = point.normalized <= TOL
        && dual_residual <= TOL
        && gap / scale <= TOL
        && complementarity <= TOL;
    Ok(Some(Kkt {
        point,
        dual_residual,
        raw_stationarity,
        gap,
        valid,
    }))
}

// LP row/dual mapping remains private; shared conic mechanics borrow originals.
use crate::builtins::math::optim::common::conic;
pub(super) fn conic_view<'a>(
    model: &'a ConicModel,
    cones: &'a [clarabel::solver::SupportedConeT<f64>],
) -> conic::Model<'a> {
    conic::Model {
        p: None,
        a: &model.a,
        b: &model.b,
        f: &model.f,
        cones,
        identity: runmat_builtins::LINPROG_CATALOG_ENTRY.identity,
        objective_scale: model.objective_scale,
    }
}
async fn finite_norm(
    values: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<f64>> {
    let mut norm = 0.0_f64;
    for chunk in values.chunks(1024) {
        for value in chunk {
            if !value.is_finite() {
                return Ok(None);
            }
            norm = norm.max(value.abs());
        }
        cooperation.completed(chunk.len() as u64).await?;
    }
    Ok(Some(norm))
}

async fn products(
    model: &ConicModel,
    x: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<(Vec<f64>, Vec<f64>)>> {
    let mut result = vec![0.0; model.b.len()];
    let mut magnitude = vec![0.0; model.b.len()];
    for col in 0..model.f.len() {
        for first in (model.a.colptr[col]..model.a.colptr[col + 1]).step_by(1024) {
            let end = first.saturating_add(1024).min(model.a.colptr[col + 1]);
            for entry in first..end {
                let row = model.a.rowval[entry];
                let value = model.a.nzval[entry] * x[col];
                result[row] += value;
                magnitude[row] += value.abs();
                if !result[row].is_finite() || !magnitude[row].is_finite() {
                    return Ok(None);
                }
            }
            cooperation.completed((end - first) as u64).await?;
        }
        cooperation.completed(1).await?;
    }
    Ok(Some((result, magnitude)))
}
async fn dot_checked(
    left: &[f64],
    right: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<(f64, f64)>> {
    let mut sum = 0.0;
    let mut magnitude = 0.0;
    for first in (0..left.len()).step_by(1024) {
        let end = first.saturating_add(1024).min(left.len());
        for i in first..end {
            let value = left[i] * right[i];
            sum += value;
            magnitude += value.abs();
        }
        if !sum.is_finite() || !magnitude.is_finite() {
            return Ok(None);
        }
        cooperation.completed((end - first) as u64).await?;
    }
    Ok(Some((sum, magnitude)))
}
