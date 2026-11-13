//! Original residual and original KKT authority, independent of lifted status.
use super::*;
use model::Model;
pub(super) struct Checked {
    pub residual: Vec<f64>,
    pub resnorm: f64,
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
    if x.len() != model.c.n || x.iter().any(|v| !v.is_finite()) {
        return Ok(None);
    }
    let mut residual = model.d.iter().map(|v| -v).collect::<Vec<_>>();
    for (col, &value) in x.iter().enumerate() {
        for k in model.c.colptr[col]..model.c.colptr[col + 1] {
            residual[model.c.rowval[k]] += model.c.nzval[k] * value;
            if k % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        cooperation.completed(1).await?;
    }
    // Keep the public unweighted arithmetic unchanged. Compiled weighted
    // models never require the possibly overflowing unweighted squared norm.
    let resnorm = match model.weighting {
        objective::Weighting::Uniform => {
            let mut norm = 0.0_f64;
            for chunk in residual.chunks(1024) {
                for &v in chunk {
                    if !v.is_finite() {
                        return Ok(None);
                    }
                    norm = norm.hypot(v);
                }
                cooperation.completed(chunk.len() as u64).await?;
            }
            norm * norm
        }
        objective::Weighting::PerRow => {
            let mut squared = 0.0;
            for (row, &value) in residual.iter().enumerate() {
                if !value.is_finite() {
                    return Ok(None);
                }
                let Some(weight) = model.weight(row) else {
                    return Ok(None);
                };
                squared += (weight * value) * value;
                if !squared.is_finite() {
                    return Ok(None);
                }
                cooperation.completed(1).await?;
            }
            squared
        }
    };
    if !resnorm.is_finite() {
        return Ok(None);
    }
    let activity = activity(model, x, cooperation).await?;
    let mut violation = 0.0_f64;
    for row in model.c.m..model.b.len() {
        let error = activity[row] - model.b[row];
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
    Ok(Some(Checked {
        residual,
        resnorm,
        violation,
        optimality: f64::NAN,
        gap: f64::NAN,
        valid: false,
    }))
}
async fn activity(
    model: &Model,
    x: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    let mut activity = vec![0.0; model.b.len()];
    for (col, &value) in x.iter().enumerate() {
        for k in model.a.colptr[col]..model.a.colptr[col + 1] {
            let row = model.a.rowval[k];
            if row >= model.c.m {
                activity[row] += model.a.nzval[k] * value;
            }
            if k % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        cooperation.completed(1).await?;
    }
    Ok(activity)
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
    let Some(mut checked) = primal(model, x, cooperation).await? else {
        return Ok(None);
    };
    let mut optimality = 0.0_f64;
    for col in 0..model.c.n {
        let mut sum = 0.0;
        let mut magnitude = 0.0;
        for k in model.c.colptr[col]..model.c.colptr[col + 1] {
            let row = model.c.rowval[k];
            let residual = match model.weighting {
                objective::Weighting::Uniform => checked.residual[row],
                objective::Weighting::PerRow => {
                    let Some(weight) = model.weight(row) else {
                        return Ok(None);
                    };
                    weight * checked.residual[row]
                }
            };
            let term = model.c.nzval[k] * residual;
            sum += term;
            magnitude += term.abs();
            if k % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        for k in model.a.colptr[col]..model.a.colptr[col + 1] {
            let row = model.a.rowval[k];
            if row < model.c.m {
                continue;
            }
            let term = model.a.nzval[k] * z[row];
            sum += term;
            magnitude += term.abs();
            if k % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        if !sum.is_finite() || !magnitude.is_finite() {
            return Ok(None);
        }
        optimality = optimality.max(sum.abs() / (1.0 + magnitude));
        cooperation.completed(1).await?;
    }
    // At a least-squares KKT point: r'r + d'r + b_external'z_external = 0.
    // Lifted residual-equation duals are never used as original multipliers.
    let mut rhs = 0.0;
    let mut dr = 0.0;
    for i in 0..model.c.m {
        let residual = match model.weighting {
            objective::Weighting::Uniform => checked.residual[i],
            objective::Weighting::PerRow => {
                let Some(weight) = model.weight(i) else {
                    return Ok(None);
                };
                weight * checked.residual[i]
            }
        };
        dr += model.d[i] * residual;
        if i % 1024 == 1023 {
            cooperation.completed(1024).await?;
        }
    }
    for (row, &dual) in z.iter().enumerate().skip(model.c.m) {
        rhs += model.b[row] * dual;
        if row % 1024 == 1023 {
            cooperation.completed(1024).await?;
        }
    }
    let gap = (checked.resnorm + dr + rhs).abs();
    let scale = 1.0 + checked.resnorm + dr.abs() + rhs.abs();
    if !gap.is_finite() || !scale.is_finite() {
        return Ok(None);
    }
    let activity = activity(model, x, cooperation).await?;
    let mut complementarity = 0.0_f64;
    let mut cone_error = 0.0_f64;
    for row in model.eq..model.b.len() {
        cone_error = cone_error.max((-z[row]).max(0.0) / (1.0 + z[row].abs()));
        let product = z[row] * (model.b[row] - activity[row]);
        if !product.is_finite() {
            return Ok(None);
        }
        complementarity = complementarity.max(product.abs() / scale);
        if row % 1024 == 1023 {
            cooperation.completed(1024).await?;
        }
    }
    checked.valid = checked.violation <= controls.feasibility
        && optimality <= controls.optimality
        && cone_error <= controls.optimality
        && gap / scale <= controls.optimality
        && complementarity <= controls.optimality;
    checked.optimality = optimality;
    checked.gap = gap;
    Ok(Some(checked))
}
