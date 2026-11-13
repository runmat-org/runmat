//! Lawson–Hanson feasible passive-set iterations, using owned RRQR/COD solves.
use super::*;
use crate::builtins::math::linalg::ops::sparse_factor::least_squares::PassiveMatrix;
async fn selected(
    passive: &[bool],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<usize>> {
    let mut count = 0;
    for chunk in passive.chunks(1024) {
        count += chunk.iter().filter(|&&v| v).count();
        cooperation.completed(chunk.len() as u64).await?;
    }
    let mut result = runmat_value::exact_capacity_vec(count).map_err(allocation)?;
    for (chunk_index, chunk) in passive.chunks(1024).enumerate() {
        for (offset, &value) in chunk.iter().enumerate() {
            if value {
                result.push(chunk_index * 1024 + offset);
            }
        }
        cooperation.completed(chunk.len() as u64).await?;
    }
    Ok(result)
}
fn physical(value: f64, scale: f64) -> BuiltinResult<()> {
    let x = value / scale;
    if !x.is_finite() || x < 0.0 || (value > 0.0 && x == 0.0) {
        return Err(numerical(
            "feasible iterate exceeds representable physical coordinate range",
        ));
    }
    Ok(())
}
fn boundary_fraction(value: f64, candidate: f64) -> f64 {
    if candidate == 0.0 {
        return 1.0;
    }
    let negative = -candidate;
    if value <= negative {
        let ratio = value / negative;
        ratio / (1.0 + ratio)
    } else {
        1.0 / (1.0 + negative / value)
    }
}
pub(super) async fn solve(
    model: &model::Model,
    controls: &mut options::Controls,
    runtime: &crate::context::RuntimeContext,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<model::Outcome> {
    let n = model.c.cols;
    let mut v = runmat_value::exact_capacity_vec(n).map_err(allocation)?;
    v.resize(n, 0.0);
    let mut passive = runmat_value::exact_capacity_vec(n).map_err(allocation)?;
    passive.resize(n, false);
    if model.scale == 0.0 {
        return validation::finalize(model, v, 1, controls.threshold, cooperation).await;
    }
    let matrix = PassiveMatrix::borrow(
        &model.scaled,
        runtime,
        runmat_builtins::LSQNONNEG_CATALOG_ENTRY.identity,
    )
    .await?;
    'outer: loop {
        let residual = validation::residual(&model.scaled, &v, &model.d, cooperation).await?;
        let dual = validation::dual(&model.scaled, &residual, cooperation).await?;
        let mut blocked = runmat_value::exact_capacity_vec(n).map_err(allocation)?;
        blocked.resize(n, false);
        loop {
            let mut entering = None;
            let mut maximum = controls.threshold;
            for j in 0..n {
                if !passive[j] && !blocked[j] && dual[j] > maximum {
                    entering = Some(j);
                    maximum = dual[j];
                }
                if j % 1024 == 1023 {
                    cooperation.completed(1024).await?;
                }
            }
            let Some(entering) = entering else {
                // Original physical verification—not a rejected probe—owns success.
                return validation::finalize(model, v, 1, controls.threshold, cooperation).await;
            };
            if controls.iterations >= controls.limit {
                return validation::finalize(model, v, 0, controls.threshold, cooperation).await;
            }
            passive[entering] = true;
            let mut indices = selected(&passive, cooperation).await?;
            let mut trial = matrix.solve(&indices, &model.d).await?;
            let entering_position = indices
                .binary_search(&entering)
                .expect("selected entering column");
            if trial.values.iter().any(|v| !v.is_finite()) {
                return Err(numerical("nonfinite passive solution"));
            }
            if trial.rank < indices.len() || trial.values[entering_position] <= 0.0 {
                passive[entering] = false;
                blocked[entering] = true;
                cooperation.completed(0).await?;
                // Rejected independence/prospective-positive probes do not count
                // as accepted secondary feasibility iterations.
                continue;
            }
            loop {
                if !controls.consume() {
                    return validation::finalize(model, v, 0, controls.threshold, cooperation)
                        .await;
                }
                let mut boundary = None;
                let mut alpha = 1.0_f64;
                for (position, &j) in indices.iter().enumerate() {
                    let z = trial.values[position];
                    if z <= 0.0 {
                        let fraction = boundary_fraction(v[j], z);
                        if boundary.is_none() || fraction < alpha {
                            alpha = fraction;
                            boundary = Some(j);
                        }
                    }
                    cooperation.completed(1).await?;
                }
                if let Some(hit) = boundary {
                    for (position, &j) in indices.iter().enumerate() {
                        if j == hit {
                            v[j] = 0.0;
                        } else {
                            v[j] = (1.0 - alpha) * v[j] + alpha * trial.values[position];
                        }
                        // This is an active constraint hit, not a post-solve clamp:
                        // nonpositive roundoff coordinates leave the passive set.
                        if v[j] <= 0.0 {
                            v[j] = 0.0;
                            passive[j] = false;
                        }
                        physical(v[j], model.scale)?;
                        cooperation.completed(1).await?;
                    }
                    if controls.iterations >= controls.limit {
                        return validation::finalize(model, v, 0, controls.threshold, cooperation)
                            .await;
                    }
                    indices = selected(&passive, cooperation).await?;
                    trial = matrix.solve(&indices, &model.d).await?;
                    if trial.rank < indices.len() || trial.values.iter().any(|v| !v.is_finite()) {
                        return Err(numerical(
                            "reduced passive solve is numerically uncertified",
                        ));
                    }
                } else {
                    for (position, &j) in indices.iter().enumerate() {
                        v[j] = trial.values[position];
                        physical(v[j], model.scale)?;
                        cooperation.completed(1).await?;
                    }
                    continue 'outer;
                }
            }
        }
    }
}
