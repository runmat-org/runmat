//! Coleman--Li distance scaling and an actual reflected bound trajectory.
//! The original Hessian is unchanged; its metric correction belongs only to
//! the transformed reflective quadratic subproblem.
use super::*;
pub(super) use trust_quadratic::bounds::Metric;
use trust_quadratic::Operator;
pub(super) struct Scaled<'a, 'model> {
    pub base: &'a mut model::Products<'model>,
    pub metric: &'a Metric,
    pub scratch: Vec<f64>,
}
impl Operator for Scaled<'_, '_> {
    fn policy(&self) -> trust_quadratic::Policy {
        model::policy()
    }
    async fn apply(
        &mut self,
        input: &[f64],
        output: &mut [f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<()>> {
        for ((to, &input), &scale) in self.scratch.iter_mut().zip(input).zip(&self.metric.scale) {
            *to = input * scale;
            if !to.is_finite() {
                return Ok(None);
            }
            cooperation.completed(1).await?;
        }
        if self
            .base
            .apply_numerical(&self.scratch, output, cooperation)
            .await?
            .is_none()
        {
            return Ok(None);
        }
        for (i, value) in output.iter_mut().enumerate() {
            *value = self.metric.scale[i] * *value + self.metric.diagonal[i] * input[i];
            if !value.is_finite() {
                return Ok(None);
            }
            cooperation.completed(1).await?;
        }
        Ok(Some(()))
    }
    async fn copy_band(
        &self,
        n: usize,
        bandwidth: usize,
        output: &mut [f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<trust_quadratic::Materialization> {
        if !self
            .base
            .copy_band(n, bandwidth, output, cooperation)
            .await?
        {
            return Ok(trust_quadratic::Materialization::Unavailable);
        }
        let stride = bandwidth
            .checked_add(1)
            .ok_or_else(|| super::super::invalid("reflective band extent overflow"))?;
        for row in 0..n {
            for column in row.saturating_sub(bandwidth)..=row {
                let index = row * stride + row - column;
                output[index] *= self.metric.scale[row] * self.metric.scale[column];
                if row == column {
                    output[index] += self.metric.diagonal[row];
                }
                if !output[index].is_finite() {
                    return Ok(trust_quadratic::Materialization::Undefined);
                }
                cooperation.completed(1).await?;
            }
        }
        Ok(trust_quadratic::Materialization::Complete)
    }
}

async fn boundary(
    x: &[f64],
    p: &[f64],
    geometry: &Geometry,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<(f64, Option<usize>)>> {
    let mut alpha = 1.0_f64;
    let mut hit = None;
    for (i, (&x, &p)) in x.iter().zip(p).enumerate() {
        let ratio = if p > 0.0 && geometry.upper[i].is_finite() {
            (geometry.upper[i] - x) / p
        } else if p < 0.0 && geometry.lower[i].is_finite() {
            (geometry.lower[i] - x) / p
        } else {
            f64::INFINITY
        };
        if ratio.is_nan() || ratio < 0.0 {
            return Ok(None);
        }
        if ratio < alpha {
            alpha = ratio;
            hit = Some(i)
        }
        cooperation.completed(1).await?;
    }
    Ok(Some((alpha, hit)))
}
pub(super) async fn limited(
    x: &[f64],
    p: &[f64],
    geometry: &Geometry,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Vec<f64>>> {
    let Some((alpha, hit)) = boundary(x, p, geometry, cooperation).await? else {
        return Ok(None);
    };
    let alpha = if hit.is_some() { 0.995 * alpha } else { alpha };
    let mut result = workspace::capacity(p.len(), IDENTITY)?;
    for (i, (&x, &p)) in x.iter().zip(p).enumerate() {
        let value = alpha * p;
        let next = x + value;
        if !next.is_finite() || next <= geometry.lower[i] || next >= geometry.upper[i] {
            return Ok(None);
        }
        result.push(value);
        cooperation.completed(1).await?;
    }
    Ok(Some(result))
}
pub(in crate::builtins::math::optim::fmincon) async fn reflected(
    x: &[f64],
    p: &[f64],
    geometry: &Geometry,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Vec<f64>>> {
    let Some((alpha, Some(hit))) = boundary(x, p, geometry, cooperation).await? else {
        return Ok(None);
    };
    let mut edge = workspace::capacity(x.len(), IDENTITY)?;
    let mut direction = workspace::capacity(x.len(), IDENTITY)?;
    for (i, (&x, &p)) in x.iter().zip(p).enumerate() {
        let value = x + alpha * p;
        if !value.is_finite() {
            return Ok(None);
        }
        edge.push(if i == hit {
            if p > 0.0 {
                geometry.upper[i]
            } else {
                geometry.lower[i]
            }
        } else {
            value
        });
        direction.push((1.0 - alpha) * if i == hit { -p } else { p });
        cooperation.completed(1).await?;
    }
    let Some((fraction, next)) = boundary(&edge, &direction, geometry, cooperation).await? else {
        return Ok(None);
    };
    let fraction = if next.is_some() {
        0.995 * fraction
    } else {
        fraction
    };
    let mut result = workspace::capacity(x.len(), IDENTITY)?;
    for (i, ((&edge, &direction), &x)) in edge.iter().zip(&direction).zip(x).enumerate() {
        let candidate = edge + fraction * direction;
        if !candidate.is_finite()
            || candidate <= geometry.lower[i]
            || candidate >= geometry.upper[i]
        {
            return Ok(None);
        }
        let value = candidate - x;
        if !value.is_finite() {
            return Ok(None);
        }
        result.push(value);
        cooperation.completed(1).await?;
    }
    Ok(Some(result))
}
