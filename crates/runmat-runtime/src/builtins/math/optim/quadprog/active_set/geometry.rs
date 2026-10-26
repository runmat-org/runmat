//! Numeric equality target checks against the supplied original H/f/E.
use super::*;
pub(super) struct Equality<'a> {
    pub model: &'a Model,
    pub rows: &'a Matrix,
    pub rhs: &'a [f64],
}
impl Equality<'_> {
    pub async fn point(
        &self,
        step: &working_set::Step,
        tolerance: f64,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        if step.direction.len() != self.model.f.len()
            || step.multipliers.len() != self.rhs.len()
            || step.multipliers.iter().any(|v| !v.is_finite())
        {
            return Ok(false);
        }
        if !working_set::rows::feasible(
            working_set::Rows {
                a: self.rows,
                b: self.rhs,
                equalities: self.rhs.len(),
            },
            &step.direction,
            tolerance,
            OWNER,
            c,
        )
        .await?
        {
            return Ok(false);
        }
        let Some(mut gradient) = hessian(self.model, &step.direction, c).await? else {
            return Ok(false);
        };
        for column in 0..gradient.len() {
            let mut magnitude = gradient[column].abs() + self.model.f[column].abs();
            gradient[column] += self.model.f[column];
            for entry in self.rows.colptr[column]..self.rows.colptr[column + 1] {
                let term = self.rows.nzval[entry] * step.multipliers[self.rows.rowval[entry]];
                gradient[column] += term;
                magnitude += term.abs();
                c.completed(1).await?;
            }
            if !gradient[column].is_finite()
                || !magnitude.is_finite()
                || gradient[column].abs() > tolerance * (1. + magnitude)
            {
                return Ok(false);
            }
        }
        Ok(true)
    }
    pub async fn recession(
        &self,
        point: &[f64],
        direction: &[f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        if direction.len() != point.len() || direction.iter().all(|&v| v == 0.) {
            return Ok(false);
        }
        let Some(delta) = working_set::rows::product(self.rows, direction, OWNER, c).await? else {
            return Ok(false);
        };
        let Some(hd) = hessian(self.model, direction, c).await? else {
            return Ok(false);
        };
        let Some(hx) = hessian(self.model, point, c).await? else {
            return Ok(false);
        };
        // This only filters proposals. The exact original authority separately
        // verifies E*d=0, curvature=0 and negative slope before Unbounded.
        let mut curvature = 0.;
        let mut magnitude = 0.;
        let mut slope = 0.;
        let mut norm = 0.0_f64;
        for ((&d, &hd), (&hx, &f)) in direction.iter().zip(&hd).zip(hx.iter().zip(&self.model.f)) {
            curvature += d * hd;
            magnitude += (d * hd).abs();
            slope += d * (hx + f);
            norm = norm.max(d.abs());
            c.completed(1).await?;
        }
        if !curvature.is_finite()
            || !magnitude.is_finite()
            || !slope.is_finite()
            || slope >= 0.
            || curvature.abs() > 32. * f64::EPSILON * magnitude.max(1.)
        {
            return Ok(false);
        }
        let mut scales = zeros(self.rows.m, c).await?;
        for column in 0..self.rows.n {
            for entry in self.rows.colptr[column]..self.rows.colptr[column + 1] {
                scales[self.rows.rowval[entry]] += self.rows.nzval[entry].abs() * norm;
                c.completed(1).await?;
            }
        }
        for (&value, scale) in delta.iter().zip(scales) {
            if !scale.is_finite() || value.abs() > 32. * f64::EPSILON * scale.max(1.) {
                return Ok(false);
            }
        }
        Ok(true)
    }
}
pub(super) async fn hessian(
    model: &Model,
    point: &[f64],
    c: &mut ScalarCooperation,
) -> BuiltinResult<Option<Vec<f64>>> {
    if point.len() != model.f.len() || point.iter().any(|v| !v.is_finite()) {
        return Ok(None);
    }
    let mut result = zeros(point.len(), c).await?;
    for column in 0..point.len() {
        for entry in model.p.colptr[column]..model.p.colptr[column + 1] {
            let row = model.p.rowval[entry];
            let value = model.p.nzval[entry];
            result[row] += value * point[column];
            if row != column {
                result[column] += value * point[row];
            }
            c.completed(1).await?;
        }
    }
    Ok(result.iter().all(|v| v.is_finite()).then_some(result))
}
