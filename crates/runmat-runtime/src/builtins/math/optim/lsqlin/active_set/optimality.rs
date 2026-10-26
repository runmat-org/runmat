//! Nonvanishing original residual-data scales, not candidate-gradient magnitudes.
use super::*;
pub(super) struct Metrics {
    pub optimality: f64,
    pub violation: f64,
}
pub(super) struct Scale {
    columns: Vec<f64>,
    maximum: f64,
}
#[derive(Clone, Copy)]
pub(super) struct Accuracy {
    pub feasibility: f64,
    pub optimality: f64,
}
impl Scale {
    pub(super) async fn new(
        model: &Model,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Self>> {
        let mut rows = zeros(model.c.m, c).await?;
        for column in 0..model.c.n {
            for entry in model.c.colptr[column]..model.c.colptr[column + 1] {
                let row = model.c.rowval[entry];
                rows[row] += model.c.nzval[entry].abs();
                if !rows[row].is_finite() {
                    return Ok(None);
                }
                c.completed(1).await?;
            }
            c.completed(1).await?;
        }
        let mut columns = capacity(model.c.n)?;
        let mut maximum = 0f64;
        for column in 0..model.c.n {
            let mut scale = 0f64;
            let mut nonzero = false;
            for entry in model.c.colptr[column]..model.c.colptr[column + 1] {
                let row = model.c.rowval[entry];
                let coefficient = model.c.nzval[entry].abs();
                let bound = rows[row] + model.d[row].abs();
                nonzero |= coefficient != 0. && model.weights[row] != 0. && bound != 0.;
                let term = model.units * model.weights[row] * coefficient * bound;
                if !term.is_finite() {
                    return Ok(None);
                }
                scale += term;
                if !scale.is_finite() {
                    return Ok(None);
                }
                c.completed(1).await?;
            }
            // A positive mathematical data scale that underflows is not a
            // zero objective and must never acquire a unit-floor certificate.
            if nonzero && scale == 0. {
                return Ok(None);
            }
            maximum = maximum.max(scale);
            columns.push(scale);
            c.completed(1).await?;
        }
        Ok(Some(Self { columns, maximum }))
    }
    pub(super) async fn checked(
        &self,
        model: &Model,
        rows: working_set::Rows<'_>,
        x: &[f64],
        z: &[f64],
        accuracy: Accuracy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Metrics>> {
        if x.len() != model.c.n || z.len() != rows.b.len() || rows.a.n != x.len() {
            return Ok(None);
        }
        if !working_set::rows::feasible(
            rows,
            x,
            accuracy.feasibility,
            runmat_builtins::LSQLIN_CATALOG_ENTRY.identity,
            c,
        )
        .await?
        {
            return Ok(None);
        }
        let Some(point) = model.evaluate(x, c).await? else {
            return Ok(None);
        };
        let mut magnitude = 0f64;
        for &value in x {
            magnitude = magnitude.max(value.abs());
            c.completed(1).await?;
        }
        let mut optimality = 0f64;
        for column in 0..x.len() {
            let mut total = point.gradient[column];
            let data = self.columns[column];
            let scale = if data == 0. {
                1.
            } else {
                data * (1. + magnitude)
            };
            if !(scale > 0. && scale.is_finite()) {
                return Ok(None);
            }
            for entry in rows.a.colptr[column]..rows.a.colptr[column + 1] {
                let value = rows.a.nzval[entry] * z[rows.a.rowval[entry]];
                if !value.is_finite() {
                    return Ok(None);
                }
                total += value;
                c.completed(1).await?;
            }
            if !total.is_finite() || (total / scale).abs() > accuracy.optimality {
                return Ok(None);
            }
            optimality = optimality.max((total / scale).abs());
            c.completed(1).await?;
        }
        let Some(activity) = working_set::rows::product(
            rows.a,
            x,
            runmat_builtins::LSQLIN_CATALOG_ENTRY.identity,
            c,
        )
        .await?
        else {
            return Ok(None);
        };
        let root = if self.maximum == 0. {
            1.
        } else {
            self.maximum.sqrt() * magnitude.max(1.)
        };
        if !(root > 0. && root.is_finite()) {
            return Ok(None);
        }
        let mut violation = 0f64;
        for (row, (&value, &rhs)) in activity.iter().zip(rows.b).enumerate() {
            let error = value - rhs;
            if !error.is_finite() {
                return Ok(None);
            }
            violation = violation.max(if row < rows.equalities {
                error.abs()
            } else {
                error.max(0.)
            });
            c.completed(1).await?;
        }
        for row in rows.equalities..rows.b.len() {
            if !z[row].is_finite() || z[row] < 0. {
                return Ok(None);
            }
            if z[row] != 0. {
                let normalized = (z[row] / root) * ((rows.b[row] - activity[row]) / root);
                if !normalized.is_finite() || normalized.abs() > accuracy.optimality {
                    return Ok(None);
                }
            }
            c.completed(1).await?;
        }
        Ok(Some(Metrics {
            optimality,
            violation,
        }))
    }
}
