//! One original CSC/residual/bound owner, without residual lifting or C'C.
use super::*;
pub(super) struct Model {
    pub c: Matrix,
    pub d: Vec<f64>,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub weights: Vec<f64>,
    pub units: f64,
    pub bound_rows: Vec<conic::LinearRowRole>,
}
pub(super) struct Point {
    pub x: Vec<f64>,
    pub residual: Vec<f64>,
    pub value: f64,
    pub gradient: Vec<f64>,
    pub optimality: f64,
}
impl Model {
    pub(super) async fn new(
        problem: Problem,
        units: objective::Units,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<(Self, options::Controls, Vec<f64>)> {
        let Problem {
            c: matrix,
            d,
            a,
            b,
            aeq,
            beq,
            lb,
            ub,
            x0,
            controls,
            weights,
        } = problem;
        if a.m != 0 || aeq.m != 0 || !b.is_empty() || !beq.is_empty() {
            return Err(invalid(
                "reflective least squares requires bounds without linear rows",
            ));
        }
        let mut rows = Vec::new();
        for (column, (&lower, &upper)) in lb.iter().zip(&ub).enumerate() {
            if lower >= upper {
                return Err(invalid(
                    "trust-region-reflective does not allow fixed or inconsistent bounds",
                ));
            }
            for (finite, role) in [
                (lower.is_finite(), conic::LinearRowRole::Lower(column)),
                (upper.is_finite(), conic::LinearRowRole::Upper(column)),
            ] {
                if finite {
                    rows.try_reserve(1).map_err(|source| {
                        build_runtime_error("lsqlin: bound row allocation failed")
                            .with_builtin(NAME)
                            .with_source(source)
                            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                            .build()
                    })?;
                    rows.push(role);
                }
            }
            c.completed(1).await?;
        }
        let (weights, _, _) = weights.into_values(matrix.m, c).await?;
        Ok((
            Self {
                c: matrix,
                d,
                lower: lb,
                upper: ub,
                weights,
                units: units.multiplier(),
                bound_rows: rows,
            },
            controls,
            x0,
        ))
    }
    pub(super) async fn evaluate(
        &self,
        x: Vec<f64>,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Point>> {
        let evaluated = super::super::residual::evaluate(
            super::super::residual::View {
                c: &self.c,
                d: &self.d,
                weights: &self.weights,
                units: self.units,
            },
            x,
            allocation,
            |column, g, x| {
                let distance = if g < 0. && self.upper[column].is_finite() {
                    self.upper[column] - x
                } else if g >= 0. && self.lower[column].is_finite() {
                    x - self.lower[column]
                } else {
                    1.
                };
                let first = (distance * g).abs();
                first.is_finite().then_some(first)
            },
            c,
        )
        .await?;
        Ok(evaluated.map(|point| Point {
            x: point.x,
            residual: point.residual,
            value: point.value,
            gradient: point.gradient,
            optimality: point.measure,
        }))
    }
    pub(super) async fn start(
        &self,
        mut x: Vec<f64>,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Vec<f64>>> {
        let default = x.is_empty();
        if default {
            x = zeros(self.c.n, c).await?;
        }
        if x.len() != self.c.n {
            return Err(invalid("x0 must have one element per C column"));
        }
        let mut reset = false;
        if default {
            for (i, &value) in x.iter().enumerate() {
                reset |= value < self.lower[i] || value > self.upper[i];
                c.completed(1).await?;
            }
        }
        for (i, value) in x.iter_mut().enumerate() {
            if !value.is_finite() {
                return Err(invalid("x0 must be finite"));
            }
            let (lower, upper) = (self.lower[i], self.upper[i]);
            if reset || *value < lower || *value > upper {
                *value = if lower.is_finite() && upper.is_finite() {
                    0.5 * lower + 0.5 * upper
                } else if lower.is_finite() {
                    lower.max(0.)
                } else if upper.is_finite() {
                    upper.min(0.)
                } else {
                    0.
                };
            }
            if *value <= lower {
                *value = lower.next_up();
            }
            if *value >= upper {
                *value = upper.next_down();
            }
            if !value.is_finite() || *value <= lower || *value >= upper {
                return Ok(None);
            }
            c.completed(1).await?;
        }
        Ok(Some(x))
    }
}
