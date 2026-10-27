//! Original-order bound multipliers and actual original-domain validation.
use super::*;
pub(super) struct Stop {
    pub flag: i32,
    pub message: &'static str,
    pub cg_iterations: usize,
    pub products: compiled::Products,
}
pub(super) struct Prepared<'a, V> {
    pub model: Model,
    pub point: Result<Point, &'static str>,
    pub stop: Stop,
    pub controls: options::Controls,
    pub validation: &'a mut V,
    pub c: &'a mut ScalarCooperation,
}
pub(super) async fn publish<V: compiled::Validation>(
    prepared: Prepared<'_, V>,
) -> BuiltinResult<Outcome> {
    let Prepared {
        model,
        point,
        stop,
        controls,
        validation,
        c,
    } = prepared;
    let point = match point {
        Ok(point) => point,
        Err(message) => {
            let mut out = Outcome::failure(-8, controls.iterations(), message);
            out.method = super::super::model::Method::Reflective {
                cg_iterations: stop.cg_iterations,
            };
            controls.summary(&out, c.runtime())?;
            return Ok(out);
        }
    };
    let Stop {
        flag,
        message,
        cg_iterations,
        products,
    } = stop;
    let mut lambda = None;
    if flag == 1 {
        let mut lower = zeros(model.c.n, c).await?;
        let mut upper = zeros(model.c.n, c).await?;
        let mut duals =
            runmat_value::exact_capacity_vec(model.bound_rows.len()).map_err(allocation)?;
        for (i, &g) in point.gradient.iter().enumerate() {
            if model.lower[i].is_finite() {
                lower[i] = g.max(0.);
            }
            if model.upper[i].is_finite() {
                upper[i] = (-g).max(0.);
            }
            c.completed(1).await?;
        }
        for role in &model.bound_rows {
            duals.push(match *role {
                conic::LinearRowRole::Lower(column) => lower[column],
                conic::LinearRowRole::Upper(column) => upper[column],
                _ => return Err(invalid("reflective bound row lost its original role")),
            });
            c.completed(1).await?;
        }
        if !validation
            .solution(
                &point.x,
                compiled::Rows::bounds(&model.bound_rows),
                &duals,
                controls.feasibility,
                controls.optimality,
                c,
            )
            .await?
        {
            let mut out = Outcome::failure(
                -8,
                controls.iterations(),
                "Candidate failed original model reflective KKT validation.",
            );
            out.method = super::super::model::Method::Reflective { cg_iterations };
            controls.summary(&out, c.runtime())?;
            return Ok(out);
        }
        if flag == 1 && products == compiled::Products::WithMultipliers {
            lambda = Some(super::super::model::Lambda {
                ineqlin: Vec::new(),
                eqlin: Vec::new(),
                lower,
                upper,
            });
        }
    } else if !validation
        .incumbent(&point.x, controls.feasibility, c)
        .await?
    {
        let mut out = Outcome::failure(
            -8,
            controls.iterations(),
            "Stopped point failed original model feasibility validation.",
        );
        out.method = super::super::model::Method::Reflective { cg_iterations };
        controls.summary(&out, c.runtime())?;
        return Ok(out);
    }
    let out = Outcome {
        x: Some(point.x),
        residual: Some(point.residual),
        resnorm: Some(point.value),
        lambda,
        flag,
        iterations: controls.iterations(),
        violation: 0.,
        optimality: point.optimality,
        gap: f64::NAN,
        message,
        method: super::super::model::Method::Reflective { cg_iterations },
    };
    controls.summary(&out, c.runtime())?;
    Ok(out)
}
