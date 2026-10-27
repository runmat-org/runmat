//! One original residual evaluation becomes the selected controller's seed.
use super::*;
#[derive(Clone, Copy)]
pub(super) enum Placement {
    Project,
    Interior,
}
pub(super) struct Seed {
    pub x: Vec<f64>,
    pub residual: Vec<f64>,
}
pub(super) enum Initial {
    Supplied(Vec<f64>),
    Evaluated(Seed),
}
pub(super) async fn prepare<E: LeastSquaresEvaluator>(
    name: &str,
    evaluator: &mut E,
    mut x: Vec<f64>,
    bounds: &LeastSquaresBounds,
    options: &LeastSquaresOptions,
    placement: Placement,
) -> BuiltinResult<Initial> {
    options.scaling.validate_variables(name, x.len())?;
    let n = x.len();
    if n == 0 {
        return Err(optim_error(
            name,
            format!("{name}: initial guess cannot be empty"),
        ));
    }
    bounds.validate(name, n)?;
    bounds.project(&mut x);
    if options.max_fun_evals > 0 && matches!(placement, Placement::Interior) {
        for (i, value) in x.iter_mut().enumerate() {
            if !value.is_finite() {
                return Err(optim_error(name, "initial guess must be finite"));
            }
            if bounds.lower[i] != bounds.upper[i] {
                if *value <= bounds.lower[i] {
                    *value = bounds.lower[i].next_up();
                }
                if *value >= bounds.upper[i] {
                    *value = bounds.upper[i].next_down();
                }
                if !value.is_finite() || *value <= bounds.lower[i] || *value >= bounds.upper[i] {
                    return Err(optim_error(
                        name,
                        "no representable strictly interior initial coordinate",
                    ));
                }
            }
            checkpoint(evaluator, 1).await?;
        }
    }

    checkpoint(evaluator, 0).await?;
    if options.max_fun_evals == 0 {
        return Ok(Initial::Supplied(x));
    }
    let residual = evaluator.residual(&x).await?;
    if residual.is_empty() {
        return Err(optim_error(
            name,
            format!("{name}: function value must not be empty"),
        ));
    }
    evaluator.accept_residual();
    Ok(Initial::Evaluated(Seed { x, residual }))
}

pub(super) struct Prepared<'a, E> {
    pub name: &'a str,
    pub evaluator: &'a mut E,
    pub seed: Initial,
    pub bounds: &'a LeastSquaresBounds,
    pub options: &'a LeastSquaresOptions,
}
pub(super) async fn invocation<'a, E: LeastSquaresEvaluator>(
    name: &'a str,
    evaluator: &'a mut E,
    x: Vec<f64>,
    bounds: &'a LeastSquaresBounds,
    options: &'a LeastSquaresOptions,
    placement: Placement,
) -> BuiltinResult<Prepared<'a, E>> {
    let seed = prepare(name, evaluator, x, bounds, options, placement).await?;
    Ok(Prepared {
        name,
        evaluator,
        seed,
        bounds,
        options,
    })
}
