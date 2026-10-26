//! One accepted point/Jacobian and one original residual evaluation budget.
use super::*;
mod terminal;
use terminal::{State, Stop};
pub(in crate::builtins::math::optim::least_squares) async fn solve<E: LeastSquaresEvaluator>(
    name: &str,
    evaluator: &mut E,
    point: initial::Seed,
    bounds: &LeastSquaresBounds,
    options: &LeastSquaresOptions,
    display: OptimDisplay,
    controls: controls::Controls,
) -> BuiltinResult<LeastSquaresResult> {
    let criteria = controls.criteria;
    let geometry = Geometry::new(bounds, evaluator).await?;
    let terminal = terminal::Terminal {
        name,
        geometry: &geometry,
        bounds,
        options,
        display,
    };
    let mut state = State {
        point,
        jacobian: None,
        iterations: 0,
        count: 1,
        cg_iterations: 0,
        optimality: f64::NAN,
        step: 0.,
    };
    let mut radius = (geometry.axes.len().max(1) as f64).sqrt().max(1.);
    loop {
        checkpoint(evaluator, 0).await?;
        if state.jacobian.is_none() {
            if let Some(jacobian) = evaluator.take_jacobian() {
                if jacobian.dimensions() != (state.point.residual.len(), state.point.x.len()) {
                    return Err(invalid(
                        "accepted Jacobian disagrees with current point".into(),
                    ));
                }
                state.jacobian = Some(jacobian);
            } else if geometry.axes.is_empty() {
                let count =
                    checked_jacobian_work(name, state.point.residual.len(), state.point.x.len())?;
                state.jacobian = Some(Jacobian::RowMajor {
                    values: zeros(count, evaluator).await?,
                    rows: state.point.residual.len(),
                    columns: state.point.x.len(),
                });
            } else {
                let measured = finite_difference_jacobian(
                    name,
                    evaluator,
                    &state.point.x,
                    bounds,
                    &state.point.residual,
                    &mut state.count,
                    options.max_fun_evals,
                    &options.scaling,
                )
                .await?;
                state.jacobian = Some(Jacobian::RowMajor {
                    values: measured.values,
                    rows: state.point.residual.len(),
                    columns: state.point.x.len(),
                });
                if measured.exhausted_budget {
                    return terminal.finish(evaluator, state, Stop::Limit).await;
                }
            }
        }
        let jacobian = state.jacobian.as_ref().expect("accepted Jacobian");
        let frame = {
            let c = cooperation(evaluator)?;
            let mut products = jacobian.products(c).await?;
            geometry
                .frame(&state.point.x, &state.point.residual, &mut products, c)
                .await?
        }
        .ok_or_else(|| invalid("reflective residual gradient is not representable".into()))?;
        state.optimality = frame.optimality;
        emit(evaluator, display, Event::Iteration, || {
            format!(
                "lsqnonlin: iteration={} funcCount={} firstorderopt={:.6e} stepsize={:.6e}",
                state.iterations, state.count, state.optimality, state.step
            )
        })?;
        if state.optimality <= criteria.optimality {
            return terminal.finish(evaluator, state, Stop::Optimality).await;
        }
        if state.iterations >= options.max_iter || state.count >= options.max_fun_evals {
            return terminal.finish(evaluator, state, Stop::Limit).await;
        }
        let metric = geometry
            .metric(&frame, options, evaluator)
            .await?
            .ok_or_else(|| {
                invalid("reflective bound-distance scaling is not representable".into())
            })?;
        let displacement = match controls.subproblem {
            controls::Subproblem::Factorization => {
                direct::step(
                    &geometry,
                    &state.point,
                    jacobian,
                    &metric,
                    radius,
                    cooperation(evaluator)?,
                )
                .await?
            }
            controls::Subproblem::Cg {
                bandwidth,
                maximum,
                tolerance,
            } => {
                let step = cg::step(
                    &geometry,
                    jacobian,
                    &metric,
                    radius,
                    bandwidth,
                    maximum,
                    tolerance,
                    cooperation(evaluator)?,
                )
                .await?;
                match step {
                    Some(step) => {
                        state.cg_iterations = state
                            .cg_iterations
                            .checked_add(step.iterations)
                            .ok_or_else(|| invalid("CG iteration count overflow".into()))?;
                        Some(step.displacement)
                    }
                    None => None,
                }
            }
        }
        .ok_or_else(|| {
            invalid("selected reflective subproblem did not qualify a trust step".into())
        })?;
        let direction = metric
            .physical(&displacement, cooperation(evaluator)?)
            .await?
            .ok_or_else(|| invalid("reflective physical direction is not representable".into()))?;
        let proposal = trajectory::propose(
            &geometry,
            &state.point,
            jacobian,
            &metric,
            &direction,
            radius,
            cooperation(evaluator)?,
        )
        .await?;
        state.iterations = state
            .iterations
            .checked_add(1)
            .ok_or_else(|| invalid("iteration count overflow".into()))?;
        let Some(proposal) = proposal else {
            radius *= 0.25;
            if !(radius > 0. && radius.is_finite()) {
                return Err(invalid(
                    "reflective trust radius is no longer representable".into(),
                ));
            }
            continue;
        };
        let next = evaluator.residual(&proposal.x).await?;
        state.count = state
            .count
            .checked_add(1)
            .ok_or_else(|| invalid("function count overflow".into()))?;
        if next.len() != state.point.residual.len() {
            return Err(invalid(
                "function output size changed during reflective iteration".into(),
            ));
        }
        let old_value = norm2_squared(&state.point.residual, evaluator).await?;
        let next_value = norm2_squared(&next, evaluator).await?;
        if !old_value.is_finite() || !next_value.is_finite() {
            return Err(invalid("reflective squared residual is not finite".into()));
        }
        let actual = 0.5 * (old_value - next_value);
        let ratio = actual / proposal.predicted;
        if ratio.is_finite() && ratio > 1e-4 && actual > 0. {
            let old_size = residual_norm_inf(&state.point.x, evaluator.cooperation()).await?;
            state.point = initial::Seed {
                x: proposal.x,
                residual: next,
            };
            evaluator.accept_residual();
            state.jacobian = None;
            state.optimality = f64::NAN;
            state.step = proposal.step;
            if ratio < 0.25 {
                radius *= 0.25;
            } else if ratio > 0.75 && proposal.length / radius >= 0.9 {
                radius = radius.min(f64::MAX / 2.) * 2.;
            }
            if state.step <= criteria.step * (1. + old_size) {
                return terminal.finish(evaluator, state, Stop::Step).await;
            }
            if old_value - next_value <= criteria.function * old_value {
                return terminal.finish(evaluator, state, Stop::Function).await;
            }
        } else {
            radius *= 0.25;
        }
        if !(radius > 0. && radius.is_finite()) {
            return Err(invalid(
                "reflective trust radius is no longer representable".into(),
            ));
        }
    }
}
