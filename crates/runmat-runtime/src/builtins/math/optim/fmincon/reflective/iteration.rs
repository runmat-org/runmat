//! Accepted-generation trust-region control. A rejected trial cannot replace
//! the model owner; a stopping accepted point needs no unused successor model.
use super::*;
use crate::builtins::math::optim::common::output_callback::OutputState;
use crate::execution::RootedValue;

async fn supplied(
    source: &controls::Source,
    raw: RootedValue,
    n: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<model::Model> {
    match source {
        controls::Source::Objective => model::matrix(raw, n, cooperation).await,
        controls::Source::Multiply { function } => Ok(model::Model::multiply(
            raw,
            function.clone(),
            model::policy(),
        )),
        controls::Source::GradientDifferences { .. } => Err(super::super::invalid(
            "gradient-difference model has an unexpected third objective result",
        )),
    }
}
pub(in crate::builtins::math::optim::fmincon) async fn minimize(
    x: Vec<f64>,
    geometry: &Geometry,
    parameters: &Parameters<'_>,
    controls: &Controls,
    mode: Mode,
    evaluator: &mut Evaluator<'_>,
) -> BuiltinResult<publication::Solved> {
    let n = x.len();
    let third = !matches!(
        controls.source,
        controls::Source::GradientDifferences { .. }
    );
    let pattern = match &controls.source {
        controls::Source::GradientDifferences {
            pattern: Some(value),
        } => Some(
            differences::Pattern::prepare(value, n, evaluator.cooperation, model::policy()).await?,
        ),
        _ => None,
    };
    let initialization::Started {
        outcome: mut out,
        equality,
        mut raw,
    } = initialization::prepare(x, geometry, parameters, mode, third, evaluator).await?;
    let mut info = publication::Info {
        cg_iterations: 0,
        radius: (n as f64).sqrt().max(1.0),
    };
    let mut last = None;
    let mut status = if out.point.is_some() {
        stationarity::refresh(&mut out, geometry, equality.as_ref(), parameters, evaluator).await?
    } else {
        stationarity::State::Continue
    };
    if publication::observe(&out, parameters, info, OutputState::Init, &[], evaluator).await? {
        out.termination = Termination::Observer;
    } else if out.point.is_some() {
        loop {
            if evaluator.time_limit_reached() {
                out.termination = Termination::Time;
                break;
            }
            match status {
                stationarity::State::Optimal => {
                    out.termination = Termination::Optimality;
                    break;
                }
                stationarity::State::Numerical => {
                    out.termination = Termination::Numerical;
                    break;
                }
                stationarity::State::Continue => {}
            }
            if out.iterations >= parameters.max_iterations || evaluator.budget.remaining() == 0 {
                out.termination = Termination::Limit;
                break;
            }
            if equality.as_ref().is_some_and(|equality| equality.free == 0) {
                out.termination = Termination::Numerical;
                break;
            }
            let next = match &controls.source {
                controls::Source::GradientDifferences { .. } => match differences::build(
                    &out.x,
                    out.point.as_ref().expect("accepted point"),
                    geometry,
                    pattern.as_ref(),
                    parameters.differences,
                    evaluator,
                )
                .await?
                {
                    Ok(model) => model,
                    Err(differences::Stop::Time) => {
                        out.termination = Termination::Time;
                        break;
                    }
                    Err(differences::Stop::Budget) => {
                        out.termination = Termination::Limit;
                        break;
                    }
                    Err(differences::Stop::Undefined) => {
                        out.termination = Termination::Numerical;
                        break;
                    }
                },
                source => {
                    supplied(
                        source,
                        raw.take()
                            .expect("selected third result belongs to accepted generation"),
                        n,
                        evaluator.cooperation,
                    )
                    .await?
                }
            };
            last = Some(next);
            let result = {
                let mut products = last
                    .as_mut()
                    .expect("constructed current model")
                    .products(evaluator.cooperation)
                    .await?;
                match mode {
                    Mode::Bounds => {
                        let Some(metric) = bounds::Metric::new(
                            &out.x,
                            out.point
                                .as_ref()
                                .expect("accepted point")
                                .gradient
                                .as_ref()
                                .expect("supplied gradient"),
                            trust_quadratic::bounds::Bounds {
                                lower: &geometry.lower,
                                upper: &geometry.upper,
                            },
                            |_| 1.0,
                            model::policy(),
                            evaluator.cooperation,
                        )
                        .await?
                        else {
                            out.termination = Termination::Numerical;
                            break;
                        };
                        let mut space = subproblem::Space::Bounds(bounds::Scaled {
                            base: &mut products,
                            metric: &metric,
                            scratch: zeros(n, evaluator.cooperation).await?,
                        });
                        trials::search(
                            &mut out,
                            &mut space,
                            &metric.gradient,
                            geometry,
                            parameters,
                            controls,
                            &mut info,
                            evaluator,
                        )
                        .await?
                    }
                    Mode::Equalities => {
                        let equality = equality.as_ref().expect("admitted equality factor");
                        let Some(gradient) = equality
                            .gradient(
                                out.point
                                    .as_ref()
                                    .expect("accepted point")
                                    .gradient
                                    .as_ref()
                                    .expect("supplied gradient"),
                                evaluator.cooperation,
                            )
                            .await?
                        else {
                            out.termination = Termination::Numerical;
                            break;
                        };
                        let mut space = subproblem::Space::Equality(equality::Projected {
                            base: &mut products,
                            equality,
                            input: zeros(n, evaluator.cooperation).await?,
                            rhs: zeros(geometry.equality.matrix.m, evaluator.cooperation).await?,
                            correction: zeros(n, evaluator.cooperation).await?,
                        });
                        trials::search(
                            &mut out, &mut space, &gradient, geometry, parameters, controls,
                            &mut info, evaluator,
                        )
                        .await?
                    }
                }
            };
            let accepted = match result {
                Ok(accepted) => accepted,
                Err(termination) => {
                    out.termination = termination;
                    break;
                }
            };
            raw = accepted.raw;
            status =
                stationarity::refresh(&mut out, geometry, equality.as_ref(), parameters, evaluator)
                    .await?;
            if publication::observe(
                &out,
                parameters,
                info,
                OutputState::Iter,
                &accepted.direction,
                evaluator,
            )
            .await?
            {
                out.termination = Termination::Observer;
                break;
            }
            if matches!(status, stationarity::State::Optimal) {
                out.termination = Termination::Optimality;
                break;
            }
            if matches!(status, stationarity::State::Numerical) {
                out.termination = Termination::Numerical;
                break;
            }
            if accepted.function_small
                && out
                    .violation
                    .is_some_and(|value| value <= parameters.feasibility)
            {
                out.termination = Termination::Function;
                break;
            }
            if accepted.radius_exhausted {
                out.termination = Termination::Numerical;
                break;
            }
        }
    }
    out.counts = evaluator.budget.counts();
    let _ = publication::observe(&out, parameters, info, OutputState::Done, &[], evaluator).await?;
    Ok(publication::Solved {
        outcome: out,
        model: last,
        info,
    })
}
