//! Numerical iteration retires before original-DAG terminal validation.
use super::*;
pub(in crate::builtins::math::optim::lsqlin) fn solve<'a, V: compiled::Validation + 'a>(
    problem: Problem,
    products: compiled::Products,
    units: objective::Units,
    c: &'a mut ScalarCooperation,
    validation: &'a mut V,
) -> impl std::future::Future<Output = BuiltinResult<Outcome>> + 'a {
    crate::execution::after_result(
        run(problem, products, units, c, validation),
        terminal::publish,
    )
}
async fn run<'a, V: compiled::Validation + 'a>(
    problem: Problem,
    products: compiled::Products,
    units: objective::Units,
    c: &'a mut ScalarCooperation,
    validation: &'a mut V,
) -> BuiltinResult<terminal::Prepared<'a, V>> {
    let (model, mut controls, initial) = Model::new(problem, units, c).await?;
    let settings = controls
        .reflective
        .as_ref()
        .expect("selected reflective controls");
    settings.typical.validate_variables(NAME, model.c.n)?;
    let maximum = settings
        .maximum_cg_iterations
        .unwrap_or((model.c.n / 2).max(1));
    let point = match model.start(initial, c).await? {
        Some(x) => model
            .evaluate(x, c)
            .await?
            .ok_or("Starting residual or gradient is not finite."),
        None => Err("No representable strictly interior starting point."),
    };
    let mut point = match point {
        Ok(point) => point,
        Err(message) => {
            return Ok(terminal::Prepared {
                model,
                point: Err(message),
                controls,
                stop: terminal::Stop {
                    flag: -8,
                    message,
                    cg_iterations: 0,
                    products,
                },
                validation,
                c,
            })
        }
    };
    let mut radius = (model.c.n as f64).sqrt().max(1.);
    let mut cg_iterations = 0usize;
    let (flag, message) = loop {
        c.completed(0).await?;
        if controls.remaining() == 0 {
            break (0, "Iteration limit reached.");
        }
        if point.optimality <= controls.optimality {
            break (1, "First-order reflective optimality tolerance satisfied.");
        }
        let settings = controls
            .reflective
            .as_ref()
            .expect("selected reflective controls");
        let Some(metric) = trust_quadratic::bounds::Metric::new(
            &point.x,
            &point.gradient,
            trust_quadratic::bounds::Bounds {
                lower: &model.lower,
                upper: &model.upper,
            },
            |i| settings.typical.typical_value(i),
            policy(),
            c,
        )
        .await?
        else {
            break (-8, "Reflective distance scaling is not representable.");
        };
        let displacement =
            match settings.method {
                options::Method::Factorization => {
                    match direct::step(&model, &point, &metric, radius, c).await? {
                        Some(step) => Some(step),
                        None => break (
                            -8,
                            "Direct residual factorization did not produce a qualified trust step.",
                        ),
                    }
                }
                options::Method::Cg { bandwidth } => {
                    let mut attempted = 0usize;
                    let step = {
                        let mut operator = products::Products {
                            model: &model,
                            metric: &metric,
                            scratch: zeros(model.c.m, c).await?,
                        };
                        let Some(preconditioner) =
                            cg::Preconditioner::new(&mut operator, model.c.n, bandwidth, c).await?
                        else {
                            break (-8, "Reflective preconditioner is not representable.");
                        };
                        cg::step_with_control(
                            &mut operator,
                            &preconditioner,
                            &metric.gradient,
                            radius,
                            cg::IterationControl {
                                maximum,
                                tolerance: settings.cg_tolerance,
                                completed: Some(&mut attempted),
                            },
                            c,
                        )
                        .await?
                    };
                    cg_iterations = cg_iterations
                        .checked_add(attempted)
                        .ok_or_else(|| invalid("CG iteration count overflow"))?;
                    step.map(|step| step.displacement)
                }
            };
        let direction = match displacement {
            Some(displacement) => metric.physical(&displacement, c).await?,
            None => None,
        };
        let trial = if let Some(direction) = direction {
            trajectory::search(&model, &point, &direction, &metric, radius, c).await?
        } else {
            None
        };
        let trial = match trial {
            Some(trial) => Some(trial),
            None => {
                let length = norm(&metric.gradient);
                if length > 0. && length.is_finite() {
                    let mut direction =
                        runmat_value::exact_capacity_vec(model.c.n).map_err(allocation)?;
                    for (&g, &scale) in metric.gradient.iter().zip(&metric.scale) {
                        direction.push(-(g / length) * radius * scale);
                        c.completed(1).await?;
                    }
                    trajectory::search(&model, &point, &direction, &metric, radius, c).await?
                } else {
                    None
                }
            }
        };
        controls.advance()?;
        if let Some(trial) = trial {
            if trial.ratio < 0.25 {
                radius *= 0.25;
            } else if trial.ratio > 0.75 && trial.boundary {
                radius = radius.min(f64::MAX / 2.) * 2.;
            }
            let reduction = point.value - trial.point.value;
            let tolerance = controls
                .reflective
                .as_ref()
                .expect("selected reflective controls")
                .function_tolerance;
            let small = reduction <= tolerance * point.value;
            point = trial.point;
            if point.optimality <= controls.optimality {
                break (1, "First-order reflective optimality tolerance satisfied.");
            }
            if small {
                break (
                    3,
                    "Change in the squared residual is below FunctionTolerance.",
                );
            }
        } else {
            radius *= 0.25;
        }
        if !(radius > 0. && radius.is_finite()) {
            break (-8, "Trust-region radius is not representable.");
        }
    };
    Ok(terminal::Prepared {
        model,
        point: Ok(point),
        controls,
        stop: terminal::Stop {
            flag,
            message,
            cg_iterations,
            products,
        },
        validation,
        c,
    })
}
