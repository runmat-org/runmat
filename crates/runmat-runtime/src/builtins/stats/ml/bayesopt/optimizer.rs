use super::*;

pub(super) async fn optimize(
    runtime: Option<&crate::context::RuntimeContext>,
    objective: Value,
    variables: Vec<Variable>,
    options: BayesoptOptions,
) -> BuiltinResult<BayesoptOutcome> {
    if !is_callable(&objective) {
        return Err(bayesopt_error(
            "bayesopt: objective must be a function handle",
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    let start = Instant::now();
    let candidates = candidate_set(runtime, &variables, options.max_evals)?;
    let mut seen = HashSet::new();
    let mut observations = Vec::with_capacity(options.max_evals);
    let mut surrogate = SurrogateSummary::default();
    let mut exhausted_candidates = false;

    while observations.len() < options.max_evals {
        check_cancellation(runtime)?;
        let Some(point) = choose_next_point(
            runtime,
            &options,
            &candidates,
            &observations,
            &seen,
            &mut surrogate,
        )?
        else {
            exhausted_candidates = true;
            break;
        };
        let key = point.key();
        if !seen.insert(key) {
            exhausted_candidates = true;
            break;
        }
        if let Some(constraint) = &options.x_constraint {
            let row = point.to_table(&variables)?;
            if !constraint_allows(constraint, row).await? {
                continue;
            }
            check_cancellation(runtime)?;
        }
        let row = point.to_table(&variables)?;
        let eval_start = Instant::now();
        let callback = call_function(&objective, vec![row]).await.map_err(|err| {
            bayesopt_error_with_source(
                format!("bayesopt: objective evaluation failed: {}", err.message()),
                &ERROR_OBJECTIVE,
                err,
            )
        })?;
        check_cancellation(runtime)?;
        let elapsed = eval_start.elapsed().as_secs_f64();
        let objective_value = callback.with_owned_async(parse_objective_value).await?;
        observations.push(Observation {
            point,
            objective: objective_value.objective,
            constraint_violations: objective_value.constraint_violations,
            objective_time: elapsed,
        });
    }

    if observations.is_empty() {
        return Err(bayesopt_error(
            "bayesopt: no objective evaluations were completed",
            &ERROR_OBJECTIVE,
        ));
    }
    let total_time = start.elapsed().as_secs_f64();
    Ok(BayesoptOutcome {
        variables,
        options,
        observations,
        surrogate,
        total_time,
        exhausted_candidates,
    })
}

fn choose_next_point(
    runtime: Option<&crate::context::RuntimeContext>,
    options: &BayesoptOptions,
    candidates: &[Point],
    observations: &[Observation],
    seen: &HashSet<PointKey>,
    surrogate: &mut SurrogateSummary,
) -> BuiltinResult<Option<Point>> {
    if observations.len() < options.num_seed_points {
        return Ok(candidates
            .iter()
            .filter(|point| !seen.contains(&point.key()))
            .max_by(|a, b| {
                let da = min_distance_to_observations(a, observations);
                let db = min_distance_to_observations(b, observations);
                da.partial_cmp(&db).unwrap_or(Ordering::Equal)
            })
            .cloned());
    }
    let model = GaussianProcess::fit(runtime, observations)?;
    surrogate.length_scale = model.length_scale;
    surrogate.noise = GP_NOISE;
    let best = best_feasible_objective(observations).unwrap_or_else(|| {
        observations
            .iter()
            .map(Observation::penalized_objective)
            .fold(f64::INFINITY, f64::min)
    });
    let next = candidates
        .iter()
        .filter(|point| !seen.contains(&point.key()))
        .map(|point| {
            let (mean, variance) = model.predict(&point.encoded);
            let score = acquisition_score(
                options.acquisition,
                best,
                mean,
                variance,
                options.exploration_ratio,
            );
            (score, point)
        })
        .max_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .map(|(_, point)| point.clone());
    check_cancellation(runtime)?;
    Ok(next)
}

fn min_distance_to_observations(point: &Point, observations: &[Observation]) -> f64 {
    if observations.is_empty() {
        return f64::INFINITY;
    }
    observations
        .iter()
        .map(|observation| squared_distance(&point.encoded, &observation.point.encoded).sqrt())
        .fold(f64::INFINITY, f64::min)
}

fn acquisition_score(
    acquisition: Acquisition,
    best: f64,
    mean: f64,
    variance: f64,
    exploration: f64,
) -> f64 {
    let std = variance.max(0.0).sqrt();
    match acquisition {
        Acquisition::LowerConfidenceBound => -(mean - (1.0 + 4.0 * exploration) * std),
        Acquisition::ExpectedImprovement | Acquisition::ExpectedImprovementPlus => {
            if std <= 1.0e-12 {
                return 0.0;
            }
            let improvement = best - mean;
            let z = improvement / std;
            let ei = improvement * normal_cdf(z) + std * normal_pdf(z);
            if matches!(acquisition, Acquisition::ExpectedImprovementPlus) {
                ei + exploration * std * 1.0e-3
            } else {
                ei
            }
        }
    }
}

fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t)
            * (-x * x).exp();
    sign * y
}
