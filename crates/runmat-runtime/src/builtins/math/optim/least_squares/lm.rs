//! Existing Levenberg--Marquardt transitions with one already accepted seed.
use super::*;
pub(super) async fn solve<E: LeastSquaresEvaluator>(
    name: &str,
    evaluator: &mut E,
    seed: initial::Seed,
    bounds: &LeastSquaresBounds,
    options: &LeastSquaresOptions,
    display: runmat_builtins::OptimDisplay,
) -> BuiltinResult<LeastSquaresResult> {
    let initial::Seed {
        mut x,
        mut residual,
    } = seed;
    let criteria = options.criteria(Method::LevenbergMarquardt);
    let n = x.len();
    let m = residual.len();
    let jacobian_len = checked_least_squares_work(name, m, n)?;
    let mut func_count = 1usize;
    let mut iterations = 0usize;
    let mut lambda = 1.0e-3;
    let mut last_optimality = f64::NAN;
    let mut last_step_size = 0.0;
    super::display::emit(
        evaluator.cooperation().and_then(|c| c.runtime()),
        display,
        super::display::Event::Iteration,
        || "lsqnonlin: iteration=0 funcCount=1 firstorderopt=NaN stepsize=0".to_owned(),
    )?;

    if options.criteria.is_none()
        && residual_norm_inf(&residual, evaluator.cooperation()).await? <= options.tol_fun
    {
        if !options.final_jacobian {
            return result(
                evaluator,
                bounds,
                options,
                x,
                residual,
                Vec::new(),
                1,
                0,
                func_count,
                f64::NAN,
                0.0,
                "Residual is within function tolerance; final Jacobian was not requested.",
            )
            .await;
        }
        let evaluated = finite_difference_jacobian(
            name,
            evaluator,
            &x,
            bounds,
            &residual,
            &mut func_count,
            options.max_fun_evals,
            &options.scaling,
        )
        .await?;
        let optimality = if evaluated.exhausted_budget {
            f64::NAN
        } else {
            first_order_optimality(&evaluated.values, &residual, n, evaluator).await?
        };
        return result(
            evaluator,bounds,options,x,residual,evaluated.values,1,iterations,func_count,
            optimality,0.0,if evaluated.exhausted_budget {
                "Residual is within tolerance; unevaluated final Jacobian coefficients and first-order optimality are NaN."
            } else { "Local minimum found. Residual is within function tolerance." },
        ).await;
    }

    for iter in 0..options.max_iter {
        checkpoint(evaluator, x.len()).await?;
        iterations = iter + 1;
        if func_count >= options.max_fun_evals {
            return result(
                evaluator,
                bounds,
                options,
                x,
                residual,
                Vec::new(),
                0,
                iterations.saturating_sub(1),
                func_count,
                last_optimality,
                last_step_size,
                "Exceeded maximum function evaluations.",
            )
            .await;
        }

        let jacobian_eval = finite_difference_jacobian(
            name,
            evaluator,
            &x,
            bounds,
            &residual,
            &mut func_count,
            options.max_fun_evals,
            &options.scaling,
        )
        .await?;
        let jacobian = jacobian_eval.values;
        if jacobian_eval.exhausted_budget {
            return result(evaluator,bounds,options,
                x,
                residual,
                jacobian,
                0,
                iterations,
                func_count,
                f64::NAN,
                last_step_size,
                "Exceeded maximum function evaluations; unevaluated Jacobian coefficients and first-order optimality are NaN.",
            )
            .await;
        }
        let optimality = if options.criteria.is_some() {
            criteria::lm_optimality(&jacobian, &residual, &x, bounds, evaluator).await?
        } else {
            first_order_optimality(&jacobian, &residual, n, evaluator).await?
        };
        last_optimality = optimality;
        super::display::emit(
            evaluator.cooperation().and_then(|c| c.runtime()),
            display,
            super::display::Event::Iteration,
            || {
                format!(
                    "lsqnonlin: iteration={} funcCount={} firstorderopt={:.6e} stepsize={:.6e}",
                    iterations, func_count, optimality, last_step_size
                )
            },
        )?;
        if optimality <= criteria.optimality {
            return result(
                evaluator,
                bounds,
                options,
                x,
                residual,
                jacobian,
                1,
                iterations,
                func_count,
                optimality,
                last_step_size,
                "Local minimum found. First-order optimality is within tolerance.",
            )
            .await;
        }

        let j = DMatrix::from_row_slice(m, n, &jacobian);
        let f = DVector::from_column_slice(&residual);
        let gradient = j.transpose() * &f;
        let current_norm = norm2_squared(&residual, evaluator).await?;
        let mut accepted = false;

        for _ in 0..12 {
            if func_count >= options.max_fun_evals {
                return result(
                    evaluator,
                    bounds,
                    options,
                    x,
                    residual,
                    jacobian,
                    0,
                    iterations,
                    func_count,
                    optimality,
                    last_step_size,
                    "Exceeded maximum function evaluations.",
                )
                .await;
            }
            checkpoint(evaluator, 0).await?;
            let normal = j.transpose() * &j + DMatrix::<f64>::identity(n, n) * lambda;
            let rhs = -&gradient;
            let delta = normal.lu().solve(&rhs);
            checkpoint(evaluator, jacobian_len).await?;
            let Some(delta) = delta else {
                lambda *= 10.0;
                continue;
            };

            let mut trial = x
                .iter()
                .zip(delta.iter())
                .map(|(xi, di)| xi + di)
                .collect::<Vec<_>>();
            bounds.project(&mut trial);
            let step_norm = max_abs_difference(&trial, &x, evaluator).await?;
            if step_norm == 0.0 {
                lambda *= 10.0;
                continue;
            }
            let trial_residual = evaluator.residual(&trial).await?;
            func_count += 1;
            if trial_residual.len() != m {
                return Err(optim_error(
                    name,
                    format!("{name}: function output size changed during iteration"),
                ));
            }
            let trial_norm = norm2_squared(&trial_residual, evaluator).await?;
            if trial_norm <= current_norm {
                let x_norm = residual_norm_inf(&x, evaluator.cooperation()).await?;
                let improvement = current_norm - trial_norm;
                x = trial;
                residual = trial_residual;
                evaluator.accept_residual();
                // The previous derivative was measured at the preceding x.
                // Keep its capacity, but never publish it as the new Jacobian.
                last_optimality = f64::NAN;
                last_step_size = step_norm;
                lambda = (lambda * 0.3).max(1.0e-12);
                accepted = true;
                if options.criteria.is_none()
                    && residual_norm_inf(&residual, evaluator.cooperation()).await?
                        <= options.tol_fun
                {
                    return final_result(
                        name,
                        evaluator,
                        x,
                        residual,
                        Vec::new(),
                        bounds,
                        options,
                        1,
                        iterations,
                        func_count,
                        optimality,
                        step_norm,
                        "Local minimum found. Residual is within function tolerance.",
                    )
                    .await;
                }
                if step_norm <= criteria.step * (1.0 + x_norm) {
                    return final_result(
                        name,
                        evaluator,
                        x,
                        residual,
                        Vec::new(),
                        bounds,
                        options,
                        2,
                        iterations,
                        func_count,
                        optimality,
                        step_norm,
                        "Local minimum possible. Step size is within tolerance.",
                    )
                    .await;
                }
                if improvement <= criteria.function * (1.0 + current_norm) {
                    return final_result(
                        name,
                        evaluator,
                        x,
                        residual,
                        Vec::new(),
                        bounds,
                        options,
                        3,
                        iterations,
                        func_count,
                        optimality,
                        step_norm,
                        "Local minimum possible. Change in residual norm is within tolerance.",
                    )
                    .await;
                }
                break;
            }

            lambda *= 10.0;
            if func_count >= options.max_fun_evals {
                return result(
                    evaluator,
                    bounds,
                    options,
                    x,
                    residual,
                    jacobian,
                    0,
                    iterations,
                    func_count,
                    optimality,
                    last_step_size,
                    "Exceeded maximum function evaluations.",
                )
                .await;
            }
        }

        if !accepted {
            return result(
                evaluator,
                bounds,
                options,
                x,
                residual,
                jacobian,
                0,
                iterations,
                func_count,
                last_optimality,
                last_step_size,
                "Iteration stalled before convergence.",
            )
            .await;
        }
    }

    final_result(
        name,
        evaluator,
        x,
        residual,
        Vec::new(),
        bounds,
        options,
        0,
        iterations,
        func_count,
        last_optimality,
        last_step_size,
        "Exceeded maximum iterations.",
    )
    .await
}
