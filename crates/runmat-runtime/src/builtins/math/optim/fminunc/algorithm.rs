//! Existing BFGS, line-search, and Hessian algorithms with invocation cooperation.

use super::reporting::{Progress, TerminationReason};
use super::*;
use crate::builtins::math::optim::common::output_callback::{notify, IterationValues, OutputState};
pub(super) mod finite_difference;
pub(super) mod vectors;
use finite_difference::Refinement;

use super::evaluation::{ObjectiveEvaluator, ObjectiveState};

pub(super) struct MinimizeOutcome {
    x: Vec<f64>,
    fval: f64,
    grad: Vec<f64>,
    hessian: Option<crate::execution::RootedValue>,
    algorithm: &'static str,
    iterations: usize,
    func_count: usize,
    exitflag: i32,
    firstorderopt: f64,
    stepsize: f64,
    message: String,
    termination: TerminationReason,
}

impl MinimizeOutcome {
    pub(super) fn candidate(&self) -> &[f64] {
        &self.x
    }
    pub(super) fn value(&self) -> f64 {
        self.fval
    }
    pub(super) fn reason(&self) -> TerminationReason {
        self.termination
    }
    pub(super) fn information(&self) -> StructValue {
        publication::build_output_struct(self)
    }
}

mod preparation;
pub(super) enum HessianProduct {
    Omit,
    Requested,
}
struct Prepared<'a> {
    evaluator: ObjectiveEvaluator<'a>,
    x0: Vec<f64>,
    shape: &'a [usize],
    scalar: bool,
    options: &'a FminuncOptions,
    callback: Option<&'a Value>,
    products: HessianProduct,
}
pub(super) fn minimize<'a>(
    function: &'a Value,
    x0: Vec<f64>,
    shape: &'a [usize],
    scalar: bool,
    options: &'a FminuncOptions,
    output_callback: Option<&'a Value>,
    cooperation: &'a mut crate::builtins::math::optim::common::ScalarCooperation,
) -> impl std::future::Future<Output = BuiltinResult<MinimizeOutcome>> + 'a {
    crate::execution::after_result(
        preparation::callable(
            function,
            x0,
            shape,
            scalar,
            options,
            output_callback,
            cooperation,
        ),
        run,
    )
}
pub(in crate::builtins::math::optim::fminunc) fn minimize_internal<'a>(
    oracle: &'a mut dyn crate::builtins::math::optim::fmincon::compiled::Oracle,
    x0: Vec<f64>,
    shape: &'a [usize],
    options: &'a FminuncOptions,
    output_callback: Option<&'a Value>,
    cooperation: &'a mut crate::builtins::math::optim::common::ScalarCooperation,
) -> impl std::future::Future<Output = BuiltinResult<MinimizeOutcome>> + 'a {
    crate::execution::after_result(
        preparation::internal(oracle, x0, shape, options, output_callback, cooperation),
        run,
    )
}
async fn run(prepared: Prepared<'_>) -> BuiltinResult<MinimizeOutcome> {
    let Prepared {
        mut evaluator,
        x0,
        shape,
        scalar,
        options,
        callback: output_callback,
        products,
    } = prepared;
    let n = x0.len();
    let mut state = evaluator.evaluate(&x0).await?;
    let mut firstorderopt = vectors::norm_inf(&state.grad, evaluator.cooperation).await?;
    let mut model = quasi_newton::Model::new(options.update, n, options.max_iter)?;
    let mut iterations = 0usize;
    let mut stepsize = 0.0f64;
    let mut last_direction = Vec::new();
    let mut last_alpha = None;
    let mut termination = None;

    if notify(
        output_callback,
        OutputState::Init,
        IterationValues {
            x: &state.x,
            shape,
            scalar,
            iteration: iterations,
            func_count: evaluator.func_count,
            fval: state.f,
            gradient: &state.grad,
            first_order_optimality: firstorderopt,
            step_size: stepsize,
            search_direction: &[],
            line_step_length: None,
            directional_derivative: None,
        },
        evaluator.cooperation,
    )
    .await?
    {
        termination = Some(TerminationReason::OutputFunction);
    }
    reporting::initial(
        options.display,
        Progress {
            iteration: iterations,
            func_count: evaluator.func_count,
            fval: state.f,
            firstorderopt,
            line_step: None,
        },
        evaluator.cooperation.runtime(),
    )?;
    if termination.is_none() && state.f <= options.objective_limit {
        termination = Some(TerminationReason::ObjectiveLimit);
    }
    if termination.is_none() && firstorderopt <= options.tol_fun {
        termination = Some(TerminationReason::Optimality);
    }

    while termination.is_none()
        && iterations < options.max_iter
        && evaluator.func_count < options.max_fun_evals
    {
        let mut direction = model.direction(&state.grad, evaluator.cooperation).await?;
        if vectors::dot(&direction, &state.grad, evaluator.cooperation).await? >= -1.0e-14
            || !vectors::all_finite(&direction, evaluator.cooperation).await?
        {
            model.reset(n, evaluator.cooperation).await?;
            direction = vectors::negated(&state.grad, evaluator.cooperation).await?;
        }

        let (next, alpha) =
            match line_search(&mut evaluator, &state, &direction, options.objective_limit).await? {
                LineSearchResult::Step(next, alpha) => (next, alpha),
                LineSearchResult::BudgetExhausted => {
                    termination = Some(TerminationReason::Limit);
                    break;
                }
                LineSearchResult::Failed => {
                    match evaluator.refine_gradient(&mut state).await? {
                        Refinement::Updated => {
                            firstorderopt =
                                vectors::norm_inf(&state.grad, evaluator.cooperation).await?;
                            // No accepted iteration or observer replay is invented.
                            // The refined derivative belongs to the same current x/f.
                            if firstorderopt <= options.tol_fun {
                                termination = Some(TerminationReason::Optimality);
                            }
                            continue;
                        }
                        Refinement::BudgetLimit => {
                            termination = Some(TerminationReason::Limit);
                            break;
                        }
                        Refinement::Unavailable => {}
                    }
                    termination = Some(TerminationReason::NumericalFailure);
                    break;
                }
            };

        let s = vectors::subtract(&next.x, &state.x, evaluator.cooperation).await?;
        stepsize = vectors::norm_inf(&s, evaluator.cooperation).await?;
        let y = if options.update == runmat_builtins::OptimQuasiNewtonUpdate::Steepest {
            Vec::new()
        } else {
            vectors::subtract(&next.grad, &state.grad, evaluator.cooperation).await?
        };
        model.update(s, y, evaluator.cooperation).await?;
        let x_scale = 1.0 + vectors::norm_inf(&state.x, evaluator.cooperation).await?;
        state = next;
        firstorderopt = vectors::norm_inf(&state.grad, evaluator.cooperation).await?;
        iterations += 1;
        last_direction = direction;
        last_alpha = Some(alpha);
        reporting::iteration(
            options.display,
            Progress {
                iteration: iterations,
                func_count: evaluator.func_count,
                fval: state.f,
                firstorderopt,
                line_step: Some(alpha),
            },
            evaluator.cooperation.runtime(),
        )?;
        if notify(
            output_callback,
            OutputState::Iter,
            IterationValues {
                x: &state.x,
                shape,
                scalar,
                iteration: iterations,
                func_count: evaluator.func_count,
                fval: state.f,
                gradient: &state.grad,
                first_order_optimality: firstorderopt,
                step_size: stepsize,
                search_direction: &last_direction,
                line_step_length: Some(alpha),
                directional_derivative: Some(
                    vectors::dot(&state.grad, &last_direction, evaluator.cooperation).await?,
                ),
            },
            evaluator.cooperation,
        )
        .await?
        {
            termination = Some(TerminationReason::OutputFunction);
            break;
        }

        if state.f <= options.objective_limit {
            termination = Some(TerminationReason::ObjectiveLimit);
        } else if firstorderopt <= options.tol_fun {
            termination = Some(TerminationReason::Optimality);
        } else if stepsize <= options.tol_x * x_scale {
            termination = Some(TerminationReason::Step);
        }
    }

    drop(model);
    let reason = termination.unwrap_or(TerminationReason::Limit);
    let hessian = match (
        matches!(products, HessianProduct::Requested),
        options.update,
    ) {
        (false, _) => None,
        (true, runmat_builtins::OptimQuasiNewtonUpdate::LimitedMemory { .. }) => {
            let runtime = evaluator.cooperation.runtime();
            Some(crate::execution::RootedValue::in_optional_context(
                crate::builtins::math::optim::common::output_callback::empty(runtime)?,
                runtime,
            )?)
        }
        (
            true,
            runmat_builtins::OptimQuasiNewtonUpdate::Bfgs
            | runmat_builtins::OptimQuasiNewtonUpdate::Dfp
            | runmat_builtins::OptimQuasiNewtonUpdate::Steepest,
        ) => {
            if reason == TerminationReason::ObjectiveLimit {
                return Err(quasi_newton::hessian_unavailable(
                    "ObjectiveLimit stopped the solve before optional Hessian probes",
                ));
            }
            Some(trust_region::requested_hessian(&state,&mut evaluator).await?.ok_or_else(||quasi_newton::hessian_unavailable("remaining MaxFunctionEvaluations cannot fund the complete final Hessian estimate"))?)
        }
    };
    notify(
        output_callback,
        OutputState::Done,
        IterationValues {
            x: &state.x,
            shape,
            scalar,
            iteration: iterations,
            func_count: evaluator.func_count,
            fval: state.f,
            gradient: &state.grad,
            first_order_optimality: firstorderopt,
            step_size: stepsize,
            search_direction: &last_direction,
            line_step_length: last_alpha,
            directional_derivative: if last_alpha.is_some() {
                Some(vectors::dot(&state.grad, &last_direction, evaluator.cooperation).await?)
            } else {
                None
            },
        },
        evaluator.cooperation,
    )
    .await?;
    let exitflag = reason.flag();
    let message = reporting::finish(
        options.display,
        reason,
        Progress {
            iteration: iterations,
            func_count: evaluator.func_count,
            fval: state.f,
            firstorderopt,
            line_step: last_alpha,
        },
        evaluator.cooperation.runtime(),
    )?;

    Ok(MinimizeOutcome {
        x: state.x,
        fval: state.f,
        grad: state.grad,
        hessian,
        algorithm: options.update.algorithm(),
        iterations,
        func_count: evaluator.func_count,
        exitflag,
        firstorderopt,
        stepsize,
        message,
        termination: reason,
    })
}

async fn line_search(
    evaluator: &mut ObjectiveEvaluator<'_>,
    state: &ObjectiveState,
    direction: &[f64],
    objective_limit: f64,
) -> BuiltinResult<LineSearchResult> {
    let dphi0 = vectors::dot(&state.grad, direction, evaluator.cooperation).await?;
    if dphi0 >= 0.0 || !dphi0.is_finite() {
        return Ok(LineSearchResult::Failed);
    }

    let mut alpha_prev = 0.0;
    let mut f_prev = state.f;
    let mut alpha = 1.0;
    for iter in 0..MAX_LINE_SEARCH_ITERS {
        if !evaluator.can_evaluate_state(state.x.len()) {
            return Ok(LineSearchResult::BudgetExhausted);
        }
        let trial_x =
            vectors::add_scaled(&state.x, direction, alpha, evaluator.cooperation).await?;
        let trial = evaluator.evaluate(&trial_x).await?;
        if trial.f <= objective_limit {
            return Ok(LineSearchResult::Step(trial, alpha));
        }
        if trial.f > state.f + C1 * alpha * dphi0 || (iter > 0 && trial.f >= f_prev) {
            return zoom(
                evaluator,
                state,
                direction,
                alpha_prev,
                alpha,
                f_prev,
                objective_limit,
            )
            .await;
        }
        let dphi = vectors::dot(&trial.grad, direction, evaluator.cooperation).await?;
        if dphi.abs() <= -C2 * dphi0 {
            return Ok(LineSearchResult::Step(trial, alpha));
        }
        if dphi >= 0.0 {
            return zoom(
                evaluator,
                state,
                direction,
                alpha,
                alpha_prev,
                trial.f,
                objective_limit,
            )
            .await;
        }
        alpha_prev = alpha;
        f_prev = trial.f;
        // The trial-count owner already bounds expansion. A fixed alpha cap
        // repeats an identical point and invents a collapsed zoom bracket.
        alpha *= 2.0;
    }
    Ok(LineSearchResult::Failed)
}

enum LineSearchResult {
    Step(ObjectiveState, f64),
    BudgetExhausted,
    Failed,
}

async fn zoom(
    evaluator: &mut ObjectiveEvaluator<'_>,
    state: &ObjectiveState,
    direction: &[f64],
    mut lo: f64,
    mut hi: f64,
    mut f_lo: f64,
    objective_limit: f64,
) -> BuiltinResult<LineSearchResult> {
    let dphi0 = vectors::dot(&state.grad, direction, evaluator.cooperation).await?;
    for _ in 0..MAX_LINE_SEARCH_ITERS {
        if !evaluator.can_evaluate_state(state.x.len()) {
            return Ok(LineSearchResult::BudgetExhausted);
        }
        let alpha = 0.5 * (lo + hi);
        let trial_x =
            vectors::add_scaled(&state.x, direction, alpha, evaluator.cooperation).await?;
        let trial = evaluator.evaluate(&trial_x).await?;
        if trial.f <= objective_limit {
            return Ok(LineSearchResult::Step(trial, alpha));
        }
        if trial.f > state.f + C1 * alpha * dphi0 || trial.f >= f_lo {
            hi = alpha;
        } else {
            let dphi = vectors::dot(&trial.grad, direction, evaluator.cooperation).await?;
            if dphi.abs() <= -C2 * dphi0 {
                return Ok(LineSearchResult::Step(trial, alpha));
            }
            if dphi * (hi - lo) >= 0.0 {
                hi = lo;
            }
            lo = alpha;
            f_lo = trial.f;
        }
        if (hi - lo).abs() <= 1.0e-12 * (1.0 + lo.abs() + hi.abs()) {
            if trial.f <= state.f + C1 * alpha * dphi0 {
                return Ok(LineSearchResult::Step(trial, alpha));
            }
            return Ok(LineSearchResult::Failed);
        }
    }
    Ok(LineSearchResult::Failed)
}

pub(in crate::builtins::math::optim::fminunc) mod publication;
pub(super) use publication::finalize;

pub(super) fn dense_len(n: usize) -> BuiltinResult<usize> {
    n.checked_mul(n).ok_or_else(|| {
        fminunc_error_with_detail(
            &FMINUNC_ERROR_INVALID_INPUT,
            "dense BFGS matrix size overflows usize",
        )
    })
}

pub(super) fn identity(n: usize) -> BuiltinResult<Vec<f64>> {
    let len = dense_len(n)?;
    let mut out = vec![0.0; len];
    for i in 0..n {
        out[i * n + i] = 1.0;
    }
    Ok(out)
}

pub(super) async fn mat_vec(
    matrix: &[f64],
    vector: &[f64],
    n: usize,
    cooperation: &mut crate::builtins::math::optim::common::ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    let mut out = vec![0.0; n];
    for row in 0..n {
        cooperation.completed(n as u64).await?;
        for col in 0..n {
            out[row] += matrix[row * n + col] * vector[col];
        }
    }
    Ok(out)
}
