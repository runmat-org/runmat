//! Callback-free equation port into the existing residual controller.
use super::*;
use crate::builtins::math::optim::common::ScalarCooperation;
pub(in crate::builtins::math::optim) struct Options {
    options: FsolveOptions,
}
pub(in crate::builtins::math::optim) async fn prepare(
    value: Option<&Value>,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Options> {
    c.completed(0).await?;
    crate::context::host_allocation::require_unbounded_workspace(
        c.runtime(),
        NAME,
        "equation Jacobian and factorization workspaces require unbounded host memory",
    )?;
    let value = if let Some(Value::Tensor(array)) = value {
        array
            .validate_shape()
            .map_err(|e| fsolve_error_with_detail(&FSOLVE_ERROR_INVALID_ARGUMENT, e))?;
        if array.is_empty() && array.numeric_dtype() == runmat_value::NumericDType::F64 {
            None
        } else {
            value
        }
    } else {
        value
    };
    let options = parse_options(value)?;
    let runtime = c.runtime().cloned();
    let prepared = crate::builtins::math::optim::common::prepare_solver_options(
        runmat_builtins::OptimSolver::Fsolve,
        options,
        Some(&FSOLVE_OPTION_EXTENSION),
        Some(&FSOLVE_RESIDENT_EXTENSION),
        runtime.as_ref(),
        c,
    )
    .await?;
    let options = FsolveOptions::from_struct(prepared.as_ref().map(|value| value.structure()))?;
    if options.specify_objective_gradient {
        return Err(fsolve_error_with_detail(&FSOLVE_ERROR_INVALID_ARGUMENT,"modeled equations currently use finite differences; a supplied callback Jacobian is not available"));
    }
    Ok(Options { options })
}
pub(in crate::builtins::math::optim) fn solve<'a, E: LeastSquaresEvaluator + 'a>(
    options: &'a Options,
    x: Vec<f64>,
    bounds: &'a LeastSquaresBounds,
    evaluator: &'a mut E,
) -> BuiltinResult<crate::execution::RuntimeContinuationFuture<'a, LeastSquaresResult>> {
    let runtime = evaluator.cooperation().and_then(|c| c.runtime()).cloned();
    crate::execution::admit_borrowed_operation_with_contextual_error(
        runtime.as_ref(),
        move || async move {
            for (&lower, &upper) in bounds.lower.iter().zip(&bounds.upper) {
                if lower != f64::NEG_INFINITY || upper != f64::INFINITY {
                    return Err(fsolve_error_with_detail(
                        &FSOLVE_ERROR_INVALID_ARGUMENT,
                        "fsolve cannot discard original variable bounds",
                    ));
                }
                if let Some(c) = evaluator.cooperation() {
                    c.completed(1).await?;
                }
            }
            let raw = &options.options;
            let mut result = solve_least_squares(
                NAME,
                evaluator,
                x,
                bounds,
                &LeastSquaresOptions {
                    criteria: None,
                    tol_x: raw.tol_x,
                    tol_fun: raw.tol_fun,
                    max_iter: raw.max_iter,
                    max_fun_evals: raw.max_fun_evals,
                    // Problem solve has no raw Jacobian output. Its fifth output is
                    // lambda, never permission to spend a final Jacobian callback.
                    final_gradient: false,
                    final_jacobian: false,
                    scaling: raw.scaling.clone(),
                },
            )
            .await
            .map_err(no_replay)?;
            crate::builtins::math::optim::least_squares::equation::require_root(
                &mut result,
                raw.tol_fun,
                evaluator.cooperation(),
            )
            .await
            .map_err(no_replay)?;
            Ok(result)
        },
    )
}
pub(in crate::builtins::math::optim) fn information(result: &LeastSquaresResult) -> Value {
    super::information(result)
}
fn no_replay(mut error: RuntimeError) -> RuntimeError {
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}
