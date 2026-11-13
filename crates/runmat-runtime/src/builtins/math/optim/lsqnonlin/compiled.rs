//! Callback-free port into the same nonlinear residual controller.
use super::*;
use crate::builtins::math::optim::common::ScalarCooperation;
pub(in crate::builtins::math::optim) struct Options {
    options: LsqOptions,
    source: Option<crate::builtins::math::optim::common::PreparedSolverOptions>,
}
pub(in crate::builtins::math::optim) fn equation_tolerance(options: &Options) -> f64 {
    options.options.solver.tol_fun
}
pub(in crate::builtins::math::optim) fn equation_options(mut options: Options) -> Options {
    // Equation solve has no terminal Jacobian or multiplier output. Jacobians
    // needed by the selected controller remain ordinary counted solver work.
    options.options.solver.final_jacobian = false;
    options.options.solver.final_gradient = false;
    options
}
pub(in crate::builtins::math::optim) fn model_options(mut options: Options) -> Options {
    options.options.solver.final_jacobian = false;
    options.options.solver.final_gradient = true;
    options
}
pub(in crate::builtins::math::optim) async fn prepare(
    value: Option<&Value>,
    variables: usize,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Options> {
    c.completed(0).await?;
    crate::context::host_allocation::require_unbounded_workspace(
        c.runtime(),
        NAME,
        "nonlinear residual Jacobian and factorization workspaces require unbounded host memory",
    )?;
    let options = parse_options(value.cloned())?;
    let runtime = c.runtime().cloned();
    let options = crate::builtins::math::optim::common::prepare_modeled_residual_options(
        options,
        runtime.as_ref(),
        c,
    )
    .await
    .map_err(|error| map_error(error, &LSQNONLIN_ERROR_INVALID_ARGUMENT))?;
    Ok(Options {
        options: LsqOptions::from_struct(
            options.as_ref().map(|value| value.structure()),
            variables,
        )?,
        source: options,
    })
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
            let mut result = crate::builtins::math::optim::least_squares::solve_controlled(
                NAME,
                evaluator,
                x,
                bounds,
                &options.options.solver,
                options.options.method,
                options.options.display,
                options.source.as_ref().map(|value| value.structure()),
            )
            .await
            .map_err(|mut error| {
                error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                map_error(error, &LSQNONLIN_ERROR_INVALID_INPUT)
            })?;
            if result.method == Method::LevenbergMarquardt {
                crate::builtins::math::optim::least_squares::original_derivative_columns(
                    NAME,
                    &mut result,
                    bounds,
                    evaluator.cooperation(),
                )
                .await?;
            }
            Ok(result)
        },
    )
}
pub(in crate::builtins::math::optim) fn information(
    result: &LeastSquaresResult,
    _options: &Options,
) -> Value {
    Value::Struct(output_struct(result))
}
pub(in crate::builtins::math::optim) struct BoundMultipliers {
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub finite_gradient: bool,
}
pub(in crate::builtins::math::optim) async fn bound_multipliers(
    result: &LeastSquaresResult,
    bounds: &LeastSquaresBounds,
    c: &mut ScalarCooperation,
) -> BuiltinResult<BoundMultipliers> {
    let gradient = crate::builtins::math::optim::least_squares::bound_gradient(result, c).await?;
    if gradient.len() != result.variable_len
        || result.x.len() != result.variable_len
        || bounds.lower.len() != result.variable_len
        || bounds.upper.len() != result.variable_len
    {
        return Err(lsq_error_with_detail(
            &LSQNONLIN_ERROR_INVALID_INPUT,
            "bound multiplier geometry disagrees with the accepted iterate",
        ));
    }
    let mut lower = runmat_value::exact_capacity_vec(result.variable_len).map_err(|e| {
        crate::build_runtime_error("nonlinear bound multiplier allocation failed")
            .with_source(e)
            .build()
    })?;
    let mut upper = runmat_value::exact_capacity_vec(result.variable_len).map_err(|e| {
        crate::build_runtime_error("nonlinear bound multiplier allocation failed")
            .with_source(e)
            .build()
    })?;
    let mut finite_gradient = true;
    for (i, &value) in gradient.iter().enumerate() {
        finite_gradient &= value.is_finite();
        lower.push(
            if bounds.lower[i].is_finite()
                && (result.x[i] - bounds.lower[i]).abs() <= 1e-8 * (1. + result.x[i].abs())
            {
                value.max(0.)
            } else {
                0.
            },
        );
        upper.push(
            if bounds.upper[i].is_finite()
                && (result.x[i] - bounds.upper[i]).abs() <= 1e-8 * (1. + result.x[i].abs())
            {
                (-value).max(0.)
            } else {
                0.
            },
        );
        c.completed(1).await?;
    }
    Ok(BoundMultipliers {
        lower,
        upper,
        finite_gradient,
    })
}
