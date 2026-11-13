//! Scalar model equations reuse the ordinary bracket and Brent controllers.
use super::*;
use crate::builtins::math::optim::brent::{zero_with_receipts, ZeroFunction, ZeroResult};
pub(in crate::builtins::math::optim) struct Options {
    controls: FzeroOptions,
}
pub(in crate::builtins::math::optim) async fn prepare(
    value: Option<&Value>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Options> {
    cooperation.completed(0).await?;
    crate::context::host_allocation::require_unbounded_workspace(
        cooperation.runtime(),
        NAME,
        "model equation compilation and scalar sample metadata require unbounded host memory",
    )?;
    let value = match value {
        Some(Value::Tensor(array)) => {
            array
                .validate_shape()
                .map_err(|e| fzero_error_with_detail(&FZERO_ERROR_INVALID_ARGUMENT, e))?;
            if array.is_empty() && array.numeric_dtype() == runmat_value::NumericDType::F64 {
                None
            } else {
                value
            }
        }
        value => value,
    };
    let options = parse_options(value.cloned())?;
    let runtime = cooperation.runtime().cloned();
    let prepared = crate::builtins::math::optim::common::prepare_solver_options(
        OptimSolver::Fzero,
        options,
        Some(&FZERO_OPTION_EXTENSION),
        Some(&FZERO_RESIDENT_EXTENSION),
        runtime.as_ref(),
        cooperation,
    )
    .await?;
    let controls = FzeroOptions::from_struct(
        prepared.as_ref().map(|value| value.structure()),
        runtime.as_ref(),
    )?;
    Ok(Options { controls })
}
pub(in crate::builtins::math::optim) fn solve<'a, F: ZeroFunction + 'a>(
    options: &'a Options,
    initial: f64,
    mut function: F,
    cooperation: &'a mut ScalarCooperation,
) -> BuiltinResult<crate::execution::RuntimeContinuationFuture<'a, ZeroResult<F::Receipt>>> {
    let runtime = cooperation.runtime().cloned();
    crate::execution::admit_borrowed_operation_with_contextual_error(
        runtime.as_ref(),
        move || async move {
            let controls = &options.controls;
            let bracket =
                super::bracket::expand(&mut function, initial, controls.max_fun_evals, cooperation)
                    .await?;
            let mut display = IterDisplay::new(controls.display);
            let observer: Option<&mut dyn BrentZeroObserver> =
                if matches!(controls.display, OptimDisplay::Iter) {
                    Some(&mut display)
                } else {
                    None
                };
            let result = zero_with_receipts(
                NAME,
                cooperation,
                function,
                bracket.numeric,
                bracket.receipts,
                BrentParams {
                    tol_x: controls.tol_x,
                    max_iter: controls.max_iter,
                    max_fun_evals: controls.max_fun_evals,
                },
                observer,
            )
            .await?;
            let flag = if result.result.converged { 1 } else { 0 };
            cooperation.completed(0).await?;
            let message = build_message(&result.result);
            emit_summary(&result.result, flag, &message, controls)?;
            Ok(result)
        },
    )
}
pub(in crate::builtins::math::optim) fn information(result: &BrentZeroResult) -> Value {
    Value::Struct(build_output_struct(result, &build_message(result)))
}
