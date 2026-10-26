//! Actual callable/model preparation shares one existing QN controller.
use super::*;
pub(super) async fn callable<'a>(
    function: &'a Value,
    x0: Vec<f64>,
    shape: &'a [usize],
    scalar: bool,
    options: &'a FminuncOptions,
    callback: Option<&'a Value>,
    cooperation: &'a mut crate::builtins::math::optim::common::ScalarCooperation,
) -> BuiltinResult<Prepared<'a>> {
    let n = x0.len();
    if options.update.dense() {
        validate_problem_size(n)?;
    }
    let mut evaluator = ObjectiveEvaluator::new(
        function,
        shape,
        scalar,
        options.specify_objective_gradient,
        options.max_fun_evals,
        cooperation,
    );
    evaluator.scaling = options.scaling.clone();
    evaluator.ensure_state_budget(n)?;
    evaluator.parallel = crate::execution::CallbackBatch::select(
        evaluator.cooperation.runtime(),
        function,
        options.parallel,
    )?;
    let products = if crate::output_count::current_output_count_in(evaluator.cooperation.runtime())
        == Some(6)
    {
        HessianProduct::Requested
    } else {
        HessianProduct::Omit
    };
    Ok(Prepared {
        evaluator,
        x0,
        shape,
        scalar,
        options,
        callback,
        products,
    })
}
pub(super) async fn internal<'a>(
    oracle: &'a mut dyn crate::builtins::math::optim::fmincon::compiled::Oracle,
    x0: Vec<f64>,
    shape: &'a [usize],
    options: &'a FminuncOptions,
    callback: Option<&'a Value>,
    cooperation: &'a mut crate::builtins::math::optim::common::ScalarCooperation,
) -> BuiltinResult<Prepared<'a>> {
    let n = x0.len();
    if options.update.dense() {
        validate_problem_size(n)?;
    }
    let mut evaluator = ObjectiveEvaluator::internal(oracle, n, options.max_fun_evals, cooperation);
    evaluator.scaling = options.scaling.clone();
    evaluator.ensure_state_budget(n)?;
    Ok(Prepared {
        evaluator,
        x0,
        shape,
        scalar: false,
        options,
        callback,
        products: HessianProduct::Omit,
    })
}
