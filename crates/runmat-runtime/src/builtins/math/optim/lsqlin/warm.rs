//! The documented ws form retains actual state on the same admitted solver.
use super::*;
use crate::builtins::math::optim::optimwarmstart;
pub(super) fn is_value(value: &Value) -> bool {
    matches!(value,Value::Object(o) if o.class_name().is(runmat_builtins::LSQLIN_WARM_START_CLASS) && o.active_set_warm_start_payload().is_some())
}
pub(super) fn is_form(args: &crate::call::BuiltinArguments) -> bool {
    args.len() == 9 && is_value(&args[8])
}
pub(super) async fn validate_options(
    value: &Value,
    c: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    if options::Controls::selected(value, false)? != runmat_builtins::OptimAlgorithm::ActiveSet {
        return Err(invalid("warm start requires active-set lsqlin options"));
    }
    options::Controls::warm_carriers(value)?;
    let _controls = options::Controls::parse_borrowed(Some(value), false, c).await?;
    Ok(())
}
pub(super) async fn execute(
    args: crate::call::BuiltinArguments,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Value> {
    let requested = crate::output_count::current_output_count_in(c.runtime());
    let timing = if requested.is_some_and(|count| count >= 5) {
        Some(SolverTiming::start(c.runtime())?)
    } else {
        None
    };
    let mut values = args.iter().cloned().collect::<Vec<_>>();
    // Keep the original rooted forest through every numerical await and the
    // successor publication: immutable Options can contain dormant GC values.
    let _inputs = args;
    let Value::Object(object) = values.pop().expect("selected nine-argument warm form") else {
        return Err(invalid("warm input lost its native object"));
    };
    let data = optimwarmstart::checked(&object, optimwarmstart::Kind::LeastSquares, c.runtime())?;
    let expected = match data.x() {
        Value::Tensor(value) => value.len(),
        _ => return Err(invalid("warm X lost its numerical storage")),
    };
    let known_columns = match &values[0] {
        Value::Num(_) => Some(1),
        Value::Tensor(value) => Some(value.cols()),
        Value::SparseTensor(value) => Some(value.cols),
        Value::GpuTensor(value) => value.shape().get(1).copied(),
        _ => None,
    };
    if known_columns.is_some_and(|n| n != expected) {
        return Err(invalid("warm start requires the same number of variables"));
    }
    values.push(data.x().clone());
    values.push(data.options().clone());
    let problem = parse::prepared_values(values, c).await?;
    if problem.controls.algorithm != runmat_builtins::OptimAlgorithm::ActiveSet {
        return Err(invalid("warm start requires active-set lsqlin options"));
    }
    if problem.c.n != expected {
        return Err(invalid("warm start requires the same number of variables"));
    }
    let mut validation = compiled::WarmSupplied {
        saved: data.working(),
        completed: None,
    };
    let outcome = solve::solve_validated(
        problem,
        compiled::Products::for_output_count(requested),
        objective::Units::HalfSquared,
        c,
        &mut validation,
    )?
    .await?;
    let working = validation.completed.take();
    drop(validation);
    output::finalize_selected(outcome, timing, requested, move |point| {
        let point = point.unwrap_or_default();
        let n = point.len();
        let x = Tensor::new(point, vec![n, 1]).map_err(invalid)?;
        optimwarmstart::updated(
            optimwarmstart::Kind::LeastSquares,
            x,
            data,
            working,
            c.runtime(),
        )
    })
}
