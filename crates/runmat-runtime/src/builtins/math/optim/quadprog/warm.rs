//! Documented immutable ws form on the same original QP execution owner.
use super::*;
use crate::builtins::math::optim::optimwarmstart::{self, Kind};
pub(super) fn is_form(args: &crate::call::BuiltinArguments) -> bool {
    args.len() == 9
        && matches!(&args[8], Value::Object(object)
        if object.class_name().is(runmat_builtins::QUADPROG_WARM_START_CLASS)
            && object.active_set_warm_start_payload().is_some())
}
pub(super) async fn validate_options(
    value: &Value,
    c: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    let Value::Struct(options) = value else {
        return Err(invalid("warm options must be a scalar struct"));
    };
    for (name, value) in &options.fields {
        if runmat_builtins::OptimOptionField::parse_for_solver(
            name,
            runmat_builtins::OptimSolver::Quadprog,
        )
        .is_some()
            && matches!(value, Value::GpuTensor(_))
        {
            return Err(invalid("warm-start options require host control carriers"));
        }
    }
    let controls = options::Controls::parse_borrowed(Some(value), c).await?;
    if controls.active.is_none() {
        return Err(invalid("warm start requires active-set quadprog"));
    }
    Ok(())
}
pub(super) async fn execute(
    args: crate::call::BuiltinArguments,
    timing: Option<SolverTiming>,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Value> {
    let requested = crate::output_count::current_output_count_in(c.runtime());
    let mut values = args.iter().cloned().collect::<Vec<_>>();
    // Original argument roots, including dormant options values, live through
    // every numerical await and the new immutable object publication.
    let _inputs = args;
    let Value::Object(object) = values.pop().expect("selected warm form") else {
        return Err(invalid("warm input lost its object"));
    };
    let data = optimwarmstart::checked(&object, Kind::Quadratic, c.runtime())?;
    let Value::Tensor(start) = data.x() else {
        return Err(invalid("warm X lost double storage"));
    };
    let expected = start.len();
    let known_columns = match &values[0] {
        Value::Num(_) => Some(1),
        Value::Tensor(value) => Some(value.cols()),
        Value::SparseTensor(value) => Some(value.cols),
        Value::GpuTensor(value) => value.shape().get(1).copied(),
        _ => None,
    };
    if known_columns.is_some_and(|columns| columns != expected) {
        return Err(invalid("warm start requires the same number of variables"));
    }
    values.push(data.x().clone());
    values.push(data.options().clone());
    let problem = parse::prepared_values(values, timing.as_ref(), c).await?;
    if problem.controls.active.is_none() || problem.f.len() != expected {
        return Err(invalid(
            "warm solve requires active-set and the same variable count",
        ));
    }
    let mut validation = compiled::WarmSupplied {
        saved: data.working(),
        completed: None,
    };
    let outcome = solve::solve_validated(
        problem,
        if requested.is_some_and(|n| n >= 5) {
            compiled::Products::WithMultipliers
        } else {
            compiled::Products::Primal
        },
        c,
        &mut validation,
    )
    .await?;
    let working = validation.completed.take();
    drop(validation);
    output::finalize_selected(outcome, timing, requested, move |point| {
        let point = point.unwrap_or_default();
        let n = point.len();
        let x = Tensor::new(point, vec![n, 1]).map_err(invalid)?;
        optimwarmstart::updated(Kind::Quadratic, x, data, working, c.runtime())
    })
}
