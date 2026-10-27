//! Real public MILP observers; state lives in workspace/scene owners, not a solver map.
use crate::builtins::math::optim::common::{output_callback::OutputState, ScalarCooperation};
use crate::{BuiltinResult, RuntimeError};
use runmat_builtins::MilpObserverKind;
use runmat_macros::runtime_builtin;
use runmat_value::Value;
mod numeric;
mod save;
mod plot;
#[cfg(test)]
mod tests;

#[runtime_builtin(
    name = "savemilpsolutions",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::milp_observers"
)]
async fn savemilpsolutions_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(MilpObserverKind::SaveSolutions, runtime.as_ref(), args).await
}
#[runtime_builtin(
    name = "optimplotmilp",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::milp_observers"
)]
async fn optimplotmilp_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(MilpObserverKind::PlotBounds, runtime.as_ref(), args).await
}
async fn execute(
    kind: MilpObserverKind,
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute_once(kind, runtime, args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}
async fn execute_once(
    kind: MilpObserverKind,
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if args.len() != 3 || crate::output_count::current_output_count().is_some_and(|n| n > 1) {
        return Err(invalid(
            kind,
            "expected (x,optimValues,state) and at most one stop output",
        ));
    }
    let state = crate::builtins::common::validation::value_text_scalar_string(&args[2])
        .and_then(|state| OutputState::parse(&state))
        .ok_or_else(|| invalid(kind, "state must be init, iter, or done"))?;
    let Value::Struct(values) = &args[1] else {
        return Err(invalid(kind, "optimValues must be a scalar struct"));
    };
    crate::context::host_allocation::require_unbounded_workspace(runtime,kind.name(),
        "finite host-memory execution is unsupported; solution history or graphics payload growth has no complete bound")?;
    let mut cooperation = ScalarCooperation::for_builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: kind.name() },
        runtime,
    );
    cooperation.completed(0).await?;
    match kind {
        MilpObserverKind::SaveSolutions => {
            save::update(&args[0], values, state, &mut cooperation).await?
        }
        MilpObserverKind::PlotBounds => plot::update(values, state, &mut cooperation).await?,
    }
    Ok(Value::Bool(false))
}
fn invalid(kind: MilpObserverKind, detail: impl std::fmt::Display) -> RuntimeError {
    crate::build_runtime_error(format!("{}: {detail}", kind.name()))
        .with_builtin(kind.name())
        .with_identifier(format!("RunMat:{}:InvalidArgument", kind.name()))
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
