use runmat_macros::runtime_builtin;
use runmat_value::Value;

fn active_context(operation: &str) -> crate::BuiltinResult<crate::context::RuntimeContext> {
    crate::context::legacy::active().ok_or_else(|| {
        crate::build_runtime_error(format!("{operation}: no active runtime context"))
            .with_builtin(operation)
            .with_identifier("RunMat:parallel:RuntimeContextUnavailable")
            .build()
    })
}

#[runtime_builtin(
    name = "getCurrentTask",
    binding_variant = "default",
    builtin_path = "crate::builtins::parallel::current"
)]
async fn get_current_task_builtin(arguments: Vec<Value>) -> crate::BuiltinResult<Value> {
    require_no_arguments("getCurrentTask", &arguments)?;
    Ok(crate::parallel::current::task(&active_context(
        "getCurrentTask",
    )?))
}

#[runtime_builtin(
    name = "getCurrentWorker",
    binding_variant = "default",
    builtin_path = "crate::builtins::parallel::current"
)]
async fn get_current_worker_builtin(arguments: Vec<Value>) -> crate::BuiltinResult<Value> {
    require_no_arguments("getCurrentWorker", &arguments)?;
    Ok(crate::parallel::current::worker(&active_context(
        "getCurrentWorker",
    )?))
}

#[runtime_builtin(
    name = "getCurrentJob",
    binding_variant = "default",
    builtin_path = "crate::builtins::parallel::current"
)]
async fn get_current_job_builtin(arguments: Vec<Value>) -> crate::BuiltinResult<Value> {
    require_no_arguments("getCurrentJob", &arguments)?;
    Ok(crate::parallel::current::job(&active_context(
        "getCurrentJob",
    )?))
}

fn require_no_arguments(name: &str, arguments: &[Value]) -> crate::BuiltinResult<()> {
    if arguments.is_empty() {
        Ok(())
    } else {
        Err(crate::runtime_error::semantic_error(
            "RunMat:parallel:TooManyInputs",
            format!("{name} does not accept input arguments"),
        ))
    }
}
