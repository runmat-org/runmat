//! Actual public helper bindings over one typed operator owner.
use super::*;
use runmat_macros::runtime_builtin;

#[runtime_builtin(
    name = "acceptancesa",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::simulannealbnd::helpers::bindings"
)]
async fn acceptancesa(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    run(Role::Acceptance, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[runtime_builtin(
    name = "annealingfast",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::simulannealbnd::helpers::bindings"
)]
async fn annealingfast(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    run(Role::FastProposal, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[runtime_builtin(
    name = "annealingboltz",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::simulannealbnd::helpers::bindings"
)]
async fn annealingboltz(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    run(Role::BoltzProposal, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[runtime_builtin(
    name = "temperatureexp",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::simulannealbnd::helpers::bindings"
)]
async fn temperatureexp(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    run(Role::Exponential, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[runtime_builtin(
    name = "temperaturefast",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::simulannealbnd::helpers::bindings"
)]
async fn temperaturefast(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    run(Role::FastTemperature, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[runtime_builtin(
    name = "temperatureboltz",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::simulannealbnd::helpers::bindings"
)]
async fn temperatureboltz(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    run(Role::BoltzTemperature, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[runtime_builtin(
    name = "saplotbestf",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::simulannealbnd::helpers::bindings"
)]
async fn saplotbestf(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    run(Role::BestFunction, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[runtime_builtin(
    name = "saplotbestx",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::simulannealbnd::helpers::bindings"
)]
async fn saplotbestx(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    run(Role::BestPoint, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[runtime_builtin(
    name = "saplotf",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::simulannealbnd::helpers::bindings"
)]
async fn saplotf(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    run(Role::CurrentFunction, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[runtime_builtin(
    name = "saplotx",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::simulannealbnd::helpers::bindings"
)]
async fn saplotx(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    run(Role::CurrentPoint, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[runtime_builtin(
    name = "saplotstopping",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::simulannealbnd::helpers::bindings"
)]
async fn saplotstopping(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    run(Role::Stopping, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[runtime_builtin(
    name = "saplottemperature",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::simulannealbnd::helpers::bindings"
)]
async fn saplottemperature(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    run(Role::Temperature, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}
