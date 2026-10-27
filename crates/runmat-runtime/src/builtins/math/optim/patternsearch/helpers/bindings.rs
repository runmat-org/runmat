//! Public generated-registry paths remain crate-visible on native and WASM.
use super::*;
use runmat_macros::runtime_builtin;

#[runtime_builtin(
    name = "psplotbestf",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::patternsearch::helpers::bindings"
)]
async fn psplotbestf(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    run(OptimPlotKind::PatternBestFunction, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[runtime_builtin(
    name = "psplotfuncount",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::patternsearch::helpers::bindings"
)]
async fn psplotfuncount(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    run(OptimPlotKind::PatternFunctionCount, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[runtime_builtin(
    name = "psplotmeshsize",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::patternsearch::helpers::bindings"
)]
async fn psplotmeshsize(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    run(OptimPlotKind::PatternMeshSize, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[runtime_builtin(
    name = "psplotbestx",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::patternsearch::helpers::bindings"
)]
async fn psplotbestx(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    run(OptimPlotKind::PatternBestPoint, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[runtime_builtin(
    name = "psplotmaxconstr",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::patternsearch::helpers::bindings"
)]
async fn psplotmaxconstr(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    run(OptimPlotKind::PatternMaxConstraint, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}
