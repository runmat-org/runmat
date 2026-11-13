//! Nonnegative least squares with genuine passive-set RRQR solves.
use crate::builtins::math::optim::common::{self, ScalarCooperation};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
pub use runmat_builtins::LSQNONNEG_DESCRIPTOR;
use runmat_builtins::{BuiltinExtensionDescriptor, LsqnonnegOption};
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};
mod input;
mod options;
mod model;
mod engine;
mod validation;
mod output;
#[cfg(test)]
mod tests;
const NAME: &str = "lsqnonneg";
const ALGORITHM: &str = "active-set";
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::optim::lsqnonneg")]
pub const GPU_SPEC:BuiltinGpuSpec=BuiltinGpuSpec {
    name:NAME,op_kind:GpuOpKind::Custom("nonnegative-least-squares"),supported_precisions:&[],broadcast:BroadcastSemantics::None,
    provider_hooks:&[],constant_strategy:ConstantStrategy::InlineLiteral,residency:ResidencyPolicy::GatherImmediately,
    nan_mode:ReductionNaN::Include,two_pass_threshold:None,workgroup_size:None,accepts_nan_mode:false,
    notes:"Host active-set solver; only authenticated automatic data placement gathers through original invocation authority.",
};
#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::optim::lsqnonneg")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Linear least squares is a numerical solver boundary, not an elementwise fusion operation.",
};

#[runtime_builtin(
    name = "lsqnonneg",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::lsqnonneg"
)]
async fn lsqnonneg_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(runtime.as_ref(), args).await
}
async fn execute(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let runtime = runtime.ok_or_else(|| {
        let mut error = crate::context::RuntimeCapabilityError::new(
            crate::context::RuntimeCapability::Builtin,
            "lsqnonneg requires its selected execution context",
        )
        .into_runtime_error();
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        error
    })?;
    crate::context::host_allocation::require_unbounded_workspace(Some(runtime),NAME,
        "finite host-memory execution is unsupported; passive-set assembly/factorization workspace has no complete finite bound")?;
    crate::output_count::validate_bounded_outputs(
        &LSQNONNEG_DESCRIPTOR,
        &runmat_builtins::LSQNONNEG_ERROR_INVALID_ARGUMENT,
        NAME,
    )?;
    let mut cooperation = ScalarCooperation::for_builtin(
        runmat_builtins::LSQNONNEG_CATALOG_ENTRY.identity,
        Some(runtime),
    );
    cooperation.completed(0).await?;
    let (c, d, mut controls) = input::parse(args, &mut cooperation).await?;
    let model = model::Model::new(c, d, &mut cooperation).await?;
    controls.initialize(&model)?;
    let outcome = engine::solve(&model, &mut controls, runtime, &mut cooperation).await?;
    controls.summary(outcome.flag, outcome.message, runtime)?;
    cooperation.completed(0).await?;
    output::finalize(outcome, controls.iterations)
}
fn invalid(detail: impl AsRef<str>) -> RuntimeError {
    build_runtime_error(format!("{NAME}: {}", detail.as_ref()))
        .with_builtin(NAME)
        .with_identifier("RunMat:lsqnonneg:InvalidInput")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
fn numerical(detail: impl AsRef<str>) -> RuntimeError {
    build_runtime_error(format!("{NAME}: {}", detail.as_ref()))
        .with_builtin(NAME)
        .with_identifier("RunMat:lsqnonneg:NumericalFailure")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
fn allocation(source: runmat_value::ExactHostNumericAllocationError) -> RuntimeError {
    build_runtime_error("lsqnonneg: workspace allocation failed")
        .with_builtin(NAME)
        .with_source(source)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
fn empty() -> Value {
    Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty geometry"))
}
