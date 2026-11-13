//! Sparse MILP branch-and-bound with exact original-linear certificates.
use crate::builtins::math::optim::common::{self, ScalarCooperation};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
pub use runmat_builtins::INTLINPROG_DESCRIPTOR;
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};
pub(in crate::builtins::math::optim) mod compiled;
mod controls;
mod domains;
mod engine;
pub(super) mod feasibility;
mod gap;
mod model;
mod normalize;
mod numeric;
mod observers;
mod output;
mod parse;
mod proofs;
mod publication;
#[cfg(test)]
mod tests;
mod tree;
const NAME: &str = "intlinprog";
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::optim::intlinprog")]
pub const GPU_SPEC:BuiltinGpuSpec=BuiltinGpuSpec {
    name:NAME,op_kind:GpuOpKind::Custom("mixed-integer-linear-programming"),supported_precisions:&[],broadcast:BroadcastSemantics::None,
    provider_hooks:&[],constant_strategy:ConstantStrategy::InlineLiteral,residency:ResidencyPolicy::GatherImmediately,
    nan_mode:ReductionNaN::Include,two_pass_threshold:None,workgroup_size:None,accepts_nan_mode:false,
    notes:"Host sparse branch-and-bound. Authenticated automatic numeric placement alone may gather through original invocation authority.",
};
#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::optim::intlinprog")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "MILP is an original-model numerical solver boundary, not elementwise fusion.",
};
#[runtime_builtin(
    name = "intlinprog",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::intlinprog"
)]
async fn intlinprog_builtin_registered(
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
            "intlinprog requires its selected execution context",
        )
        .into_runtime_error();
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        error
    })?;
    crate::output_count::validate_bounded_outputs(
        &INTLINPROG_DESCRIPTOR,
        &runmat_builtins::INTLINPROG_ERROR_INVALID_ARGUMENT,
        NAME,
    )?;
    crate::context::host_allocation::require_unbounded_workspace(Some(runtime),NAME,
        "finite host-memory execution is unsupported; tree, sparse factorization and exact-rational growth workspace have no complete finite bound")?;
    let mut cooperation = ScalarCooperation::for_builtin(
        runmat_builtins::INTLINPROG_CATALOG_ENTRY.identity,
        Some(runtime),
    );
    cooperation.completed(0).await?;
    let (problem, mut controls) = parse::parse(args, runtime, &mut cooperation).await?;
    let workspace = model::Workspace::new(problem, &mut cooperation).await?;
    let (outcome, workspace) = engine::solve(workspace, &mut controls, &mut cooperation)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })?;
    output::finalize(outcome, &workspace, &controls, &mut cooperation).await
}
fn invalid(detail: impl AsRef<str>) -> RuntimeError {
    build_runtime_error(format!("{NAME}: {}", detail.as_ref()))
        .with_builtin(NAME)
        .with_identifier("RunMat:intlinprog:InvalidInput")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
fn argument(detail: impl AsRef<str>) -> RuntimeError {
    build_runtime_error(format!("{NAME}: {}", detail.as_ref()))
        .with_builtin(NAME)
        .with_identifier("RunMat:intlinprog:InvalidArgument")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
fn numerical(detail: impl AsRef<str>) -> RuntimeError {
    build_runtime_error(format!("{NAME}: {}", detail.as_ref()))
        .with_builtin(NAME)
        .with_identifier("RunMat:intlinprog:NumericalFailure")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
fn allocation(source: runmat_value::ExactHostNumericAllocationError) -> RuntimeError {
    build_runtime_error("intlinprog: allocation failed")
        .with_builtin(NAME)
        .with_source(source)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
fn reserve_error(source: std::collections::TryReserveError, role: &str) -> RuntimeError {
    build_runtime_error(format!("intlinprog: {role} allocation failed"))
        .with_builtin(NAME)
        .with_source(source)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
fn empty() -> Value {
    Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty geometry"))
}
