//! Constrained linear least squares on the shared cooperative sparse conic kernel.
use crate::builtins::math::optim::common::{conic, solver_timing::SolverTiming, ScalarCooperation};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use clarabel::algebra::CscMatrix;
pub use runmat_builtins::LSQLIN_DESCRIPTOR;
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

mod active_set;
pub(in crate::builtins::math::optim) mod compiled;
mod model;
mod numeric;
mod objective;
mod options;
mod output;
mod parse;
mod reflective;
mod residual;
mod solve;
#[cfg(test)]
mod tests;
mod validation;
mod warm;
pub(in crate::builtins::math::optim) fn validate_warm_options<'a>(
    value: &'a Value,
    c: &'a mut ScalarCooperation,
) -> impl std::future::Future<Output = BuiltinResult<()>> + 'a {
    warm::validate_options(value, c)
}

const NAME: &str = "lsqlin";
const ALGORITHM: &str = "interior-point";
type Matrix = CscMatrix<f64>;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use model::{Outcome, Problem};
#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::optim::lsqlin")]
pub const GPU_SPEC:BuiltinGpuSpec=BuiltinGpuSpec {
    name:NAME,op_kind:GpuOpKind::Custom("linear-least-squares"),supported_precisions:&[],broadcast:BroadcastSemantics::None,
    provider_hooks:&[],constant_strategy:ConstantStrategy::InlineLiteral,residency:ResidencyPolicy::GatherImmediately,
    nan_mode:ReductionNaN::Include,two_pass_threshold:None,workgroup_size:None,accepts_nan_mode:false,
    notes:"Host sparse residual solver; resident inputs gather using their original invocation authority.",
};
#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::optim::lsqlin")]
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
    name = "lsqlin",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::lsqlin"
)]
async fn lsqlin_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(runtime.as_ref(), args).await
}
async fn execute(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::context::host_allocation::require_unbounded_workspace(runtime, NAME,
        "finite host-memory execution is unsupported; lifted residual factorization and certificate workspace have no complete finite bound")?;
    crate::output_count::validate_bounded_outputs_in(
        &LSQLIN_DESCRIPTOR,
        &runmat_builtins::LSQLIN_ERROR_INVALID_ARGUMENT,
        NAME,
        runtime,
    )?;
    let mut cooperation = ScalarCooperation::new(runmat_builtins::OptimSolver::Lsqlin, runtime);
    cooperation.completed(0).await?;
    if warm::is_form(&args) {
        return warm::execute(args, &mut cooperation).await;
    }
    let timing = if crate::output_count::current_output_count().is_some_and(|count| count >= 5) {
        Some(SolverTiming::start(runtime)?)
    } else {
        None
    };
    let problem = parse::problem(args, &mut cooperation).await?;
    let outcome = solve::solve(problem, &mut cooperation).await?;
    output::finalize(outcome, timing)
}
fn invalid(detail: impl AsRef<str>) -> RuntimeError {
    build_runtime_error(format!("lsqlin: {}", detail.as_ref()))
        .with_builtin(NAME)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .with_identifier("RunMat:lsqlin:InvalidInput")
        .build()
}
fn argument(detail: impl AsRef<str>) -> RuntimeError {
    build_runtime_error(format!("lsqlin: {}", detail.as_ref()))
        .with_builtin(NAME)
        .with_identifier("RunMat:lsqlin:InvalidArgument")
        .build()
}
fn empty() -> Value {
    Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty geometry"))
}
