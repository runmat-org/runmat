//! Convex quadratic programming on the shared cooperative sparse conic kernel.
use crate::builtins::math::optim::common::{conic, solver_timing::SolverTiming, ScalarCooperation};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use clarabel::algebra::CscMatrix;
pub use runmat_builtins::QUADPROG_DESCRIPTOR;
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

mod active_set;
pub(in crate::builtins::math::optim) mod compiled;
mod model;
mod numeric;
mod options;
mod output;
mod parse;
mod psd;
mod solve;
#[cfg(test)]
mod tests;
mod validation;

const NAME: &str = "quadprog";
const ALGORITHM: &str = "interior-point-convex";
type Matrix = CscMatrix<f64>;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use model::{Outcome, Problem};
#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::optim::quadprog")]
pub const GPU_SPEC:BuiltinGpuSpec=BuiltinGpuSpec {
    name:NAME,op_kind:GpuOpKind::Custom("quadratic-programming"),supported_precisions:&[],broadcast:BroadcastSemantics::None,
    provider_hooks:&[],constant_strategy:ConstantStrategy::InlineLiteral,residency:ResidencyPolicy::GatherImmediately,
    nan_mode:ReductionNaN::Include,two_pass_threshold:None,workgroup_size:None,accepts_nan_mode:false,
    notes:"Host sparse quadratic solver; resident inputs gather using their original invocation authority.",
};
#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::optim::quadprog")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Quadratic programming is a numerical solver boundary, not an elementwise fusion operation.",
};

#[runtime_builtin(
    name = "quadprog",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::quadprog"
)]
async fn quadprog_builtin_registered(
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
        "finite host-memory execution is unsupported; quadratic factorization and certificate workspace have no complete finite bound")?;
    crate::output_count::validate_bounded_outputs(
        &QUADPROG_DESCRIPTOR,
        &runmat_builtins::QUADPROG_ERROR_INVALID_ARGUMENT,
        NAME,
    )?;
    let mut cooperation = ScalarCooperation::new(runmat_builtins::OptimSolver::Quadprog, runtime);
    cooperation.completed(0).await?;
    let timing = if crate::output_count::current_output_count().is_some_and(|count| count >= 4) {
        Some(SolverTiming::start(runtime)?)
    } else {
        None
    };
    if warm::is_form(&args) {
        return warm::execute(args, timing, &mut cooperation).await;
    }
    let problem = parse::problem(args, timing.as_ref(), &mut cooperation).await?;
    let outcome = solve::solve(problem, &mut cooperation).await?;
    output::finalize(outcome, timing)
}
mod warm;
pub(in crate::builtins::math::optim) fn validate_warm_options<'a>(
    value: &'a Value,
    c: &'a mut ScalarCooperation,
) -> impl std::future::Future<Output = BuiltinResult<()>> + 'a {
    warm::validate_options(value, c)
}
fn invalid(detail: impl AsRef<str>) -> RuntimeError {
    build_runtime_error(format!("quadprog: {}", detail.as_ref()))
        .with_builtin(NAME)
        .with_identifier("RunMat:quadprog:InvalidInput")
        .build()
}
fn argument(detail: impl AsRef<str>) -> RuntimeError {
    build_runtime_error(format!("quadprog: {}", detail.as_ref()))
        .with_builtin(NAME)
        .with_identifier("RunMat:quadprog:InvalidArgument")
        .build()
}
fn empty() -> Value {
    Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty geometry"))
}
