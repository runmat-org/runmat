//! Sparse second-order cone optimization with original public sign conventions.
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::math::optim::common::{conic, ScalarCooperation};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::{
    BuiltinErrorDescriptor, CONEPROG_ERROR_INVALID_ARGUMENT as ERROR_INVALID_ARGUMENT,
    CONEPROG_ERROR_INVALID_INPUT as ERROR_INVALID_INPUT,
};
pub use runmat_builtins::{CONEPROG_DESCRIPTOR, SECONDORDERCONE_DESCRIPTOR};
use runmat_macros::runtime_builtin;
use runmat_value::{StructValue, Tensor, Value};

const CONEPROG: &str = "coneprog";
const ALGORITHM: &str = "interior point";
const DEFAULT_TOL: f64 = 1e-7;
const DEFAULT_MAX_ITER: usize = 200;
type MatrixInput = clarabel::algebra::CscMatrix<f64>;

mod constructor;
mod model;
mod numeric;
mod options;
mod output;
mod parse;
mod preflight;
mod solve;
mod validation;
use model::{ConeOutcome, ConeProgram};
use numeric::{matrix as numeric_matrix, FiniteMode};
use parse::{empty_value, is_empty_value, optional_field, required_field};
use solve::solve_coneprog;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod regression_tests;
#[cfg(test)]
mod context_tests;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::optim::coneprog")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "coneprog",
    op_kind: GpuOpKind::Custom("conic-programming"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Host conic interior-point solver. GPU-resident numeric inputs are gathered before solving.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::optim::coneprog")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "coneprog",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Conic programming is a solver boundary and terminates fusion planning.",
};

#[runtime_builtin(
    name = "secondordercone",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::coneprog"
)]
async fn secondordercone_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::context::host_allocation::require_unbounded_workspace(runtime.as_ref(),"secondordercone",
        "finite host-memory execution is unsupported; this constructor has no complete descriptor backing bound")?;
    crate::output_count::validate_fixed_outputs(
        &SECONDORDERCONE_DESCRIPTOR,
        &ERROR_INVALID_ARGUMENT,
        "secondordercone",
    )?;
    let mut cooperation = ScalarCooperation::for_builtin(
        runmat_builtins::SECONDORDERCONE_CATALOG_ENTRY.identity,
        runtime.as_ref(),
    );
    cooperation.completed(0).await?;
    if args.len() != 4 {
        return Err(coneprog_error(
            &ERROR_INVALID_ARGUMENT,
            "secondordercone: expected A, b, d, and gamma",
        ));
    }
    crate::builtins::math::optim::common::preflight_catalog_inputs(
        &runmat_builtins::SECONDORDERCONE_CATALOG_ENTRY,
        &ERROR_INVALID_INPUT,
        args.view(),
    )?;
    let mut args = args.into_iter();
    constructor::construct(
        args.next().unwrap(),
        args.next().unwrap(),
        args.next().unwrap(),
        args.next().unwrap(),
        &mut cooperation,
    )
    .await
}

#[cfg(test)]
async fn secondordercone_builtin(
    a: Value,
    b: Value,
    d: Value,
    gamma: Value,
) -> BuiltinResult<Value> {
    let mut cooperation = ScalarCooperation::for_builtin(
        runmat_builtins::SECONDORDERCONE_CATALOG_ENTRY.identity,
        None,
    );
    constructor::construct(a, b, d, gamma, &mut cooperation).await
}

#[runtime_builtin(
    name = "coneprog",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::coneprog"
)]
async fn coneprog_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    coneprog_in_runtime(runtime.as_ref(), args).await
}
#[cfg(test)]
async fn coneprog_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let args = crate::call::BuiltinArguments::from_borrowed(&args, None)?;
    coneprog_in_runtime(None, args).await
}
async fn coneprog_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::context::host_allocation::require_unbounded_workspace(runtime,CONEPROG,
        "finite host-memory execution is unsupported; this optimizer has no complete workspace and factorization bound")?;
    crate::output_count::validate_bounded_outputs(
        &CONEPROG_DESCRIPTOR,
        &ERROR_INVALID_ARGUMENT,
        CONEPROG,
    )?;
    let mut cooperation =
        ScalarCooperation::for_builtin(runmat_builtins::CONEPROG_CATALOG_ENTRY.identity, runtime);
    cooperation.completed(0).await?;
    preflight::arguments(args.view())?;
    crate::builtins::math::optim::common::preflight_catalog_inputs(
        &runmat_builtins::CONEPROG_CATALOG_ENTRY,
        &ERROR_INVALID_INPUT,
        args.view(),
    )?;
    let problem = parse::problem(args, runtime, &mut cooperation).await?;
    Ok(output::finalize(
        solve_coneprog(problem, &mut cooperation).await?,
    ))
}
fn coneprog_error(error: &'static BuiltinErrorDescriptor, detail: impl AsRef<str>) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.starts_with("coneprog:") || detail.starts_with("secondordercone:") {
        detail.to_string()
    } else {
        format!("{}: {detail}", error.message)
    };
    let mut builder = build_runtime_error(message).with_builtin(CONEPROG);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
