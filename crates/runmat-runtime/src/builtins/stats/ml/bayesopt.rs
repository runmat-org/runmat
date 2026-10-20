//! Deterministic MATLAB-compatible `bayesopt` surface for small in-memory workflows.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::time::Instant;

use runmat_builtins::{
    BayesoptAcquisition as Acquisition, BayesoptOption, BuiltinErrorDescriptor,
    OptimizationVariableTransform as Transform, OptimizationVariableType as VariableType,
    BAYESOPT_ERROR_FLOW as ERROR_FLOW, BAYESOPT_ERROR_INVALID_ARGUMENT as ERROR_INVALID_ARGUMENT,
    BAYESOPT_ERROR_OBJECTIVE as ERROR_OBJECTIVE, BAYESOPT_ERROR_UNSUPPORTED as ERROR_UNSUPPORTED,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CellArray, LogicalArray, ObjectInstance, StringArray, StructValue, Tensor, Value,
};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::common::validation::native_integer_value_is_exact_f64;
use crate::{
    build_runtime_error, gather_if_needed_async, BuiltinResult, GpuGatherRetry, RuntimeError,
};

mod admission;
mod callbacks;
mod candidate;
mod optimizer;
mod options;
mod result;
mod surrogate;
mod variables;
#[cfg(test)]
mod tests;

use admission::*;
use callbacks::*;
use candidate::*;
use optimizer::*;
use options::*;
use result::*;
use surrogate::*;
use variables::*;

const NAME: &str = "bayesopt";
const RESULT_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::standard::BAYESIAN_OPTIMIZATION;
const OPTIMIZABLE_VARIABLE_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::standard::OPTIMIZABLE_VARIABLE;
const DEFAULT_MAX_EVALS: usize = 30;
const DEFAULT_SEED_POINTS: usize = 4;
const MAX_CANDIDATES: usize = 2048;
const MAX_EVALS_LIMIT: usize = MAX_CANDIDATES;
const GP_NOISE: f64 = 1.0e-8;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::stats::ml::bayesopt")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("bayesian-optimization-loop"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "bayesopt is a host control-flow solver; variable metadata and option inputs are gathered before deterministic callback evaluation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::stats::ml::bayesopt")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "bayesopt repeatedly invokes callbacks and terminates fusion planning.",
};

#[runtime_builtin(
    name = "bayesopt",
    context = true,
    requested_outputs = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::ml::bayesopt"
)]
async fn bayesopt_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    objective: Value,
    variables: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    bayesopt_in_runtime(
        runtime.as_ref(),
        requested_outputs,
        objective,
        variables,
        rest,
    )
    .await
}

#[cfg(test)]
async fn bayesopt_builtin(
    objective: Value,
    variables: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    bayesopt_in_runtime(
        None,
        crate::output_count::current_output_count(),
        objective,
        variables,
        rest,
    )
    .await
}

async fn bayesopt_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    objective: Value,
    variables: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    check_cancellation(runtime)?;
    if requested_outputs.is_some_and(|count| count > 1) {
        return Err(bayesopt_error(
            "bayesopt supports one result output",
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    if !is_callable(&objective) {
        return Err(bayesopt_error(
            "bayesopt: objective must be callable",
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    let _control_sources = rest.fork();
    let headers = BayesoptOptions::headers(rest)?;
    let admitted_options = BayesoptOptions::admit(headers)?;
    let options = BayesoptOptions::materialize(admitted_options).await?;
    let variables = gather_if_needed_async(&variables)
        .await
        .map_err(|err| map_flow(err, "variables"))?;
    let vars = parse_variables(variables)?;
    check_cancellation(runtime)?;
    let outcome = optimize(runtime, objective, vars, options).await?;
    finalize(runtime, outcome, requested_outputs)
}

fn check_cancellation(runtime: Option<&crate::context::RuntimeContext>) -> BuiltinResult<()> {
    if let Some(runtime) = runtime.filter(|runtime| runtime.is_cancelled()) {
        return Err(build_runtime_error("bayesopt: execution was cancelled")
            .with_builtin(NAME)
            .with_identifier(format!("{}:Cancelled", runtime.error_namespace()))
            .build());
    }
    Ok(())
}

fn map_flow(err: RuntimeError, label: &str) -> RuntimeError {
    bayesopt_error_with_source(
        format!("bayesopt: failed to gather {label}: {}", err.message()),
        &ERROR_FLOW,
        err,
    )
}

fn bayesopt_error(
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn bayesopt_error_with_source(
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
    source: RuntimeError,
) -> RuntimeError {
    let mut builder = build_runtime_error(message)
        .with_builtin(NAME)
        .with_gpu_gather_retry(GpuGatherRetry::Never)
        .with_source(source);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

async fn call_function(
    handle: &Value,
    args: Vec<Value>,
) -> BuiltinResult<crate::execution::RootedValue> {
    let callback = crate::canonicalize_callback_handle_for_semantic_resolution(handle.clone())?;
    crate::call_feval_async_with_outputs(callback, &args, 1).await
}

fn objective_scalar(value: Value) -> BuiltinResult<f64> {
    if !native_integer_value_is_exact_f64(&value) {
        return Err(bayesopt_error(
            "bayesopt: objective integer value must be exactly representable as double",
            &ERROR_OBJECTIVE,
        ));
    }
    let number = match value {
        Value::Num(value) => value,
        Value::Int(value) => value.to_f64(),
        Value::Bool(value) => f64::from(u8::from(value)),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(&tensor) => {
            tensor::tensor_value_f64(&tensor, 0)
        }
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            f64::from(u8::from(logical.data[0] != 0))
        }
        other => {
            return Err(bayesopt_error(
                format!("bayesopt: objective must return a real numeric scalar, got {other:?}"),
                &ERROR_OBJECTIVE,
            ))
        }
    };
    if !number.is_finite() {
        return Err(bayesopt_error(
            "bayesopt: objective must be finite",
            &ERROR_OBJECTIVE,
        ));
    }
    Ok(number)
}
