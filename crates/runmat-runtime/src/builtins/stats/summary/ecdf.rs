//! Empirical distribution function helpers.

use runmat_builtins::{
    BuiltinErrorDescriptor, BuiltinExtensionDescriptor, EcdfBounds, EcdfFunction as FunctionKind,
    EcdfOptionName, EmpiricalBuiltin, CDFPLOT_ERROR_INTERNAL, CDFPLOT_ERROR_INVALID_ARGUMENT,
    ECDF_ERROR_INTERNAL, ECDF_ERROR_INVALID_ARGUMENT, ECDF_ERROR_UNSUPPORTED,
    ECDF_INTEGER_CENSORING_EXTENSION, ECDF_INTEGER_FREQUENCY_EXTENSION, ECDF_INTEGER_Y_EXTENSION,
    ECDF_LOGICAL_FREQUENCY_EXTENSION, ECDF_LOGICAL_Y_EXTENSION,
};
use runmat_builtins::{
    CDFPLOT_GPU_INPUT_EXTENSION, CDFPLOT_INTEGER_INPUT_EXTENSION, CDFPLOT_LOGICAL_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
#[cfg(feature = "plot-core")]
use runmat_plot::plots::{LinePlot, LineStyle};
#[cfg(feature = "plot-core")]
use runmat_value::{IntValue, NumericScalar, StructValue};
use runmat_value::{Tensor, Value};

use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinGpuSpec, ConstantStrategy, GpuOpKind, ReductionNaN, ResidencyPolicy,
};
use crate::builtins::common::tensor;
#[cfg(feature = "plot-core")]
use crate::builtins::plotting::op_common::{apply_axes_target, split_leading_axes_arguments};
#[cfg(feature = "plot-core")]
use crate::builtins::plotting::state::{render_active_plot, PlotRenderOptions};
use crate::builtins::stats::summary::distribution_math::standard_normal_inv;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const ECDF_NAME: &str = "ecdf";
const CDFPLOT_NAME: &str = "cdfplot";
const ECDF_OWNER: EmpiricalBuiltin = EmpiricalBuiltin::Ecdf;
const CDFPLOT_OWNER: EmpiricalBuiltin = EmpiricalBuiltin::Cdfplot;

#[path = "ecdf/options.rs"]
mod options;
use options::EcdfOptionPlan;
#[path = "ecdf/engine.rs"]
mod engine;
use engine::evaluate_ecdf;
#[cfg(feature = "plot-core")]
#[path = "ecdf/plot.rs"]
mod plot;
#[cfg(feature = "plot-core")]
use plot::{cdfplot_stats, plot_ecdf_line};
#[path = "ecdf/values.rs"]
mod values;
use values::*;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::stats::summary::ecdf")]
pub const CDFPLOT_GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: CDFPLOT_NAME,
    op_kind: GpuOpKind::Custom("empirical-cdf-plot"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "The builtin owns resident input so compatibility and exactness checks run before provider access; admitted input gathers explicitly for host statistics and plotting.",
};

fn ecdf_error(
    builtin: EmpiricalBuiltin,
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(builtin.name());
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid_argument(builtin: EmpiricalBuiltin, message: impl Into<String>) -> RuntimeError {
    let descriptor = match builtin {
        EmpiricalBuiltin::Ecdf => &ECDF_ERROR_INVALID_ARGUMENT,
        EmpiricalBuiltin::Cdfplot => &CDFPLOT_ERROR_INVALID_ARGUMENT,
    };
    ecdf_error(builtin, message, descriptor)
}

fn unsupported(message: impl Into<String>) -> RuntimeError {
    ecdf_error(ECDF_OWNER, message, &ECDF_ERROR_UNSUPPORTED)
}

fn internal_error(builtin: EmpiricalBuiltin, message: impl Into<String>) -> RuntimeError {
    let descriptor = match builtin {
        EmpiricalBuiltin::Ecdf => &ECDF_ERROR_INTERNAL,
        EmpiricalBuiltin::Cdfplot => &CDFPLOT_ERROR_INTERNAL,
    };
    ecdf_error(builtin, message, descriptor)
}

#[derive(Clone, Debug)]
struct Options {
    function: FunctionKind,
    censoring: Option<Vec<f64>>,
    frequency: Option<Vec<f64>>,
    alpha: f64,
    bounds_on: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            function: FunctionKind::Cdf,
            censoring: None,
            frequency: None,
            alpha: 0.05,
            bounds_on: false,
        }
    }
}

#[derive(Clone, Debug)]
struct Observation {
    time: f64,
    censor: f64,
    frequency: f64,
}

#[derive(Clone, Debug)]
struct EcdfEvaluation {
    f: Vec<f64>,
    x: Vec<f64>,
    flo: Vec<f64>,
    fup: Vec<f64>,
    hazard: Vec<f64>,
    hazard_flo: Vec<f64>,
    hazard_fup: Vec<f64>,
    function: FunctionKind,
    bounds_on: bool,
}

impl EcdfEvaluation {
    fn f_value(&self) -> BuiltinResult<Value> {
        column_tensor(self.f.clone()).map(Value::Tensor)
    }

    fn x_value(&self) -> BuiltinResult<Value> {
        column_tensor(self.x.clone()).map(Value::Tensor)
    }

    fn flo_value(&self) -> BuiltinResult<Value> {
        column_tensor(self.flo.clone()).map(Value::Tensor)
    }

    fn fup_value(&self) -> BuiltinResult<Value> {
        column_tensor(self.fup.clone()).map(Value::Tensor)
    }

    fn outputs_for_count(&self, out_count: usize) -> BuiltinResult<Vec<Value>> {
        let mut outputs = Vec::with_capacity(out_count.min(4));
        if out_count >= 1 {
            outputs.push(self.f_value()?);
        }
        if out_count >= 2 {
            outputs.push(self.x_value()?);
        }
        if out_count >= 3 {
            outputs.push(self.flo_value()?);
        }
        if out_count >= 4 {
            outputs.push(self.fup_value()?);
        }
        Ok(outputs)
    }
}

#[runtime_builtin(
    name = "ecdf",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::summary::ecdf"
)]
pub(crate) async fn ecdf_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let (axes_target, args) = split_ecdf_axes_target(args, ECDF_OWNER)?;
    let mut args = args;
    let _argument_sources = args.fork();
    let y = args
        .pop_front()
        .ok_or_else(|| invalid_argument(ECDF_OWNER, "ecdf: expected sample data"))?;
    ensure_ecdf_value_extension(&y, &ECDF_INTEGER_Y_EXTENSION, &ECDF_LOGICAL_Y_EXTENSION)?;
    let options = EcdfOptionPlan::admit(args)?;
    options.ensure_extensions()?;
    let options = options.preflight_known()?;
    preflight_ecdf_sample(&y)?;
    let options = options.preflight_lengths(&y)?;
    #[cfg(not(feature = "plot-core"))]
    if is_statement_form_call()
        && (crate::value_contains_gpu(&y) || !options.borrowed_handles()?.is_empty())
    {
        return Err(crate::plotting_hooks::unavailable_presentation());
    }
    let mut borrowed = Vec::new();
    crate::dispatcher::try_visit_gpu_handles(&y, &mut |handle| borrowed.push(handle.clone()))
        .map_err(|cause| map_error(ECDF_OWNER, cause))?;
    borrowed.extend(options.borrowed_handles()?);
    let gpu_source = match &y {
        Value::GpuTensor(handle) => Some(handle.clone()),
        _ => None,
    };
    let y = gather_ecdf_value(y).await?;
    ensure_exact_ecdf_integer_boundary(&y, "sample")?;
    let options = options.resolve().await?;
    let eval = evaluate_ecdf(y, options)?;

    if is_statement_form_call() {
        emit_ecdf_plot(axes_target, &eval)?;
        return Ok(Value::output_list(Vec::new()));
    }

    match crate::output_count::current_output_count() {
        Some(0) => {
            emit_ecdf_plot(axes_target, &eval)?;
            Ok(Value::output_list(Vec::new()))
        }
        Some(out_count) => {
            let outputs = eval.outputs_for_count(out_count)?;
            Ok(crate::output_count::output_list_with_padding(
                out_count,
                restore_ecdf_outputs(outputs, gpu_source.as_ref(), &borrowed)?,
            ))
        }
        None => restore_ecdf_output(eval.f_value()?, gpu_source.as_ref(), &borrowed),
    }
}

#[cfg(feature = "plot-core")]
type EcdfAxesTarget = crate::builtins::plotting::op_common::AxesTarget;
#[cfg(not(feature = "plot-core"))]
type EcdfAxesTarget = ();

#[cfg(feature = "plot-core")]
fn emit_ecdf_plot(target: EcdfAxesTarget, eval: &EcdfEvaluation) -> BuiltinResult<()> {
    apply_axes_target(target, ECDF_NAME).map_err(|err| map_error(ECDF_OWNER, err))?;
    plot_ecdf_line(ECDF_OWNER, eval).map(|_| ())
}

#[cfg(not(feature = "plot-core"))]
fn emit_ecdf_plot(_target: EcdfAxesTarget, _eval: &EcdfEvaluation) -> BuiltinResult<()> {
    Err(crate::plotting_hooks::unavailable_presentation())
}

fn ensure_ecdf_value_extension(
    value: &Value,
    integer: &'static BuiltinExtensionDescriptor,
    logical: &'static BuiltinExtensionDescriptor,
) -> BuiltinResult<()> {
    if is_typed_integer_value(value) {
        crate::compatibility::ensure_builtin_extension_enabled(integer, ECDF_NAME)?;
    }
    if is_logical_value(value) {
        crate::compatibility::ensure_builtin_extension_enabled(logical, ECDF_NAME)?;
    }
    Ok(())
}

fn ensure_exact_ecdf_integer_boundary(value: &Value, role: &str) -> BuiltinResult<()> {
    if crate::builtins::common::validation::native_integer_value_is_exact_f64(value) {
        Ok(())
    } else {
        Err(invalid_argument(
            ECDF_OWNER,
            format!("ecdf: integer {role} values must be exactly representable as double"),
        ))
    }
}

fn preflight_ecdf_sample(value: &Value) -> BuiltinResult<()> {
    let shape = match value {
        Value::Tensor(tensor) => tensor.shape(),
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_storage(handle)
                != runmat_accelerate_api::GpuTensorStorage::Real
            {
                return Err(invalid_argument(
                    ECDF_OWNER,
                    "ecdf: expected real numeric sample data",
                ));
            }
            handle.shape()
        }
        Value::LogicalArray(array) => &array.shape,
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => return Ok(()),
        _ => {
            return Err(invalid_argument(
                ECDF_OWNER,
                "ecdf: expected real numeric sample data",
            ));
        }
    };
    let rows = shape.first().copied().unwrap_or(1);
    let cols = shape.get(1).copied().unwrap_or(1);
    if rows > 1 && cols > 1 {
        return if cols == 2 {
            Err(unsupported(
                "ecdf: interval-censored two-column input is not supported yet",
            ))
        } else {
            Err(invalid_argument(
                ECDF_OWNER,
                "ecdf: expected a vector or two-column interval matrix",
            ))
        };
    }
    if shape.contains(&0) {
        return Err(invalid_argument(
            ECDF_OWNER,
            "ecdf: sample data must be nonempty",
        ));
    }
    Ok(())
}

fn restore_ecdf_outputs(
    outputs: Vec<Value>,
    source: Option<&runmat_accelerate_api::GpuTensorHandle>,
    borrowed: &[runmat_accelerate_api::GpuTensorHandle],
) -> BuiltinResult<Vec<Value>> {
    let mut staged = Vec::with_capacity(outputs.len());
    for value in outputs {
        let mut protected = borrowed.iter().collect::<Vec<_>>();
        protected.extend(staged.iter().filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle),
            _ => None,
        }));
        match restore_ecdf_output_protecting(value, source, &protected) {
            Ok(value) => staged.push(value),
            Err(mut error) => {
                for staged_value in staged {
                    if let Value::GpuTensor(handle) = staged_value {
                        error =
                            crate::builtins::common::gpu_helpers::reject_gpu_output(handle, error);
                    }
                }
                return Err(error);
            }
        }
    }
    Ok(staged)
}

fn restore_ecdf_output(
    value: Value,
    source: Option<&runmat_accelerate_api::GpuTensorHandle>,
    borrowed: &[runmat_accelerate_api::GpuTensorHandle],
) -> BuiltinResult<Value> {
    let protected = borrowed.iter().collect::<Vec<_>>();
    restore_ecdf_output_protecting(value, source, &protected)
}

fn restore_ecdf_output_protecting(
    value: Value,
    source: Option<&runmat_accelerate_api::GpuTensorHandle>,
    siblings: &[&runmat_accelerate_api::GpuTensorHandle],
) -> BuiltinResult<Value> {
    let Some(source) = source else {
        return Ok(value);
    };
    let tensor = match value {
        Value::Tensor(tensor) => tensor,
        Value::Num(number) => Tensor::new(vec![number], vec![1, 1])
            .map_err(|err| internal_error(ECDF_OWNER, format!("ecdf: {err}")))?,
        other => return Ok(other),
    };
    let provider = runmat_accelerate_api::provider_for_handle(source)
        .ok_or_else(|| internal_error(ECDF_OWNER, "ecdf: source GPU provider is unavailable"))?;
    let mut protected = Vec::with_capacity(siblings.len() + 1);
    protected.push(source);
    protected.extend_from_slice(siblings);
    let mut handle = crate::builtins::common::gpu_helpers::upload_tensor_protecting_typed(
        provider, &tensor, &protected,
    )
    .map_err(|err| {
        crate::runtime_error::with_provider_execution_source(
            internal_error(ECDF_OWNER, format!("ecdf: {err}")),
            runmat_accelerate_api::ProviderExecutionError::from_source(err),
        )
    })?;
    if let Err(cause) =
        crate::builtins::common::gpu_helpers::propagate_output_provenance(&mut handle, [source])
    {
        return Err(crate::builtins::common::gpu_helpers::reject_gpu_output(
            handle,
            internal_error_with_cause(cause),
        ));
    }
    let cleanup_reference = handle.clone();
    crate::builtins::common::gpu_helpers::resident_gpu_value(handle).map_err(|cause| {
        crate::builtins::common::gpu_helpers::reject_gpu_output(
            cleanup_reference,
            internal_error_with_cause(cause),
        )
    })
}

fn internal_error_with_cause(cause: RuntimeError) -> RuntimeError {
    let mut owner = internal_error(ECDF_OWNER, format!("ecdf: {cause}"));
    owner.span = cause.span;
    owner.gpu_gather_retry = cause.gpu_gather_retry;
    owner.source = Some(Box::new(cause));
    owner
}

#[runtime_builtin(
    name = "cdfplot",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::summary::ecdf"
)]
pub(crate) async fn cdfplot_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    cdfplot_with_presentation(args).await
}

#[cfg(feature = "plot-core")]
async fn cdfplot_with_presentation(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let (axes_target, args) = split_ecdf_axes_target(args, CDFPLOT_OWNER)?;
    if args.len() != 1 {
        return Err(invalid_argument(
            CDFPLOT_OWNER,
            "cdfplot: expected exactly one numeric vector after optional axes handle",
        ));
    }
    ensure_cdfplot_extensions_enabled(&args[0])?;
    preflight_cdfplot_sample(&args[0])?;
    let sample = gather_if_needed_async(&args[0])
        .await
        .map_err(|cause| wrap_gather_error(CDFPLOT_OWNER, cause))?;
    ensure_exact_cdfplot_integer_boundary(&sample)?;
    let data = cdfplot_numeric_vector(&sample)?;
    let stats = cdfplot_stats(&data)?;
    let eval_input = Value::Tensor(
        Tensor::new(data.clone(), vec![data.len(), 1])
            .map_err(|err| internal_error(CDFPLOT_OWNER, format!("cdfplot: {err}")))?,
    );
    let eval = evaluate_ecdf(eval_input, Options::default())?;
    apply_axes_target(axes_target, CDFPLOT_NAME).map_err(|err| map_error(CDFPLOT_OWNER, err))?;
    let handle = plot_ecdf_line(CDFPLOT_OWNER, &eval)?;

    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(1) => Ok(Value::output_list(vec![Value::Num(handle)])),
        Some(out_count) => Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![Value::Num(handle), Value::Struct(stats)],
        )),
        None => Ok(Value::Num(handle)),
    }
}

#[cfg(not(feature = "plot-core"))]
async fn cdfplot_with_presentation(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    if args.len() != 1 {
        return Err(invalid_argument(
            CDFPLOT_OWNER,
            "cdfplot: expected exactly one numeric vector after optional axes handle",
        ));
    }
    ensure_cdfplot_extensions_enabled(&args[0])?;
    preflight_cdfplot_sample(&args[0])?;
    Err(crate::plotting_hooks::unavailable_presentation())
}

#[cfg(feature = "plot-core")]
fn split_ecdf_axes_target(
    args: crate::call::BuiltinArguments,
    builtin: EmpiricalBuiltin,
) -> BuiltinResult<crate::builtins::plotting::op_common::SplitAxesArgs> {
    let first_is_data_only = args.len() == 1
        || args.first().is_some_and(|value| {
            is_typed_integer_value(value)
                || is_logical_value(value)
                || matches!(value, Value::GpuTensor(_))
        });
    if first_is_data_only {
        return Ok((None, args));
    }
    split_leading_axes_arguments(args, builtin.name()).map_err(|err| map_error(builtin, err))
}

#[cfg(not(feature = "plot-core"))]
fn split_ecdf_axes_target(
    args: crate::call::BuiltinArguments,
    _builtin: EmpiricalBuiltin,
) -> BuiltinResult<(EcdfAxesTarget, crate::call::BuiltinArguments)> {
    Ok(((), args))
}

fn ensure_cdfplot_extensions_enabled(value: &Value) -> BuiltinResult<()> {
    if is_typed_integer_value(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CDFPLOT_INTEGER_INPUT_EXTENSION,
            CDFPLOT_NAME,
        )?;
    }
    if is_logical_value(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CDFPLOT_LOGICAL_INPUT_EXTENSION,
            CDFPLOT_NAME,
        )?;
    }
    if matches!(value, Value::GpuTensor(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CDFPLOT_GPU_INPUT_EXTENSION,
            CDFPLOT_NAME,
        )?;
    }
    Ok(())
}

#[cfg(feature = "plot-core")]
fn cdfplot_numeric_vector(value: &Value) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Tensor(tensor) => {
            ensure_cdfplot_vector_shape(tensor.shape())?;
            Ok(tensor::tensor_values_f64(tensor))
        }
        Value::Int(value) => Ok(vec![value.to_f64()]),
        Value::Num(value) => Ok(vec![*value]),
        Value::Bool(value) => Ok(vec![if *value { 1.0 } else { 0.0 }]),
        Value::LogicalArray(array) => {
            ensure_cdfplot_vector_shape(&array.shape)?;
            Ok(array
                .data
                .iter()
                .map(|value| if *value == 0 { 0.0 } else { 1.0 })
                .collect())
        }
        _ => numeric_vector(value, CDFPLOT_OWNER),
    }
}

fn preflight_cdfplot_sample(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_storage(handle)
                != runmat_accelerate_api::GpuTensorStorage::Real
            {
                return Err(invalid_argument(
                    CDFPLOT_OWNER,
                    "cdfplot: expected real numeric data",
                ));
            }
            ensure_cdfplot_vector_shape(handle.shape())
        }
        Value::Tensor(tensor) => ensure_cdfplot_vector_shape(tensor.shape()),
        Value::LogicalArray(array) => ensure_cdfplot_vector_shape(&array.shape),
        _ => Ok(()),
    }
}

fn ensure_cdfplot_vector_shape(shape: &[usize]) -> BuiltinResult<()> {
    let rows = shape.first().copied().unwrap_or(1);
    let cols = shape.get(1).copied().unwrap_or(1);
    if rows > 1 && cols > 1 {
        return Err(invalid_argument(
            CDFPLOT_OWNER,
            format!("cdfplot: expected a vector, got {rows}x{cols} matrix"),
        ));
    }
    if shape.iter().skip(2).any(|dim| *dim > 1) {
        return Err(invalid_argument(
            CDFPLOT_OWNER,
            format!("cdfplot: expected a vector, got {shape:?} data"),
        ));
    }
    if shape.contains(&0) {
        return Err(invalid_argument(
            CDFPLOT_OWNER,
            "cdfplot: sample data must contain at least one non-NaN value",
        ));
    }
    Ok(())
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::SparseTensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn is_logical_value(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::SparseTensor(tensor) if tensor.is_logical())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}

#[cfg(feature = "plot-core")]
fn ensure_exact_cdfplot_integer_boundary(value: &Value) -> BuiltinResult<()> {
    const MAX_EXACT_INTEGER: i128 = 1_i128 << 53;
    let in_range = |value: i128| (-MAX_EXACT_INTEGER..=MAX_EXACT_INTEGER).contains(&value);
    match value {
        Value::Int(value) if !in_range(int_value_i128(value)) => {
            return Err(cdfplot_inexact_integer_error())
        }
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            for index in 0..tensor.len() {
                if tensor
                    .numeric_value_at(index)
                    .and_then(numeric_scalar_i128)
                    .is_some_and(|value| !in_range(value))
                {
                    return Err(cdfplot_inexact_integer_error());
                }
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(feature = "plot-core")]
fn cdfplot_inexact_integer_error() -> RuntimeError {
    invalid_argument(CDFPLOT_OWNER,
        "cdfplot: integer sample values must lie within the contiguous exact binary64 integer range",
    )
}

#[cfg(feature = "plot-core")]
fn int_value_i128(value: &IntValue) -> i128 {
    match value {
        IntValue::I8(value) => i128::from(*value),
        IntValue::I16(value) => i128::from(*value),
        IntValue::I32(value) => i128::from(*value),
        IntValue::I64(value) => i128::from(*value),
        IntValue::U8(value) => i128::from(*value),
        IntValue::U16(value) => i128::from(*value),
        IntValue::U32(value) => i128::from(*value),
        IntValue::U64(value) => i128::from(*value),
    }
}

#[cfg(feature = "plot-core")]
fn numeric_scalar_i128(value: NumericScalar) -> Option<i128> {
    match value {
        NumericScalar::I8(value) => Some(i128::from(value)),
        NumericScalar::I16(value) => Some(i128::from(value)),
        NumericScalar::I32(value) => Some(i128::from(value)),
        NumericScalar::I64(value) => Some(i128::from(value)),
        NumericScalar::U8(value) => Some(i128::from(value)),
        NumericScalar::U16(value) => Some(i128::from(value)),
        NumericScalar::U32(value) => Some(i128::from(value)),
        NumericScalar::U64(value) => Some(i128::from(value)),
        NumericScalar::F32(_) | NumericScalar::F64(_) => None,
    }
}

fn is_statement_form_call() -> bool {
    matches!(crate::output_count::current_output_count(), Some(0))
        || (crate::output_context::requested_output_count() == Some(0)
            && crate::output_count::current_output_count().is_none())
}

fn map_error(builtin: EmpiricalBuiltin, err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        let mut owner = invalid_argument(builtin, err.message().to_owned());
        owner.gpu_gather_retry = err.gpu_gather_retry;
        owner.span = err.span;
        owner.source = Some(Box::new(err));
        owner
    }
}

async fn gather_ecdf_value(value: Value) -> BuiltinResult<Value> {
    gather_if_needed_async(&value)
        .await
        .map_err(|cause| wrap_gather_error(ECDF_OWNER, cause))
}

fn wrap_gather_error(builtin: EmpiricalBuiltin, cause: RuntimeError) -> RuntimeError {
    let mut owner = invalid_argument(builtin, format!("{builtin}: {cause}"));
    owner.gpu_gather_retry = cause.gpu_gather_retry;
    owner.span = cause.span;
    owner.source = Some(Box::new(cause));
    owner
}

#[cfg(test)]
#[path = "ecdf/tests.rs"]
mod tests;
#[cfg(test)]
#[path = "ecdf/provider_tests.rs"]
mod provider_tests;
