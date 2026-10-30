//! MATLAB-compatible `sum` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, ProviderPrecision, ReductionFlavor};
use runmat_builtins::{
    BuiltinErrorDescriptor, SumControl, SumOutputClass as OutputTemplate, SUM_ERROR_INTERNAL,
    SUM_ERROR_INVALID_ARGUMENT, SUM_ERROR_INVALID_INPUT,
};
use runmat_types::SumInvocation;
use runmat_value::{ComplexTensor, IntValue, NumericDType, Tensor, Value};
const NAME: &str = "sum";

use crate::builtins::common::random_args::{complex_tensor_into_value, keyword_of};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType,
    ShapeRequirements,
};
use crate::builtins::common::{
    gpu_helpers,
    shape::{is_scalar_shape, normalize_scalar_shape},
    tensor,
};
use crate::context::RuntimeContext;
use crate::execution::{RootedValue, RootedValueLease};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::reduction::sum")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "sum",
    op_kind: GpuOpKind::Reduction,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Reduction {
            name: "reduce_sum_dim",
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: Some(256),
    workgroup_size: Some(256),
    accepts_nan_mode: true,
    notes:
        "Providers may specialise reduce_sum_dim / reduce_sum; omitnan and multi-axis reductions fall back to the CPU path when unsupported.",
};

fn sum_descriptor_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    sum_descriptor_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn sum_descriptor_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn sum_invalid_argument(detail: impl AsRef<str>) -> RuntimeError {
    sum_descriptor_error_with_detail(&SUM_ERROR_INVALID_ARGUMENT, detail)
}

fn sum_invalid_input(detail: impl AsRef<str>) -> RuntimeError {
    sum_descriptor_error_with_detail(&SUM_ERROR_INVALID_INPUT, detail)
}

fn sum_internal_error(detail: impl AsRef<str>) -> RuntimeError {
    sum_descriptor_error_with_detail(&SUM_ERROR_INTERNAL, detail)
}

#[path = "sum/fusion_contract.rs"]
mod fusion_contract;

#[path = "sum/provider_output.rs"]
mod provider_output;

#[path = "sum/preparation.rs"]
mod preparation;

#[path = "sum/omit.rs"]
mod omit;
#[path = "sum/geometry.rs"]
mod geometry;
pub(in crate::builtins::math) use geometry::{
    reduction_geometry, reduction_geometry_for_axes, ReductionGeometry,
};

#[cfg(test)]
#[path = "sum/omit_tests.rs"]
mod omit_tests;

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::reduction::sum", reduction = fusion_contract::DECLARATION)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "sum",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: crate::builtins::common::spec::FusionExprBuilder::new(|ctx, writer| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            writer.write_formatted(format_args!("accumulator += {input};"))
        }),
    }),
    emits_nan: false,
    notes: "Planner emits a standard column-major reduction template; providers can substitute custom kernels.",
};

#[runtime_builtin(
    name = "sum",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::reduction::sum",
    context = true,
    call_observation = true
)]
pub(crate) async fn sum_builtin(
    runtime: Option<RuntimeContext>,
    observation: Option<crate::dispatcher::direct_call_observation::CallAdmissionSink>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<RootedValue> {
    sum_for_invocation_observed(
        runtime.as_ref(),
        value,
        rest,
        None,
        SumInvocation::Sum,
        observation,
    )
    .await
}

#[cfg(test)]
async fn sum_with_nan_policy(
    value: Value,
    rest: crate::call::BuiltinArguments,
    imposed_nan_mode: Option<ReductionNaN>,
) -> crate::BuiltinResult<Value> {
    sum_for_invocation(None, value, rest, imposed_nan_mode, SumInvocation::Sum).await
}

#[cfg(test)]
async fn sum_for_invocation(
    runtime: Option<&RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
    imposed_nan_mode: Option<ReductionNaN>,
    invocation: SumInvocation,
) -> crate::BuiltinResult<Value> {
    sum_for_invocation_observed(runtime, value, rest, imposed_nan_mode, invocation, None)
        .await
        .map(|owned| owned.value().clone())
}

pub(crate) async fn sum_for_invocation_observed(
    runtime: Option<&RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
    imposed_nan_mode: Option<ReductionNaN>,
    invocation: SumInvocation,
    observation: Option<crate::dispatcher::direct_call_observation::CallAdmissionSink>,
) -> crate::BuiltinResult<RootedValue> {
    let _inputs = RootedValueLease::for_values(std::iter::once(&value).chain(rest.iter()))
        .map_err(|cause| cause.into_runtime_error())?;
    let input_count = rest.len() + 1;
    if crate::builtins::common::validation::is_typed_complex_integer(&value) {
        return Err(sum_invalid_input(
            "operations involving complex numbers with integer types are not supported",
        ));
    }
    let input_meta = InputMeta::from_value(&value);
    let mut parsed = parse_arguments(rest.view()).await?;
    parsed.invocation = invocation;
    parsed.nan_mode = match invocation {
        SumInvocation::NanSum => ReductionNaN::Omit,
        SumInvocation::Sum => imposed_nan_mode.unwrap_or(parsed.nan_mode),
    };
    if preparation::eligible(&value, parsed.output) {
        return preparation::execute(runtime, value, parsed, input_meta, input_count, observation)
            .await;
    }
    let raw_result = execute_raw(value, &parsed).await?;
    let published = apply_output_template(raw_result, &parsed.output, &input_meta).await?;
    RootedValue::new(published).map_err(|cause| cause.into_runtime_error())
}

async fn execute_raw(value: Value, parsed: &ParsedArguments) -> BuiltinResult<Value> {
    Ok(match value {
        Value::GpuTensor(handle) => sum_gpu(handle, parsed).await?,
        Value::ComplexTensor(ct) => sum_host_complex_tensor(ct, parsed)?,
        Value::Complex(re, im) => sum_host_complex_scalar(re, im, parsed)?,
        other => sum_host(other, parsed)?,
    })
}

fn sum_native_integer(value: &Value, parsed: &ParsedArguments) -> BuiltinResult<Option<Value>> {
    let (storage, shape) = match value {
        Value::Int(value) => (
            crate::builtins::math::reduction::integer_native::storage_from_scalar(value),
            vec![1, 1],
        ),
        Value::Tensor(tensor) => {
            let Some(storage) = tensor.integer_storage() else {
                return Ok(None);
            };
            (storage.clone(), tensor.shape().to_vec())
        }
        _ => return Ok(None),
    };
    let resolved = resolve_dims(&shape, &parsed.selection)?;
    crate::builtins::math::reduction::integer_native::sum(
        &storage,
        &shape,
        &resolved.dims_in_bounds,
    )
    .map(Some)
    .map_err(sum_internal_error)
}

fn numeric_dtype_from_value(value: &Value) -> Option<NumericDType> {
    match value {
        Value::Tensor(t) => Some(t.numeric_dtype()),
        Value::ComplexTensor(t) => Some(t.numeric_dtype()),
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_is_logical(handle) {
                // Logical storage may be U8 or floating, but sum's numerical
                // class is double in both default and native modes.
                Some(NumericDType::F64)
            } else {
                runmat_accelerate_api::handle_precision(handle).map(precision_to_dtype)
            }
        }
        Value::Num(_) => Some(NumericDType::F64),
        Value::LogicalArray(_) => Some(NumericDType::F64),
        _ => None,
    }
}

fn precision_to_dtype(precision: ProviderPrecision) -> NumericDType {
    match precision {
        ProviderPrecision::F32 => NumericDType::F32,
        ProviderPrecision::F64 => NumericDType::F64,
    }
}

#[derive(Clone, PartialEq, Eq)]
enum DimSelection {
    Auto,
    Dim(usize),
    Vec(Vec<usize>),
    All,
}

#[derive(Clone)]
struct ResolvedDims {
    dims_in_bounds: Vec<usize>,
}

struct ParsedArguments {
    invocation: SumInvocation,
    selection: DimSelection,
    nan_mode: ReductionNaN,
    output: OutputTemplate,
}

#[derive(Clone, Copy)]
enum InputClass {
    Double,
    Complex,
    Logical,
    Integer(IntClass),
    Bool,
}

#[derive(Clone, Copy)]
enum IntClass {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

struct InputMeta {
    class: InputClass,
    source: Option<GpuTensorHandle>,
    numeric_dtype: Option<NumericDType>,
}

impl InputMeta {
    fn from_value(value: &Value) -> Self {
        let class = match value {
            Value::Complex(_, _) | Value::ComplexTensor(_) => InputClass::Complex,
            Value::LogicalArray(_) => InputClass::Logical,
            Value::Bool(_) => InputClass::Bool,
            Value::Int(i) => InputClass::Integer(IntClass::from_int_value(i)),
            Value::Tensor(tensor) => IntClass::from_numeric_dtype(tensor.numeric_dtype())
                .map(InputClass::Integer)
                .unwrap_or(InputClass::Double),
            Value::GpuTensor(handle) => {
                if runmat_accelerate_api::handle_is_logical(handle) {
                    InputClass::Logical
                } else if let Some(integer) = runmat_accelerate_api::handle_integer_type(handle) {
                    InputClass::Integer(IntClass::from_integer_element_type(integer))
                } else if runmat_accelerate_api::handle_storage(handle)
                    == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
                {
                    InputClass::Complex
                } else {
                    InputClass::Double
                }
            }
            _ => InputClass::Double,
        };
        let source = match value {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        };
        let numeric_dtype = numeric_dtype_from_value(value);
        Self {
            class,
            source,
            numeric_dtype,
        }
    }
}

impl IntClass {
    fn from_numeric_dtype(dtype: NumericDType) -> Option<Self> {
        match dtype {
            NumericDType::I8 => Some(Self::I8),
            NumericDType::I16 => Some(Self::I16),
            NumericDType::I32 => Some(Self::I32),
            NumericDType::I64 => Some(Self::I64),
            NumericDType::U8 => Some(Self::U8),
            NumericDType::U16 => Some(Self::U16),
            NumericDType::U32 => Some(Self::U32),
            NumericDType::U64 => Some(Self::U64),
            NumericDType::F32 | NumericDType::F64 => None,
        }
    }

    fn from_integer_element_type(dtype: runmat_accelerate_api::IntegerElementType) -> Self {
        match dtype {
            runmat_accelerate_api::IntegerElementType::I8 => Self::I8,
            runmat_accelerate_api::IntegerElementType::I16 => Self::I16,
            runmat_accelerate_api::IntegerElementType::I32 => Self::I32,
            runmat_accelerate_api::IntegerElementType::I64 => Self::I64,
            runmat_accelerate_api::IntegerElementType::U8 => Self::U8,
            runmat_accelerate_api::IntegerElementType::U16 => Self::U16,
            runmat_accelerate_api::IntegerElementType::U32 => Self::U32,
            runmat_accelerate_api::IntegerElementType::U64 => Self::U64,
        }
    }

    fn from_int_value(value: &IntValue) -> Self {
        match value {
            IntValue::I8(_) => IntClass::I8,
            IntValue::I16(_) => IntClass::I16,
            IntValue::I32(_) => IntClass::I32,
            IntValue::I64(_) => IntClass::I64,
            IntValue::U8(_) => IntClass::U8,
            IntValue::U16(_) => IntClass::U16,
            IntValue::U32(_) => IntClass::U32,
            IntValue::U64(_) => IntClass::U64,
        }
    }

    fn to_value(self, scalar: f64) -> BuiltinResult<Value> {
        if scalar.is_nan() {
            return Err(sum_internal_error(
                "sum: cannot represent NaN as an integer output",
            ));
        }
        let rounded = scalar.round();
        if !rounded.is_finite() {
            return Err(sum_internal_error(
                "sum: integer output overflowed the target type",
            ));
        }
        Ok(match self {
            IntClass::I8 => Value::Int(IntValue::I8(rounded as i8)),
            IntClass::I16 => Value::Int(IntValue::I16(rounded as i16)),
            IntClass::I32 => Value::Int(IntValue::I32(rounded as i32)),
            IntClass::I64 => Value::Int(IntValue::I64(rounded as i64)),
            IntClass::U8 => Value::Int(IntValue::U8(rounded as u8)),
            IntClass::U16 => Value::Int(IntValue::U16(rounded as u16)),
            IntClass::U32 => Value::Int(IntValue::U32(rounded as u32)),
            IntClass::U64 => Value::Int(IntValue::U64(rounded as u64)),
        })
    }
}

async fn parse_arguments(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<ParsedArguments> {
    let mut selection = DimSelection::Auto;
    let mut selection_set = false;
    let mut nan_mode = ReductionNaN::Include;
    let mut output = OutputTemplate::Default;
    let mut output_set = false;
    let mut idx = 0;
    while idx < args.len() {
        let arg = &args[idx];
        if let Some(keyword) = keyword_of(arg)? {
            match SumControl::from_public_option(&keyword) {
                Some(SumControl::Missing(mode)) => {
                    nan_mode = mode;
                    idx += 1;
                    continue;
                }
                Some(SumControl::All) => {
                    if selection_set && !matches!(selection, DimSelection::Auto) {
                        return Err(sum_invalid_argument(
                            "sum: 'all' cannot be combined with an explicit dimension",
                        ));
                    }
                    selection = DimSelection::All;
                    selection_set = true;
                    idx += 1;
                    continue;
                }
                Some(SumControl::Output(class)) => {
                    if output_set {
                        return Err(sum_invalid_argument(
                            "sum: multiple output class specifications provided",
                        ));
                    }
                    output = class;
                    output_set = true;
                    idx += 1;
                    continue;
                }
                None => {}
            }
        }

        if !selection_set || matches!(selection, DimSelection::Auto) {
            if let Some(sel) = parse_dimension_spec(arg).await? {
                selection = sel;
                selection_set = true;
                idx += 1;
                continue;
            }
        }

        return Err(sum_invalid_argument(format!(
            "sum: unrecognised argument {arg:?}"
        )));
    }

    Ok(ParsedArguments {
        invocation: SumInvocation::Sum,
        selection,
        nan_mode,
        output,
    })
}

async fn parse_dimension_spec(value: &Value) -> BuiltinResult<Option<DimSelection>> {
    match value {
        Value::Int(_) | Value::Num(_) => {
            let dim = tensor::dimension_from_value_async(value, "sum", false)
                .await
                .map_err(sum_invalid_argument)?;
            Ok(dim.map(DimSelection::Dim))
        }
        Value::Tensor(t) => parse_dimension_tensor(value, t.shape()).await,
        Value::LogicalArray(logical) => parse_dimension_tensor(value, &logical.shape).await,
        Value::GpuTensor(handle) => parse_dimension_tensor(value, handle.shape()).await,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => Ok(None),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Ok(None),
        _ => Ok(None),
    }
}

async fn parse_dimension_tensor(
    value: &Value,
    shape: &[usize],
) -> BuiltinResult<Option<DimSelection>> {
    if !is_vector_shape(shape) {
        return Err(sum_invalid_argument(
            "sum: dimension vector must be a row or column vector",
        ));
    }
    let dims = tensor::dims_from_value_async_typed(value)
        .await
        .map_err(map_dims_error)?;
    let Some(dims) = dims else {
        return Ok(None);
    };
    if dims.is_empty() {
        return Ok(Some(DimSelection::Auto));
    }
    for &dim in &dims {
        if dim < 1 {
            return Err(sum_invalid_argument("sum: dimension indices must be >= 1"));
        }
    }
    Ok(Some(DimSelection::Vec(dims)))
}

fn map_dims_error(error: tensor::DimensionDecodeError) -> RuntimeError {
    use tensor::DimensionDecodeError;
    let message = match &error {
        DimensionDecodeError::Negative
        | DimensionDecodeError::IntegerNegative
        | DimensionDecodeError::IntegerOutOfPlatformRange
        | DimensionDecodeError::BelowMinimum => "sum: dimension indices must be >= 1".to_string(),
        DimensionDecodeError::NonFinite => "sum: dimensions must be finite".to_string(),
        DimensionDecodeError::Fractional => "sum: dimensions must contain integers".to_string(),
        DimensionDecodeError::OutOfPlatformRange | DimensionDecodeError::Gather(_) => {
            error.to_string()
        }
    };
    error.attach_cause(sum_invalid_argument(message))
}

fn is_vector_shape(shape: &[usize]) -> bool {
    if is_scalar_shape(shape) {
        return true;
    }
    match shape.len() {
        1 => true,
        2 => shape[0] == 1 || shape[1] == 1,
        _ => shape.iter().filter(|&&d| d > 1).count() <= 1,
    }
}

fn sum_host(value: Value, parsed: &ParsedArguments) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor(value).map_err(sum_invalid_input)?;
    let resolved = resolve_dims(tensor.shape(), &parsed.selection)?;
    let reduced = sum_tensor(&tensor, &resolved, parsed.nan_mode)?;
    Ok(tensor::tensor_into_value(reduced))
}

fn sum_host_complex_tensor(ct: ComplexTensor, parsed: &ParsedArguments) -> BuiltinResult<Value> {
    let resolved = resolve_dims(ct.shape(), &parsed.selection)?;
    let reduced = sum_complex_tensor(&ct, &resolved, parsed.nan_mode)?;
    Ok(complex_tensor_into_value(reduced))
}

fn sum_host_complex_scalar(re: f64, im: f64, parsed: &ParsedArguments) -> BuiltinResult<Value> {
    let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
        .map_err(|e| sum_internal_error(format!("sum: {e}")))?;
    sum_host_complex_tensor(tensor, parsed)
}

async fn sum_gpu(handle: GpuTensorHandle, parsed: &ParsedArguments) -> BuiltinResult<Value> {
    gpu_helpers::require_resident_compute_owner(&handle, || {
        sum_internal_error("sum: resident input has no exact owning provider")
    })?;
    if matches!(parsed.nan_mode, ReductionNaN::Omit) {
        return omit::sum_gpu_with_omitnan(handle, parsed).await;
    }
    sum_gpu_include(handle, parsed).await
}

async fn sum_gpu_include(
    handle: GpuTensorHandle,
    parsed: &ParsedArguments,
) -> BuiltinResult<Value> {
    let Some(provider) = gpu_helpers::exact_provider_for_handle(&handle) else {
        return sum_gpu_fallback(&handle, parsed).await;
    };

    let resolved = resolve_dims(handle.shape(), &parsed.selection)?;
    if resolved.dims_in_bounds.is_empty() {
        return Ok(Value::GpuTensor(handle));
    }
    if matches!(parsed.output, OutputTemplate::Native)
        && runmat_accelerate_api::handle_integer_type(&handle).is_some()
        && !runmat_accelerate_api::handle_is_logical(&handle)
    {
        if resolved.dims_in_bounds.len() == handle.shape().len() && !is_scalar_shape(handle.shape())
        {
            let output = provider
                .reduce_integer_sum_native(&handle)
                .await
                .map_err(sum_provider_error)?;
            let output = provider_output::admit_reduction(
                provider,
                &handle,
                &handle,
                output,
                None,
                parsed.output,
            )?;
            return publish_submitted_sum(provider, output, &handle);
        }
        let mut current = handle.clone();
        for &dim in &resolved.dims_in_bounds {
            let next = match provider.reduce_integer_sum_native_dim(&current, dim).await {
                Ok(next) => next,
                Err(error) => {
                    return Err(provider_output::fail_stage(
                        current,
                        sum_provider_error(error),
                    ))
                }
            };
            current = provider_output::advance_reduction(
                provider,
                &handle,
                current,
                next,
                dim,
                parsed.output,
            )?;
        }
        return publish_submitted_sum(provider, current, &handle);
    }

    if resolved.dims_in_bounds.len() == handle.shape().len() && !is_scalar_shape(handle.shape()) {
        match provider.reduce_sum(&handle).await {
            Ok(reduced) => {
                let reduced = provider_output::admit_reduction(
                    provider,
                    &handle,
                    &handle,
                    reduced,
                    None,
                    parsed.output,
                )?;
                return publish_submitted_sum(provider, reduced, &handle);
            }
            Err(error) if error.is_unsupported() => {}
            Err(error) => return Err(sum_provider_error(error)),
        }
    }

    let mut current = handle.clone();
    for &dim in &resolved.dims_in_bounds {
        match provider.reduce_sum_dim(&current, dim).await {
            Ok(next) => {
                current = provider_output::advance_reduction(
                    provider,
                    &handle,
                    current,
                    next,
                    dim,
                    parsed.output,
                )?;
            }
            Err(error) if error.is_unsupported() => {
                provider_output::retire_before_fallback(current)?;
                return sum_gpu_fallback(&handle, parsed).await;
            }
            Err(error) => {
                return Err(provider_output::fail_stage(
                    current,
                    sum_provider_error(error),
                ))
            }
        }
    }
    publish_submitted_sum(provider, current, &handle)
}

fn sum_observation_identity(
    invocation: SumInvocation,
) -> runmat_accelerate_api::ProviderOperationIdentity {
    runmat_accelerate_api::ProviderOperationIdentity::new(
        runmat_builtins::sum_invocation_entry(invocation)
            .identity
            .name,
    )
}

fn publish_submitted_sum(
    _provider: &'static dyn AccelProvider,
    output: GpuTensorHandle,
    input: &GpuTensorHandle,
) -> BuiltinResult<Value> {
    provider_output::publish(output, input)
}

fn sum_provider_error(
    source: impl Into<runmat_accelerate_api::ProviderExecutionError>,
) -> RuntimeError {
    let source = source.into();
    let mut error = sum_internal_error(format!("sum: provider execution failed: {source}"));
    error = crate::runtime_error::with_provider_execution_source(error, source);
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}

#[cfg(test)]
#[path = "sum/owning_tests.rs"]
mod owning_tests;

#[cfg(test)]
#[path = "sum/omit_effect_tests.rs"]
mod omit_effect_tests;

#[cfg(test)]
#[path = "sum/single_cast_tests.rs"]
mod single_cast_tests;

#[allow(dead_code)]
fn reduce_omitnan_2d(
    provider: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
    axis: usize,
    expected_shape: &[usize],
) -> Option<GpuTensorHandle> {
    let rows = handle.shape().first().copied()?;
    let cols = *handle.shape().get(1)?;
    let (reduce_len, num_slices, axis_is_row) = if axis == 0 {
        (rows, cols, false)
    } else if axis == 1 {
        (cols, rows, true)
    } else {
        return None;
    };
    let scalar_ty = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F32 => "f32",
        runmat_accelerate_api::ProviderPrecision::F64 => "f64",
    };
    let shader = build_omitnan_shader(scalar_ty, axis_is_row);
    provider
        .fused_reduction(
            &shader,
            match provider.precision() {
                runmat_accelerate_api::ProviderPrecision::F32 => {
                    runmat_accelerate_api::ProviderShaderRequirements::Basic
                }
                runmat_accelerate_api::ProviderPrecision::F64 => {
                    runmat_accelerate_api::ProviderShaderRequirements::F64IeeeWords
                }
            },
            std::slice::from_ref(handle),
            expected_shape,
            reduce_len,
            num_slices,
            256,
            ReductionFlavor::Sum,
        )
        .ok()
}

#[allow(dead_code)]
fn build_omitnan_shader(scalar_ty: &str, axis_is_row: bool) -> String {
    let mut shader = String::new();
    shader.push_str(&format!("struct Tensor {{ data: array<{scalar_ty}>; }}\n"));
    shader.push_str("struct MParams { nrows: u32, ncols: u32, ld: u32, flags: u32 }\n\n");
    shader.push_str("@group(0) @binding(0) var<storage, read> input0: Tensor;\n");
    shader.push_str("@group(0) @binding(1) var<storage, read_write> output: Tensor;\n");
    shader.push_str("@group(0) @binding(2) var<uniform> params: MParams;\n\n");
    shader.push_str("var<workgroup> tile: array<f32, 256u>;\n");
    shader.push_str("fn toBoolNan(x: f32) -> bool { return x != x; }\n\n");
    shader.push_str("@compute @workgroup_size(256)\n");
    if axis_is_row {
        shader.push_str(
            "fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {\n",
        );
        shader.push_str("  let row = wid.x; if (row >= params.nrows) { return; }\n");
        shader.push_str(&format!(
            "  var acc: {scalar_ty} = {}0.0;\n",
            if scalar_ty == "f64" { "f64(" } else { "" }
        ));
        if scalar_ty == "f64" {
            shader.push_str("  // close literal\n");
        }
        shader.push_str(
            "  var c = lid.x;\n  while (c < params.ncols) {\n    let v = input0.data[row + (c * params.ld)];\n    if (!toBoolNan(v)) { acc = acc + v; }\n    c += 256u;\n  }\n",
        );
        shader.push_str("  tile[lid.x] = acc;\n  workgroupBarrier();\n");
    } else {
        shader.push_str(
            "fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {\n",
        );
        shader.push_str("  let col = wid.x; if (col >= params.ncols) { return; }\n");
        shader.push_str(&format!(
            "  var acc: {scalar_ty} = {}0.0;\n",
            if scalar_ty == "f64" { "f64(" } else { "" }
        ));
        if scalar_ty == "f64" {
            shader.push_str("  // close literal\n");
        }
        shader.push_str(
            "  var r = lid.x;\n  while (r < params.nrows) {\n    let v = input0.data[(col * params.ld) + r];\n    if (!toBoolNan(v)) { acc = acc + v; }\n    r += 256u;\n  }\n",
        );
        shader.push_str("  tile[lid.x] = acc;\n  workgroupBarrier();\n");
    }
    shader.push_str(
        "  var off = 128u;\n  loop {\n    if (off == 0u) { break; }\n    if (lid.x < off) {\n      let a = tile[lid.x];\n      let b = tile[lid.x + off];\n      tile[lid.x] = a + b;\n    }\n    workgroupBarrier();\n    off = off / 2u;\n  }\n",
    );
    if axis_is_row {
        shader.push_str("  if (lid.x == 0u) { output.data[row] = tile[0u]; }\n}\n");
    } else {
        shader.push_str("  if (lid.x == 0u) { output.data[col] = tile[0u]; }\n}\n");
    }
    shader
}

#[allow(dead_code)]
fn reduced_shape(shape: &[usize], dims: &[usize]) -> Vec<usize> {
    if is_scalar_shape(shape) {
        return normalize_scalar_shape(shape);
    }
    let mut out = shape.to_vec();
    for &dim in dims {
        if dim < out.len() {
            out[dim] = 1;
        }
    }
    out
}

async fn sum_gpu_fallback(
    handle: &GpuTensorHandle,
    parsed: &ParsedArguments,
) -> BuiltinResult<Value> {
    mark_sum_runtime_fallback(parsed.invocation);
    match gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone())).await? {
        Value::ComplexTensor(tensor) => sum_host_complex_tensor(tensor, parsed),
        Value::Complex(real, imaginary) => sum_host_complex_scalar(real, imaginary, parsed),
        Value::GpuTensor(_) => Err(sum_internal_error(
            "sum: exact-owner gather left the input resident",
        )),
        host => sum_host(host, parsed),
    }
}

fn mark_sum_runtime_fallback(invocation: SumInvocation) {
    if let Some(recorder) = runmat_accelerate_api::observation::current_provider_observation() {
        if recorder.expects_operation(&sum_observation_identity(invocation)) {
            recorder.execution_disposition(
                runmat_accelerate_api::observation::ProviderExecutionDisposition::SharedRuntimeFallback,
            );
        }
    }
}

fn resolve_dims(shape: &[usize], selection: &DimSelection) -> BuiltinResult<ResolvedDims> {
    let dims_1_based: Vec<usize> = match selection {
        DimSelection::Auto => vec![default_dimension_from_shape(shape)],
        DimSelection::Dim(d) => vec![*d],
        DimSelection::Vec(v) => {
            if v.is_empty() {
                vec![default_dimension_from_shape(shape)]
            } else {
                v.clone()
            }
        }
        DimSelection::All => {
            let ndims = if is_scalar_shape(shape) {
                1
            } else {
                shape.len()
            };
            (1..=ndims).collect()
        }
    };

    // A bounded flat axis vector needs no opaque hash-table allocation.
    // Sorting/dedup below preserves the original set and ascending order.
    let mut dims_in_bounds = Vec::with_capacity(dims_1_based.len());
    let ndims = shape.len();

    for dim1 in dims_1_based {
        if dim1 == 0 {
            return Err(sum_invalid_argument("sum: dimension indices must be >= 1"));
        }
        let zero = dim1 - 1;
        if zero < ndims {
            dims_in_bounds.push(zero);
        }
    }

    dims_in_bounds.sort_unstable();
    dims_in_bounds.dedup();

    Ok(ResolvedDims { dims_in_bounds })
}

fn default_dimension_from_shape(shape: &[usize]) -> usize {
    if is_scalar_shape(shape) {
        return 1;
    }
    shape
        .iter()
        .position(|&extent| extent != 1)
        .map(|idx| idx + 1)
        .unwrap_or(1)
}

fn sum_tensor(
    tensor: &Tensor,
    dims: &ResolvedDims,
    nan_mode: ReductionNaN,
) -> BuiltinResult<Tensor> {
    let shape = if is_scalar_shape(tensor.shape()) {
        normalize_scalar_shape(tensor.shape())
    } else {
        tensor.shape().to_vec()
    };

    if dims.dims_in_bounds.is_empty() {
        return Ok(tensor.clone());
    }

    let mut output_shape = shape.clone();
    for &dim in &dims.dims_in_bounds {
        if dim < output_shape.len() {
            output_shape[dim] = 1;
        }
    }

    let out_len = tensor::element_count(&output_shape);
    let mut sums = vec![0.0f64; out_len];
    let mut saw_value = vec![false; out_len];
    let mut saw_nan = vec![false; out_len];
    let mut coords = vec![0usize; shape.len()];
    let mut out_coords = vec![0usize; shape.len()];
    let mut reduce_mask = vec![false; shape.len()];
    for &dim in &dims.dims_in_bounds {
        if dim < reduce_mask.len() {
            reduce_mask[dim] = true;
        }
    }

    let values = tensor::tensor_values_f64_cow(tensor);
    for (linear, &value) in values.iter().enumerate() {
        linear_to_multi(linear, &shape, &mut coords);
        for (i, coord) in coords.iter().enumerate() {
            out_coords[i] = if reduce_mask[i] { 0 } else { *coord };
        }
        let out_idx = multi_to_linear(&out_coords, &output_shape);
        match nan_mode {
            ReductionNaN::Include => {
                if value.is_nan() {
                    saw_nan[out_idx] = true;
                } else {
                    sums[out_idx] += value;
                    saw_value[out_idx] = true;
                }
            }
            ReductionNaN::Omit => {
                if !value.is_nan() {
                    sums[out_idx] += value;
                    saw_value[out_idx] = true;
                }
            }
        }
    }

    let mut output = Vec::with_capacity(out_len);
    for idx in 0..out_len {
        let result = match nan_mode {
            ReductionNaN::Include => {
                if saw_nan[idx] {
                    f64::NAN
                } else if saw_value[idx] {
                    sums[idx]
                } else {
                    0.0
                }
            }
            ReductionNaN::Omit => {
                if saw_value[idx] {
                    sums[idx]
                } else {
                    0.0
                }
            }
        };
        output.push(result);
    }

    let result_shape = if dims.dims_in_bounds.len() == shape.len() {
        vec![1, 1]
    } else {
        output_shape
    };
    Tensor::new(output, result_shape).map_err(|e| sum_internal_error(format!("sum: {e}")))
}

fn sum_complex_tensor(
    tensor: &ComplexTensor,
    dims: &ResolvedDims,
    nan_mode: ReductionNaN,
) -> BuiltinResult<ComplexTensor> {
    let shape = if is_scalar_shape(tensor.shape()) {
        normalize_scalar_shape(tensor.shape())
    } else {
        tensor.shape().to_vec()
    };

    if dims.dims_in_bounds.is_empty() {
        return Ok(tensor.clone());
    }

    let mut output_shape = shape.clone();
    for &dim in &dims.dims_in_bounds {
        if dim < output_shape.len() {
            output_shape[dim] = 1;
        }
    }

    let out_len = tensor::element_count(&output_shape);
    let mut sums = vec![(0.0f64, 0.0f64); out_len];
    let mut saw_value = vec![false; out_len];
    let mut saw_nan = vec![false; out_len];
    let mut coords = vec![0usize; shape.len()];
    let mut out_coords = vec![0usize; shape.len()];
    let mut reduce_mask = vec![false; shape.len()];
    for &dim in &dims.dims_in_bounds {
        if dim < reduce_mask.len() {
            reduce_mask[dim] = true;
        }
    }

    for (linear, &(re, im)) in tensor.materialize_f64().iter().enumerate() {
        linear_to_multi(linear, &shape, &mut coords);
        for (i, coord) in coords.iter().enumerate() {
            out_coords[i] = if reduce_mask[i] { 0 } else { *coord };
        }
        let out_idx = multi_to_linear(&out_coords, &output_shape);
        let is_nan = re.is_nan() || im.is_nan();
        match nan_mode {
            ReductionNaN::Include => {
                if is_nan {
                    saw_nan[out_idx] = true;
                } else {
                    sums[out_idx].0 += re;
                    sums[out_idx].1 += im;
                    saw_value[out_idx] = true;
                }
            }
            ReductionNaN::Omit => {
                if !is_nan {
                    sums[out_idx].0 += re;
                    sums[out_idx].1 += im;
                    saw_value[out_idx] = true;
                }
            }
        }
    }

    let mut output = Vec::with_capacity(out_len);
    for idx in 0..out_len {
        let (re, im) = sums[idx];
        let result = match nan_mode {
            ReductionNaN::Include => {
                if saw_nan[idx] {
                    (f64::NAN, f64::NAN)
                } else if saw_value[idx] {
                    (re, im)
                } else {
                    (0.0, 0.0)
                }
            }
            ReductionNaN::Omit => {
                if saw_value[idx] {
                    (re, im)
                } else {
                    (0.0, 0.0)
                }
            }
        };
        output.push(result);
    }

    let result_shape = if dims.dims_in_bounds.len() == shape.len() {
        vec![1, 1]
    } else {
        output_shape
    };
    ComplexTensor::new(output, result_shape).map_err(|e| sum_internal_error(format!("sum: {e}")))
}

fn linear_to_multi(index: usize, shape: &[usize], out: &mut [usize]) {
    let mut remainder = index;
    for (dim, &size) in shape.iter().enumerate() {
        if size == 0 {
            out[dim] = 0;
        } else {
            out[dim] = remainder % size;
            remainder /= size;
        }
    }
}

fn multi_to_linear(coords: &[usize], shape: &[usize]) -> usize {
    let mut stride = 1usize;
    let mut index = 0usize;
    for (dim, &size) in shape.iter().enumerate() {
        if size == 0 {
            continue;
        }
        index += coords[dim] * stride;
        stride *= size;
    }
    index
}

async fn apply_output_template(
    value: Value,
    template: &OutputTemplate,
    meta: &InputMeta,
) -> BuiltinResult<Value> {
    match template {
        OutputTemplate::Default
            if meta.numeric_dtype == Some(NumericDType::F32)
                && !matches!(meta.class, InputClass::Logical | InputClass::Bool) =>
        {
            let value = apply_native_template(value, meta).await?;
            ensure_device(value, meta).await
        }
        OutputTemplate::Default | OutputTemplate::Double => {
            apply_double_template(value, meta).await
        }
        OutputTemplate::Native if matches!(meta.class, InputClass::Logical | InputClass::Bool) => {
            apply_double_template(value, meta).await
        }
        OutputTemplate::Native => {
            let value = apply_native_template(value, meta).await?;
            ensure_device(value, meta).await
        }
    }
}

async fn apply_double_template(value: Value, meta: &InputMeta) -> BuiltinResult<Value> {
    let value = match value {
        Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_precision(&handle) != Some(ProviderPrecision::F64)
                || runmat_accelerate_api::handle_integer_type(&handle).is_some()
                || runmat_accelerate_api::handle_is_logical(&handle) =>
        {
            gpu_helpers::gather_value_async(&Value::GpuTensor(handle)).await?
        }
        other => other,
    };
    let value = match value {
        Value::Int(value) => Value::Num(value.to_f64()),
        Value::Bool(value) => Value::Num(f64::from(u8::from(value))),
        Value::Tensor(tensor) if tensor.numeric_dtype() != NumericDType::F64 => {
            Value::Tensor(tensor::coerce_tensor_dtype(tensor, NumericDType::F64))
        }
        Value::LogicalArray(logical) => {
            Value::Tensor(tensor::logical_to_tensor(&logical).map_err(sum_internal_error)?)
        }
        Value::ComplexTensor(tensor) if tensor.numeric_dtype() == NumericDType::F32 => {
            let shape = tensor.shape().to_vec();
            ComplexTensor::from_f64_values_with_dtype(
                tensor.materialize_f64(),
                shape,
                NumericDType::F64,
            )
            .map(Value::ComplexTensor)
            .map_err(sum_internal_error)?
        }
        other => other,
    };
    ensure_device(value, meta).await
}

async fn apply_native_template(value: Value, meta: &InputMeta) -> BuiltinResult<Value> {
    match meta.class {
        InputClass::Integer(class) => match value {
            Value::Num(n) => class.to_value(n),
            Value::Tensor(t) if tensor::is_scalar_tensor(&t) => {
                class.to_value(tensor::tensor_value_f64(&t, 0))
            }
            other => Ok(other),
        },
        _ => {
            if let Some(dtype) = meta.numeric_dtype {
                coerce_value_to_dtype(value, dtype, meta.source.as_ref()).await
            } else {
                Ok(value)
            }
        }
    }
}

async fn coerce_value_to_dtype(
    value: Value,
    dtype: NumericDType,
    borrowed_source: Option<&GpuTensorHandle>,
) -> BuiltinResult<Value> {
    match dtype {
        NumericDType::F64 => Ok(value),
        NumericDType::F32
        | NumericDType::I8
        | NumericDType::I16
        | NumericDType::I32
        | NumericDType::I64
        | NumericDType::U8
        | NumericDType::U16
        | NumericDType::U32
        | NumericDType::U64 => match value {
            Value::Tensor(tensor) => {
                let tensor = tensor::coerce_tensor_dtype(tensor, dtype);
                Ok(Value::Tensor(tensor))
            }
            Value::Num(n) => {
                let tensor = Tensor::new(vec![n], vec![1, 1])
                    .map_err(|e| sum_internal_error(format!("{NAME}: {e}")))?;
                let tensor = tensor::coerce_tensor_dtype(tensor, dtype);
                Ok(Value::Tensor(tensor))
            }
            Value::Complex(real, imaginary) if dtype == NumericDType::F32 => {
                ComplexTensor::from_f32(vec![(real as f32, imaginary as f32)], vec![1, 1])
                    .map(Value::ComplexTensor)
                    .map_err(sum_internal_error)
            }
            Value::LogicalArray(logical) => {
                let tensor = tensor::logical_to_tensor(&logical)
                    .map_err(|e| sum_internal_error(format!("{NAME}: {e}")))?;
                let tensor = tensor::coerce_tensor_dtype(tensor, dtype);
                Ok(Value::Tensor(tensor))
            }
            Value::ComplexTensor(tensor) if tensor.numeric_dtype() == dtype => {
                Ok(Value::ComplexTensor(tensor))
            }
            Value::ComplexTensor(tensor) if matches!(dtype, NumericDType::F32) => {
                let shape = tensor.shape().to_vec();
                ComplexTensor::from_f64_values_with_dtype(tensor.materialize_f64(), shape, dtype)
                    .map(Value::ComplexTensor)
                    .map_err(sum_internal_error)
            }
            Value::GpuTensor(handle)
                if dtype == NumericDType::F32
                    && runmat_accelerate_api::handle_precision(&handle)
                        == Some(ProviderPrecision::F32)
                    && runmat_accelerate_api::handle_integer_type(&handle).is_none()
                    && !runmat_accelerate_api::handle_is_logical(&handle) =>
            {
                Ok(Value::GpuTensor(handle))
            }
            Value::GpuTensor(handle) => {
                if dtype == NumericDType::F32
                    && runmat_accelerate_api::handle_storage(&handle)
                        == runmat_accelerate_api::GpuTensorStorage::Real
                {
                    if let Some(cast) =
                        provider_output::cast_public_single(&handle, borrowed_source).await?
                    {
                        return provider_output::publish(cast, &handle);
                    }
                }
                match gpu_helpers::gather_value_async(&Value::GpuTensor(handle)).await? {
                    Value::Tensor(tensor) => {
                        Ok(Value::Tensor(tensor::coerce_tensor_dtype(tensor, dtype)))
                    }
                    Value::ComplexTensor(tensor) if dtype == NumericDType::F32 => {
                        let shape = tensor.shape().to_vec();
                        ComplexTensor::from_f64_values_with_dtype(
                            tensor.materialize_f64(),
                            shape,
                            dtype,
                        )
                        .map(Value::ComplexTensor)
                        .map_err(sum_internal_error)
                    }
                    other => Err(sum_internal_error(format!(
                        "sum: unexpected gathered output class {other:?}"
                    ))),
                }
            }
            other => Ok(other),
        },
    }
}

async fn ensure_device(value: Value, meta: &InputMeta) -> BuiltinResult<Value> {
    match &meta.source {
        Some(source) if !matches!(value, Value::GpuTensor(_)) => {
            gpu_helpers::restore_class_preserving_value(source, value, NAME)
        }
        None if matches!(value, Value::GpuTensor(_)) => {
            gpu_helpers::gather_value_async(&value).await
        }
        _ => Ok(value),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::{
        HostIntegerDataOwned, HostIntegerDataView, HostIntegerTensorView, HostLogicalTensorView,
        IntegerElementType,
    };
    use runmat_value::{IntValue, IntegerStorage};

    fn sum_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::sum_builtin(
            None,
            None,
            value,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
        .map(|owned| owned.value().clone())
    }

    fn values(tensor: &Tensor) -> Vec<f64> {
        tensor.materialize_f64()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sum_scalar_num() {
        let result = sum_builtin(Value::Num(5.0), Vec::new()).expect("sum");
        assert_eq!(result, Value::Num(5.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sum_matrix_default_dimension() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result = sum_builtin(Value::Tensor(tensor), Vec::new()).expect("sum");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [1, 3]);
                assert_eq!(values(&out), vec![5.0, 7.0, 9.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sum_matrix_dimension_two() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result =
            sum_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(2))]).expect("sum");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [2, 1]);
                assert_eq!(values(&out), vec![6.0, 15.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn sum_double_path_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::I16(vec![1, 4, 2, 5, 3, 6]), vec![2, 3])
            .expect("tensor");

        let result = sum_builtin(Value::Tensor(tensor), Vec::new()).expect("sum");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [1, 3]);
                assert_eq!(values(&out), vec![5.0, 7.0, 9.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn sum_single_default_native_and_explicit_double_have_distinct_classes() {
        let input = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("single input");
        for options in [
            vec![],
            vec![Value::from("default")],
            vec![Value::from("native")],
        ] {
            let Value::Tensor(output) =
                sum_builtin(Value::Tensor(input.clone()), options).expect("single sum")
            else {
                panic!("expected tensor output");
            };
            assert_eq!(output.numeric_dtype(), NumericDType::F32);
            assert_eq!(values(&output), vec![3.0, 7.0]);
        }
        let Value::Tensor(explicit) =
            sum_builtin(Value::Tensor(input.clone()), vec![Value::from("double")])
                .expect("explicit double sum")
        else {
            panic!("expected tensor output");
        };
        assert_eq!(explicit.numeric_dtype(), NumericDType::F64);
        assert_eq!(values(&explicit), vec![3.0, 7.0]);

        let Value::Tensor(noop) = sum_builtin(
            Value::Tensor(input),
            vec![Value::Int(IntValue::I32(5)), Value::from("double")],
        )
        .expect("explicit double no-op") else {
            panic!("expected tensor output");
        };
        assert_eq!(noop.numeric_dtype(), NumericDType::F64);
    }

    #[test]
    fn sum_complex_single_respects_native_and_noop_classes() {
        let input = ComplexTensor::from_f32(
            vec![(1.0, 0.5), (2.0, 1.0), (3.0, 1.5), (4.0, 2.0)],
            vec![2, 2],
        )
        .expect("complex single input");
        let ordinary =
            sum_builtin(Value::ComplexTensor(input.clone()), Vec::new()).expect("ordinary sum");
        let Value::ComplexTensor(ordinary) = ordinary else {
            panic!("expected complex tensor");
        };
        assert_eq!(ordinary.numeric_dtype(), NumericDType::F32);
        assert_eq!(ordinary.materialize_f64(), vec![(3.0, 1.5), (7.0, 3.5)]);

        let explicit = sum_builtin(
            Value::ComplexTensor(input.clone()),
            vec![Value::from("double")],
        )
        .expect("explicit double complex sum");
        let Value::ComplexTensor(explicit) = explicit else {
            panic!("expected complex tensor");
        };
        assert_eq!(explicit.numeric_dtype(), NumericDType::F64);

        let native = sum_builtin(
            Value::ComplexTensor(input.clone()),
            vec![Value::from("native")],
        )
        .expect("native sum");
        let Value::ComplexTensor(native) = native else {
            panic!("expected complex tensor");
        };
        assert_eq!(native.numeric_dtype(), NumericDType::F32);
        assert_eq!(native.materialize_f64(), vec![(3.0, 1.5), (7.0, 3.5)]);

        let noop = sum_builtin(
            Value::ComplexTensor(input.clone()),
            vec![Value::Int(IntValue::I32(5))],
        )
        .expect("no-op sum");
        assert_eq!(noop, Value::ComplexTensor(input));

        let scalar_input =
            ComplexTensor::from_f32(vec![(1.0, 0.5), (2.0, 1.0)], vec![1, 2]).expect("complex row");
        let scalar = sum_builtin(Value::ComplexTensor(scalar_input), Vec::new())
            .expect("single complex scalar result");
        let Value::ComplexTensor(scalar) = scalar else {
            panic!("single complex scalar must retain its class");
        };
        assert_eq!(scalar.numeric_dtype(), NumericDType::F32);
        assert_eq!(scalar.materialize_f64(), vec![(3.0, 1.5)]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sum_all_dimension() {
        let tensor =
            Tensor::new((1..=6).map(|v| v as f64).collect::<Vec<_>>(), vec![2, 3]).unwrap();
        let result = sum_builtin(Value::Tensor(tensor), vec![Value::from("all")]).expect("sum");
        assert_eq!(result, Value::Num(21.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sum_vecdim_multiple_axes() {
        let tensor = Tensor::new(
            (1..=24).map(|v| v as f64).collect::<Vec<_>>(),
            vec![3, 4, 2],
        )
        .unwrap();
        let dims = Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap();
        let result = sum_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect("sum");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [1, 4, 1]);
                assert_eq!(values(&out), vec![48.0, 66.0, 84.0, 102.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sum_with_omit_nan_default_dimension() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 3.0], vec![3, 1]).unwrap();
        let result = sum_builtin(Value::Tensor(tensor), vec![Value::from("omitnan")]).expect("sum");
        assert_eq!(result, Value::Num(4.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sum_generic_missing_flags_match_legacy_aliases() {
        let tensor = || Tensor::new(vec![1.0, f64::NAN, 3.0], vec![3, 1]).unwrap();
        let omitted = sum_builtin(Value::Tensor(tensor()), vec![Value::from("omitmissing")])
            .expect("sum omitmissing");
        assert_eq!(omitted, Value::Num(4.0));
        let included = sum_builtin(Value::Tensor(tensor()), vec![Value::from("includemissing")])
            .expect("sum includemissing");
        match included {
            Value::Num(value) => assert!(value.is_nan()),
            other => panic!("expected scalar NaN, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sum_with_include_nan_propagates() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 3.0], vec![3, 1]).unwrap();
        let result = sum_builtin(Value::Tensor(tensor), Vec::new()).expect("sum");
        match result {
            Value::Num(n) => assert!(n.is_nan()),
            other => panic!("expected scalar NaN, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sum_dimension_greater_than_ndims_returns_input() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let original = tensor.clone();
        let value = Value::Tensor(tensor);
        let result = sum_builtin(value, vec![Value::Int(IntValue::I32(5))]).expect("sum");
        match result {
            Value::Tensor(out) => assert_eq!(out, original),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn sum_integer_dimension_beyond_rank_still_honors_output_class() {
        let storages = vec![
            IntegerStorage::I8(vec![2, 3]),
            IntegerStorage::I16(vec![2, 3]),
            IntegerStorage::I32(vec![2, 3]),
            IntegerStorage::I64(vec![2, 3]),
            IntegerStorage::U8(vec![2, 3]),
            IntegerStorage::U16(vec![2, 3]),
            IntegerStorage::U32(vec![2, 3]),
            IntegerStorage::U64(vec![2, 3]),
        ];
        for storage in storages {
            let tensor = Tensor::new_integer(storage.clone(), vec![1, 2]).expect("integer tensor");
            let default = sum_builtin(
                Value::Tensor(tensor.clone()),
                vec![Value::Int(IntValue::I32(4))],
            )
            .expect("default sum");
            let explicit_double = sum_builtin(
                Value::Tensor(tensor.clone()),
                vec![Value::Int(IntValue::I32(4)), Value::from("double")],
            )
            .expect("double sum");
            let explicit_default = sum_builtin(
                Value::Tensor(tensor.clone()),
                vec![Value::Int(IntValue::I32(4)), Value::from("default")],
            )
            .expect("explicit default sum");
            for result in [default, explicit_default, explicit_double] {
                let Value::Tensor(output) = result else {
                    panic!("expected tensor output");
                };
                assert_eq!(output.shape(), [1, 2]);
                assert_eq!(output.numeric_dtype(), NumericDType::F64);
                assert_eq!(values(&output), vec![2.0, 3.0]);
            }
            let native = sum_builtin(
                Value::Tensor(tensor),
                vec![Value::Int(IntValue::I32(4)), Value::from("native")],
            )
            .expect("native sum");
            let Value::Tensor(output) = native else {
                panic!("expected native tensor output");
            };
            assert_eq!(output.integer_storage(), Some(&storage));
        }
    }

    #[test]
    fn sum_empty_integer_identities_preserve_shape_and_output_class() {
        let storages = vec![
            IntegerStorage::I8(Vec::new()),
            IntegerStorage::I16(Vec::new()),
            IntegerStorage::I32(Vec::new()),
            IntegerStorage::I64(Vec::new()),
            IntegerStorage::U8(Vec::new()),
            IntegerStorage::U16(Vec::new()),
            IntegerStorage::U32(Vec::new()),
            IntegerStorage::U64(Vec::new()),
        ];
        for storage in storages {
            let tensor =
                Tensor::new_integer(storage.clone(), vec![0, 3]).expect("empty integer tensor");
            let default =
                sum_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("default empty sum");
            let Value::Tensor(default) = default else {
                panic!("expected default tensor output");
            };
            assert_eq!(default.shape(), [1, 3]);
            assert_eq!(default.numeric_dtype(), NumericDType::F64);
            assert_eq!(values(&default), vec![0.0; 3]);

            let native = sum_builtin(Value::Tensor(tensor.clone()), vec![Value::from("native")])
                .expect("native empty sum");
            let Value::Tensor(native) = native else {
                panic!("expected native tensor output");
            };
            assert_eq!(native.shape(), [1, 3]);
            assert_eq!(native.integer_storage(), Some(&storage.zeros_like(3)));

            let default_dim_two = sum_builtin(
                Value::Tensor(tensor.clone()),
                vec![Value::Int(IntValue::I32(2))],
            )
            .expect("default empty sum along dimension two");
            let Value::Tensor(default_dim_two) = default_dim_two else {
                panic!("expected default empty tensor output");
            };
            assert_eq!(default_dim_two.shape(), [0, 1]);
            assert_eq!(default_dim_two.numeric_dtype(), NumericDType::F64);
            assert!(default_dim_two.is_empty());

            let native_dim_two = sum_builtin(
                Value::Tensor(tensor.clone()),
                vec![Value::Int(IntValue::I32(2)), Value::from("native")],
            )
            .expect("native empty sum along dimension two");
            let Value::Tensor(native_dim_two) = native_dim_two else {
                panic!("expected native empty tensor output");
            };
            assert_eq!(native_dim_two.shape(), [0, 1]);
            assert_eq!(native_dim_two.integer_storage(), Some(&storage));

            let default_all = sum_builtin(Value::Tensor(tensor.clone()), vec![Value::from("all")])
                .expect("default empty sum across all dimensions");
            assert_eq!(default_all, Value::Num(0.0));

            let native_all = sum_builtin(
                Value::Tensor(tensor),
                vec![Value::from("all"), Value::from("native")],
            )
            .expect("native empty sum across all dimensions");
            assert_eq!(
                native_all,
                Value::Int(storage.zeros_like(1).value_at(0).unwrap())
            );

            let zero_by_zero =
                Tensor::new_integer(storage.clone(), vec![0, 0]).expect("zero-by-zero tensor");
            let zero_by_zero =
                sum_builtin(Value::Tensor(zero_by_zero), vec![Value::from("native")])
                    .expect("zero-by-zero sum");
            let Value::Tensor(zero_by_zero) = zero_by_zero else {
                panic!("expected zero-by-zero tensor output");
            };
            assert_eq!(zero_by_zero.shape(), [1, 0]);
            assert_eq!(zero_by_zero.integer_storage(), Some(&storage));
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sum_native_integer_scalar() {
        let value = Value::Int(IntValue::I16(42));
        let result = sum_builtin(value, vec![Value::from("native")]).expect("sum");
        assert_eq!(result, Value::Int(IntValue::I16(42)));
    }

    #[test]
    fn sum_native_integer_tensor_forms_read_typed_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::I16(vec![1, 4, 2, 5, 3, 6]), vec![2, 3])
            .expect("tensor");

        let result = sum_builtin(Value::Tensor(tensor), vec![Value::from("native")]).expect("sum");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [1, 3]);
                assert_eq!(
                    out.integer_storage(),
                    Some(&IntegerStorage::I16(vec![5, 7, 9]))
                );
            }
            other => panic!("expected tensor result, got {other:?}"),
        }

        let tensor = Tensor::new_integer(IntegerStorage::U16(vec![10, 20, 30, 40]), vec![2, 2])
            .expect("tensor");
        let result = sum_builtin(
            Value::Tensor(tensor),
            vec![Value::from("all"), Value::from("native")],
        )
        .expect("sum all");
        assert_eq!(result, Value::Int(IntValue::U16(100)));

        let tensor = Tensor::new_integer(
            IntegerStorage::I16((1..=12).map(|value| value as i16).collect()),
            vec![2, 3, 2],
        )
        .expect("tensor");
        let dims = Tensor::new_integer(IntegerStorage::U8(vec![1, 3]), vec![1, 2]).expect("dims");

        let result = sum_builtin(
            Value::Tensor(tensor),
            vec![Value::Tensor(dims), Value::from("native")],
        )
        .expect("sum vecdim");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [1, 3, 1]);
                assert_eq!(
                    out.integer_storage(),
                    Some(&IntegerStorage::I16(vec![18, 26, 34]))
                );
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn sum_native_template_reads_typed_integer_scalar_storage_exactly() {
        let tensor =
            Tensor::new_integer(IntegerStorage::I32(vec![-9]), vec![1, 1]).expect("tensor");
        let meta = InputMeta {
            class: InputClass::Integer(IntClass::I32),
            source: None,
            numeric_dtype: None,
        };

        let result =
            block_on(apply_native_template(Value::Tensor(tensor), &meta)).expect("native template");

        assert_eq!(result, Value::Int(IntValue::I32(-9)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sum_rejects_undocumented_like_option() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = sum_builtin(
            Value::Tensor(tensor),
            vec![Value::from("like"), Value::Num(0.0)],
        )
        .expect_err("expected error");
        assert_eq!(err.identifier(), SUM_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn sum_typed_option_admission_retains_output_conflict_and_axis_order() {
        let input = || Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap());
        let result = sum_builtin(
            input(),
            vec![Value::from("NATIVE"), Value::from("OMITMISSING")],
        )
        .expect("case-insensitive options remain admitted");
        assert_eq!(result, Value::Num(3.0));

        let conflict = sum_builtin(input(), vec![Value::from("default"), Value::from("native")])
            .expect_err("two output classes must conflict");
        assert_eq!(conflict.identifier(), SUM_ERROR_INVALID_ARGUMENT.identifier);
        assert!(conflict.message().contains("multiple output class"));

        let axis_conflict = sum_builtin(input(), vec![Value::Num(1.0), Value::from("all")])
            .expect_err("all after an explicit axis must conflict");
        assert_eq!(
            axis_conflict.identifier(),
            SUM_ERROR_INVALID_ARGUMENT.identifier
        );
        assert!(axis_conflict.message().contains("'all' cannot be combined"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sum_invalid_dim_identifier_is_descriptor_backed() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = sum_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(0))])
            .expect_err("sum");
        assert_eq!(err.identifier(), SUM_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sum_all_omitnan() {
        let tensor = Tensor::new(vec![2.0, f64::NAN, 10.0, 4.0], vec![2, 2]).unwrap();
        let result = sum_builtin(
            Value::Tensor(tensor),
            vec![Value::from("all"), Value::from("omitnan")],
        )
        .expect("sum");
        assert_eq!(result, Value::Num(16.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sum_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let result = sum_builtin(Value::GpuTensor(handle), Vec::new()).expect("sum");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape(), [1, 3]);
            assert_eq!(values(&gathered), vec![5.0, 7.0, 9.0]);
        });
    }

    #[test]
    fn sum_gpu_single_default_and_explicit_double_preserve_output_contract() {
        test_support::with_test_provider(|provider| {
            let tensor =
                Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("single input");
            let input = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let default =
                sum_builtin(Value::GpuTensor(input.clone()), Vec::new()).expect("default sum");
            let gathered = test_support::gather(default).expect("gather default");
            assert_eq!(gathered.numeric_dtype(), NumericDType::F32);
            assert_eq!(values(&gathered), vec![3.0, 7.0]);

            let explicit = sum_builtin(Value::GpuTensor(input), vec![Value::from("double")])
                .expect("explicit double sum");
            let gathered = test_support::gather(explicit).expect("gather explicit");
            assert_eq!(gathered.numeric_dtype(), NumericDType::F64);
            assert_eq!(values(&gathered), vec![3.0, 7.0]);
        });
    }

    #[test]
    fn sum_single_accumulates_in_double_before_default_and_explicit_output_classes() {
        let input = Tensor::from_f32(vec![16_777_216.0, 1.0, -16_777_216.0], vec![3, 1]).unwrap();
        let host_default = sum_builtin(Value::Tensor(input.clone()), Vec::new()).unwrap();
        let host_double =
            sum_builtin(Value::Tensor(input.clone()), vec![Value::from("double")]).unwrap();
        assert_eq!(
            values(&test_support::gather(host_default).unwrap()),
            vec![1.0]
        );
        assert_eq!(
            values(&test_support::gather(host_double).unwrap()),
            vec![1.0]
        );
        test_support::with_test_provider(|provider| {
            let handle = gpu_helpers::upload_tensor(provider, &input).unwrap();
            for options in [vec![], vec![Value::from("double")]] {
                let result = sum_builtin(Value::GpuTensor(handle.clone()), options).unwrap();
                assert!(matches!(&result, Value::GpuTensor(_)));
                let gathered = test_support::gather(result).unwrap();
                assert_eq!(values(&gathered), vec![1.0]);
            }
        });
    }

    #[test]
    fn sum_all_normalizes_nd_single_and_complex_scalar_shape() {
        let single = Tensor::from_f32(vec![1.0; 8], vec![2, 2, 2]).unwrap();
        let Value::Tensor(host) =
            sum_builtin(Value::Tensor(single.clone()), vec![Value::from("all")]).unwrap()
        else {
            panic!("expected single tensor");
        };
        assert_eq!(host.shape(), [1, 1]);
        assert_eq!(host.numeric_dtype(), NumericDType::F32);
        test_support::with_test_provider(|provider| {
            let handle = gpu_helpers::upload_tensor(provider, &single).unwrap();
            let gathered = test_support::gather(
                sum_builtin(Value::GpuTensor(handle), vec![Value::from("all")]).unwrap(),
            )
            .unwrap();
            assert_eq!(gathered.shape(), [1, 1]);
            assert_eq!(gathered.numeric_dtype(), NumericDType::F32);
            assert_eq!(values(&gathered), vec![8.0]);
        });

        let complex = ComplexTensor::from_f32(vec![(1.0, 2.0); 8], vec![2, 2, 2]).unwrap();
        let Value::ComplexTensor(host) = sum_builtin(
            Value::ComplexTensor(complex.clone()),
            vec![Value::from("all")],
        )
        .unwrap() else {
            panic!("expected complex tensor");
        };
        assert_eq!(host.shape(), [1, 1]);
        assert_eq!(host.materialize_f64(), vec![(8.0, 16.0)]);
        test_support::with_test_provider(|provider| {
            let handle = gpu_helpers::upload_complex_tensor(provider, &complex).unwrap();
            let result = sum_builtin(Value::GpuTensor(handle), vec![Value::from("all")]).unwrap();
            let Value::ComplexTensor(gathered) =
                block_on(gpu_helpers::gather_value_async(&result)).unwrap()
            else {
                panic!("expected complex resident reduction");
            };
            assert_eq!(gathered.shape(), [1, 1]);
            assert_eq!(gathered.numeric_dtype(), NumericDType::F32);
            assert_eq!(gathered.materialize_f64(), vec![(8.0, 16.0)]);
        });
    }

    #[test]
    fn sum_complex_resident_include_and_omitnan_follow_pairwise_host_policy() {
        let input =
            ComplexTensor::from_f32(vec![(1.0, 2.0), (f32::NAN, 3.0), (4.0, 5.0)], vec![3, 1])
                .unwrap();
        test_support::with_test_provider(|provider| {
            let handle = gpu_helpers::upload_complex_tensor(provider, &input).unwrap();
            let include = sum_builtin(Value::GpuTensor(handle.clone()), Vec::new()).unwrap();
            let Value::ComplexTensor(include) =
                block_on(gpu_helpers::gather_value_async(&include)).unwrap()
            else {
                panic!("expected complex include result");
            };
            let include = include.materialize_f64();
            assert!(include[0].0.is_nan() && include[0].1.is_nan());
            let omit = sum_builtin(Value::GpuTensor(handle), vec![Value::from("omitnan")]).unwrap();
            let Value::ComplexTensor(omit) =
                block_on(gpu_helpers::gather_value_async(&omit)).unwrap()
            else {
                panic!("expected complex omit result");
            };
            assert_eq!(omit.materialize_f64(), vec![(5.0, 7.0)]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sum_gpu_row_vector_auto_dim() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let result = sum_builtin(Value::GpuTensor(handle), Vec::new()).expect("sum");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape(), [1, 1]);
            assert_eq!(values(&gathered), vec![10.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sum_gpu_all_reduction() {
        test_support::with_test_provider(|provider| {
            let tensor =
                Tensor::new((1..=8).map(|v| v as f64).collect::<Vec<_>>(), vec![2, 4]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let result =
                sum_builtin(Value::GpuTensor(handle), vec![Value::from("all")]).expect("sum");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape(), [1, 1]);
            assert_eq!(values(&gathered), vec![36.0]);
        });
    }

    #[test]
    fn sum_integer_gpu_dimension_beyond_rank_returns_resident_double() {
        test_support::with_test_provider(|provider| {
            let handle = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&[9_007_199_254_740_992, 5]),
                    shape: &[1, 2],
                })
                .expect("upload integer gpuArray");
            let result = sum_builtin(Value::GpuTensor(handle), vec![Value::Int(IntValue::I32(4))])
                .expect("sum");
            let Value::GpuTensor(output) = &result else {
                panic!("expected resident gpuArray output");
            };
            assert_eq!(output.shape(), vec![1, 2]);
            assert_eq!(runmat_accelerate_api::handle_integer_type(output), None);
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.numeric_dtype(), NumericDType::F64);
            assert_eq!(values(&gathered), vec![9_007_199_254_740_992.0, 5.0]);
        });
    }

    #[test]
    fn sum_integer_gpu_omitnan_fallback_restores_resident_double() {
        test_support::with_test_provider(|provider| {
            let handle = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&[1, 3, 5, 7]),
                    shape: &[2, 2],
                })
                .expect("upload integer gpuArray");
            let result = sum_builtin(Value::GpuTensor(handle), vec![Value::from("omitnan")])
                .expect("sum omitnan");
            let Value::GpuTensor(output) = &result else {
                panic!("expected resident double output");
            };
            assert_eq!(runmat_accelerate_api::handle_integer_type(output), None);
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.numeric_dtype(), NumericDType::F64);
            assert_eq!(values(&gathered), vec![4.0, 12.0]);
        });
    }

    #[test]
    fn sum_integer_gpu_default_accumulates_double_before_native_saturation() {
        test_support::with_test_provider(|provider| {
            let upload = || {
                provider
                    .upload_integer(&HostIntegerTensorView {
                        data: HostIntegerDataView::I8(&[100, 100, 100, 100]),
                        shape: &[2, 2],
                    })
                    .expect("upload int8 gpuArray")
            };
            for options in [vec![], vec![Value::from("double")]] {
                let result =
                    sum_builtin(Value::GpuTensor(upload()), options).expect("ordinary integer sum");
                let gathered = test_support::gather(result).expect("gather double sum");
                assert_eq!(gathered.numeric_dtype(), NumericDType::F64);
                assert_eq!(values(&gathered), vec![200.0, 200.0]);
            }
            let native = sum_builtin(Value::GpuTensor(upload()), vec![Value::from("native")])
                .expect("native integer sum");
            let Value::GpuTensor(native) = native else {
                panic!("native result should remain resident");
            };
            assert_eq!(
                block_on(provider.download_integer(&native))
                    .expect("native int8 sum")
                    .data,
                HostIntegerDataOwned::I8(vec![127, 127])
            );
        });
    }

    #[test]
    fn native_sum_of_resident_logical_u8_is_numeric_double_not_integer() {
        test_support::with_test_provider(|provider| {
            let input = provider
                .upload_logical(&HostLogicalTensorView {
                    data: &[1, 1, 0, 1],
                    shape: &[2, 2],
                })
                .expect("upload logical gpuArray");
            assert!(runmat_accelerate_api::handle_is_logical(&input));
            let result = sum_builtin(Value::GpuTensor(input), vec![Value::from("native")])
                .expect("logical native sum");
            let Value::GpuTensor(output) = &result else {
                panic!("logical sum should remain resident on the F64 owner");
            };
            assert!(!runmat_accelerate_api::handle_is_logical(output));
            assert_eq!(runmat_accelerate_api::handle_integer_type(output), None);
            let gathered = test_support::gather(result).expect("gather logical sum");
            assert_eq!(gathered.numeric_dtype(), NumericDType::F64);
            assert_eq!(values(&gathered), vec![2.0, 1.0]);
        });
    }

    #[test]
    fn resident_native_integer_omitnan_keeps_wide_exact_payload() {
        test_support::with_test_provider(|provider| {
            let wide = (1_u64 << 53) + 1;
            let input = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&[wide, 0]),
                    shape: &[2, 1],
                })
                .expect("upload wide uint64");
            let result = sum_builtin(
                Value::GpuTensor(input),
                vec![Value::from("omitnan"), Value::from("native")],
            )
            .expect("native integer omitnan");
            let Value::GpuTensor(result) = result else {
                panic!("native integer result should remain resident");
            };
            assert_eq!(
                block_on(provider.download_integer(&result))
                    .expect("exact native result")
                    .data,
                HostIntegerDataOwned::U64(vec![wide])
            );
        });
    }

    #[test]
    fn sum_empty_integer_gpu_identities_preserve_class_shape_and_residency() {
        test_support::with_test_provider(|provider| {
            let upload = || {
                provider
                    .upload_integer(&HostIntegerTensorView {
                        data: HostIntegerDataView::U16(&[]),
                        shape: &[0, 3],
                    })
                    .expect("upload empty integer gpuArray")
            };
            let default = sum_builtin(Value::GpuTensor(upload()), Vec::new()).expect("default sum");
            let Value::GpuTensor(default_handle) = &default else {
                panic!("expected resident default output");
            };
            assert_eq!(default_handle.shape(), vec![1, 3]);
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(default_handle),
                None
            );
            let default = test_support::gather(default).expect("gather default sum");
            assert_eq!(default.numeric_dtype(), NumericDType::F64);
            assert_eq!(values(&default), vec![0.0; 3]);

            let native = sum_builtin(Value::GpuTensor(upload()), vec![Value::from("native")])
                .expect("native sum");
            let Value::GpuTensor(native) = native else {
                panic!("expected resident native output");
            };
            assert_eq!(native.shape(), vec![1, 3]);
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&native),
                Some(IntegerElementType::U16)
            );
            assert_eq!(
                block_on(provider.download_integer(&native))
                    .expect("download native sum")
                    .data,
                HostIntegerDataOwned::U16(vec![0; 3])
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn sum_native_integer_gpu_stays_resident() {
        match runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) {
            Ok(_) => {}
            Err(error) if error.to_string() == "wgpu: no compatible adapter found" => return,
            Err(error) => panic!("register wgpu provider failed: {error:?}"),
        }
        let provider = runmat_accelerate_api::provider().expect("provider");
        let handle = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::I16(&[10, 20, 30, 40]),
                shape: &[2, 2],
            })
            .expect("upload native integer gpuArray");
        let result = sum_builtin(
            Value::GpuTensor(handle),
            vec![Value::Int(IntValue::I32(1)), Value::from("native")],
        )
        .expect("sum native integer gpuArray");
        let Value::GpuTensor(out) = result else {
            panic!("expected resident gpuArray result");
        };
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&out),
            Some(IntegerElementType::I16)
        );
        assert_eq!(
            block_on(provider.download_integer(&out))
                .expect("download native sum")
                .data,
            HostIntegerDataOwned::I16(vec![30, 70])
        );
    }

    #[test]
    fn sum_native_integer_gpu_vecdim_nd_stays_resident() {
        test_support::with_test_provider(|provider| {
            let values: Vec<u16> = (1..=12).collect();
            let handle = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U16(&values),
                    shape: &[2, 3, 2],
                })
                .expect("upload native integer gpuArray");
            let dims = Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap();
            let result = sum_builtin(
                Value::GpuTensor(handle),
                vec![Value::Tensor(dims), Value::from("native")],
            )
            .expect("sum native integer gpuArray vecdim");
            let Value::GpuTensor(out) = result else {
                panic!("expected resident gpuArray result");
            };
            assert_eq!(out.shape(), vec![1, 3, 1]);
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&out),
                Some(IntegerElementType::U16)
            );
            assert_eq!(
                block_on(provider.download_integer(&out))
                    .expect("download native vecdim sum")
                    .data,
                HostIntegerDataOwned::U16(vec![18, 26, 34])
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn sum_wgpu_dim1_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let t = Tensor::new(vec![1.0, 4.0, 2.0, 6.0], vec![2, 2]).unwrap();
        let cpu = sum_host(
            Value::Tensor(t.clone()),
            &ParsedArguments {
                invocation: SumInvocation::Sum,
                selection: DimSelection::Dim(1),
                nan_mode: ReductionNaN::Include,
                output: OutputTemplate::Double,
            },
        )
        .unwrap();
        let h = gpu_helpers::upload_tensor(runmat_accelerate_api::provider().unwrap(), &t).unwrap();
        let gpu = block_on(sum_gpu(
            h,
            &ParsedArguments {
                invocation: SumInvocation::Sum,
                selection: DimSelection::Dim(1),
                nan_mode: ReductionNaN::Include,
                output: OutputTemplate::Double,
            },
        ))
        .unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match cpu {
            Value::Tensor(ct) => {
                assert_eq!(gathered.shape(), ct.shape());
                assert_eq!(values(&gathered), values(&ct));
            }
            other => panic!("unexpected shapes {other:?}"),
        }
    }
}
