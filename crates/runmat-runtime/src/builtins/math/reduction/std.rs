//! MATLAB-compatible `std` builtin with GPU-aware semantics for RunMat.
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, NumericElementType, ProviderNanMode,
    ProviderStdNormalization,
};
use runmat_builtins::{
    BuiltinErrorDescriptor, DispersionControl, DispersionNormalization as StdNormalization,
    STD_ERROR_COMPLEX_UNSUPPORTED, STD_ERROR_INTERNAL, STD_ERROR_INVALID_ARGUMENT,
    STD_ERROR_INVALID_INPUT, STD_TYPED_INTEGER_CONTROL_EXTENSION,
};
use runmat_value::{Tensor, Value};
const NAME: &str = "std";

use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{
    gpu_helpers,
    shape::{is_scalar_shape, normalize_scalar_shape},
    tensor,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::reduction::std")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "std",
    op_kind: GpuOpKind::Reduction,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Reduction {
            name: "reduce_std_dim",
        },
        ProviderHook::Reduction {
            name: "reduce_std",
        },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: Some(256),
    workgroup_size: Some(256),
    accepts_nan_mode: true,
    notes: "Providers may offer reduce_std_dim/reduce_std implementations; host fallback ensures correctness when they are unavailable.",
};

fn std_error_with_descriptor(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn std_invalid_argument(message: impl Into<String>) -> RuntimeError {
    std_error_with_descriptor(message, &STD_ERROR_INVALID_ARGUMENT)
}

fn std_invalid_input(message: impl Into<String>) -> RuntimeError {
    std_error_with_descriptor(message, &STD_ERROR_INVALID_INPUT)
}

fn std_complex_unsupported_default() -> RuntimeError {
    std_error_with_descriptor(
        STD_ERROR_COMPLEX_UNSUPPORTED.message,
        &STD_ERROR_COMPLEX_UNSUPPORTED,
    )
}

fn std_internal_error(message: impl Into<String>) -> RuntimeError {
    std_error_with_descriptor(message, &STD_ERROR_INTERNAL)
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::reduction::std")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "std",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Fusion currently gathers to the host; future kernels can reuse the variance accumulator directly.",
};

#[derive(Clone, Debug)]
enum StdAxes {
    Default,
    Dim(usize),
    Vec(Vec<usize>),
    All,
}

#[derive(Clone)]
struct ParsedArguments {
    axes: StdAxes,
    normalization: StdNormalization,
    nan_mode: ReductionNaN,
}

enum NormParse {
    NotMatched,
    Placeholder,
    Value(StdNormalization),
}

#[runtime_builtin(
    name = "std",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::reduction::std"
)]
pub(crate) async fn std_builtin(
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    std_with_nan_policy(value, rest, None).await
}

pub(crate) async fn std_with_nan_policy(
    value: Value,
    rest: crate::call::BuiltinArguments,
    imposed_nan_mode: Option<ReductionNaN>,
) -> crate::BuiltinResult<Value> {
    reject_integer_data_input(&value)?;
    if rest.iter().any(is_real_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &STD_TYPED_INTEGER_CONTROL_EXTENSION,
            NAME,
        )?;
    }
    let mut parsed = parse_arguments(rest.view()).await?;
    if let Some(mode) = imposed_nan_mode {
        parsed.nan_mode = mode;
    }
    Ok(match value {
        Value::GpuTensor(handle) => std_gpu(handle, &parsed).await?,
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            return Err(std_complex_unsupported_default());
        }
        other => std_host(other, &parsed)?,
    })
}

fn is_real_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(
            value,
            Value::GpuTensor(handle)
                if runmat_accelerate_api::handle_integer_type(handle).is_some()
        )
}

fn reject_integer_data_input(value: &Value) -> BuiltinResult<()> {
    let is_integer = match value {
        Value::Int(_) => true,
        Value::Tensor(tensor) => tensor.integer_storage().is_some(),
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_integer_type(handle).is_some(),
        _ => false,
    };
    if is_integer {
        return Err(std_invalid_input(
            "std: integer data inputs are not supported",
        ));
    }
    Ok(())
}

async fn parse_arguments(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<ParsedArguments> {
    let mut axes = StdAxes::Default;
    let mut axes_set = false;
    let mut normalization = StdNormalization::Sample;
    let mut normalization_consumed = false;
    let mut nan_mode = ReductionNaN::Include;
    let mut idx = 0;
    while idx < args.len() {
        let arg = &args[idx];
        if let Some(keyword) = keyword_of(arg)? {
            match DispersionControl::from_std_public_option(&keyword) {
                Some(DispersionControl::Missing(mode)) => {
                    nan_mode = mode;
                    idx += 1;
                    continue;
                }
                Some(DispersionControl::All) => {
                    if axes_set && !matches!(axes, StdAxes::Default) {
                        return Err(std_invalid_argument(
                            "std: 'all' cannot be combined with an explicit dimension",
                        ));
                    }
                    axes = StdAxes::All;
                    axes_set = true;
                    idx += 1;
                    continue;
                }
                None => {}
            }
        }

        if !normalization_consumed {
            match parse_normalization(arg)? {
                NormParse::Value(norm) => {
                    normalization = norm;
                    normalization_consumed = true;
                    idx += 1;
                    continue;
                }
                NormParse::Placeholder => {
                    normalization_consumed = true;
                    idx += 1;
                    continue;
                }
                NormParse::NotMatched => {}
            }
        }

        if !axes_set || matches!(axes, StdAxes::Default) {
            if let Some(selection) = parse_axes(arg).await? {
                if matches!(selection, StdAxes::All)
                    && axes_set
                    && !matches!(axes, StdAxes::Default)
                {
                    return Err(std_invalid_argument(
                        "std: 'all' cannot be combined with an explicit dimension",
                    ));
                }
                axes = selection;
                axes_set = true;
                idx += 1;
                continue;
            }
        } else if let Some(selection) = parse_axes(arg).await? {
            if matches!(selection, StdAxes::All) {
                return Err(std_invalid_argument(
                    "std: 'all' cannot be combined with an explicit dimension",
                ));
            }
            return Err(std_invalid_argument(
                "std: multiple dimension specifications provided",
            ));
        }

        return Err(std_invalid_argument(format!(
            "std: unrecognised argument {arg:?}"
        )));
    }

    Ok(ParsedArguments {
        axes,
        normalization,
        nan_mode,
    })
}

fn parse_normalization(value: &Value) -> BuiltinResult<NormParse> {
    match value {
        Value::Tensor(tensor) => {
            let values = tensor::tensor_values_f64(tensor);
            if values.is_empty() {
                return Ok(NormParse::Placeholder);
            }
            if values.len() == 1 {
                return parse_normalization_scalar(values[0]);
            }
            Ok(NormParse::NotMatched)
        }
        Value::LogicalArray(logical) => {
            if logical.data.is_empty() {
                return Ok(NormParse::Placeholder);
            }
            if logical.data.len() == 1 {
                return parse_normalization_scalar(if logical.data[0] != 0 { 1.0 } else { 0.0 });
            }
            Ok(NormParse::NotMatched)
        }
        Value::Bool(flag) => Ok(NormParse::Value(if *flag {
            StdNormalization::Population
        } else {
            StdNormalization::Sample
        })),
        Value::Int(i) => match i.try_to_u64() {
            Some(0) => Ok(NormParse::Value(StdNormalization::Sample)),
            Some(1) => Ok(NormParse::Value(StdNormalization::Population)),
            _ => Ok(NormParse::NotMatched),
        },
        Value::Num(n) => parse_normalization_scalar(*n),
        Value::GpuTensor(_) => Err(std_invalid_argument(
            "std: normalisation flag must reside on the host",
        )),
        _ => Ok(NormParse::NotMatched),
    }
}

fn parse_normalization_scalar(value: f64) -> BuiltinResult<NormParse> {
    if !value.is_finite() {
        return Err(std_invalid_argument(
            "std: normalisation flag must be finite",
        ));
    }
    Ok(StdNormalization::from_finite_flag(value)
        .map(NormParse::Value)
        .unwrap_or(NormParse::NotMatched))
}

async fn parse_axes(value: &Value) -> BuiltinResult<Option<StdAxes>> {
    if let Some(text) = value_as_str(value)? {
        if text.trim().is_empty() {
            return Err(std_invalid_argument(
                "std: dimension string must not be empty",
            ));
        }
        return Ok(matches!(
            DispersionControl::from_std_public_option(&text),
            Some(DispersionControl::All)
        )
        .then_some(StdAxes::All));
    }

    let (scalar_hint, is_empty) = match value {
        Value::Num(_) | Value::Int(_) => (true, false),
        Value::Tensor(t) => (tensor::is_scalar_tensor(t), tensor_len(t) == 0),
        Value::LogicalArray(logical) => (logical.data.len() == 1, logical.data.is_empty()),
        Value::GpuTensor(handle) => {
            let count = tensor::element_count(handle.shape());
            (is_scalar_shape(handle.shape()) || count == 1, count == 0)
        }
        _ => (false, false),
    };
    if is_empty {
        return Ok(Some(StdAxes::Default));
    }

    let dims = match value {
        Value::Tensor(_)
        | Value::LogicalArray(_)
        | Value::Int(_)
        | Value::Num(_)
        | Value::GpuTensor(_) => tensor::dims_from_value_async_typed(value)
            .await
            .map_err(|err| map_dims_error(err, scalar_hint))?,
        Value::Bool(_) => {
            return Err(std_invalid_argument("std: dimension must be numeric"));
        }
        _ => return Ok(None),
    };

    let Some(dims) = dims else {
        return Ok(None);
    };
    if dims.is_empty() {
        return Err(std_invalid_argument(
            "std: dimension vector must not be empty",
        ));
    }
    if dims.len() == 1 {
        let dim = dims[0];
        if dim < 1 {
            return Err(std_invalid_argument("std: dimension must be >= 1"));
        }
        return Ok(Some(StdAxes::Dim(dim)));
    }
    for &dim in &dims {
        if dim < 1 {
            return Err(std_invalid_argument("std: dimension entries must be >= 1"));
        }
    }
    Ok(Some(StdAxes::Vec(dims)))
}

fn tensor_len(tensor: &Tensor) -> usize {
    tensor.len()
}

fn value_as_str(value: &Value) -> BuiltinResult<Option<String>> {
    Ok(match value {
        Value::String(runmat_value::StringElement::Text(s)) => Some(s.as_str().to_owned()),
        Value::StringArray(sa) if sa.data().len() == 1 && !sa.data()[0].is_missing() => {
            sa.data()[0].as_str().map(str::to_owned)
        }
        Value::CharArray(ca) if ca.rows == 1 => Some(
            runmat_value::decode_utf16_code_units(ca.data()).map_err(|source| {
                let mut error = std_invalid_argument(source.to_string());
                error.source = Some(Box::new(source));
                error
            })?,
        ),
        _ => None,
    })
}

fn map_dims_error(error: tensor::DimensionDecodeError, scalar: bool) -> RuntimeError {
    use tensor::DimensionDecodeError;
    let message = match (&error, scalar) {
        (
            DimensionDecodeError::Negative
            | DimensionDecodeError::IntegerNegative
            | DimensionDecodeError::IntegerOutOfPlatformRange
            | DimensionDecodeError::BelowMinimum,
            true,
        ) => "std: dimension must be >= 1".to_string(),
        (
            DimensionDecodeError::Negative
            | DimensionDecodeError::IntegerNegative
            | DimensionDecodeError::IntegerOutOfPlatformRange
            | DimensionDecodeError::BelowMinimum,
            false,
        ) => "std: dimension entries must be >= 1".to_string(),
        (DimensionDecodeError::NonFinite, true) => "std: dimension must be finite".to_string(),
        (DimensionDecodeError::NonFinite, false) => {
            "std: dimension entries must be finite integers".to_string()
        }
        (DimensionDecodeError::Fractional, true) => "std: dimension must be an integer".to_string(),
        (DimensionDecodeError::Fractional, false) => {
            "std: dimension entries must be integers".to_string()
        }
        (DimensionDecodeError::OutOfPlatformRange | DimensionDecodeError::Gather(_), _) => {
            error.to_string()
        }
    };
    error.attach_cause(std_invalid_argument(message))
}

fn std_host(value: Value, args: &ParsedArguments) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("std", value).map_err(std_invalid_input)?;
    let reduced = std_tensor(tensor, &args.axes, args.normalization, args.nan_mode)?;
    Ok(tensor::tensor_into_value(reduced))
}

fn std_tensor(
    tensor: Tensor,
    axes: &StdAxes,
    normalization: StdNormalization,
    nan_mode: ReductionNaN,
) -> BuiltinResult<Tensor> {
    let (dims, _) = resolve_axes(tensor.shape(), axes)?;
    if dims.is_empty() {
        if tensor::is_scalar_tensor(&tensor) {
            return std_scalar_tensor(&tensor, nan_mode);
        }
        if !matches!(axes, StdAxes::Default) {
            return tensor::zeros_with_dtype(tensor.shape(), tensor.numeric_dtype())
                .map_err(|error| std_internal_error(format!("std: {error}")));
        }
        return Ok(tensor);
    }
    std_tensor_reduce(&tensor, &dims, normalization, nan_mode)
}

fn std_scalar_tensor(tensor: &Tensor, nan_mode: ReductionNaN) -> BuiltinResult<Tensor> {
    let value = tensor::tensor_values_f64(tensor)
        .into_iter()
        .next()
        .unwrap_or(f64::NAN);
    let result = if value.is_nan() {
        f64::NAN
    } else {
        match nan_mode {
            ReductionNaN::Include | ReductionNaN::Omit => 0.0,
        }
    };
    Tensor::new_with_dtype(vec![result], vec![1, 1], tensor.numeric_dtype())
        .map_err(|e| std_internal_error(format!("std: {e}")))
}

fn std_tensor_reduce(
    tensor: &Tensor,
    dims: &[usize],
    normalization: StdNormalization,
    nan_mode: ReductionNaN,
) -> BuiltinResult<Tensor> {
    let mut dims_sorted = dims.to_vec();
    dims_sorted.sort_unstable();
    dims_sorted.dedup();
    if dims_sorted.is_empty() {
        return Ok(tensor.clone());
    }

    let output_shape = reduced_shape(tensor.shape(), &dims_sorted);
    let out_len = tensor::element_count(&output_shape);
    let values = tensor::tensor_values_f64(tensor);
    if values.is_empty() {
        let fill = vec![f64::NAN; out_len];
        return Tensor::new_with_dtype(fill, output_shape, tensor.numeric_dtype())
            .map_err(|e| std_internal_error(format!("std: {e}")));
    }

    let mut counts = vec![0usize; out_len];
    let mut means = vec![0.0f64; out_len];
    let mut m2 = vec![0.0f64; out_len];
    let mut saw_nan = vec![false; out_len];
    let mut coords = vec![0usize; tensor.shape().len()];
    let mut out_coords = vec![0usize; tensor.shape().len()];
    let mut reduce_mask = vec![false; tensor.shape().len()];
    for &dim in &dims_sorted {
        if dim < reduce_mask.len() {
            reduce_mask[dim] = true;
        }
    }

    for (linear, value) in values.into_iter().enumerate() {
        linear_to_multi(linear, tensor.shape(), &mut coords);
        for (i, coord) in coords.iter().enumerate() {
            out_coords[i] = if reduce_mask[i] { 0 } else { *coord };
        }
        let out_idx = multi_to_linear(&out_coords, &output_shape);
        if value.is_nan() {
            if matches!(nan_mode, ReductionNaN::Include) {
                saw_nan[out_idx] = true;
            }
            continue;
        }

        let mean = &mut means[out_idx];
        let m2_slot = &mut m2[out_idx];
        counts[out_idx] += 1;
        let count = counts[out_idx];
        let delta = value - *mean;
        *mean += delta / (count as f64);
        let delta2 = value - *mean;
        *m2_slot += delta * delta2;
    }

    let mut output = vec![0.0f64; out_len];
    for idx in 0..out_len {
        output[idx] =
            if (saw_nan[idx] && matches!(nan_mode, ReductionNaN::Include)) || counts[idx] == 0 {
                f64::NAN
            } else {
                let count = counts[idx];
                let variance = match normalization {
                    StdNormalization::Sample => {
                        if count > 1 {
                            (m2[idx] / (count - 1) as f64).max(0.0)
                        } else {
                            0.0
                        }
                    }
                    StdNormalization::Population => (m2[idx] / (count as f64)).max(0.0),
                };
                variance.sqrt()
            };
    }

    Tensor::new_with_dtype(output, output_shape, tensor.numeric_dtype())
        .map_err(|e| std_internal_error(format!("std: {e}")))
}

fn resolve_axes(shape: &[usize], axes: &StdAxes) -> BuiltinResult<(Vec<usize>, bool)> {
    match axes {
        StdAxes::Default => {
            if is_scalar_shape(shape) {
                Ok((Vec::new(), true))
            } else {
                let dim = default_dimension_from_shape(shape);
                let zero = dim.saturating_sub(1);
                if zero < shape.len() {
                    Ok((vec![zero], true))
                } else {
                    Ok((Vec::new(), true))
                }
            }
        }
        StdAxes::Dim(dim) => {
            if *dim == 0 {
                return Err(std_invalid_argument("std: dimension must be >= 1"));
            }
            let zero = dim - 1;
            if zero < shape.len() {
                Ok((vec![zero], true))
            } else {
                Ok((Vec::new(), true))
            }
        }
        StdAxes::Vec(dims) => {
            if dims.is_empty() {
                return resolve_axes(shape, &StdAxes::Default);
            }
            let mut out = Vec::with_capacity(dims.len());
            for &dim in dims {
                if dim == 0 {
                    return Err(std_invalid_argument("std: dimension must be >= 1"));
                }
                let zero = dim - 1;
                if zero < shape.len() {
                    out.push(zero);
                }
            }
            out.sort_unstable();
            out.dedup();
            Ok((out, true))
        }
        StdAxes::All => {
            if is_scalar_shape(shape) {
                Ok((Vec::new(), true))
            } else {
                Ok(((0..shape.len()).collect(), true))
            }
        }
    }
}

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
    for (&coord, &size) in coords.iter().zip(shape.iter()) {
        if size == 0 {
            continue;
        }
        index += coord * stride;
        stride *= size;
    }
    index
}

async fn std_gpu(handle: GpuTensorHandle, args: &ParsedArguments) -> BuiltinResult<Value> {
    gpu_helpers::require_resident_compute_owner(&handle, || {
        std_internal_error("std: resident input has no exact owning provider")
    })?;
    if let Some(provider) = gpu_helpers::exact_provider_for_handle(&handle) {
        handle
            .allocation_lease()
            .validate_live()
            .map_err(|source| {
                let mut error = std_internal_error(format!(
                    "std: resident input allocation is unavailable: {source}"
                ));
                error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                error.source = Some(Box::new(source));
                error
            })?;
        if let Some(device_value) = std_gpu_reduce(provider, &handle, args).await? {
            return Ok(Value::GpuTensor(device_value));
        }
    }
    std_gpu_fallback(&handle, args).await
}

async fn std_gpu_reduce(
    provider: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
    args: &ParsedArguments,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    let (dims, _) = resolve_axes(handle.shape(), &args.axes)?;
    let normalization = match args.normalization {
        StdNormalization::Sample => ProviderStdNormalization::Sample,
        StdNormalization::Population => ProviderStdNormalization::Population,
    };
    let nan_mode = match args.nan_mode {
        ReductionNaN::Include => ProviderNanMode::Include,
        ReductionNaN::Omit => ProviderNanMode::Omit,
    };

    if dims.is_empty() {
        if is_scalar_shape(handle.shape()) {
            return accepted_std_result(
                provider,
                handle,
                &normalize_scalar_shape(handle.shape()),
                "reduce_std",
                provider.reduce_std(handle, normalization, nan_mode).await,
            );
        }
        return Ok(None);
    }

    if dims.len() == handle.shape().len() {
        return accepted_std_result(
            provider,
            handle,
            &reduced_shape(handle.shape(), &dims),
            "reduce_std",
            provider.reduce_std(handle, normalization, nan_mode).await,
        );
    }

    if dims.len() == 1 {
        let dim = dims[0] + 1;
        return reduce_std_dim_gpu(provider, handle, dim, normalization, nan_mode).await;
    }

    Ok(None)
}

fn initial_std_result(
    operation: &'static str,
    result: runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle>,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    match result {
        Ok(output) => Ok(Some(output)),
        Err(source) if source.is_unsupported() => Ok(None),
        Err(source) => {
            let mut error =
                std_internal_error(format!("std: provider {operation} failed: {source}"));
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            Err(crate::runtime_error::with_provider_execution_source(
                error, source,
            ))
        }
    }
}

pub(super) fn std_output_matches(
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    output: &GpuTensorHandle,
    expected_shape: &[usize],
) -> bool {
    let element = input.physical().element_type();
    if input.allocation_lease().validate_live().is_err()
        || !matches!(element, NumericElementType::F32 | NumericElementType::F64)
        || runmat_accelerate_api::handle_storage(input) != GpuTensorStorage::Real
        || runmat_accelerate_api::handle_integer_type(input).is_some()
        || runmat_accelerate_api::handle_is_logical(input)
        || !gpu_helpers::numeric_descriptor_matches(input, element, GpuTensorStorage::Real)
    {
        return false;
    }
    gpu_helpers::unary_gpu_output_matches_shape(
        output,
        input,
        provider,
        expected_shape,
        gpu_helpers::UnaryGpuOutputContract {
            storage: GpuTensorStorage::Real,
            precision: runmat_accelerate_api::handle_precision(input),
            integer: None,
            logical: false,
            alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation,
        },
    ) && output.physical().element_type() == element
        && gpu_helpers::numeric_descriptor_matches(output, element, GpuTensorStorage::Real)
}

fn accepted_std_result(
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    expected_shape: &[usize],
    operation: &'static str,
    result: runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle>,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    let Some(output) = initial_std_result(operation, result)? else {
        return Ok(None);
    };
    if !std_output_matches(provider, input, &output, expected_shape) {
        let mut error = std_internal_error(format!(
            "std: provider {operation} returned an invalid resident reduction"
        ));
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        return Err(gpu_helpers::reject_gpu_output(output, error));
    }
    Ok(Some(output))
}

async fn reduce_std_dim_gpu(
    provider: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
    dim: usize,
    normalization: ProviderStdNormalization,
    nan_mode: ProviderNanMode,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    if dim == 0 {
        return Ok(None);
    }
    if handle.shape().len() < dim {
        return Ok(None);
    }
    accepted_std_result(
        provider,
        handle,
        &reduced_shape(handle.shape(), &[dim - 1]),
        "reduce_std_dim",
        provider
            .reduce_std_dim(handle, dim - 1, normalization, nan_mode)
            .await,
    )
}

async fn std_gpu_fallback(
    handle: &GpuTensorHandle,
    args: &ParsedArguments,
) -> BuiltinResult<Value> {
    let tensor = gpu_helpers::gather_tensor_async(handle).await?;
    let reduced = std_tensor(tensor, &args.axes, args.normalization, args.nan_mode)?;
    Ok(tensor::tensor_into_value(reduced))
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

#[cfg(all(test, any(feature = "wgpu", target_arch = "wasm32")))]
#[path = "std/wgpu_required_tests.rs"]
mod wgpu_required_tests;

#[cfg(test)]
#[path = "std/typed_phase_tests.rs"]
mod typed_phase_tests;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_value::{IntValue, IntegerStorage, Tensor};
    use std::f64::consts::{FRAC_1_SQRT_2, SQRT_2};

    fn std_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        block_on(super::std_builtin(
            value,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
    }

    fn values(tensor: &Tensor) -> Vec<f64> {
        tensor.materialize_f64()
    }

    fn integer_inputs() -> Vec<Value> {
        vec![
            IntegerStorage::I8(vec![0, 2, 4]),
            IntegerStorage::I16(vec![0, 2, 4]),
            IntegerStorage::I32(vec![0, 2, 4]),
            IntegerStorage::I64(vec![0, 2, 4]),
            IntegerStorage::U8(vec![0, 2, 4]),
            IntegerStorage::U16(vec![0, 2, 4]),
            IntegerStorage::U32(vec![0, 2, 4]),
            IntegerStorage::U64(vec![0, 2, 4]),
        ]
        .into_iter()
        .map(|storage| {
            Value::Tensor(Tensor::new_integer(storage, vec![3, 1]).expect("integer tensor"))
        })
        .collect()
    }

    #[test]
    fn std_typed_normalization_requires_exact_zero_or_one() {
        assert!(matches!(
            parse_normalization(&Value::Int(IntValue::U64(0))),
            Ok(NormParse::Value(StdNormalization::Sample))
        ));
        assert!(matches!(
            parse_normalization(&Value::Int(IntValue::U64(1))),
            Ok(NormParse::Value(StdNormalization::Population))
        ));
        assert!(matches!(
            parse_normalization(&Value::Int(IntValue::U64(u64::MAX))),
            Ok(NormParse::NotMatched)
        ));
        assert!(matches!(
            parse_normalization(&Value::Int(IntValue::I64(-1))),
            Ok(NormParse::NotMatched)
        ));
    }

    #[test]
    fn std_typed_integer_tensor_normalization_reads_exact_storage() {
        let normalization =
            Tensor::new_integer(IntegerStorage::U16(vec![1]), vec![1, 1]).expect("normalization");

        assert!(matches!(
            parse_normalization(&Value::Tensor(normalization)),
            Ok(NormParse::Value(StdNormalization::Population))
        ));
    }

    #[test]
    fn std_typed_integer_controls_follow_compatibility_mode() {
        let tensor = Tensor::new(vec![1.0, 3.0], vec![2, 1]).expect("input");
        let args = vec![Value::Int(IntValue::U8(1))];
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(super::std_builtin(
                Value::Tensor(tensor.clone()),
                crate::builtins::common::test_support::builtin_arguments(args.clone()),
            ))
            .expect_err("MATLAB-compatible mode");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:StdTypedIntegerControlExtension")
            );
        }
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            block_on(super::std_builtin(
                Value::Tensor(tensor),
                crate::builtins::common::test_support::builtin_arguments(args),
            ))
            .expect("RunMat extension mode");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn std_vector_sample_default() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5, 1]).unwrap();
        let result = std_builtin(Value::Tensor(tensor), Vec::new()).expect("std");
        match result {
            Value::Num(v) => {
                let diff = (v - 1.58113883008).abs();
                assert!(diff < 1e-10, "value={v}, diff={diff}");
            }
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn host_single_output_boundary_preserves_class_across_scalar_empty_and_nd() {
        let scalar = Tensor::new_with_dtype(vec![7.0], vec![1, 1], runmat_value::NumericDType::F32)
            .expect("single scalar");
        let Value::Tensor(scalar_out) =
            std_builtin(Value::Tensor(scalar), Vec::new()).expect("single std scalar")
        else {
            panic!("single scalar result must remain a tensor")
        };
        assert_eq!(scalar_out.numeric_dtype(), runmat_value::NumericDType::F32);
        assert_eq!(scalar_out.shape(), [1, 1]);
        assert_eq!(values(&scalar_out), vec![0.0]);

        let input = Tensor::new_with_dtype(
            (1..=8).map(f64::from).collect(),
            vec![2, 2, 2],
            runmat_value::NumericDType::F32,
        )
        .expect("single N-D input");
        let axes = Tensor::new(vec![1.0, 3.0], vec![1, 2]).expect("axes");
        let Value::Tensor(nd) = std_builtin(
            Value::Tensor(input),
            vec![Value::Num(0.0), Value::Tensor(axes)],
        )
        .expect("single N-D std") else {
            panic!("single N-D result must remain a tensor")
        };
        assert_eq!(nd.numeric_dtype(), runmat_value::NumericDType::F32);
        assert_eq!(nd.shape(), [1, 2, 1]);
        for value in values(&nd) {
            assert!((value - (17.0_f64 / 3.0).sqrt()).abs() < 1e-6);
        }

        let empty = Tensor::new_with_dtype(Vec::new(), vec![0, 2], runmat_value::NumericDType::F32)
            .expect("single empty input");
        let Value::Tensor(empty_out) =
            std_builtin(Value::Tensor(empty.clone()), Vec::new()).expect("single empty std")
        else {
            panic!("single empty result must remain a tensor")
        };
        assert_eq!(empty_out.numeric_dtype(), runmat_value::NumericDType::F32);
        assert_eq!(empty_out.shape(), [1, 2]);
        assert!(values(&empty_out).iter().all(|value| value.is_nan()));
        let Value::Tensor(empty_all) = std_builtin(Value::Tensor(empty), vec![Value::from("all")])
            .expect("single empty all std")
        else {
            panic!("single empty all result must remain a tensor")
        };
        assert_eq!(empty_all.numeric_dtype(), runmat_value::NumericDType::F32);
        assert_eq!(empty_all.shape(), [1, 1]);
        assert!(values(&empty_all)[0].is_nan());

        let beyond =
            Tensor::new_with_dtype(vec![1.0, 2.0], vec![1, 2], runmat_value::NumericDType::F32)
                .expect("single beyond-rank input");
        let Value::Tensor(beyond_out) = std_builtin(
            Value::Tensor(beyond),
            vec![Value::Num(0.0), Value::Num(4.0)],
        )
        .expect("single beyond-rank std") else {
            panic!("single beyond-rank result must remain a tensor")
        };
        assert_eq!(beyond_out.numeric_dtype(), runmat_value::NumericDType::F32);
        assert_eq!(beyond_out.shape(), [1, 2]);
        assert_eq!(values(&beyond_out), vec![0.0, 0.0]);

        let population = Tensor::new_with_dtype(
            (1..=4).map(f64::from).collect(),
            vec![2, 2],
            runmat_value::NumericDType::F32,
        )
        .expect("single population input");
        let Value::Tensor(population_out) = std_builtin(
            Value::Tensor(population),
            vec![Value::Num(1.0), Value::from("all")],
        )
        .expect("single population std") else {
            panic!("single population result must remain a tensor")
        };
        assert_eq!(
            population_out.numeric_dtype(),
            runmat_value::NumericDType::F32
        );
        assert_eq!(population_out.shape(), [1, 1]);
        assert!((values(&population_out)[0] - (1.25_f64).sqrt()).abs() < 1e-6);

        assert_eq!(
            std_builtin(Value::Bool(true), Vec::new()).unwrap(),
            Value::Num(0.0)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn host_single_omitnan_rounds_once_at_output() {
        let input = Tensor::new_with_dtype(
            vec![f64::NAN, 2.0, 4.0],
            vec![3, 1],
            runmat_value::NumericDType::F32,
        )
        .expect("single NaN input");
        let Value::Tensor(output) = std_builtin(Value::Tensor(input), vec![Value::from("omitnan")])
            .expect("single omitnan std")
        else {
            panic!("single result must remain a tensor")
        };
        assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F32);
        assert_eq!(output.shape(), [1, 1]);
        assert!((values(&output)[0] - (2.0_f64).sqrt()).abs() < 1e-6);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn std_population_columns() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let result =
            std_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(1))]).expect("std");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [1, 3]);
                for value in values(&out) {
                    let diff = (value - 0.5).abs();
                    assert!(diff < 1e-12, "value={value}, diff={diff}");
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn std_all_dimensions() {
        let tensor = Tensor::new((1..=12).map(|v| v as f64).collect(), vec![3, 4]).unwrap();
        let result = std_builtin(Value::Tensor(tensor), vec![Value::from("all")]).expect("std");
        match result {
            Value::Num(v) => {
                let diff = (v - 3.60555127546).abs();
                assert!(diff < 1e-10, "value={v}, diff={diff}");
            }
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn std_repeated_axis_vector_reduces_once() {
        let input = Tensor::new((1..=12).map(f64::from).collect(), vec![3, 4]).unwrap();
        let axes = Tensor::new(vec![2.0, 2.0], vec![1, 2]).unwrap();
        let result = std_builtin(
            Value::Tensor(input),
            vec![Value::Num(0.0), Value::Tensor(axes)],
        )
        .expect("repeated axis vector");
        let Value::Tensor(output) = result else {
            panic!("expected column tensor");
        };
        assert_eq!(output.shape(), [3, 1]);
        for value in values(&output) {
            assert!((value - 15.0_f64.sqrt()).abs() < 1e-12);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn std_vecdim_multiple_axes() {
        let tensor = Tensor::new(
            (1..=24).map(|v| v as f64).collect::<Vec<_>>(),
            vec![3, 4, 2],
        )
        .unwrap();
        let dims = Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap();
        let result = std_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect("std");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [1, 4, 1]);
                for value in values(&out) {
                    let diff = (value - 6.63324958071).abs();
                    assert!(diff < 1e-8, "value={value}, diff={diff}");
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn std_with_omit_nan_dimension_two() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 3.0, 2.0, 4.0, f64::NAN], vec![3, 2]).unwrap();
        let result = std_builtin(
            Value::Tensor(tensor),
            vec![
                Value::Int(IntValue::I32(0)),
                Value::Int(IntValue::I32(2)),
                Value::from("omitnan"),
            ],
        )
        .expect("std");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [3, 1]);
                let expected = [FRAC_1_SQRT_2, 0.0, 0.0];
                for (value, exp) in values(&out).iter().zip(expected.iter()) {
                    let diff = (value - exp).abs();
                    assert!(diff < 1e-9, "value={value}, expected={exp}, diff={diff}");
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn std_with_omit_nan_all_nan_slice() {
        let tensor = Tensor::new(vec![f64::NAN, f64::NAN], vec![2, 1]).unwrap();
        let result = std_builtin(Value::Tensor(tensor), vec![Value::from("omitnan")]).expect("std");
        match result {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected NaN scalar, got {other:?}"),
        }
    }

    #[test]
    fn std_rejects_every_integer_data_class_for_all_argument_forms() {
        let forms = [
            Vec::new(),
            vec![Value::Int(IntValue::I32(0)), Value::Int(IntValue::I32(2))],
            vec![Value::Int(IntValue::I32(0)), Value::from("all")],
            vec![Value::Int(IntValue::I32(0)), Value::from("omitmissing")],
        ];
        for input in integer_inputs() {
            for args in &forms {
                let err = std_builtin(input.clone(), args.clone()).expect_err("integer data");
                assert_eq!(err.identifier(), STD_ERROR_INVALID_INPUT.identifier);
                assert!(err.message().contains("integer data inputs"));
            }
        }
        let err =
            std_builtin(Value::Int(IntValue::I16(42)), Vec::new()).expect_err("integer scalar");
        assert_eq!(err.identifier(), STD_ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn std_accepts_typed_integer_population_control_exactly() {
        let tensor = Tensor::new(vec![0.0, 2.0, 4.0], vec![1, 3]).expect("tensor");
        let normalization =
            Tensor::new_integer(IntegerStorage::U16(vec![1]), vec![1, 1]).expect("normalization");

        let result =
            std_builtin(Value::Tensor(tensor), vec![Value::Tensor(normalization)]).expect("std");

        match result {
            Value::Num(value) => assert!((value - (8.0_f64 / 3.0).sqrt()).abs() < 1e-12),
            other => panic!("expected numeric result, got {other:?}"),
        }
    }

    #[test]
    fn std_rejects_undocumented_output_template_options() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        for args in [
            vec![Value::from("native")],
            vec![Value::from("double")],
            vec![Value::from("like"), Value::Num(0.0)],
        ] {
            let err =
                std_builtin(Value::Tensor(tensor.clone()), args).expect_err("unsupported option");
            assert_eq!(err.identifier(), STD_ERROR_INVALID_ARGUMENT.identifier);
        }
    }

    #[test]
    fn std_missing_aliases_match_nan_aliases() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 3.0], vec![3, 1]).unwrap();
        let omit_nan =
            std_builtin(Value::Tensor(tensor.clone()), vec![Value::from("omitnan")]).unwrap();
        let omit_missing = std_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::from("omitmissing")],
        )
        .unwrap();
        assert_eq!(omit_nan, omit_missing);
        let include_nan = std_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::from("includenan")],
        )
        .unwrap();
        let include_missing =
            std_builtin(Value::Tensor(tensor), vec![Value::from("includemissing")]).unwrap();
        assert!(matches!(include_nan, Value::Num(value) if value.is_nan()));
        assert!(matches!(include_missing, Value::Num(value) if value.is_nan()));
    }

    #[test]
    fn std_typed_text_controls_preserve_axis_and_whitespace_admission() {
        let parsed = block_on(parse_arguments(
            crate::call::FunctionArgumentView::from_slice(&[Value::from(" all ")]),
        ))
        .expect("std has historically admitted spaces around all");
        assert!(matches!(parsed.axes, StdAxes::All));

        let conflict = block_on(parse_arguments(
            crate::call::FunctionArgumentView::from_slice(&[Value::Num(2.0), Value::from("all")]),
        ))
        .err()
        .expect("all after an axis must conflict");
        assert_eq!(conflict.identifier(), STD_ERROR_INVALID_ARGUMENT.identifier);
        assert!(conflict.message().contains("'all' cannot be combined"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn std_dimension_greater_than_ndims_returns_same_shaped_zeros() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let result = std_builtin(
            Value::Tensor(tensor),
            vec![Value::Int(IntValue::I32(0)), Value::Int(IntValue::I32(5))],
        )
        .expect("std");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [3, 1]);
                assert_eq!(values(&out), vec![0.0, 0.0, 0.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn std_invalid_weight_vector_reports_error() {
        let weights = Tensor::new(vec![0.2, 0.8], vec![1, 2]).unwrap();
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = std_builtin(Value::Tensor(tensor), vec![Value::Tensor(weights)]).unwrap_err();
        assert_eq!(err.identifier(), STD_ERROR_INVALID_ARGUMENT.identifier);
        assert!(
            err.message()
                .contains("std: dimension entries must be integers")
                || err
                    .message()
                    .contains("std: dimension vector must not be empty"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn std_complex_identifier_is_stable() {
        let err = std_builtin(Value::Complex(1.0, 2.0), Vec::new()).unwrap_err();
        assert_eq!(err.identifier(), STD_ERROR_COMPLEX_UNSUPPORTED.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn std_invalid_input_identifier_is_stable() {
        let err = std_builtin(Value::String("abc".to_string().into()), Vec::new()).unwrap_err();
        assert_eq!(err.identifier(), STD_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn std_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let result = std_builtin(Value::GpuTensor(handle.clone()), Vec::new()).expect("std");
            let gathered = test_support::gather(result).expect("gather");
            let host = std_builtin(Value::Tensor(tensor), Vec::new()).expect("std");
            match (host, gathered) {
                (Value::Num(expected), actual) => {
                    assert_eq!(actual.shape(), [1, 1]);
                    assert!((values(&actual)[0] - expected).abs() < 1e-10);
                }
                _ => panic!("unexpected shapes"),
            }
            handle.release().unwrap();
        });
    }

    #[test]
    fn std_gpu_rejects_integer_data_before_provider_dispatch() {
        test_support::with_test_provider(|provider| {
            let handle = provider
                .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                    data: runmat_accelerate_api::HostIntegerDataView::U64(&[0, 2, u64::MAX]),
                    shape: &[3, 1],
                })
                .expect("upload integer");
            let err = std_builtin(
                Value::GpuTensor(handle.clone()),
                vec![Value::Int(IntValue::I32(0)), Value::from("all")],
            )
            .expect_err("integer GPU data");
            assert_eq!(err.identifier(), STD_ERROR_INVALID_INPUT.identifier);
            assert!(err.message().contains("integer data inputs"));
            handle.release().unwrap();
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn std_gpu_all_population_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor =
                Tensor::new((1..=12).map(|v| v as f64).collect::<Vec<_>>(), vec![3, 4]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let result = std_builtin(
                Value::GpuTensor(handle.clone()),
                vec![Value::Int(IntValue::I32(1)), Value::from("all")],
            )
            .expect("std");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape(), [1, 1]);
            let value = values(&gathered)[0];
            let diff = (value - 3.45205252953).abs();
            assert!(diff < 1e-6, "value={value}, diff={diff}");
            handle.release().unwrap();
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn std_gpu_omit_nan_falls_back() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, f64::NAN, 3.0], vec![3, 1]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let result = std_builtin(
                Value::GpuTensor(handle.clone()),
                vec![Value::Int(IntValue::I32(0)), Value::from("omitnan")],
            )
            .expect("std");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape(), [1, 1]);
            assert!((values(&gathered)[0] - SQRT_2).abs() < 1e-8);
            handle.release().unwrap();
        });
    }
}
