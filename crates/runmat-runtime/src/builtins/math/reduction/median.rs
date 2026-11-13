//! MATLAB-compatible `median` builtin with GPU-aware semantics for RunMat.

use std::cmp::Ordering;

use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};

#[path = "median/resident.rs"]
mod resident;
use runmat_builtins::{
    BuiltinErrorDescriptor, MedianControl, MEDIAN_ERROR_INTERNAL, MEDIAN_ERROR_INVALID_ARGUMENT,
    MEDIAN_ERROR_INVALID_INPUT,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    IntValue, IntegerStorage, LogicalArray, NumericDType, NumericScalar, NumericStorage, Tensor,
    Value,
};

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "median";

use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::reduction::median")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "median",
    op_kind: GpuOpKind::Reduction,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Reduction {
            name: "reduce_median_dim",
        },
        ProviderHook::Reduction {
            name: "reduce_median",
        },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may execute medians entirely on device; runtimes fall back to host when hooks are missing or omitnan is requested.",
};

fn median_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn median_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    median_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn median_invalid_argument(detail: impl AsRef<str>) -> RuntimeError {
    median_error_with_detail(&MEDIAN_ERROR_INVALID_ARGUMENT, detail)
}

fn median_invalid_input(detail: impl AsRef<str>) -> RuntimeError {
    median_error_with_detail(&MEDIAN_ERROR_INVALID_INPUT, detail)
}

fn median_internal_error(detail: impl AsRef<str>) -> RuntimeError {
    median_error_with_detail(&MEDIAN_ERROR_INTERNAL, detail)
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::reduction::median")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "median",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes:
        "Fusion planner gathers to the host; future kernels may expose order-statistic reductions.",
};

#[derive(Clone)]
enum MedianAxes {
    Default,
    Dim(usize),
    Vec(Vec<usize>),
    All,
}

struct ParsedArguments {
    axes: MedianAxes,
    nan_mode: ReductionNaN,
    weights: Option<MedianWeights>,
}

struct MedianWeights {
    values: Vec<f64>,
    shape: Vec<usize>,
}

#[runtime_builtin(
    name = "median",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::reduction::median"
)]
pub(crate) async fn median_builtin(
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    median_with_nan_policy(value, rest, None).await
}

pub(crate) async fn median_with_nan_policy(
    value: Value,
    rest: crate::call::BuiltinArguments,
    imposed_nan_mode: Option<ReductionNaN>,
) -> crate::BuiltinResult<Value> {
    let protected = rest
        .iter()
        .filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle),
            _ => None,
        })
        .collect::<Vec<_>>();
    let _residency = protected
        .iter()
        .map(|handle| gpu_helpers::ResidencyRestoreGuard::new(handle))
        .collect::<Vec<_>>();
    let mut admission =
        resident::InputAdmission::new(std::iter::once(&value).chain(rest.iter()).filter_map(
            |value| match value {
                Value::GpuTensor(handle) => Some(handle),
                _ => None,
            },
        ));
    let mut parsed = parse_arguments(rest.view(), &mut admission).await?;
    if let Some(mode) = imposed_nan_mode {
        parsed.nan_mode = mode;
    }
    admission.before_effect()?;
    drop(admission);
    match value {
        Value::GpuTensor(handle) => median_gpu(handle, &parsed, &protected).await,
        other => median_host(other, &parsed),
    }
}

async fn parse_arguments(
    args: crate::call::FunctionArgumentView<'_>,
    admission: &mut resident::InputAdmission<'_>,
) -> BuiltinResult<ParsedArguments> {
    let mut axes = MedianAxes::Default;
    let mut axes_set = false;
    let mut nan_mode = ReductionNaN::Include;
    let mut weights = None;
    let mut idx = 0;
    while idx < args.len() {
        let arg = &args[idx];

        if let Some(keyword) = keyword_of(arg)? {
            match MedianControl::from_public_option(&keyword) {
                Some(MedianControl::Weights) => {
                    if weights.is_some() {
                        return Err(median_invalid_argument(
                            "median: Weights may be specified only once",
                        ));
                    }
                    let value = args.get(idx + 1).ok_or_else(|| {
                        median_invalid_argument("median: Weights requires a value")
                    })?;
                    weights = Some(parse_weights(value, admission).await?);
                    idx += 2;
                    continue;
                }
                Some(MedianControl::Missing(mode)) => {
                    nan_mode = mode;
                    idx += 1;
                    continue;
                }
                Some(MedianControl::All) => {
                    if axes_set && !matches!(axes, MedianAxes::Default) {
                        return Err(median_invalid_argument(
                            "median: 'all' cannot be combined with an explicit dimension",
                        ));
                    }
                    axes = MedianAxes::All;
                    axes_set = true;
                    idx += 1;
                    continue;
                }
                None if keyword.is_empty() => {
                    return Err(median_invalid_argument(
                        "median: keyword arguments must not be empty strings",
                    ));
                }
                None => {
                    if let Some(original) = value_as_str(arg)? {
                        return Err(median_invalid_argument(format!(
                            "median: unrecognised argument '{original}'"
                        )));
                    } else {
                        return Err(median_invalid_argument(format!(
                            "median: unrecognised argument {arg:?}"
                        )));
                    }
                }
            }
        }

        if !axes_set || matches!(axes, MedianAxes::Default) {
            if let Some(selection) = parse_axes(arg, admission).await? {
                if matches!(selection, MedianAxes::All) {
                    if axes_set && !matches!(axes, MedianAxes::Default) {
                        return Err(median_invalid_argument(
                            "median: 'all' cannot be combined with an explicit dimension",
                        ));
                    }
                    axes = MedianAxes::All;
                } else {
                    axes = selection;
                }
                axes_set = true;
                idx += 1;
                continue;
            }
        } else if parse_axes(arg, admission).await?.is_some() {
            return Err(median_invalid_argument(
                "median: multiple dimension specifications provided",
            ));
        }

        return Err(median_invalid_argument(format!(
            "median: unrecognised argument {arg:?}"
        )));
    }

    if weights.is_some() && matches!(axes, MedianAxes::Vec(_) | MedianAxes::All) {
        return Err(median_invalid_argument(
            "median: Weights cannot be combined with vecdim or 'all'",
        ));
    }

    Ok(ParsedArguments {
        axes,
        nan_mode,
        weights,
    })
}

async fn parse_weights(
    value: &Value,
    admission: &mut resident::InputAdmission<'_>,
) -> BuiltinResult<MedianWeights> {
    let tensor = match value {
        Value::Num(value) => Tensor::new(vec![*value], vec![1, 1])
            .map_err(|error| median_internal_error(format!("median: {error}")))?,
        Value::Tensor(tensor) => tensor.clone(),
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_integer_type(handle).is_some()
                || runmat_accelerate_api::handle_is_logical(handle)
            {
                return Err(median_invalid_argument(
                    "median: Weights must be single or double",
                ));
            }
            admission.before_effect()?;
            gpu_helpers::gather_tensor_async(handle).await?
        }
        _ => {
            return Err(median_invalid_argument(
                "median: Weights must be a single- or double-precision numeric array",
            ));
        }
    };

    if !matches!(
        tensor.numeric_dtype(),
        NumericDType::F32 | NumericDType::F64
    ) {
        return Err(median_invalid_argument(
            "median: Weights must be single or double",
        ));
    }

    let shape = tensor.shape().to_vec();
    let values = tensor.materialize_f64();
    for (index, &weight) in values.iter().enumerate() {
        if weight.is_nan() || weight < 0.0 {
            return Err(median_invalid_argument(format!(
                "median: Weights must contain nonnegative values (index {})",
                index + 1
            )));
        }
    }
    Ok(MedianWeights { values, shape })
}

fn median_host(value: Value, args: &ParsedArguments) -> BuiltinResult<Value> {
    match value {
        Value::LogicalArray(logical) => median_logical_host(logical, args),
        Value::Bool(value) => median_logical_host(
            LogicalArray::new(vec![u8::from(value)], vec![1, 1]).map_err(median_internal_error)?,
            args,
        ),
        other => {
            let tensor =
                tensor::value_into_tensor_for("median", other).map_err(median_invalid_input)?;
            let reduced = median_tensor(
                tensor,
                args.axes.clone(),
                args.nan_mode,
                args.weights.as_ref(),
            )?;
            Ok(tensor::tensor_into_value(reduced))
        }
    }
}

fn median_logical_host(logical: LogicalArray, args: &ParsedArguments) -> BuiltinResult<Value> {
    let tensor = Tensor::new_integer(
        IntegerStorage::U8(logical.data.into_vec()),
        logical.shape.clone(),
    )
    .map_err(|error| median_internal_error(format!("median: {error}")))?;
    let reduced = median_tensor(
        tensor,
        args.axes.clone(),
        args.nan_mode,
        args.weights.as_ref(),
    )?;
    logical_value_from_reduction(reduced)
}

fn logical_value_from_reduction(tensor: Tensor) -> BuiltinResult<Value> {
    let logical = logical_array_from_reduction(tensor)?;
    if logical.data.len() == 1 {
        Ok(Value::Bool(logical.data[0] != 0))
    } else {
        Ok(Value::LogicalArray(logical))
    }
}

// Consume the native payload while retaining the complete reduction shape.
// Resident restoration must not apply the host wrapper's scalar collapse.
fn logical_array_from_reduction(tensor: Tensor) -> BuiltinResult<LogicalArray> {
    let shape = tensor.shape().to_vec();
    let storage = tensor
        .into_numeric_storage()
        .map_err(median_internal_error)?;
    let NumericStorage::U8(values) = storage else {
        return Err(median_internal_error(
            "median: logical reduction did not preserve its logical payload",
        ));
    };
    LogicalArray::new(values, shape).map_err(median_internal_error)
}

fn logical_tensor_from_gathered(tensor: Tensor) -> BuiltinResult<Tensor> {
    let shape = tensor.shape().to_vec();
    let values = tensor
        .materialize_f64()
        .into_iter()
        .map(|value| u8::from(value != 0.0))
        .collect();
    Tensor::new_integer(IntegerStorage::U8(values), shape)
        .map_err(|error| median_internal_error(format!("median: {error}")))
}

async fn median_gpu(
    handle: GpuTensorHandle,
    args: &ParsedArguments,
    protected: &[&GpuTensorHandle],
) -> BuiltinResult<Value> {
    let _residency = gpu_helpers::ResidencyRestoreGuard::new(&handle);
    let provider = gpu_helpers::exact_provider_for_handle(&handle)
        .ok_or_else(|| resident::terminal("median: resident input has no exact owning provider"))?;
    handle.allocation_lease().validate_live().map_err(|error| {
        resident::provider_error(runmat_accelerate_api::ProviderExecutionError::from_source(
            error,
        ))
    })?;
    let is_logical = runmat_accelerate_api::handle_is_logical(&handle);
    let is_integer = runmat_accelerate_api::handle_integer_type(&handle).is_some();
    if !is_integer
        && !is_logical
        && args.weights.is_none()
        && args.nan_mode == ReductionNaN::Include
    {
        if let Some(device_result) =
            resident::try_reduce(provider, &handle, &args.axes, protected).await?
        {
            return gpu_helpers::resident_gpu_value(device_result).map_err(|error| {
                resident::provider_error(
                    runmat_accelerate_api::ProviderExecutionError::from_source(error),
                )
            });
        }
    }

    let gathered = gpu_helpers::gather_tensor_async(&handle).await?;
    let gathered = if is_logical {
        logical_tensor_from_gathered(gathered)?
    } else {
        gathered
    };
    let reduced = median_tensor(
        gathered,
        args.axes.clone(),
        args.nan_mode,
        args.weights.as_ref(),
    )?;
    let result = if is_logical {
        Value::LogicalArray(logical_array_from_reduction(reduced)?)
    } else {
        Value::Tensor(reduced)
    };
    gpu_helpers::restore_class_preserving_value_protecting(&handle, result, "median", protected)
}

fn median_tensor(
    tensor: Tensor,
    axes: MedianAxes,
    nan_mode: ReductionNaN,
    weights: Option<&MedianWeights>,
) -> BuiltinResult<Tensor> {
    if let Some(weights) = weights {
        return match axes {
            MedianAxes::Default => {
                let dim = default_dimension(&tensor);
                reduce_tensor_weighted_median_dim(tensor, dim, nan_mode, weights)
            }
            MedianAxes::Dim(dim) => {
                reduce_tensor_weighted_median_dim(tensor, dim, nan_mode, weights)
            }
            MedianAxes::Vec(_) | MedianAxes::All => Err(median_invalid_argument(
                "median: Weights cannot be combined with vecdim or 'all'",
            )),
        };
    }

    match axes {
        MedianAxes::Default => {
            let dim = default_dimension(&tensor);
            reduce_tensor_median_dim(tensor, dim, nan_mode)
        }
        MedianAxes::Dim(dim) => reduce_tensor_median_dim(tensor, dim, nan_mode),
        MedianAxes::Vec(mut dims) => {
            let mut current = tensor;
            dims.sort_unstable();
            dims.dedup();
            if dims.is_empty() {
                let dim = default_dimension(&current);
                current = reduce_tensor_median_dim(current, dim, nan_mode)?;
                return Ok(current);
            }
            for dim in dims {
                current = reduce_tensor_median_dim(current, dim, nan_mode)?;
            }
            Ok(current)
        }
        MedianAxes::All => {
            if tensor.shape().is_empty() {
                Ok(tensor)
            } else {
                let len = tensor.len();
                let flat = tensor
                    .reshape(vec![len, 1])
                    .map_err(median_internal_error)?;
                reduce_tensor_median_dim(flat, 1, nan_mode)
            }
        }
    }
}

async fn parse_axes(
    value: &Value,
    admission: &mut resident::InputAdmission<'_>,
) -> BuiltinResult<Option<MedianAxes>> {
    if let Some(text) = value_as_str(value)? {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Err(median_invalid_argument(
                "median: dimension string must not be empty",
            ));
        }
        return match MedianControl::from_public_option(&text) {
            Some(MedianControl::All) => Ok(Some(MedianAxes::All)),
            Some(MedianControl::Missing(_)) => Ok(None),
            Some(MedianControl::Weights) | None => Err(median_invalid_argument(format!(
                "median: unrecognised argument '{trimmed}'"
            ))),
        };
    }

    let (scalar_hint, is_empty) = match value {
        Value::Num(_) | Value::Int(_) => (true, false),
        Value::Tensor(t) => (tensor::is_scalar_tensor(t), tensor_len(t) == 0),
        Value::LogicalArray(logical) => (logical.data.len() == 1, logical.data.is_empty()),
        Value::GpuTensor(handle) => {
            let count = tensor::element_count(handle.shape());
            (handle.shape().is_empty() || count == 1, count == 0)
        }
        _ => (false, false),
    };
    if is_empty {
        return Ok(Some(MedianAxes::Default));
    }

    if matches!(value, Value::GpuTensor(_)) {
        admission.before_effect()?;
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
            return Err(median_invalid_argument("median: dimension must be numeric"));
        }
        _ => return Ok(None),
    };

    let Some(dims) = dims else {
        return Ok(None);
    };
    if dims.is_empty() {
        return Ok(Some(MedianAxes::Default));
    }
    if dims.len() == 1 {
        let dim = dims[0];
        if dim < 1 {
            return Err(median_invalid_argument("median: dimension must be >= 1"));
        }
        return Ok(Some(MedianAxes::Dim(dim)));
    }
    for &dim in &dims {
        if dim < 1 {
            return Err(median_invalid_argument(
                "median: dimension entries must be >= 1",
            ));
        }
    }
    Ok(Some(MedianAxes::Vec(dims)))
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
        ) => "median: dimension must be >= 1".to_string(),
        (
            DimensionDecodeError::Negative
            | DimensionDecodeError::IntegerNegative
            | DimensionDecodeError::IntegerOutOfPlatformRange
            | DimensionDecodeError::BelowMinimum,
            false,
        ) => "median: dimension entries must be >= 1".to_string(),
        (DimensionDecodeError::NonFinite, true) => "median: dimension must be finite".to_string(),
        (DimensionDecodeError::NonFinite, false) => {
            "median: dimension entries must be finite integers".to_string()
        }
        (DimensionDecodeError::Fractional, true) => {
            "median: dimension must be an integer".to_string()
        }
        (DimensionDecodeError::Fractional, false) => {
            "median: dimension entries must be integers".to_string()
        }
        (DimensionDecodeError::OutOfPlatformRange | DimensionDecodeError::Gather(_), _) => {
            error.to_string()
        }
    };
    error.attach_cause(median_invalid_argument(message))
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
                let mut error = median_invalid_argument(source.to_string());
                error.source = Some(Box::new(source));
                error
            })?,
        ),
        _ => None,
    })
}

#[derive(Clone, Copy)]
enum WeightLayout {
    Vector,
    Full,
}

fn reduce_tensor_weighted_median_dim(
    tensor: Tensor,
    dim: usize,
    nan_mode: ReductionNaN,
    weights: &MedianWeights,
) -> BuiltinResult<Tensor> {
    if dim == 0 {
        return Err(median_invalid_argument("median: dimension must be >= 1"));
    }

    let input_shape = tensor.shape().to_vec();
    let reduce_len = if input_shape.is_empty() {
        1
    } else if dim <= input_shape.len() {
        input_shape[dim - 1]
    } else {
        1
    };
    let weight_layout = validate_weight_shape(weights, &input_shape, reduce_len)?;

    if input_shape.is_empty() {
        return tensor
            .reshape(vec![1, 1])
            .map_err(|error| median_internal_error(format!("median: {error}")));
    }
    if dim > input_shape.len() {
        return Ok(tensor);
    }

    let output_shape = reduction_shape(&input_shape, dim).expect("in-range reduction dimension");
    if reduce_len == 0 || tensor.is_empty() {
        return reduce_tensor_median_dim(tensor, dim, nan_mode);
    }

    let dim_index = dim - 1;
    let stride_before = dim_product(&input_shape[..dim_index]);
    let stride_after = dim_product(&input_shape[dim..]);
    let mut selected = Vec::with_capacity(tensor::element_count(&output_shape));

    for after in 0..stride_after {
        for before in 0..stride_before {
            let mut candidates = Vec::with_capacity(reduce_len);
            let mut first_missing = None;

            for k in 0..reduce_len {
                let index = before + k * stride_before + after * stride_before * reduce_len;
                let value = tensor
                    .numeric_value_at(index)
                    .expect("weighted median index is in bounds");
                if numeric_scalar_is_nan(value) {
                    first_missing.get_or_insert(index);
                    if nan_mode == ReductionNaN::Include {
                        candidates.clear();
                        break;
                    }
                    continue;
                }
                let weight = match weight_layout {
                    WeightLayout::Vector => weights.values[k],
                    WeightLayout::Full => weights.values[index],
                };
                candidates.push((index, weight));
            }

            if nan_mode == ReductionNaN::Include {
                if let Some(index) = first_missing {
                    selected.push(index);
                    continue;
                }
            }
            if candidates.is_empty() {
                if let Some(index) = first_missing {
                    selected.push(index);
                    continue;
                }
                return Err(median_invalid_argument(
                    "median: each weighted slice must contain a positive total weight",
                ));
            }

            candidates.sort_by(|(left, _), (right, _)| {
                compare_same_class_numeric_scalars(
                    tensor
                        .numeric_value_at(*left)
                        .expect("weighted median left index is in bounds"),
                    tensor
                        .numeric_value_at(*right)
                        .expect("weighted median right index is in bounds"),
                )
            });

            let scale = candidates
                .iter()
                .map(|(_, weight)| *weight)
                .fold(0.0_f64, f64::max);
            if scale == 0.0 {
                return Err(median_invalid_argument(
                    "median: each weighted slice must contain a positive total weight",
                ));
            }
            let normalized_weight = |weight: f64| {
                if scale.is_infinite() {
                    if weight.is_infinite() {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    weight / scale
                }
            };
            let total: f64 = candidates
                .iter()
                .map(|(_, weight)| normalized_weight(*weight))
                .sum();
            let threshold = total / 2.0;
            let mut cumulative = 0.0;
            let mut chosen = candidates
                .last()
                .map(|(index, _)| *index)
                .expect("nonempty weighted median candidates");
            for (index, weight) in candidates {
                cumulative += normalized_weight(weight);
                if cumulative >= threshold {
                    chosen = index;
                    break;
                }
            }
            selected.push(chosen);
        }
    }

    let storage = tensor
        .into_numeric_storage()
        .map_err(median_internal_error)?
        .gather(&selected)
        .map_err(median_internal_error)?;
    Tensor::from_numeric_storage(storage, output_shape)
        .map_err(|error| median_internal_error(format!("median: {error}")))
}

fn validate_weight_shape(
    weights: &MedianWeights,
    input_shape: &[usize],
    reduce_len: usize,
) -> BuiltinResult<WeightLayout> {
    if is_vector_shape(&weights.shape) {
        if weights.values.len() != reduce_len {
            return Err(median_invalid_argument(format!(
                "median: vector Weights length {} must match operating dimension length {reduce_len}",
                weights.values.len()
            )));
        }
        return Ok(WeightLayout::Vector);
    }
    if !matlab_shape_equal(&weights.shape, input_shape) {
        return Err(median_invalid_argument(format!(
            "median: nonvector Weights shape {:?} must match input shape {:?}",
            weights.shape, input_shape
        )));
    }
    Ok(WeightLayout::Full)
}

fn is_vector_shape(shape: &[usize]) -> bool {
    match shape {
        [] | [_] => true,
        [rows, cols] => *rows == 1 || *cols == 1,
        _ => false,
    }
}

fn matlab_shape_equal(left: &[usize], right: &[usize]) -> bool {
    let rank = left.len().max(right.len());
    (0..rank).all(|index| {
        left.get(index).copied().unwrap_or(1) == right.get(index).copied().unwrap_or(1)
    })
}

fn numeric_scalar_is_nan(value: NumericScalar) -> bool {
    match value {
        NumericScalar::F64(value) => value.is_nan(),
        NumericScalar::F32(value) => value.is_nan(),
        NumericScalar::I8(_)
        | NumericScalar::I16(_)
        | NumericScalar::I32(_)
        | NumericScalar::I64(_)
        | NumericScalar::U8(_)
        | NumericScalar::U16(_)
        | NumericScalar::U32(_)
        | NumericScalar::U64(_) => false,
    }
}

fn compare_same_class_numeric_scalars(left: NumericScalar, right: NumericScalar) -> Ordering {
    match (left, right) {
        (NumericScalar::F64(left), NumericScalar::F64(right)) => {
            left.partial_cmp(&right).unwrap_or(Ordering::Equal)
        }
        (NumericScalar::F32(left), NumericScalar::F32(right)) => {
            left.partial_cmp(&right).unwrap_or(Ordering::Equal)
        }
        (NumericScalar::I8(left), NumericScalar::I8(right)) => left.cmp(&right),
        (NumericScalar::I16(left), NumericScalar::I16(right)) => left.cmp(&right),
        (NumericScalar::I32(left), NumericScalar::I32(right)) => left.cmp(&right),
        (NumericScalar::I64(left), NumericScalar::I64(right)) => left.cmp(&right),
        (NumericScalar::U8(left), NumericScalar::U8(right)) => left.cmp(&right),
        (NumericScalar::U16(left), NumericScalar::U16(right)) => left.cmp(&right),
        (NumericScalar::U32(left), NumericScalar::U32(right)) => left.cmp(&right),
        (NumericScalar::U64(left), NumericScalar::U64(right)) => left.cmp(&right),
        _ => unreachable!("numeric tensor storage is homogeneous"),
    }
}

fn reduce_tensor_median_dim(
    tensor: Tensor,
    dim: usize,
    nan_mode: ReductionNaN,
) -> BuiltinResult<Tensor> {
    if dim == 0 {
        return Err(median_invalid_argument("median: dimension must be >= 1"));
    }

    if tensor.shape().is_empty() {
        return tensor
            .reshape(vec![1, 1])
            .map_err(|e| median_internal_error(format!("median: {e}")));
    }

    if dim > tensor.shape().len() {
        return Ok(tensor);
    }

    let dim_index = dim - 1;
    let reduce_len = tensor.shape()[dim_index];
    let Some(output_shape) = reduction_shape(tensor.shape(), dim) else {
        return Ok(tensor);
    };

    if reduce_len == 0 || tensor.is_empty() {
        let fill = vec![f64::NAN; tensor::element_count(&output_shape)];
        return Tensor::new(fill, output_shape)
            .map_err(|e| median_internal_error(format!("median: {e}")));
    }

    if tensor.integer_storage().is_some() {
        let input_shape = tensor.shape().to_vec();
        let storage = tensor
            .into_numeric_storage()
            .map_err(median_internal_error)?
            .into_integer_storage()
            .expect("checked integer storage");
        return reduce_integer_tensor_median_dim(
            storage,
            input_shape,
            dim,
            output_shape,
            reduce_len,
        );
    }

    if reduce_len == 1 {
        return Ok(tensor);
    }

    let input_shape = tensor.shape().to_vec();
    let storage = tensor
        .into_numeric_storage()
        .map_err(median_internal_error)?;
    match storage {
        runmat_value::NumericStorage::F64(values) => reduce_floating_tensor_median_dim(
            values,
            input_shape,
            dim,
            output_shape,
            reduce_len,
            nan_mode,
            f64::NAN,
            f64::is_nan,
            2.0,
            runmat_value::NumericStorage::F64,
        ),
        runmat_value::NumericStorage::F32(values) => reduce_floating_tensor_median_dim(
            values,
            input_shape,
            dim,
            output_shape,
            reduce_len,
            nan_mode,
            f32::NAN,
            f32::is_nan,
            2.0,
            runmat_value::NumericStorage::F32,
        ),
        _ => unreachable!("integer storage handled before floating median"),
    }
}

#[allow(clippy::too_many_arguments)]
fn reduce_floating_tensor_median_dim<T>(
    values: Vec<T>,
    input_shape: Vec<usize>,
    dim: usize,
    output_shape: Vec<usize>,
    reduce_len: usize,
    nan_mode: ReductionNaN,
    nan: T,
    is_nan: fn(T) -> bool,
    two: T,
    wrap: fn(Vec<T>) -> runmat_value::NumericStorage,
) -> BuiltinResult<Tensor>
where
    T: Copy + PartialOrd + std::ops::Add<Output = T> + std::ops::Div<Output = T>,
{
    let dim_index = dim - 1;
    let stride_before = dim_product(&input_shape[..dim_index]);
    let stride_after = dim_product(&input_shape[dim..]);
    let mut output = vec![nan; tensor::element_count(&output_shape)];
    for after in 0..stride_after {
        for before in 0..stride_before {
            let mut slice = Vec::with_capacity(reduce_len);
            let mut saw_nan = false;

            for k in 0..reduce_len {
                let idx = before + k * stride_before + after * stride_before * reduce_len;
                let value = values[idx];
                match nan_mode {
                    ReductionNaN::Include => {
                        if is_nan(value) {
                            saw_nan = true;
                            break;
                        }
                        slice.push(value);
                    }
                    ReductionNaN::Omit => {
                        if is_nan(value) {
                            continue;
                        }
                        slice.push(value);
                    }
                }
            }

            let out_idx = after * stride_before + before;
            if saw_nan || slice.is_empty() {
                continue;
            }
            slice.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
            let middle = slice.len() / 2;
            output[out_idx] = if slice.len() % 2 == 1 {
                slice[middle]
            } else {
                (slice[middle - 1] + slice[middle]) / two
            };
        }
    }
    Tensor::from_numeric_storage(wrap(output), output_shape)
        .map_err(|e| median_internal_error(format!("median: {e}")))
}

fn reduce_integer_tensor_median_dim(
    storage: IntegerStorage,
    input_shape: Vec<usize>,
    dim: usize,
    output_shape: Vec<usize>,
    reduce_len: usize,
) -> BuiltinResult<Tensor> {
    if reduce_len == 1 {
        return Tensor::new_integer(storage, input_shape)
            .map_err(|e| median_internal_error(format!("median: {e}")));
    }

    let dim_index = dim - 1;
    let stride_before = dim_product(&input_shape[..dim_index]);
    let stride_after = dim_product(&input_shape[dim..]);
    let exact = storage.exact_values();
    let mut output = Vec::with_capacity(tensor::element_count(&output_shape));

    for after in 0..stride_after {
        for before in 0..stride_before {
            let mut slice = Vec::with_capacity(reduce_len);
            for k in 0..reduce_len {
                let index = before + k * stride_before + after * stride_before * reduce_len;
                slice.push(exact[index].clone());
            }
            slice.sort_by(compare_same_class_integer);
            output.push(integer_median_from_sorted(&slice));
        }
    }

    Tensor::new_integer(
        storage
            .from_same_class_values(output)
            .map_err(median_internal_error)?,
        output_shape,
    )
    .map_err(|e| median_internal_error(format!("median: {e}")))
}

pub(super) fn compare_same_class_integer(left: &IntValue, right: &IntValue) -> Ordering {
    match (left, right) {
        (IntValue::I8(a), IntValue::I8(b)) => a.cmp(b),
        (IntValue::I16(a), IntValue::I16(b)) => a.cmp(b),
        (IntValue::I32(a), IntValue::I32(b)) => a.cmp(b),
        (IntValue::I64(a), IntValue::I64(b)) => a.cmp(b),
        (IntValue::U8(a), IntValue::U8(b)) => a.cmp(b),
        (IntValue::U16(a), IntValue::U16(b)) => a.cmp(b),
        (IntValue::U32(a), IntValue::U32(b)) => a.cmp(b),
        (IntValue::U64(a), IntValue::U64(b)) => a.cmp(b),
        _ => unreachable!("integer storage supplies one homogeneous class"),
    }
}

pub(super) fn integer_median_from_sorted(values: &[IntValue]) -> IntValue {
    let middle = values.len() / 2;
    if values.len() % 2 == 1 {
        return values[middle].clone();
    }
    macro_rules! signed_median {
        ($variant:ident, $ty:ty) => {{
            let IntValue::$variant(lower) = &values[middle - 1] else {
                unreachable!()
            };
            let IntValue::$variant(upper) = &values[middle] else {
                unreachable!()
            };
            let sum = *lower as i128 + *upper as i128;
            let rounded = if sum >= 0 {
                (sum + 1) / 2
            } else {
                (sum - 1) / 2
            };
            IntValue::$variant(rounded as $ty)
        }};
    }
    macro_rules! unsigned_median {
        ($variant:ident, $ty:ty) => {{
            let IntValue::$variant(lower) = &values[middle - 1] else {
                unreachable!()
            };
            let IntValue::$variant(upper) = &values[middle] else {
                unreachable!()
            };
            IntValue::$variant((*lower as u128 + *upper as u128).div_ceil(2) as $ty)
        }};
    }
    match values.first().expect("nonempty median slice") {
        IntValue::I8(_) => signed_median!(I8, i8),
        IntValue::I16(_) => signed_median!(I16, i16),
        IntValue::I32(_) => signed_median!(I32, i32),
        IntValue::I64(_) => signed_median!(I64, i64),
        IntValue::U8(_) => unsigned_median!(U8, u8),
        IntValue::U16(_) => unsigned_median!(U16, u16),
        IntValue::U32(_) => unsigned_median!(U32, u32),
        IntValue::U64(_) => unsigned_median!(U64, u64),
    }
}

pub fn compute_median_inplace(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| partial_cmp_f64(*a, *b));
    let len = values.len();
    if len % 2 == 1 {
        values[len / 2]
    } else {
        let upper = values[len / 2];
        let lower = values[len / 2 - 1];
        0.5 * (lower + upper)
    }
}

fn partial_cmp_f64(a: f64, b: f64) -> Ordering {
    a.partial_cmp(&b).unwrap_or(Ordering::Less)
}

fn reduction_shape(shape: &[usize], dim: usize) -> Option<Vec<usize>> {
    if dim == 0 {
        return None;
    }
    if shape.is_empty() {
        return Some(vec![1, 1]);
    }
    if dim > shape.len() {
        return None;
    }
    let mut out = shape.to_vec();
    out[dim - 1] = 1;
    Some(out)
}

fn dim_product(dims: &[usize]) -> usize {
    dims.iter()
        .copied()
        .fold(1usize, |acc, v| acc.saturating_mul(v))
}

fn default_dimension(tensor: &Tensor) -> usize {
    default_dimension_from_shape(tensor.shape())
}

fn default_dimension_from_shape(shape: &[usize]) -> usize {
    if shape.is_empty() {
        return 1;
    }
    shape
        .iter()
        .position(|&extent| extent != 1)
        .map(|idx| idx + 1)
        .unwrap_or(1)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_value::{IntValue, IntegerStorage, NumericStorage};

    #[test]
    fn logical_reduction_payload_preserves_resident_shape_and_host_scalar_policy() {
        for (values, shape) in [
            (vec![1], vec![1, 1, 1]),
            (vec![0, 1], vec![1, 2, 1]),
            (vec![], vec![0, 1, 1]),
        ] {
            let reduced =
                || Tensor::new_integer(IntegerStorage::U8(values.clone()), shape.clone()).unwrap();
            let logical = logical_array_from_reduction(reduced()).unwrap();
            assert_eq!(logical.shape, shape);
            assert_eq!(logical.data.as_slice(), values.as_slice());
            match logical_value_from_reduction(reduced()).unwrap() {
                Value::Bool(bit) => {
                    assert_eq!(values.len(), 1);
                    assert_eq!(bit, values[0] != 0);
                }
                Value::LogicalArray(array) => {
                    assert_ne!(values.len(), 1);
                    assert_eq!(array.shape, shape);
                    assert_eq!(array.data.as_slice(), values.as_slice());
                }
                other => panic!("unexpected logical reduction: {other:?}"),
            }
        }
        assert!(logical_array_from_reduction(Tensor::new(vec![1.0], vec![1, 1]).unwrap()).is_err());
    }

    fn median_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::median_builtin(
            value,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn median_scalar_num() {
        let result = median_builtin(Value::Num(5.0), Vec::new()).expect("median");
        assert_eq!(result, Value::Num(5.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn median_vector_odd_length() {
        let tensor = Tensor::new(vec![7.0, 2.0, 9.0, 4.0, 5.0], vec![5, 1]).unwrap();
        let result = median_builtin(Value::Tensor(tensor), Vec::new()).expect("median");
        assert_eq!(result, Value::Num(5.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn median_vector_even_length() {
        let tensor = Tensor::new(vec![1.0, 4.0, 9.0, 10.0], vec![4, 1]).unwrap();
        let result = median_builtin(Value::Tensor(tensor), Vec::new()).expect("median");
        assert_eq!(result, Value::Num(6.5));
    }

    #[test]
    fn median_preserves_native_single_storage_and_arithmetic() {
        let tensor = Tensor::from_f32(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).expect("input");
        let result = median_builtin(Value::Tensor(tensor), Vec::new()).expect("median");
        let Value::Tensor(result) = result else {
            panic!("expected native single tensor");
        };
        assert_eq!(
            result.into_numeric_storage().expect("native storage"),
            NumericStorage::F32(vec![2.0, 3.0])
        );
    }

    #[test]
    fn median_preserves_every_integer_class_and_rounds_even_pairs() {
        let cases = [
            IntegerStorage::I8(vec![-4, -3, 2, 3]),
            IntegerStorage::I16(vec![-400, -300, 200, 300]),
            IntegerStorage::I32(vec![i32::MIN, -1, 2, i32::MAX]),
            IntegerStorage::I64(vec![i64::MIN, -3, 2, i64::MAX]),
            IntegerStorage::U8(vec![0, 1, 2, u8::MAX]),
            IntegerStorage::U16(vec![0, 1, 2, u16::MAX]),
            IntegerStorage::U32(vec![0, 1, 2, u32::MAX]),
            IntegerStorage::U64(vec![0, 9_007_199_254_740_993, u64::MAX - 1, u64::MAX]),
        ];

        for storage in cases {
            let input = Tensor::new_integer(storage.clone(), vec![4, 1]).expect("typed input");
            let result = median_builtin(Value::Tensor(input), Vec::new()).expect("median");
            let expected = integer_median_from_sorted(&storage.exact_values());
            assert_eq!(result, Value::Int(expected));
        }
    }

    #[test]
    fn median_typed_integer_dimensions_and_all_retain_exact_storage() {
        let large = 9_007_199_254_740_993_u64;
        let tensor = Tensor::new_integer(
            IntegerStorage::U64(vec![large, u64::MAX, 4, 8, 1, 3]),
            vec![2, 3],
        )
        .expect("typed input");

        let Value::Tensor(by_column) =
            median_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("median by column")
        else {
            panic!("expected typed tensor result");
        };
        assert_eq!(
            by_column.integer_storage(),
            Some(&IntegerStorage::U64(vec![
                (large as u128 + u64::MAX as u128).div_ceil(2) as u64,
                6,
                2,
            ]))
        );

        let result =
            median_builtin(Value::Tensor(tensor), vec![Value::from("all")]).expect("median all");
        assert_eq!(result, Value::Int(IntValue::U64(6)));
    }

    #[test]
    fn median_reads_typed_integer_storage_without_mirror() {
        let tensor = Tensor::new_integer(IntegerStorage::I16(vec![9, 1, 4, 8, 6, 3]), vec![3, 2])
            .expect("typed input");

        let Value::Tensor(result) =
            median_builtin(Value::Tensor(tensor), Vec::new()).expect("median by column")
        else {
            panic!("expected typed tensor result");
        };
        assert_eq!(
            result.integer_storage(),
            Some(&IntegerStorage::I16(vec![4, 6]))
        );
    }

    #[test]
    fn median_typed_integer_length_one_dimension_keeps_exact_storage() {
        let large = 9_007_199_254_740_993_u64;
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![large, u64::MAX]), vec![1, 2])
            .expect("typed input");

        let Value::Tensor(result) =
            median_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(1))])
                .expect("median along singleton dimension")
        else {
            panic!("expected typed tensor result");
        };
        assert_eq!(
            result.integer_storage(),
            Some(&IntegerStorage::U64(vec![large, u64::MAX]))
        );
    }

    #[test]
    fn weighted_median_matches_documented_column_example() {
        let input = Tensor::new(
            vec![1.0, 7.0, 1.0, 1.0, 6.0, 1.0, 9.0, 9.0, 9.0, 2.0],
            vec![5, 2],
        )
        .expect("input");
        let weights = Tensor::new(vec![1.0, 2.0, 1.0, 2.0, 3.0], vec![5, 1]).expect("weights");
        let result = median_builtin(
            Value::Tensor(input),
            vec![Value::from("Weights"), Value::Tensor(weights)],
        )
        .expect("weighted median");
        let Value::Tensor(result) = result else {
            panic!("expected weighted matrix result");
        };
        assert_eq!(result.materialize_f64(), vec![6.0, 9.0]);
    }

    #[test]
    fn weighted_median_selects_threshold_value_and_preserves_native_single() {
        let input = Tensor::from_f32(vec![1.0, 9.0, 20.0], vec![3, 1]).expect("single input");
        let weights = Tensor::from_f32(vec![1.0, 1.0, 2.0], vec![3, 1]).expect("single weights");
        let result = median_builtin(
            Value::Tensor(input),
            vec![Value::from("Weights"), Value::Tensor(weights)],
        )
        .expect("weighted median");
        let Value::Tensor(result) = result else {
            panic!("expected native single scalar tensor");
        };
        assert_eq!(
            result.into_numeric_storage().expect("storage"),
            NumericStorage::F32(vec![9.0])
        );
    }

    #[test]
    fn weighted_median_preserves_exact_u64_values_above_flintmax() {
        let wide = 9_007_199_254_740_993_u64;
        let input = Tensor::new_integer(
            IntegerStorage::U64(vec![wide, wide + 2, u64::MAX]),
            vec![3, 1],
        )
        .expect("integer input");
        let weights = Tensor::new(vec![1.0, 1.0, 10.0], vec![3, 1]).expect("weights");
        let result = median_builtin(
            Value::Tensor(input),
            vec![Value::from("Weights"), Value::Tensor(weights)],
        )
        .expect("weighted median");
        assert_eq!(result, Value::Int(IntValue::U64(u64::MAX)));
    }

    #[test]
    fn weighted_median_preserves_every_integer_class() {
        let cases = [
            (IntegerStorage::I8(vec![9, -4, 2]), IntValue::I8(-4)),
            (
                IntegerStorage::I16(vec![900, -400, 200]),
                IntValue::I16(-400),
            ),
            (
                IntegerStorage::I32(vec![90_000, -40_000, 20_000]),
                IntValue::I32(-40_000),
            ),
            (
                IntegerStorage::I64(vec![i64::MAX, i64::MIN, 0]),
                IntValue::I64(i64::MIN),
            ),
            (IntegerStorage::U8(vec![9, 4, 2]), IntValue::U8(4)),
            (IntegerStorage::U16(vec![900, 400, 200]), IntValue::U16(400)),
            (
                IntegerStorage::U32(vec![90_000, 40_000, 20_000]),
                IntValue::U32(40_000),
            ),
            (
                IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993, 0]),
                IntValue::U64(9_007_199_254_740_993),
            ),
        ];
        for (storage, expected) in cases {
            let input = Tensor::new_integer(storage, vec![3, 1]).expect("integer input");
            let weights = Tensor::new(vec![1.0, 10.0, 1.0], vec![3, 1]).expect("weights");
            let result = median_builtin(
                Value::Tensor(input),
                vec![Value::from("Weights"), Value::Tensor(weights)],
            )
            .expect("weighted median");
            assert_eq!(result, Value::Int(expected));
        }
    }

    #[test]
    fn weighted_median_supports_full_size_weights_and_explicit_dimension() {
        let input = Tensor::new(vec![1.0, 4.0, 10.0, 2.0, 8.0, 20.0], vec![3, 2]).expect("input");
        let weights = Tensor::new(vec![1.0, 1.0, 8.0, 8.0, 1.0, 1.0], vec![3, 2]).expect("weights");
        let result = median_builtin(
            Value::Tensor(input),
            vec![
                Value::Int(IntValue::I32(1)),
                Value::from("Weights"),
                Value::Tensor(weights),
            ],
        )
        .expect("weighted median");
        let Value::Tensor(result) = result else {
            panic!("expected matrix result");
        };
        assert_eq!(result.shape(), [1, 2]);
        assert_eq!(result.materialize_f64(), vec![10.0, 2.0]);
    }

    #[test]
    fn weighted_median_applies_missing_policy_before_weight_threshold() {
        let input = Tensor::new(vec![1.0, f64::NAN, 10.0], vec![3, 1]).expect("input");
        let weights = Tensor::new(vec![1.0, 100.0, 1.0], vec![3, 1]).expect("weights");
        let included = median_builtin(
            Value::Tensor(input.clone()),
            vec![
                Value::from("Weights"),
                Value::Tensor(weights.clone()),
                Value::from("includemissing"),
            ],
        )
        .expect("included median");
        let Value::Num(included) = included else {
            panic!("expected double scalar");
        };
        assert!(included.is_nan());

        let omitted = median_builtin(
            Value::Tensor(input),
            vec![
                Value::from("omitmissing"),
                Value::from("Weights"),
                Value::Tensor(weights),
            ],
        )
        .expect("omitted median");
        assert_eq!(omitted, Value::Num(1.0));
    }

    #[test]
    fn weighted_median_preserves_logical_class() {
        let input = LogicalArray::new(vec![0, 1, 1, 0, 0, 1], vec![3, 2]).expect("logical");
        let weights = Tensor::new(vec![1.0, 5.0, 1.0], vec![3, 1]).expect("weights");
        let result = median_builtin(
            Value::LogicalArray(input),
            vec![Value::from("Weights"), Value::Tensor(weights)],
        )
        .expect("weighted logical median");
        assert_eq!(
            result,
            Value::LogicalArray(LogicalArray::new(vec![1, 0], vec![1, 2]).expect("logical result"))
        );
    }

    #[test]
    fn weighted_median_validates_documented_weight_contract() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).expect("input");
        let integer_weights = Tensor::new_integer(IntegerStorage::U8(vec![1, 1, 1]), vec![3, 1])
            .expect("integer weights");
        let error = median_builtin(
            Value::Tensor(input.clone()),
            vec![Value::from("Weights"), Value::Tensor(integer_weights)],
        )
        .expect_err("integer weights must fail");
        assert!(error.message().contains("single or double"));

        let negative = Tensor::new(vec![1.0, -1.0, 1.0], vec![3, 1]).expect("weights");
        let error = median_builtin(
            Value::Tensor(input.clone()),
            vec![Value::from("Weights"), Value::Tensor(negative)],
        )
        .expect_err("negative weights must fail");
        assert!(error.message().contains("nonnegative"));

        let wrong_length = Tensor::new(vec![1.0, 1.0], vec![2, 1]).expect("weights");
        let error = median_builtin(
            Value::Tensor(input.clone()),
            vec![Value::from("Weights"), Value::Tensor(wrong_length)],
        )
        .expect_err("wrong weight length must fail");
        assert!(error.message().contains("operating dimension length"));

        let vecdim = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("vecdim");
        let weights = Tensor::new(vec![1.0, 1.0, 1.0], vec![3, 1]).expect("weights");
        let error = median_builtin(
            Value::Tensor(input),
            vec![
                Value::Tensor(vecdim),
                Value::from("Weights"),
                Value::Tensor(weights),
            ],
        )
        .expect_err("vecdim with weights must fail");
        assert!(error.message().contains("vecdim"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn median_matrix_default_dimension() {
        let tensor = Tensor::new(vec![1.0, 7.0, 2.0, 9.0, 5.0, 11.0], vec![3, 2]).expect("tensor");
        let result = median_builtin(Value::Tensor(tensor), Vec::new()).expect("median");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [1, 2]);
                assert_eq!(out.materialize_f64(), vec![2.0, 9.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn median_matrix_dimension_two() {
        let tensor = Tensor::new(vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0], vec![3, 2]).expect("tensor");
        let result = median_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(2))])
            .expect("median");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [3, 1]);
                assert_eq!(out.materialize_f64(), vec![4.0, 6.0, 8.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn median_all_across_matrix() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![3, 2]).unwrap();
        let result =
            median_builtin(Value::Tensor(tensor), vec![Value::from("all")]).expect("median");
        match result {
            Value::Num(v) => assert!((v - 3.5).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn median_vecdim_multiple_axes() {
        let tensor =
            Tensor::new((1..=8).map(|v| v as f64).collect::<Vec<_>>(), vec![2, 2, 2]).unwrap();
        let dims = Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap();
        let result =
            median_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect("median");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [1, 2, 1]);
                assert_eq!(out.materialize_f64().len(), 2);
                assert!((out.materialize_f64()[0] - 3.5).abs() < 1e-12);
                assert!((out.materialize_f64()[1] - 5.5).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn median_with_omit_nan() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 5.0], vec![3, 1]).unwrap();
        let result =
            median_builtin(Value::Tensor(tensor), vec![Value::from("omitnan")]).expect("median");
        assert_eq!(result, Value::Num(3.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn median_accepts_generic_missing_aliases() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 5.0], vec![3, 1]).unwrap();
        let omitted = median_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::from("omitmissing")],
        )
        .expect("omitmissing");
        assert_eq!(omitted, Value::Num(3.0));

        let included = median_builtin(Value::Tensor(tensor), vec![Value::from("includemissing")])
            .expect("includemissing");
        let Value::Num(included) = included else {
            panic!("expected scalar median");
        };
        assert!(included.is_nan());
    }

    #[test]
    fn median_typed_text_controls_preserve_conflict_diagnostics() {
        let input = || Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap());
        let omitted = median_builtin(input(), vec![Value::from("OMITMISSING")])
            .expect("case-insensitive missing control");
        assert_eq!(omitted, Value::Num(2.0));

        let conflict = median_builtin(input(), vec![Value::Num(1.0), Value::from("all")])
            .expect_err("all after dimension must conflict");
        assert_eq!(
            conflict.identifier(),
            MEDIAN_ERROR_INVALID_ARGUMENT.identifier
        );
        assert!(conflict.message().contains("'all' cannot be combined"));

        let missing_weight = median_builtin(input(), vec![Value::from("Weights")])
            .expect_err("Weights requires a value");
        assert_eq!(
            missing_weight.identifier(),
            MEDIAN_ERROR_INVALID_ARGUMENT.identifier
        );
        assert!(missing_weight
            .message()
            .contains("Weights requires a value"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn median_with_include_nan_propagates() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 5.0], vec![3, 1]).unwrap();
        let result = median_builtin(Value::Tensor(tensor), Vec::new()).expect("median");
        match result {
            Value::Num(n) => assert!(n.is_nan()),
            other => panic!("expected scalar NaN, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn median_empty_returns_nan() {
        let tensor = Tensor::new(vec![], vec![0, 1]).unwrap();
        let result = median_builtin(Value::Tensor(tensor), Vec::new()).expect("median");
        match result {
            Value::Num(n) => assert!(n.is_nan()),
            other => panic!("expected NaN scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn median_dimension_greater_than_ndims_returns_input() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let original = tensor.clone();
        let result = median_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(5))])
            .expect("median");
        match result {
            Value::Tensor(out) => assert_eq!(out, original),
            Value::Num(n) => assert_eq!(n, original.materialize_f64()[0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn median_rejects_unknown_keyword() {
        let err = median_builtin(Value::Num(1.0), vec![Value::from("like")]).unwrap_err();
        assert_eq!(err.identifier(), MEDIAN_ERROR_INVALID_ARGUMENT.identifier);
        assert!(
            err.message().contains("unrecognised argument"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn median_invalid_input_identifier() {
        let err = median_builtin(Value::String("abc".to_string().into()), Vec::new()).unwrap_err();
        assert_eq!(err.identifier(), MEDIAN_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn median_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 4.0, 9.0, 16.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let result = median_builtin(Value::GpuTensor(handle), Vec::new()).expect("median");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape(), [1, 1]);
            assert_eq!(gathered.materialize_f64()[0], 6.5);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn median_gpu_omit_nan_host_compute_returns_resident_result() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![f64::NAN, 2.0, f64::NAN, 4.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let result = median_builtin(Value::GpuTensor(handle), vec![Value::from("omitnan")])
                .expect("median");
            assert!(matches!(result, Value::GpuTensor(_)));
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape(), [1, 1]);
            assert_eq!(gathered.materialize_f64()[0], 3.0);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn median_integer_gpu_fallback_preserves_exact_class_and_residency() {
        test_support::with_test_provider(|provider| {
            let wide = 9_007_199_254_740_993_u64;
            let tensor = Tensor::new_integer(
                IntegerStorage::U64(vec![wide, u64::MAX, wide, wide + 2]),
                vec![4, 1],
            )
            .expect("integer tensor");
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("integer upload");
            let result = median_builtin(Value::GpuTensor(handle), Vec::new()).expect("median");
            let Value::GpuTensor(result_handle) = &result else {
                panic!("expected resident integer median");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(result_handle),
                Some(runmat_accelerate_api::IntegerElementType::U64)
            );
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(
                gathered.integer_storage(),
                Some(&IntegerStorage::U64(vec![wide + 1]))
            );
        });
    }

    #[test]
    fn weighted_median_gpu_fallback_preserves_exact_class_and_residency() {
        test_support::with_test_provider(|provider| {
            let wide = 9_007_199_254_740_993_u64;
            let input = Tensor::new_integer(
                IntegerStorage::U64(vec![wide, wide + 2, u64::MAX]),
                vec![3, 1],
            )
            .expect("integer input");
            let handle = gpu_helpers::upload_tensor(provider, &input).expect("input upload");
            let weights = Tensor::from_f32(vec![1.0, 1.0, 10.0], vec![3, 1]).expect("weights");
            let result = median_builtin(
                Value::GpuTensor(handle),
                vec![Value::from("Weights"), Value::Tensor(weights)],
            )
            .expect("weighted median");
            let Value::GpuTensor(result_handle) = &result else {
                panic!("expected resident result");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(result_handle),
                Some(runmat_accelerate_api::IntegerElementType::U64)
            );
            let gathered = test_support::gather(result).expect("gather result");
            assert_eq!(
                gathered.integer_storage(),
                Some(&IntegerStorage::U64(vec![u64::MAX]))
            );
        });
    }

    #[test]
    fn weighted_median_accepts_resident_weights_and_preserves_logical_metadata() {
        test_support::with_test_provider(|provider| {
            let input_handle = provider
                .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                    data: &[0, 1, 0],
                    shape: &[3, 1],
                })
                .expect("input upload");
            let input =
                gpu_helpers::logical_gpu_value(input_handle).expect("admit logical producer");
            let Value::GpuTensor(input_handle) = input else {
                unreachable!()
            };

            let weights = Tensor::new(vec![1.0, 5.0, 1.0], vec![3, 1]).expect("weights");
            let weights_handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &weights.materialize_f64(),
                    shape: weights.shape(),
                })
                .expect("weights upload");
            let result = median_builtin(
                Value::GpuTensor(input_handle),
                vec![Value::from("Weights"), Value::GpuTensor(weights_handle)],
            )
            .expect("weighted median");
            let Value::GpuTensor(result_handle) = &result else {
                panic!("expected resident logical result");
            };
            assert!(runmat_accelerate_api::handle_is_logical(result_handle));
            let gathered = block_on(crate::dispatcher::gather_if_needed_async(&result))
                .expect("gather logical result");
            assert_eq!(
                gathered,
                Value::LogicalArray(LogicalArray::new(vec![1], vec![1, 1]).unwrap())
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn median_wgpu_dim_matches_cpu() {
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            return;
        }
        let Some(provider) = runmat_accelerate_api::provider() else {
            return;
        };
        let tensor = Tensor::new(vec![1.0, 5.0, 9.0, 2.0, 6.0, 10.0], vec![3, 2]).unwrap();
        let args_dim1 = ParsedArguments {
            axes: MedianAxes::Dim(1),
            nan_mode: ReductionNaN::Include,
            weights: None,
        };
        let cpu = median_host(Value::Tensor(tensor.clone()), &args_dim1).expect("cpu median");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_value = block_on(median_gpu(handle, &args_dim1, &[])).expect("gpu median");
        let gathered = test_support::gather(gpu_value).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(ct.shape(), gt.shape());
                let tol = match provider.precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 5e-5,
                };
                for (a, b) in ct.materialize_f64().iter().zip(gt.materialize_f64().iter()) {
                    assert!((a - b).abs() < tol, "|{} - {}| >= {}", a, b, tol);
                }
            }
            _ => panic!("unexpected shapes"),
        }

        // Global median ('all') remains consistent
        let args_all = ParsedArguments {
            axes: MedianAxes::All,
            nan_mode: ReductionNaN::Include,
            weights: None,
        };
        let cpu_all =
            median_host(Value::Tensor(tensor.clone()), &args_all).expect("cpu median all");
        let gpu_all = block_on(median_gpu(
            provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &tensor.materialize_f64(),
                    shape: tensor.shape(),
                })
                .expect("upload"),
            &args_all,
            &[],
        ))
        .expect("gpu median all");
        let gathered_all = test_support::gather(gpu_all).expect("gather");
        match cpu_all {
            Value::Num(a) => {
                assert_eq!(gathered_all.materialize_f64().len(), 1);
                assert!((a - gathered_all.materialize_f64()[0]).abs() < 1e-12);
            }
            Value::Tensor(t) => {
                assert_eq!(
                    t.materialize_f64().len(),
                    gathered_all.materialize_f64().len()
                );
                for (a, b) in t
                    .materialize_f64()
                    .iter()
                    .zip(gathered_all.materialize_f64().iter())
                {
                    assert!((a - b).abs() < 1e-12);
                }
            }
            other => panic!("unexpected CPU output for all: {other:?}"),
        }
    }
}
