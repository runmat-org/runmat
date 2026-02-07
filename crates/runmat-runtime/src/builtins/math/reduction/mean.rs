//! MATLAB-compatible `mean` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, HostTensorView, ProviderPrecision};
use runmat_builtins::{ComplexTensor, IntValue, NumericDType, Tensor, Type, Value};
const NAME: &str = "mean";

use runmat_builtins::ResolveContext;

fn mean_type(args: &[Type], ctx: &ResolveContext) -> Type {
    reduce_numeric_type(args, ctx)
}

use runmat_macros::runtime_builtin;

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use crate::builtins::common::arg_tokens::tokens_from_values;
use crate::builtins::common::random_args::{complex_tensor_into_value, keyword_of};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{
    gpu_helpers,
    shape::{canonical_scalar_shape, is_scalar_shape, normalize_scalar_shape},
    tensor,
};
use crate::builtins::math::reduction::type_resolvers::reduce_numeric_type;
use crate::dispatcher;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::reduction::mean")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "mean",
    op_kind: GpuOpKind::Reduction,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Reduction {
            name: "reduce_mean_dim",
        },
        ProviderHook::Reduction {
            name: "reduce_mean",
        },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: Some(256),
    workgroup_size: Some(256),
    accepts_nan_mode: true,
    notes: "Providers can specialise mean reductions; omitnan currently falls back to the host.",
};

fn mean_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(NAME).build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::reduction::mean")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "mean",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Fusion fallback currently gathers to host; future kernels will divide the accumulated sum by slice size.",
};

#[derive(Clone)]
struct ParsedArguments {
    axes: MeanAxes,
    nan_mode: ReductionNaN,
    output: OutputTemplate,
}

#[derive(Clone)]
enum OutputTemplate {
    Double,
    Native,
    Like(Value),
}

#[derive(Clone, Copy)]
enum DevicePreference {
    Host,
    Gpu,
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
    device: DevicePreference,
    numeric_dtype: Option<NumericDType>,
}

impl InputMeta {
    fn from_value(value: &Value) -> Self {
        let class = match value {
            Value::Complex(_, _) | Value::ComplexTensor(_) => InputClass::Complex,
            Value::LogicalArray(_) => InputClass::Logical,
            Value::Bool(_) => InputClass::Bool,
            Value::Int(i) => InputClass::Integer(IntClass::from_int_value(i)),
            _ => InputClass::Double,
        };
        let device = match value {
            Value::GpuTensor(_) => DevicePreference::Gpu,
            _ => DevicePreference::Host,
        };
        let numeric_dtype = numeric_dtype_from_value(value);
        Self {
            class,
            device,
            numeric_dtype,
        }
    }
}

fn numeric_dtype_from_value(value: &Value) -> Option<NumericDType> {
    match value {
        Value::Tensor(t) => Some(t.dtype),
        Value::GpuTensor(handle) => {
            let precision = runmat_accelerate_api::handle_precision(handle).or_else(|| {
                runmat_accelerate_api::provider_for_handle(handle)
                    .map(|provider| provider.precision())
            });
            precision.map(precision_to_dtype)
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

impl IntClass {
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
            return Err(mean_error(
                "mean: cannot represent NaN as an integer output",
            ));
        }
        let rounded = scalar.round();
        if !rounded.is_finite() {
            return Err(mean_error(
                "mean: integer output overflowed the target type",
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

#[derive(Clone, Debug)]
enum MeanAxes {
    Default,
    Dim(usize),
    Vec(Vec<usize>),
    All,
}

#[runtime_builtin(
    name = "mean",
    category = "math/reduction",
    summary = "Average elements of scalars, vectors, matrices, or N-D tensors.",
    keywords = "mean,average,reduction,gpu,omitnan",
    accel = "reduction",
    type_resolver(mean_type),
    builtin_path = "crate::builtins::math::reduction::mean"
)]
async fn mean_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    // Normalise argument order defensively:
    // If the primary 'value' is not data-like (e.g., 'all'), but a data-like
    // argument exists in 'rest', swap them so we interpret calls like
    // mean('all', X) as mean(X, 'all').
    let (value, rest) = normalise_mean_call_args(value, rest);

    let input_meta = InputMeta::from_value(&value);
    let parsed = parse_arguments(&rest).await?;
    let raw = match value {
        Value::GpuTensor(handle) => mean_gpu(handle, &parsed).await?,
        Value::Complex(re, im) => mean_host_complex_scalar(re, im, &parsed)?,
        Value::ComplexTensor(ct) => mean_host_complex_tensor(ct, &parsed)?,
        other => mean_host(other, &parsed)?,
    };
    apply_output_template(raw, &parsed.output, &input_meta).await
}

fn normalise_mean_call_args(value: Value, rest: Vec<Value>) -> (Value, Vec<Value>) {
    if is_data_like(&value) {
        return (value, rest);
    }
    if let Some(idx) = rest.iter().position(is_data_like) {
        let mut rest_mut = rest;
        let new_value = rest_mut.remove(idx);
        let mut new_rest = Vec::with_capacity(rest_mut.len() + 1);
        // Keep the original non-data 'value' (e.g., 'all') in rest so it can be parsed as a keyword
        new_rest.push(value);
        // Append the remaining rest args
        new_rest.extend(rest_mut);
        return (new_value, new_rest);
    }
    (value, rest)
}

fn is_data_like(v: &Value) -> bool {
    matches!(
        v,
        Value::Tensor(_)
            | Value::GpuTensor(_)
            | Value::Num(_)
            | Value::Int(_)
            | Value::LogicalArray(_)
            | Value::Bool(_)
            | Value::Complex(_, _)
            | Value::ComplexTensor(_)
    )
}

async fn parse_arguments(args: &[Value]) -> BuiltinResult<ParsedArguments> {
    let mut axes = MeanAxes::Default;
    let mut axes_set = false;
    let mut nan_mode = ReductionNaN::Include;
    let mut output = OutputTemplate::Double;
    let mut output_set = false;
    let tokens = tokens_from_values(args);

    let mut idx = 0;
    while idx < args.len() {
        let arg = &args[idx];
        if let Some(token) = tokens.get(idx) {
            if let crate::builtins::common::arg_tokens::ArgToken::String(text) = token {
                match text.as_str() {
                    "omitnan" => {
                        nan_mode = ReductionNaN::Omit;
                        idx += 1;
                        continue;
                    }
                    "includenan" => {
                        nan_mode = ReductionNaN::Include;
                        idx += 1;
                        continue;
                    }
                    "all" => {
                        if axes_set && !matches!(axes, MeanAxes::Default) {
                            return Err(mean_error(
                                "mean: 'all' cannot be combined with an explicit dimension",
                            ));
                        }
                        axes = MeanAxes::All;
                        axes_set = true;
                        idx += 1;
                        continue;
                    }
                    _ => {}
                }
            }
        }
        if let Some(keyword) = keyword_of(arg) {
            match keyword.as_str() {
                "omitnan" => {
                    nan_mode = ReductionNaN::Omit;
                    idx += 1;
                    continue;
                }
                "includenan" => {
                    nan_mode = ReductionNaN::Include;
                    idx += 1;
                    continue;
                }
                "all" => {
                    if axes_set && !matches!(axes, MeanAxes::Default) {
                        return Err(mean_error(
                            "mean: 'all' cannot be combined with an explicit dimension",
                        ));
                    }
                    axes = MeanAxes::All;
                    axes_set = true;
                    idx += 1;
                    continue;
                }
                "double" | "default" => {
                    if output_set {
                        return Err(mean_error(
                            "mean: multiple output class specifications provided",
                        ));
                    }
                    output = OutputTemplate::Double;
                    output_set = true;
                    idx += 1;
                    continue;
                }
                "native" => {
                    if output_set {
                        return Err(mean_error(
                            "mean: multiple output class specifications provided",
                        ));
                    }
                    output = OutputTemplate::Native;
                    output_set = true;
                    idx += 1;
                    continue;
                }
                "like" => {
                    if output_set {
                        return Err(mean_error(
                            "mean: cannot combine 'like' with another output class specifier",
                        ));
                    }
                    let Some(proto) = args.get(idx + 1).cloned() else {
                        return Err(mean_error("mean: expected prototype after 'like'"));
                    };
                    output = OutputTemplate::Like(proto);
                    idx += 2;
                    if idx < args.len() {
                        return Err(mean_error("mean: 'like' must be the final argument"));
                    }
                    break;
                }
                _ => {}
            }
        }

        if !axes_set || matches!(axes, MeanAxes::Default) {
            if let Some(selection) = parse_axes(arg).await? {
                if matches!(selection, MeanAxes::All)
                    && axes_set
                    && !matches!(axes, MeanAxes::Default)
                {
                    return Err(mean_error(
                        "mean: 'all' cannot be combined with an explicit dimension",
                    ));
                }
                axes = selection;
                axes_set = true;
                idx += 1;
                continue;
            }
        }

        if axes_set && !matches!(axes, MeanAxes::Default) {
            if let Some(selection) = parse_axes(arg).await? {
                if matches!(selection, MeanAxes::All) {
                    return Err(mean_error(
                        "mean: 'all' cannot be combined with an explicit dimension",
                    ));
                }
                return Err(mean_error(
                    "mean: multiple dimension specifications provided",
                ));
            }
        }

        return Err(mean_error(format!("mean: unrecognised argument {arg:?}")));
    }

    Ok(ParsedArguments {
        axes,
        nan_mode,
        output,
    })
}

async fn parse_axes(value: &Value) -> BuiltinResult<Option<MeanAxes>> {
    if let Some(text) = value_as_str(value) {
        let lowered = text.trim().to_ascii_lowercase();
        return match lowered.as_str() {
            "all" => Ok(Some(MeanAxes::All)),
            "omitnan" | "includenan" | "double" | "native" | "default" | "like" => Ok(None),
            "" => Err(mean_error("mean: dimension string must not be empty")),
            _ => Ok(None),
        };
    }

    let scalar_hint = match value {
        Value::Num(_) | Value::Int(_) => true,
        Value::Tensor(t) => t.data.len() == 1,
        Value::LogicalArray(logical) => logical.data.len() == 1,
        Value::GpuTensor(handle) => {
            is_scalar_shape(&handle.shape) || tensor::element_count(&handle.shape) == 1
        }
        _ => false,
    };

    let dims = match value {
        Value::Tensor(_)
        | Value::LogicalArray(_)
        | Value::Int(_)
        | Value::Num(_)
        | Value::GpuTensor(_) => tensor::dims_from_value_async(value)
            .await
            .map_err(|err| map_dims_error(err, scalar_hint))?,
        Value::Bool(_) => {
            return Err(mean_error("mean: dimension must be numeric"));
        }
        _ => return Ok(None),
    };

    let Some(mut dims) = dims else {
        return Ok(None);
    };
    if dims.is_empty() {
        return Ok(Some(MeanAxes::Default));
    }
    if dims.len() == 1 {
        let dim = dims[0];
        if dim < 1 {
            return Err(mean_error("mean: dimension must be >= 1"));
        }
        return Ok(Some(MeanAxes::Dim(dim)));
    }
    for dim in &mut dims {
        if *dim == 0 {
            *dim = 1;
        }
        if *dim < 1 {
            return Err(mean_error("mean: dimension entries must be >= 1"));
        }
    }
    Ok(Some(MeanAxes::Vec(dims)))
}

fn value_as_str(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        _ => None,
    }
}

fn map_dims_error(message: String, scalar: bool) -> RuntimeError {
    if message.contains("non-negative") {
        if scalar {
            return mean_error("mean: dimension must be >= 1");
        }
        return mean_error("mean: dimension entries must be >= 1");
    }
    if scalar {
        if message.contains("finite") {
            return mean_error("mean: dimension must be finite");
        }
        if message.contains("integer") {
            return mean_error("mean: dimension must be an integer");
        }
    }
    mean_error("mean: dimension entries must be finite integers")
}

fn mean_host(value: Value, args: &ParsedArguments) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("mean", value).map_err(mean_error)?;
    let reduced = mean_tensor(tensor, args.axes.clone(), args.nan_mode)?;
    Ok(tensor::tensor_into_value(reduced))
}

fn mean_host_complex_scalar(re: f64, im: f64, args: &ParsedArguments) -> BuiltinResult<Value> {
    let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
        .map_err(|e| mean_error(format!("mean: {e}")))?;
    mean_host_complex_tensor(tensor, args)
}

fn mean_host_complex_tensor(tensor: ComplexTensor, args: &ParsedArguments) -> BuiltinResult<Value> {
    let reduced = mean_complex_tensor(tensor, args.axes.clone(), args.nan_mode)?;
    Ok(complex_tensor_into_value(reduced))
}

async fn mean_gpu(handle: GpuTensorHandle, args: &ParsedArguments) -> BuiltinResult<Value> {
    #[cfg(all(test, feature = "wgpu"))]
    {
        if handle.device_id != 0 {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    if let Some(provider) = runmat_accelerate_api::provider() {
        // Include-NaN: use provider reduce_mean_* hooks
        if args.nan_mode == ReductionNaN::Include {
            if let Some(device_result) = mean_gpu_try(provider, &handle, &args.axes).await {
                return Ok(Value::GpuTensor(device_result));
            }
        } else {
            // Omit-NaN: compute fully on device via cleaned sum and non-NaN counts
            if let Some(device_result) = mean_gpu_omitnan(provider, &handle, &args.axes).await {
                return Ok(Value::GpuTensor(device_result));
            }
        }
    }

    let gathered = gpu_helpers::gather_tensor_async(&handle).await?;
    let reduced = mean_tensor(gathered, args.axes.clone(), args.nan_mode)?;
    Ok(tensor::tensor_into_value(reduced))
}

async fn mean_gpu_try(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
    axes: &MeanAxes,
) -> Option<GpuTensorHandle> {
    match axes {
        MeanAxes::Default => {
            if is_scalar_shape(&handle.shape) {
                return Some(handle.clone());
            }
            let dim = default_dimension_from_shape(&handle.shape);
            reduce_mean_dim_gpu(provider, handle.clone(), dim).await
        }
        MeanAxes::Dim(dim) => reduce_mean_dim_gpu(provider, handle.clone(), *dim).await,
        MeanAxes::Vec(dims) => {
            // Prefer provider N-D reduce if available
            let mut dims0: Vec<usize> = dims
                .iter()
                .filter_map(|&d| if d > 0 { Some(d - 1) } else { None })
                .collect();
            dims0.sort_unstable();
            dims0.dedup();
            if !dims0.is_empty() {
                if let Ok(out) = provider.reduce_mean_nd(handle, &dims0).await {
                    return Some(out);
                }
            }
            // Try fast permute+2D fallback
            if let Some(nd) = reduce_mean_vecdim_nd_gpu(provider, handle, dims).await {
                return Some(nd);
            }
            // Sequential per-dimension reductions
            let mut result = handle.clone();
            let mut dims_sorted = dims.clone();
            dims_sorted.sort_unstable();
            dims_sorted.dedup();
            for dim in dims_sorted {
                if is_scalar_shape(&result.shape) {
                    break;
                }
                result = reduce_mean_dim_gpu(provider, result, dim).await?;
            }
            Some(result)
        }
        MeanAxes::All => {
            if is_scalar_shape(&handle.shape) {
                return Some(handle.clone());
            }
            match provider.reduce_mean(handle).await {
                Ok(out) => Some(out),
                Err(err) => {
                    log::trace!("mean: provider reduce_mean fallback triggered: {err}");
                    let rank = handle.shape.len();
                    if is_scalar_shape(&handle.shape) || rank == 0 {
                        Some(handle.clone())
                    } else {
                        let dims: Vec<usize> = (1..=rank).collect();
                        if let Some(result) =
                            reduce_mean_vecdim_nd_gpu(provider, handle, &dims).await
                        {
                            Some(result)
                        } else {
                            let mut result = handle.clone();
                            for dim in 1..=rank {
                                match reduce_mean_dim_gpu(provider, result, dim).await {
                                    Some(updated) => result = updated,
                                    None => return None,
                                }
                            }
                            Some(result)
                        }
                    }
                }
            }
        }
    }
}

async fn reduce_mean_dim_gpu(
    provider: &dyn AccelProvider,
    handle: GpuTensorHandle,
    dim: usize,
) -> Option<GpuTensorHandle> {
    if dim == 0 {
        return None;
    }
    if handle.shape.len() < dim {
        return Some(handle);
    }
    provider
        .reduce_mean_dim(&handle, dim - 1)
        .await
        .map_err(|err| {
            log::trace!("mean: provider reduce_mean_dim fallback triggered: {err}");
            err
        })
        .ok()
}

/// Reduce mean across multiple (1-based) dimensions in a single device pass by
/// permuting reduce dims to the front, reshaping to 2-D, reducing rows, and
/// reshaping/permuting back to the original order with size-1 dims preserved.
// (N-D mean fast path omitted for now; sequential per-dimension GPU reductions used instead.)
async fn reduce_mean_vecdim_nd_gpu(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
    dims_1based: &[usize],
) -> Option<GpuTensorHandle> {
    let rank = handle.shape.len();
    if rank == 0 || dims_1based.is_empty() {
        return Some(handle.clone());
    }
    // Convert to 0-based and filter in-bounds
    let mut reduce_dims: Vec<usize> = dims_1based
        .iter()
        .filter_map(|&d| {
            if d > 0 && d <= rank {
                Some(d - 1)
            } else {
                None
            }
        })
        .collect();
    if reduce_dims.is_empty() {
        return Some(handle.clone());
    }
    reduce_dims.sort_unstable();
    reduce_dims.dedup();
    // Kept dims
    let kept_dims: Vec<usize> = (0..rank).filter(|i| !reduce_dims.contains(i)).collect();
    // Permute reduced dims first
    let mut order: Vec<usize> = Vec::with_capacity(rank);
    order.extend_from_slice(&reduce_dims);
    order.extend_from_slice(&kept_dims);
    let permuted = provider.permute(handle, &order).ok()?;
    // Compute rows/cols
    let mut reduce_len: usize = 1;
    for &d in &reduce_dims {
        reduce_len = reduce_len.saturating_mul(handle.shape[d]);
    }
    let total_elems: usize = handle.shape.iter().copied().product();
    if reduce_len == 0 || total_elems == 0 {
        let _ = provider.free(&permuted);
        return provider.fill(&[1, 1], f64::NAN).ok();
    }
    let num_slices = total_elems / reduce_len;
    // Reshape permuted view to [rows, cols]
    let reshaped2d = provider
        .reshape(&permuted, &[reduce_len, num_slices])
        .ok()?;
    // Reduce along rows (dim 0) -> [1, num_slices]
    let reduced_rows = provider.reduce_mean_dim(&reshaped2d, 0).await.ok()?;
    let _ = provider.free(&reshaped2d);
    let _ = provider.free(&permuted);
    // Reshape to kept sizes (permuted order)
    let kept_sizes: Vec<usize> = kept_dims.iter().map(|&d| handle.shape[d]).collect();
    let kept_shape = if kept_sizes.is_empty() {
        vec![1, 1]
    } else {
        kept_sizes.clone()
    };
    let reshaped_kept = provider.reshape(&reduced_rows, &kept_shape).ok()?;
    let _ = provider.free(&reduced_rows);
    // Expand permuted shape by inserting ones for reduced axes
    let mut expanded_perm_shape: Vec<usize> = Vec::with_capacity(rank);
    expanded_perm_shape.extend(std::iter::repeat_n(1usize, reduce_dims.len()));
    expanded_perm_shape.extend_from_slice(&kept_sizes);
    let expanded = provider
        .reshape(&reshaped_kept, &expanded_perm_shape)
        .ok()?;
    let _ = provider.free(&reshaped_kept);
    // Inverse permute back to original axis order
    let mut inv_order = vec![0usize; rank];
    for (dst, &src) in order.iter().enumerate() {
        inv_order[src] = dst;
    }
    let out = provider.permute(&expanded, &inv_order).ok()?;
    let _ = provider.free(&expanded);
    Some(out)
}

async fn mean_gpu_omitnan(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
    axes: &MeanAxes,
) -> Option<GpuTensorHandle> {
    // Early return for empty dim selection
    let dims_in_bounds: Vec<usize> = match axes {
        MeanAxes::Default => {
            if is_scalar_shape(&handle.shape) {
                return Some(handle.clone());
            }
            vec![default_dimension_from_shape(&handle.shape) - 1]
        }
        MeanAxes::Dim(d) => {
            if *d == 0 || *d > handle.shape.len() {
                return Some(handle.clone());
            }
            vec![*d - 1]
        }
        MeanAxes::Vec(v) => {
            let mut dims: Vec<usize> = v
                .iter()
                .filter_map(|&d| {
                    if d > 0 && d <= handle.shape.len() {
                        Some(d - 1)
                    } else {
                        None
                    }
                })
                .collect();
            dims.sort_unstable();
            dims.dedup();
            dims
        }
        MeanAxes::All => {
            if is_scalar_shape(&handle.shape) {
                return Some(handle.clone());
            }
            (0..handle.shape.len()).collect()
        }
    };

    if dims_in_bounds.is_empty() {
        return Some(handle.clone());
    }

    // Build cleaned values and not-NaN counts on device
    let cleaned = provider.map_nan_to_zero(handle).ok()?;
    let mask = provider.not_nan_mask(handle).ok()?;

    // Reduce cleaned (sum) and mask (count) along the requested dims
    let mut sum_h = cleaned.clone();
    let mut cnt_h = mask.clone();
    for &dim in &dims_in_bounds {
        sum_h = provider.reduce_sum_dim(&sum_h, dim).await.ok()?;
        cnt_h = provider.reduce_sum_dim(&cnt_h, dim).await.ok()?;
    }

    // mean = sum ./ count (0/0 -> NaN when all NaN)
    let out = provider.elem_div(&sum_h, &cnt_h).await.ok()?;

    // Free intermediates
    let _ = provider.free(&cleaned);
    let _ = provider.free(&mask);
    let _ = provider.free(&sum_h);
    let _ = provider.free(&cnt_h);

    Some(out)
}

fn mean_tensor(tensor: Tensor, axes: MeanAxes, nan_mode: ReductionNaN) -> BuiltinResult<Tensor> {
    match axes {
        MeanAxes::Default => {
            let dim = default_dimension(&tensor);
            reduce_tensor_mean_dim(&tensor, dim, nan_mode)
        }
        MeanAxes::Dim(dim) => reduce_tensor_mean_dim(&tensor, dim, nan_mode),
        MeanAxes::Vec(dims) => {
            let mut current = tensor;
            let mut dims_sorted = dims;
            dims_sorted.sort_unstable();
            dims_sorted.dedup();
            for dim in dims_sorted {
                current = reduce_tensor_mean_dim(&current, dim, nan_mode)?;
            }
            Ok(current)
        }
        MeanAxes::All => mean_tensor_all(&tensor, nan_mode),
    }
}

fn mean_tensor_all(tensor: &Tensor, nan_mode: ReductionNaN) -> BuiltinResult<Tensor> {
    if is_scalar_shape(&tensor.shape) {
        return Ok(tensor.clone());
    }
    let total_elems = tensor
        .shape
        .iter()
        .copied()
        .map(|dim| dim.max(1))
        .fold(1usize, |acc, dim| acc.saturating_mul(dim));
    if total_elems == 0 || tensor.data.is_empty() {
        return Tensor::new(vec![f64::NAN], vec![1, 1])
            .map_err(|e| mean_error(format!("mean: {e}")));
    }
    let mut sum = 0.0f64;
    let mut count = 0usize;
    let mut saw_nan = false;
    match nan_mode {
        ReductionNaN::Include => {
            for &value in &tensor.data {
                if value.is_nan() {
                    saw_nan = true;
                    break;
                }
                sum += value;
            }
            let result = if saw_nan {
                f64::NAN
            } else {
                sum / (total_elems as f64)
            };
            Tensor::new(vec![result], vec![1, 1]).map_err(|e| mean_error(format!("mean: {e}")))
        }
        ReductionNaN::Omit => {
            for &value in &tensor.data {
                if value.is_nan() {
                    continue;
                }
                sum += value;
                count += 1;
            }
            let result = if count == 0 {
                f64::NAN
            } else {
                sum / (count as f64)
            };
            Tensor::new(vec![result], vec![1, 1]).map_err(|e| mean_error(format!("mean: {e}")))
        }
    }
}

fn reduce_tensor_mean_dim(
    tensor: &Tensor,
    dim: usize,
    nan_mode: ReductionNaN,
) -> BuiltinResult<Tensor> {
    if dim == 0 {
        return Err(mean_error("mean: dimension must be >= 1"));
    }

    if is_scalar_shape(&tensor.shape) {
        let value = tensor.data.first().copied().unwrap_or(f64::NAN);
        let result = match nan_mode {
            ReductionNaN::Include => value,
            ReductionNaN::Omit => {
                if value.is_nan() {
                    f64::NAN
                } else {
                    value
                }
            }
        };
        return Tensor::new(vec![result], vec![1, 1]).map_err(|e| mean_error(format!("mean: {e}")));
    }

    let Some(output_shape) = reduction_shape(&tensor.shape, dim) else {
        return Ok(tensor.clone());
    };

    if tensor.data.is_empty() {
        let fill = vec![f64::NAN; tensor::element_count(&output_shape)];
        return Tensor::new(fill, output_shape).map_err(|e| mean_error(format!("mean: {e}")));
    }

    let dim_index = dim - 1;
    let reduce_len = tensor.shape[dim_index];
    let stride_before = dim_product(&tensor.shape[..dim_index]);
    let stride_after = dim_product(&tensor.shape[dim..]);
    let out_len = tensor::element_count(&output_shape);
    let mut output = vec![0.0f64; out_len];

    for after in 0..stride_after {
        for before in 0..stride_before {
            let mut sum = 0.0;
            let mut count = 0usize;
            let mut saw_nan = false;

            for k in 0..reduce_len {
                let idx = before + k * stride_before + after * stride_before * reduce_len;
                let value = tensor.data[idx];
                match nan_mode {
                    ReductionNaN::Include => {
                        if value.is_nan() {
                            saw_nan = true;
                            break;
                        }
                        sum += value;
                    }
                    ReductionNaN::Omit => {
                        if value.is_nan() {
                            continue;
                        }
                        sum += value;
                        count += 1;
                    }
                }
            }

            let out_idx = after * stride_before + before;
            output[out_idx] = match nan_mode {
                ReductionNaN::Include => {
                    if reduce_len == 0 || saw_nan {
                        f64::NAN
                    } else {
                        sum / (reduce_len as f64)
                    }
                }
                ReductionNaN::Omit => {
                    if count == 0 {
                        f64::NAN
                    } else {
                        sum / (count as f64)
                    }
                }
            };
        }
    }

    Tensor::new(output, output_shape).map_err(|e| mean_error(format!("mean: {e}")))
}

fn mean_complex_tensor(
    tensor: ComplexTensor,
    axes: MeanAxes,
    nan_mode: ReductionNaN,
) -> BuiltinResult<ComplexTensor> {
    match axes {
        MeanAxes::Default => {
            let dim = default_dimension_from_shape(&tensor.shape);
            reduce_complex_tensor_mean_dim(&tensor, dim, nan_mode)
        }
        MeanAxes::Dim(dim) => reduce_complex_tensor_mean_dim(&tensor, dim, nan_mode),
        MeanAxes::Vec(mut dims) => {
            dims.sort_unstable();
            dims.dedup();
            let mut current = tensor;
            for dim in dims {
                current = reduce_complex_tensor_mean_dim(&current, dim, nan_mode)?;
            }
            Ok(current)
        }
        MeanAxes::All => {
            if is_scalar_shape(&tensor.shape) {
                Ok(tensor)
            } else {
                let mut current = tensor;
                let ndims = current.shape.len();
                for dim in 1..=ndims {
                    current = reduce_complex_tensor_mean_dim(&current, dim, nan_mode)?;
                }
                Ok(current)
            }
        }
    }
}

fn reduce_complex_tensor_mean_dim(
    tensor: &ComplexTensor,
    dim: usize,
    nan_mode: ReductionNaN,
) -> BuiltinResult<ComplexTensor> {
    if dim == 0 {
        return Err(mean_error("mean: dimension must be >= 1"));
    }

    let shape = if is_scalar_shape(&tensor.shape) {
        normalize_scalar_shape(&tensor.shape)
    } else {
        tensor.shape.clone()
    };

    if is_scalar_shape(&shape) {
        let (re, im) = tensor.data.first().copied().unwrap_or((f64::NAN, f64::NAN));
        let result = match nan_mode {
            ReductionNaN::Include => (re, im),
            ReductionNaN::Omit => {
                if re.is_nan() || im.is_nan() {
                    (f64::NAN, f64::NAN)
                } else {
                    (re, im)
                }
            }
        };
        return ComplexTensor::new(vec![result], canonical_scalar_shape())
            .map_err(|e| mean_error(format!("mean: {e}")));
    }

    let Some(output_shape) = reduction_shape(&shape, dim) else {
        return Ok(tensor.clone());
    };

    if tensor.data.is_empty() {
        let fill = vec![(f64::NAN, f64::NAN); tensor::element_count(&output_shape)];
        return ComplexTensor::new(fill, output_shape)
            .map_err(|e| mean_error(format!("mean: {e}")));
    }

    let dim_index = dim - 1;
    if dim_index >= shape.len() {
        return Ok(tensor.clone());
    }

    let reduce_len = shape[dim_index];
    let stride_before = dim_product(&shape[..dim_index]);
    let stride_after = dim_product(&shape[dim..]);
    let out_len = tensor::element_count(&output_shape);
    let mut output = vec![(0.0f64, 0.0f64); out_len];

    for after in 0..stride_after {
        for before in 0..stride_before {
            let mut sum_re = 0.0;
            let mut sum_im = 0.0;
            let mut count = 0usize;
            let mut saw_nan = false;

            for k in 0..reduce_len {
                let idx = before + k * stride_before + after * stride_before * reduce_len;
                let (re, im) = tensor.data[idx];
                let is_nan = re.is_nan() || im.is_nan();
                match nan_mode {
                    ReductionNaN::Include => {
                        if is_nan {
                            saw_nan = true;
                            break;
                        }
                        sum_re += re;
                        sum_im += im;
                    }
                    ReductionNaN::Omit => {
                        if is_nan {
                            continue;
                        }
                        sum_re += re;
                        sum_im += im;
                        count += 1;
                    }
                }
            }

            let out_idx = after * stride_before + before;
            output[out_idx] = match nan_mode {
                ReductionNaN::Include => {
                    if reduce_len == 0 || saw_nan {
                        (f64::NAN, f64::NAN)
                    } else {
                        (sum_re / (reduce_len as f64), sum_im / (reduce_len as f64))
                    }
                }
                ReductionNaN::Omit => {
                    if count == 0 {
                        (f64::NAN, f64::NAN)
                    } else {
                        (sum_re / (count as f64), sum_im / (count as f64))
                    }
                }
            };
        }
    }

    ComplexTensor::new(output, output_shape).map_err(|e| mean_error(format!("mean: {e}")))
}

fn reduction_shape(shape: &[usize], dim: usize) -> Option<Vec<usize>> {
    if dim == 0 {
        return None;
    }
    if is_scalar_shape(shape) {
        if dim == 1 {
            return Some(canonical_scalar_shape());
        }
        return None;
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
    default_dimension_from_shape(&tensor.shape)
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

async fn apply_output_template(
    value: Value,
    template: &OutputTemplate,
    meta: &InputMeta,
) -> BuiltinResult<Value> {
    match template {
        OutputTemplate::Double => Ok(value),
        OutputTemplate::Native => {
            let value = apply_native_template(value, meta).await?;
            ensure_device(value, meta.device).await
        }
        OutputTemplate::Like(proto) => apply_like_template(value, proto).await,
    }
}

async fn apply_native_template(value: Value, meta: &InputMeta) -> BuiltinResult<Value> {
    match meta.class {
        InputClass::Integer(class) => match value {
            Value::Num(n) => class.to_value(n),
            Value::Tensor(t) if t.data.len() == 1 => class.to_value(t.data[0]),
            other => Ok(other),
        },
        _ => {
            if let Some(dtype) = meta.numeric_dtype {
                coerce_value_to_dtype(value, dtype).await
            } else {
                Ok(value)
            }
        }
    }
}

async fn coerce_value_to_dtype(value: Value, dtype: NumericDType) -> BuiltinResult<Value> {
    match dtype {
        NumericDType::F64 => Ok(value),
        NumericDType::F32 => match value {
            Value::Tensor(tensor) => {
                let tensor = coerce_tensor_dtype(tensor, NumericDType::F32);
                Ok(Value::Tensor(tensor))
            }
            Value::Num(n) => {
                let tensor = Tensor::new_with_dtype(vec![n], vec![1, 1], NumericDType::F32)
                    .map_err(|e| mean_error(format!("{NAME}: {e}")))?;
                Ok(Value::Tensor(tensor))
            }
            Value::LogicalArray(logical) => {
                let tensor = tensor::logical_to_tensor(&logical)
                    .map_err(|e| mean_error(format!("{NAME}: {e}")))?;
                let tensor = coerce_tensor_dtype(tensor, NumericDType::F32);
                Ok(Value::Tensor(tensor))
            }
            Value::GpuTensor(handle) => {
                let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
                let tensor = coerce_tensor_dtype(tensor, NumericDType::F32);
                Ok(Value::Tensor(tensor))
            }
            other => Ok(other),
        },
    }
}

fn coerce_tensor_dtype(mut tensor: Tensor, dtype: NumericDType) -> Tensor {
    match dtype {
        NumericDType::F64 => {
            tensor.dtype = NumericDType::F64;
        }
        NumericDType::F32 => {
            for value in &mut tensor.data {
                *value = (*value as f32) as f64;
            }
            tensor.dtype = NumericDType::F32;
        }
    }
    tensor
}

async fn ensure_device(value: Value, device: DevicePreference) -> BuiltinResult<Value> {
    match device {
        DevicePreference::Host => match value {
            Value::GpuTensor(handle) => {
                let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
                Ok(tensor::tensor_into_value(tensor))
            }
            _ => Ok(value),
        },
        DevicePreference::Gpu => match value {
            Value::GpuTensor(_) => Ok(value),
            Value::Tensor(t) => upload_tensor(t),
            Value::Num(n) => {
                let tensor = Tensor::new(vec![n], vec![1, 1])
                    .map_err(|e| mean_error(format!("mean: {e}")))?;
                upload_tensor(tensor)
            }
            Value::LogicalArray(logical) => {
                let tensor = tensor::logical_to_tensor(&logical)?;
                upload_tensor(tensor)
            }
            other => Err(mean_error(format!(
                "mean: cannot place value {other:?} on the GPU"
            ))),
        },
    }
}

fn upload_tensor(tensor: Tensor) -> BuiltinResult<Value> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Err(mean_error(
            "mean: no acceleration provider available to honour GPU output",
        ));
    };
    let view = HostTensorView {
        data: &tensor.data,
        shape: &tensor.shape,
    };
    let handle = provider
        .upload(&view)
        .map_err(|e| mean_error(format!("mean: failed to upload GPU result: {e}")))?;
    Ok(Value::GpuTensor(handle))
}

async fn apply_like_template(value: Value, prototype: &Value) -> BuiltinResult<Value> {
    let analysed = analyse_like_prototype(prototype).await?;
    match analysed.class {
        PrototypeClass::Real => match analysed.device {
            DevicePreference::Host => ensure_device(value, DevicePreference::Host).await,
            DevicePreference::Gpu => ensure_device(value, DevicePreference::Gpu).await,
        },
        PrototypeClass::Complex => {
            let host_value = ensure_device(value, DevicePreference::Host).await?;
            real_to_complex(host_value)
        }
    }
}

fn real_to_complex(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Complex(_, _) | Value::ComplexTensor(_) => Ok(value),
        Value::Num(n) => Ok(Value::Complex(n, 0.0)),
        Value::Tensor(t) => {
            let data: Vec<(f64, f64)> = t.data.iter().map(|&v| (v, 0.0)).collect();
            let tensor = ComplexTensor::new(data, t.shape.clone())
                .map_err(|e| mean_error(format!("mean: {e}")))?;
            Ok(complex_tensor_into_value(tensor))
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)?;
            real_to_complex(Value::Tensor(tensor))
        }
        other => Err(mean_error(format!(
            "mean: cannot convert value {other:?} to a complex result"
        ))),
    }
}

struct LikeAnalysis {
    device: DevicePreference,
    class: PrototypeClass,
}

enum PrototypeClass {
    Real,
    Complex,
}

#[async_recursion::async_recursion(?Send)]
async fn analyse_like_prototype(proto: &Value) -> BuiltinResult<LikeAnalysis> {
    match proto {
        Value::GpuTensor(_) => Ok(LikeAnalysis {
            device: DevicePreference::Gpu,
            class: PrototypeClass::Real,
        }),
        Value::Tensor(_)
        | Value::Num(_)
        | Value::Int(_)
        | Value::LogicalArray(_)
        | Value::Bool(_) => Ok(LikeAnalysis {
            device: DevicePreference::Host,
            class: PrototypeClass::Real,
        }),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Ok(LikeAnalysis {
            device: DevicePreference::Host,
            class: PrototypeClass::Complex,
        }),
        other => {
            let gathered = dispatcher::gather_if_needed_async(other)
                .await
                .map_err(|e| mean_error(format!("mean: {e}")))?;
            analyse_like_prototype(&gathered).await
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::IntValue;

    fn mean_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::mean_builtin(value, rest))
    }

    #[test]
    fn mean_type_reduces_first_dim() {
        let out = mean_type(
            &[Type::Tensor {
                shape: Some(vec![Some(2), Some(4)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(1), Some(4)])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_scalar_num() {
        let result = mean_builtin(Value::Num(6.0), Vec::new()).expect("mean");
        assert_eq!(result, Value::Num(6.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_matrix_default_dimension() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result = mean_builtin(Value::Tensor(tensor), Vec::new()).expect("mean");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 3]);
                assert_eq!(out.data, vec![2.5, 3.5, 4.5]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_matrix_dimension_two() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result =
            mean_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(2))]).expect("mean");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert_eq!(out.data, vec![2.0, 5.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_with_omit_nan_default_dimension() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 5.0], vec![3, 1]).unwrap();
        let result =
            mean_builtin(Value::Tensor(tensor), vec![Value::from("omitnan")]).expect("mean");
        match result {
            Value::Num(v) => assert!((v - 3.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_with_omit_nan_all_nan_returns_nan() {
        let tensor = Tensor::new(vec![f64::NAN, f64::NAN], vec![2, 1]).unwrap();
        let result =
            mean_builtin(Value::Tensor(tensor), vec![Value::from("omitnan")]).expect("mean");
        match result {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected NaN result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_with_include_nan_propagates_nan() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 3.0], vec![3, 1]).unwrap();
        let result =
            mean_builtin(Value::Tensor(tensor), vec![Value::from("includenan")]).expect("mean");
        match result {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected NaN result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_dimension_greater_than_ndims_returns_input() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let original = tensor.clone();
        let result =
            mean_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(5))]).expect("mean");
        match result {
            Value::Tensor(out) => assert_eq!(out, original),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_native_integer_scalar() {
        let value = Value::Int(IntValue::I16(42));
        let result = mean_builtin(value, vec![Value::from("native")]).expect("mean");
        assert_eq!(result, Value::Int(IntValue::I16(42)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_like_complex_prototype() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let proto = Value::Complex(0.0, 1.0);
        let result = mean_builtin(
            Value::Tensor(tensor),
            vec![Value::from("all"), Value::from("like"), proto.clone()],
        )
        .expect("mean");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 2.0).abs() < 1e-12);
                assert!(im.abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_like_without_prototype_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = mean_builtin(Value::Tensor(tensor), vec![Value::from("like")])
            .expect_err("expected error");
        assert!(err.message().contains("prototype"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_dimension_with_omit_nan() {
        let tensor =
            Tensor::new(vec![1.0, f64::NAN, 3.0, 4.0], vec![2, 2]).expect("tensor construction");
        let result = mean_builtin(
            Value::Tensor(tensor),
            vec![Value::Int(IntValue::I32(1)), Value::from("omitnan")],
        )
        .expect("mean");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(out.data, vec![1.0, 3.5]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_all_dimension_reduces_to_scalar() {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let result = mean_builtin(Value::Tensor(tensor), vec![Value::from("all")]).expect("mean");
        match result {
            Value::Num(v) => assert!((v - 2.5).abs() < 1e-12),
            Value::Tensor(t) => {
                assert_eq!(t.data.len(), 1);
                assert!((t.data[0] - 2.5).abs() < 1e-12);
            }
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_all_keyword_first_arg_swapped_ok() {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let a = mean_builtin(Value::Tensor(tensor.clone()), vec![Value::from("all")]).unwrap();
        // Provide 'all' as the first argument (char/string), then the tensor
        let b = mean_builtin(Value::from("all"), vec![Value::Tensor(tensor)]).unwrap();
        match (a, b) {
            (Value::Num(x), Value::Num(y)) => assert!((x - y).abs() < 1e-12),
            (Value::Tensor(tx), Value::Tensor(ty)) => {
                assert_eq!(tx.shape, ty.shape);
                for (x, y) in tx.data.iter().zip(ty.data.iter()) {
                    assert!((x - y).abs() < 1e-12);
                }
            }
            (ax, bx) => panic!("shape mismatch a={ax:?} b={bx:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_all_with_omit_nan() {
        let tensor = Tensor::new(vec![f64::NAN, 2.0, 4.0, f64::NAN], vec![2, 2]).expect("tensor");
        let result = mean_builtin(
            Value::Tensor(tensor),
            vec![Value::from("all"), Value::from("omitnan")],
        )
        .expect("mean");
        match result {
            Value::Num(v) => assert!((v - 3.0).abs() < 1e-12),
            other => panic!("expected numeric result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_all_matches_sequential_for_nd_tensor() {
        let data: Vec<f64> = (1..=24).map(|v| v as f64).collect();
        let tensor = Tensor::new(data, vec![2, 3, 4]).expect("tensor");
        let fused =
            mean_builtin(Value::Tensor(tensor.clone()), vec![Value::from("all")]).expect("mean");
        let sequential = mean_builtin(
            mean_builtin(Value::Tensor(tensor.clone()), vec![Value::Num(1.0)]).expect("mean"),
            vec![Value::Num(2.0)],
        )
        .and_then(|v| mean_builtin(v, vec![Value::Num(3.0)]))
        .expect("mean");
        assert_eq!(fused, sequential);
        if let Value::Num(v) = fused {
            assert!((v - 12.5).abs() < 1e-12);
        } else if let Value::Tensor(t) = fused {
            assert_eq!(t.data.len(), 1);
            assert!((t.data[0] - 12.5).abs() < 1e-12);
        } else {
            panic!("unexpected result {fused:?}");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_vector_dimensions_match_sequential() {
        let tensor =
            Tensor::new(vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0], vec![2, 2, 2]).unwrap();
        let dims = Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap();
        let combined = mean_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::Tensor(dims.clone())],
        )
        .expect("mean");
        let first = mean_builtin(Value::Tensor(tensor), vec![Value::Num(1.0)]).expect("mean");
        let sequential = mean_builtin(first, vec![Value::Num(3.0)]).expect("mean");
        assert_eq!(combined, sequential);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_complex_scalar_passthrough() {
        let result = mean_builtin(Value::Complex(2.0, -3.0), Vec::new()).expect("mean");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 2.0).abs() < 1e-12);
                assert!((im + 3.0).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_complex_matrix_along_rows() {
        let tensor = ComplexTensor::new(
            vec![(1.0, 1.0), (5.0, -1.0), (2.0, 2.0), (6.0, -2.0)],
            vec![2, 2],
        )
        .unwrap();
        let result =
            mean_builtin(Value::ComplexTensor(tensor), vec![Value::Num(1.0)]).expect("mean");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                let expected = [(3.0, 0.0), (4.0, 0.0)];
                for (got, exp) in out.data.iter().zip(expected.iter()) {
                    assert!((got.0 - exp.0).abs() < 1e-12);
                    assert!((got.1 - exp.1).abs() < 1e-12);
                }
            }
            Value::Complex(re, im) => {
                panic!("expected tensor result, got scalar {re}+{im}i");
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_complex_omit_nan_returns_nan() {
        let tensor =
            ComplexTensor::new(vec![(f64::NAN, 0.0), (1.0, f64::NAN)], vec![2, 1]).unwrap();
        let result = mean_builtin(
            Value::ComplexTensor(tensor),
            vec![Value::Int(IntValue::I32(1)), Value::from("omitnan")],
        )
        .expect("mean");
        match result {
            Value::Complex(re, im) => {
                assert!(re.is_nan());
                assert!(im.is_nan());
            }
            Value::ComplexTensor(out) => {
                let (re, im) = out.data[0];
                assert!(re.is_nan());
                assert!(im.is_nan());
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = mean_builtin(Value::GpuTensor(handle), Vec::new()).expect("mean");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![1, 3]);
            assert_eq!(gathered.data, vec![2.5, 3.5, 4.5]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_gpu_omit_nan_falls_back_to_host() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![f64::NAN, 2.0, f64::NAN, 4.0], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result =
                mean_builtin(Value::GpuTensor(handle), vec![Value::from("omitnan")]).expect("mean");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![1, 2]);
            assert_eq!(gathered.data, vec![2.0, 4.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_gpu_all_dimension_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result =
                mean_builtin(Value::GpuTensor(handle), vec![Value::from("all")]).expect("mean");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![1, 1]);
            assert!((gathered.data[0] - 2.5).abs() < 1e-12);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_gpu_vector_dimensions_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor =
                Tensor::new(vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0], vec![2, 2, 2]).unwrap();
            let cpu_dims = Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap();
            let cpu_result = mean_builtin(
                Value::Tensor(tensor.clone()),
                vec![Value::Tensor(cpu_dims.clone())],
            )
            .expect("mean cpu");

            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let gpu_result = mean_builtin(Value::GpuTensor(handle), vec![Value::Tensor(cpu_dims)])
                .expect("mean gpu");

            let cpu_tensor = match cpu_result {
                Value::Tensor(t) => t,
                Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
                other => panic!("unexpected cpu result {other:?}"),
            };
            let gpu_tensor = test_support::gather(gpu_result).expect("gather");
            assert_eq!(gpu_tensor.shape, cpu_tensor.shape);
            for (a, b) in gpu_tensor.data.iter().zip(cpu_tensor.data.iter()) {
                assert!((a - b).abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_nested_dim2_then_dim3_host_matches_vecdim() {
        let t = Tensor::new((0..(2 * 3 * 4)).map(|i| i as f64).collect(), vec![2, 3, 4]).unwrap();
        let vecdim = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
        let a = mean_builtin(Value::Tensor(t.clone()), vec![Value::Tensor(vecdim)]).unwrap();
        let b1 = mean_builtin(Value::Tensor(t), vec![Value::Num(2.0)]).unwrap();
        let b2 = mean_builtin(b1, vec![Value::Num(3.0)]).unwrap();
        assert_eq!(a, b2);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_like_gpu_prototype_residency() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let input = provider.upload(&view).expect("upload");
            let prototype = provider.upload(&view).expect("upload");
            let result = mean_builtin(
                Value::GpuTensor(input),
                vec![
                    Value::from("omitnan"),
                    Value::from("like"),
                    Value::GpuTensor(prototype),
                ],
            )
            .expect("mean");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered =
                        test_support::gather(Value::GpuTensor(handle.clone())).expect("gather");
                    assert_eq!(gathered.data.len(), 1);
                    assert!((gathered.data[0] - 1.5).abs() < 1e-12);
                }
                other => panic!("expected GPU tensor result, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn mean_wgpu_dim1_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let t = Tensor::new(vec![1.0, 4.0, 2.0, 6.0], vec![2, 2]).unwrap();
        let args = ParsedArguments {
            axes: MeanAxes::Dim(1),
            nan_mode: ReductionNaN::Include,
            output: OutputTemplate::Double,
        };
        let cpu = mean_host(Value::Tensor(t.clone()), &args).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &t.data,
            shape: &t.shape,
        };
        let h = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = block_on(mean_gpu(h, &args)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(gt.shape, ct.shape);
                for (a, b) in gt.data.iter().zip(ct.data.iter()) {
                    assert!((a - b).abs() < 1e-12);
                }
            }
            _ => panic!("unexpected shapes"),
        }
    }
}
