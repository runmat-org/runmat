//! MATLAB-compatible `var` builtin with GPU-aware semantics for RunMat.
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, ProviderNanMode, ProviderStdNormalization,
};
use runmat_builtins::{
    BuiltinErrorDescriptor, DispersionControl, DispersionNormalization as VarNormalization,
    VAR_ERROR_COMPLEX_UNSUPPORTED, VAR_ERROR_INTERNAL, VAR_ERROR_INVALID_ARGUMENT,
    VAR_ERROR_INVALID_INPUT, VAR_TYPED_INTEGER_CONTROL_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

use crate::builtins::common::random_args::{extract_dims, keyword_of};
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

const NAME: &str = "var";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::reduction::var")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "var",
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
    notes: "Providers compute variance via standard-deviation reductions followed by an in-device squaring pass.",
};

fn var_error_with_descriptor(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn var_invalid_argument(message: impl Into<String>) -> RuntimeError {
    var_error_with_descriptor(message, &VAR_ERROR_INVALID_ARGUMENT)
}

fn var_invalid_input(message: impl Into<String>) -> RuntimeError {
    var_error_with_descriptor(message, &VAR_ERROR_INVALID_INPUT)
}

fn var_complex_unsupported_default() -> RuntimeError {
    var_error_with_descriptor(
        VAR_ERROR_COMPLEX_UNSUPPORTED.message,
        &VAR_ERROR_COMPLEX_UNSUPPORTED,
    )
}

fn var_internal_error(message: impl Into<String>) -> RuntimeError {
    var_error_with_descriptor(message, &VAR_ERROR_INTERNAL)
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::reduction::var")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "var",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Fusion currently gathers to the host; future kernels can reuse the variance accumulator directly.",
};

#[derive(Clone, Debug)]
enum VarAxes {
    Default,
    Dim(usize),
    Vec(Vec<usize>),
    All,
}

#[derive(Clone)]
struct ParsedArguments {
    axes: VarAxes,
    normalization: VarNormalization,
    nan_mode: ReductionNaN,
}

enum NormParse {
    NotMatched,
    Placeholder,
    Value(VarNormalization),
    Weighted,
}

#[runtime_builtin(
    name = "var",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::reduction::var"
)]
pub(crate) async fn var_builtin(
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    var_with_nan_policy(value, rest, None).await
}

pub(crate) async fn var_with_nan_policy(
    value: Value,
    rest: crate::call::BuiltinArguments,
    imposed_nan_mode: Option<ReductionNaN>,
) -> crate::BuiltinResult<Value> {
    reject_integer_data_input(&value)?;
    if rest.iter().any(is_real_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &VAR_TYPED_INTEGER_CONTROL_EXTENSION,
            NAME,
        )?;
    }
    let mut parsed = parse_arguments(rest.view()).await?;
    if let Some(mode) = imposed_nan_mode {
        parsed.nan_mode = mode;
    }
    match value {
        Value::GpuTensor(handle) => var_gpu(handle, &parsed).await,
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(var_complex_unsupported_default()),
        other => var_host(other, &parsed),
    }
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
        return Err(var_invalid_input(
            "var: integer data inputs are not supported",
        ));
    }
    Ok(())
}

async fn parse_arguments(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<ParsedArguments> {
    let mut axes = VarAxes::Default;
    let mut axes_set = false;
    let mut normalization = VarNormalization::Sample;
    let mut normalization_consumed = false;
    let mut nan_mode = ReductionNaN::Include;
    let mut idx = 0;
    while idx < args.len() {
        let arg = &args[idx];

        if let Some(keyword) = keyword_of(arg)? {
            match DispersionControl::from_var_public_option(&keyword) {
                Some(DispersionControl::Missing(mode)) => {
                    nan_mode = mode;
                    idx += 1;
                    continue;
                }
                Some(DispersionControl::All) => {
                    if axes_set && !matches!(axes, VarAxes::Default) {
                        return Err(var_invalid_argument(
                            "var: 'all' cannot be combined with an explicit dimension",
                        ));
                    }
                    axes = VarAxes::All;
                    axes_set = true;
                    idx += 1;
                    continue;
                }
                None => {
                    return Err(var_invalid_argument(format!(
                        "var: unrecognised option '{keyword}'"
                    )));
                }
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
                NormParse::Weighted => {
                    return Err(var_invalid_argument(
                        "var: weighted variance is not implemented yet",
                    ));
                }
                NormParse::NotMatched => {}
            }
        }

        if !axes_set || matches!(axes, VarAxes::Default) {
            if let Some(selection) = parse_axes(arg).await? {
                if axes_set && !matches!(axes, VarAxes::Default) {
                    return Err(var_invalid_argument(
                        "var: multiple dimension specifications provided",
                    ));
                }
                if matches!(selection, VarAxes::All)
                    && axes_set
                    && !matches!(axes, VarAxes::Default)
                {
                    return Err(var_invalid_argument(
                        "var: 'all' cannot be combined with an explicit dimension",
                    ));
                }
                axes = selection;
                axes_set = true;
                idx += 1;
                continue;
            }
        } else if parse_axes(arg).await?.is_some() {
            return Err(var_invalid_argument(
                "var: multiple dimension specifications provided",
            ));
        }

        return Err(var_invalid_argument(format!(
            "var: unrecognised argument {arg:?}"
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
            Ok(NormParse::Weighted)
        }
        Value::LogicalArray(logical) => {
            if logical.data.is_empty() {
                return Ok(NormParse::Placeholder);
            }
            if logical.data.len() == 1 {
                let flag = if logical.data[0] != 0 { 1.0 } else { 0.0 };
                return parse_normalization_scalar(flag);
            }
            Ok(NormParse::Weighted)
        }
        Value::Bool(flag) => Ok(NormParse::Value(if *flag {
            VarNormalization::Population
        } else {
            VarNormalization::Sample
        })),
        Value::Int(i) => match i.try_to_u64() {
            Some(0) => Ok(NormParse::Value(VarNormalization::Sample)),
            Some(1) => Ok(NormParse::Value(VarNormalization::Population)),
            _ => Err(var_invalid_argument(
                "var: normalisation flag must be 0, 1, or []",
            )),
        },
        Value::Num(n) => parse_normalization_scalar(*n),
        Value::GpuTensor(_) => Err(var_invalid_argument(
            "var: normalisation flag must reside on the host",
        )),
        _ => Ok(NormParse::NotMatched),
    }
}

fn parse_normalization_scalar(value: f64) -> BuiltinResult<NormParse> {
    if !value.is_finite() {
        return Err(var_invalid_argument(
            "var: normalisation flag must be finite",
        ));
    }
    VarNormalization::from_finite_flag(value)
        .map(NormParse::Value)
        .ok_or_else(|| var_invalid_argument("var: normalisation flag must be 0, 1, or []"))
}

async fn parse_axes(value: &Value) -> BuiltinResult<Option<VarAxes>> {
    if let Some(keyword) = keyword_of(value)? {
        if matches!(
            DispersionControl::from_var_public_option(&keyword),
            Some(DispersionControl::All)
        ) {
            return Ok(Some(VarAxes::All));
        }
        return Ok(None);
    }

    let Some(dims) = extract_dims(value, "var")
        .await
        .map_err(var_invalid_argument)?
    else {
        return Ok(None);
    };
    if dims.is_empty() {
        return Err(var_invalid_argument(
            "var: dimension vector must not be empty",
        ));
    }
    let mut cleaned = Vec::with_capacity(dims.len());
    for dim in dims {
        if dim == 0 {
            return Err(var_invalid_argument("var: dimensions must be >= 1"));
        }
        cleaned.push(dim);
    }
    if cleaned.len() == 1 {
        Ok(Some(VarAxes::Dim(cleaned[0])))
    } else {
        Ok(Some(VarAxes::Vec(cleaned)))
    }
}

fn var_host(value: Value, args: &ParsedArguments) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("var", value).map_err(var_invalid_input)?;
    let reduced = var_tensor(tensor, &args.axes, args.normalization, args.nan_mode)?;
    Ok(tensor::tensor_into_value(reduced))
}

fn var_tensor(
    tensor: Tensor,
    axes: &VarAxes,
    normalization: VarNormalization,
    nan_mode: ReductionNaN,
) -> BuiltinResult<Tensor> {
    let (dims, _) = resolve_axes(tensor.shape(), axes)?;
    if dims.is_empty() {
        if tensor::is_scalar_tensor(&tensor) {
            return var_scalar_tensor(&tensor, nan_mode);
        }
        if !matches!(axes, VarAxes::Default) {
            return tensor::zeros_with_dtype(tensor.shape(), tensor.numeric_dtype())
                .map_err(|error| var_internal_error(format!("var: {error}")));
        }
        return Ok(tensor);
    }
    var_tensor_reduce(&tensor, &dims, normalization, nan_mode)
}

fn var_scalar_tensor(tensor: &Tensor, nan_mode: ReductionNaN) -> BuiltinResult<Tensor> {
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
        .map_err(|e| var_internal_error(format!("var: {e}")))
}

fn var_tensor_reduce(
    tensor: &Tensor,
    dims: &[usize],
    normalization: VarNormalization,
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
            .map_err(|e| var_internal_error(format!("var: {e}")));
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
                match normalization {
                    VarNormalization::Sample => {
                        if count > 1 {
                            (m2[idx] / (count - 1) as f64).max(0.0)
                        } else {
                            0.0
                        }
                    }
                    VarNormalization::Population => (m2[idx] / (count as f64)).max(0.0),
                }
            };
    }

    Tensor::new_with_dtype(output, output_shape, tensor.numeric_dtype())
        .map_err(|e| var_internal_error(format!("var: {e}")))
}

fn resolve_axes(shape: &[usize], axes: &VarAxes) -> BuiltinResult<(Vec<usize>, bool)> {
    match axes {
        VarAxes::Default => {
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
        VarAxes::Dim(dim) => {
            if *dim == 0 {
                return Err(var_invalid_argument("var: dimension must be >= 1"));
            }
            let zero = dim - 1;
            if zero < shape.len() {
                Ok((vec![zero], true))
            } else {
                Ok((Vec::new(), true))
            }
        }
        VarAxes::Vec(dims) => {
            if dims.is_empty() {
                return resolve_axes(shape, &VarAxes::Default);
            }
            let mut out = Vec::with_capacity(dims.len());
            for &dim in dims {
                if dim == 0 {
                    return Err(var_invalid_argument("var: dimension must be >= 1"));
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
        VarAxes::All => {
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
    let mut idx = 0usize;
    let mut stride = 1usize;
    for (coord, &extent) in coords.iter().zip(shape.iter()) {
        if extent == 0 {
            return 0;
        }
        idx += coord * stride;
        stride *= extent;
    }
    idx
}

async fn var_gpu(handle: GpuTensorHandle, args: &ParsedArguments) -> BuiltinResult<Value> {
    gpu_helpers::require_resident_compute_owner(&handle, || {
        var_internal_error("var: resident input has no exact owning provider")
    })?;
    if let Some(provider) = gpu_helpers::exact_provider_for_handle(&handle) {
        handle
            .allocation_lease()
            .validate_live()
            .map_err(|source| {
                let mut error = var_internal_error(format!(
                    "var: resident input allocation is unavailable: {source}"
                ));
                error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                error.source = Some(Box::new(source));
                error
            })?;
        if let Some(device_value) = var_gpu_reduce(provider, &handle, args).await? {
            return Ok(Value::GpuTensor(device_value));
        }
    }
    var_gpu_fallback(&handle, args).await
}

async fn var_gpu_reduce(
    provider: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
    args: &ParsedArguments,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    let (dims, _) = resolve_axes(handle.shape(), &args.axes)?;
    if dims.is_empty() {
        return Ok(None);
    }

    let normalization = match args.normalization {
        VarNormalization::Sample => ProviderStdNormalization::Sample,
        VarNormalization::Population => ProviderStdNormalization::Population,
    };
    let nan_mode = match args.nan_mode {
        ReductionNaN::Include => ProviderNanMode::Include,
        ReductionNaN::Omit => ProviderNanMode::Omit,
    };

    let expected_shape = reduced_shape(handle.shape(), &dims);

    let std_handle = if dims.len() == handle.shape().len() {
        if is_scalar_shape(handle.shape()) {
            return Ok(None);
        }
        match accepted_std_result(
            provider,
            handle,
            &expected_shape,
            "reduce_std",
            provider.reduce_std(handle, normalization, nan_mode).await,
        )? {
            Some(output) => output,
            None => return Ok(None),
        }
    } else if dims.len() == 1 {
        let dim = dims[0] + 1;
        match reduce_std_dim_gpu(
            provider,
            handle,
            dim,
            normalization,
            nan_mode,
            &expected_shape,
        )
        .await?
        {
            Some(output) => output,
            None => return Ok(None),
        }
    } else {
        return Ok(None);
    };

    gpu_cleanup::complete_std_square(provider, handle, std_handle, &expected_shape)
        .await
        .map(Some)
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
    if !super::std::std_output_matches(provider, input, &output, expected_shape) {
        let mut error = var_internal_error(format!(
            "var: provider {operation} returned an invalid resident reduction"
        ));
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        return Err(gpu_helpers::reject_gpu_output(output, error));
    }
    Ok(Some(output))
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
                var_internal_error(format!("var: provider {operation} failed: {source}"));
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            Err(crate::runtime_error::with_provider_execution_source(
                error, source,
            ))
        }
    }
}

async fn reduce_std_dim_gpu(
    provider: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
    dim: usize,
    normalization: ProviderStdNormalization,
    nan_mode: ProviderNanMode,
    expected_shape: &[usize],
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
        expected_shape,
        "reduce_std_dim",
        provider
            .reduce_std_dim(handle, dim - 1, normalization, nan_mode)
            .await,
    )
}

async fn var_gpu_fallback(
    handle: &GpuTensorHandle,
    args: &ParsedArguments,
) -> BuiltinResult<Value> {
    let tensor = gpu_helpers::gather_tensor_async(handle).await?;
    let reduced = var_tensor(tensor, &args.axes, args.normalization, args.nan_mode)?;
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

#[path = "var/gpu_cleanup.rs"]
mod gpu_cleanup;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_value::{IntValue, IntegerStorage, Tensor, Value};

    fn var_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        block_on(super::var_builtin(
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
    fn var_typed_normalization_requires_exact_zero_or_one() {
        assert!(matches!(
            parse_normalization(&Value::Int(IntValue::U64(0))),
            Ok(NormParse::Value(VarNormalization::Sample))
        ));
        assert!(matches!(
            parse_normalization(&Value::Int(IntValue::U64(1))),
            Ok(NormParse::Value(VarNormalization::Population))
        ));
        assert!(parse_normalization(&Value::Int(IntValue::U64(u64::MAX))).is_err());
        assert!(parse_normalization(&Value::Int(IntValue::I64(-1))).is_err());
    }

    #[test]
    fn empty_axis_after_normalization_has_owner_specific_admission() {
        let input =
            Value::Tensor(Tensor::new((1..=12).map(f64::from).collect(), vec![3, 4]).unwrap());
        let empty = Value::Tensor(Tensor::new(Vec::<f64>::new(), vec![0, 0]).unwrap());
        let error = var_builtin(input.clone(), vec![empty.clone(), empty.clone()])
            .expect_err("var rejects an empty dimension vector");
        assert_eq!(error.identifier(), VAR_ERROR_INVALID_ARGUMENT.identifier);
        let std = block_on(super::super::std::std_builtin(
            input,
            crate::builtins::common::test_support::builtin_arguments(vec![empty.clone(), empty]),
        ))
        .expect("std accepts a default-axis placeholder");
        let Value::Tensor(std) = std else {
            panic!("expected row of standard deviations");
        };
        assert_eq!(std.shape(), [1, 4]);
        assert_eq!(std.materialize_f64(), vec![1.0; 4]);
    }

    #[test]
    fn var_typed_integer_tensor_normalization_reads_exact_storage() {
        let normalization =
            Tensor::new_integer(IntegerStorage::U16(vec![1]), vec![1, 1]).expect("normalization");

        assert!(matches!(
            parse_normalization(&Value::Tensor(normalization)),
            Ok(NormParse::Value(VarNormalization::Population))
        ));
    }

    #[test]
    fn var_typed_integer_controls_follow_compatibility_mode() {
        let tensor = Tensor::new(vec![1.0, 3.0], vec![2, 1]).expect("input");
        let args = vec![Value::Int(IntValue::U8(1))];
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(super::var_builtin(
                Value::Tensor(tensor.clone()),
                crate::builtins::common::test_support::builtin_arguments(args.clone()),
            ))
            .expect_err("MATLAB-compatible mode");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:VarTypedIntegerControlExtension")
            );
        }
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            block_on(super::var_builtin(
                Value::Tensor(tensor),
                crate::builtins::common::test_support::builtin_arguments(args),
            ))
            .expect("RunMat extension mode");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn var_scalar_num() {
        let result = var_builtin(Value::Num(6.0), Vec::new()).expect("var");
        assert_eq!(result, Value::Num(0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn host_single_output_boundary_preserves_class_across_scalar_empty_and_nd() {
        let scalar = Tensor::new_with_dtype(vec![7.0], vec![1, 1], runmat_value::NumericDType::F32)
            .expect("single scalar");
        let Value::Tensor(scalar_out) =
            var_builtin(Value::Tensor(scalar), Vec::new()).expect("single var scalar")
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
        let Value::Tensor(nd) = var_builtin(
            Value::Tensor(input),
            vec![Value::Num(0.0), Value::Tensor(axes)],
        )
        .expect("single N-D var") else {
            panic!("single N-D result must remain a tensor")
        };
        assert_eq!(nd.numeric_dtype(), runmat_value::NumericDType::F32);
        assert_eq!(nd.shape(), [1, 2, 1]);
        for value in values(&nd) {
            assert!((value - 17.0_f64 / 3.0).abs() < 1e-6);
        }

        let empty = Tensor::new_with_dtype(Vec::new(), vec![0, 2], runmat_value::NumericDType::F32)
            .expect("single empty input");
        let Value::Tensor(empty_out) =
            var_builtin(Value::Tensor(empty.clone()), Vec::new()).expect("single empty var")
        else {
            panic!("single empty result must remain a tensor")
        };
        assert_eq!(empty_out.numeric_dtype(), runmat_value::NumericDType::F32);
        assert_eq!(empty_out.shape(), [1, 2]);
        assert!(values(&empty_out).iter().all(|value| value.is_nan()));
        let Value::Tensor(empty_all) = var_builtin(Value::Tensor(empty), vec![Value::from("all")])
            .expect("single empty all var")
        else {
            panic!("single empty all result must remain a tensor")
        };
        assert_eq!(empty_all.numeric_dtype(), runmat_value::NumericDType::F32);
        assert_eq!(empty_all.shape(), [1, 1]);
        assert!(values(&empty_all)[0].is_nan());

        let beyond =
            Tensor::new_with_dtype(vec![1.0, 2.0], vec![1, 2], runmat_value::NumericDType::F32)
                .expect("single beyond-rank input");
        let Value::Tensor(beyond_out) = var_builtin(
            Value::Tensor(beyond),
            vec![Value::Num(0.0), Value::Num(4.0)],
        )
        .expect("single beyond-rank var") else {
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
        let Value::Tensor(population_out) = var_builtin(
            Value::Tensor(population),
            vec![Value::Num(1.0), Value::from("all")],
        )
        .expect("single population var") else {
            panic!("single population result must remain a tensor")
        };
        assert_eq!(
            population_out.numeric_dtype(),
            runmat_value::NumericDType::F32
        );
        assert_eq!(population_out.shape(), [1, 1]);
        assert!((values(&population_out)[0] - 1.25).abs() < 1e-6);

        assert_eq!(
            var_builtin(Value::Bool(true), Vec::new()).unwrap(),
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
        let Value::Tensor(output) = var_builtin(Value::Tensor(input), vec![Value::from("omitnan")])
            .expect("single omitnan var")
        else {
            panic!("single result must remain a tensor")
        };
        assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F32);
        assert_eq!(output.shape(), [1, 1]);
        assert!((values(&output)[0] - 2.0).abs() < 1e-6);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn var_matrix_default_dimension() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result = var_builtin(Value::Tensor(tensor), Vec::new()).expect("var");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [1, 3]);
                assert_eq!(values(&out), vec![4.5, 4.5, 4.5]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn var_matrix_population() {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let result =
            var_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(1))]).expect("var");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [1, 2]);
                assert_eq!(values(&out), vec![1.0, 1.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn var_rejects_every_integer_data_class_for_all_argument_forms() {
        let forms = [
            Vec::new(),
            vec![Value::Int(IntValue::I32(0)), Value::Int(IntValue::I32(2))],
            vec![Value::Int(IntValue::I32(0)), Value::from("all")],
            vec![Value::Int(IntValue::I32(0)), Value::from("omitmissing")],
        ];
        for input in integer_inputs() {
            for args in &forms {
                let err = var_builtin(input.clone(), args.clone()).expect_err("integer data");
                assert_eq!(err.identifier(), VAR_ERROR_INVALID_INPUT.identifier);
                assert!(err.message().contains("integer data inputs"));
            }
        }
        let err =
            var_builtin(Value::Int(IntValue::I16(42)), Vec::new()).expect_err("integer scalar");
        assert_eq!(err.identifier(), VAR_ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn var_accepts_typed_integer_population_control_exactly() {
        let tensor = Tensor::new(vec![0.0, 2.0, 4.0], vec![1, 3]).expect("tensor");
        let normalization =
            Tensor::new_integer(IntegerStorage::U16(vec![1]), vec![1, 1]).expect("normalization");

        let result =
            var_builtin(Value::Tensor(tensor), vec![Value::Tensor(normalization)]).expect("var");

        match result {
            Value::Num(value) => assert!((value - (8.0 / 3.0)).abs() < 1e-12),
            other => panic!("expected numeric result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn var_all_reduces_entire_tensor() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let result = var_builtin(Value::Tensor(tensor), vec![Value::from("all")]).expect("var");
        match result {
            Value::Num(v) => assert!((v - 1.666_666_666_666_666_7).abs() < 1e-12),
            other => panic!("expected scalar variance, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn var_repeated_axis_vector_reduces_once() {
        let input = Tensor::new((1..=12).map(f64::from).collect(), vec![3, 4]).unwrap();
        let axes = Tensor::new(vec![2.0, 2.0], vec![1, 2]).unwrap();
        let result = var_builtin(
            Value::Tensor(input),
            vec![Value::Num(0.0), Value::Tensor(axes)],
        )
        .expect("repeated axis vector");
        let Value::Tensor(output) = result else {
            panic!("expected column tensor");
        };
        assert_eq!(output.shape(), [3, 1]);
        for value in values(&output) {
            assert!((value - 15.0).abs() < 1e-12);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn var_vector_dimension_selection() {
        let tensor = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            vec![2, 3, 2],
        )
        .unwrap();
        let placeholder = Tensor::new(vec![], vec![0, 0]).unwrap();
        let vecdim = Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap();
        let args = vec![Value::Tensor(placeholder), Value::Tensor(vecdim)];
        let result = var_builtin(Value::Tensor(tensor), args).expect("var");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [1, 3, 1]);
                for value in values(&out) {
                    assert!((value - (37.0 / 3.0)).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn var_placeholder_flag_is_accepted() {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let placeholder = Tensor::new(vec![], vec![0, 0]).unwrap();
        let args = vec![Value::Tensor(placeholder), Value::Int(IntValue::I32(2))];
        let result = var_builtin(Value::Tensor(tensor), args).expect("var");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [2, 1]);
                for value in values(&out) {
                    assert!((value - 0.5).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn var_weighted_flag_not_supported() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let weights = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
        let err = var_builtin(Value::Tensor(tensor), vec![Value::Tensor(weights)])
            .expect_err("var should reject weighted inputs");
        assert_eq!(err.identifier(), VAR_ERROR_INVALID_ARGUMENT.identifier);
        assert!(err.message().contains("weighted variance"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn var_complex_identifier_is_stable() {
        let err = var_builtin(Value::Complex(1.0, 2.0), Vec::new()).unwrap_err();
        assert_eq!(err.identifier(), VAR_ERROR_COMPLEX_UNSUPPORTED.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn var_invalid_input_identifier_is_stable() {
        let err = var_builtin(Value::String("abc".to_string().into()), Vec::new()).unwrap_err();
        assert_eq!(err.identifier(), VAR_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn var_matrix_dimension_two() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result = var_builtin(
            Value::Tensor(tensor),
            vec![Value::Int(IntValue::I32(0)), Value::Int(IntValue::I32(2))],
        )
        .expect("var");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [2, 1]);
                assert_eq!(values(&out), vec![1.0, 1.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn var_with_omit_nan_default_dimension() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 5.0], vec![3, 1]).unwrap();
        let result = var_builtin(Value::Tensor(tensor), vec![Value::from("omitnan")]).expect("var");
        match result {
            Value::Num(v) => assert!((v - 8.0).abs() < 1e-12),
            other => panic!("expected scalar variance, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn var_with_include_nan_propagates() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 3.0], vec![3, 1]).unwrap();
        let result = var_builtin(Value::Tensor(tensor), Vec::new()).expect("var");
        match result {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected scalar NaN, got {other:?}"),
        }
    }

    #[test]
    fn var_missing_aliases_match_nan_aliases() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 5.0], vec![3, 1]).unwrap();
        let omit_nan =
            var_builtin(Value::Tensor(tensor.clone()), vec![Value::from("omitnan")]).unwrap();
        let omit_missing = var_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::from("omitmissing")],
        )
        .unwrap();
        assert_eq!(omit_nan, omit_missing);
        let include_nan = var_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::from("includenan")],
        )
        .unwrap();
        let include_missing =
            var_builtin(Value::Tensor(tensor), vec![Value::from("includemissing")]).unwrap();
        assert!(matches!(include_nan, Value::Num(value) if value.is_nan()));
        assert!(matches!(include_missing, Value::Num(value) if value.is_nan()));
    }

    #[test]
    fn var_typed_text_controls_preserve_owner_specific_invalid_option() {
        let parsed = block_on(parse_arguments(
            crate::call::FunctionArgumentView::from_slice(&[Value::from("ALL")]),
        ))
        .expect("case-insensitive all remains valid");
        assert!(matches!(parsed.axes, VarAxes::All));

        let spaced = block_on(parse_arguments(
            crate::call::FunctionArgumentView::from_slice(&[Value::from(" all ")]),
        ))
        .err()
        .expect("var does not accept spaces around all");
        assert_eq!(spaced.identifier(), VAR_ERROR_INVALID_ARGUMENT.identifier);
        assert!(spaced.message().contains("unrecognised option"));

        let conflict = block_on(parse_arguments(
            crate::call::FunctionArgumentView::from_slice(&[
                Value::Num(0.0),
                Value::Num(2.0),
                Value::from("all"),
            ]),
        ))
        .err()
        .expect("all after an explicit axis must conflict");
        assert_eq!(conflict.identifier(), VAR_ERROR_INVALID_ARGUMENT.identifier);
        assert!(conflict.message().contains("'all' cannot be combined"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn var_dimension_greater_than_ndims_returns_same_shaped_zeros() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let result = var_builtin(
            Value::Tensor(tensor),
            vec![Value::Int(IntValue::I32(0)), Value::Int(IntValue::I32(5))],
        )
        .expect("var");
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
    fn var_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let result = var_builtin(Value::GpuTensor(handle), Vec::new()).expect("var");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape(), [1, 3]);
            assert_eq!(values(&gathered), vec![4.5, 4.5, 4.5]);
        });
    }

    #[test]
    fn var_gpu_rejects_integer_data_before_provider_dispatch() {
        test_support::with_test_provider(|provider| {
            let handle = provider
                .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                    data: runmat_accelerate_api::HostIntegerDataView::U64(&[0, 2, u64::MAX]),
                    shape: &[3, 1],
                })
                .expect("upload integer");
            let err = var_builtin(
                Value::GpuTensor(handle.clone()),
                vec![Value::Int(IntValue::I32(0)), Value::from("all")],
            )
            .expect_err("integer GPU data");
            assert_eq!(err.identifier(), VAR_ERROR_INVALID_INPUT.identifier);
            assert!(err.message().contains("integer data inputs"));
            handle.release().unwrap();
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn var_gpu_omit_nan_falls_back_to_host() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![f64::NAN, 2.0, f64::NAN, 4.0], vec![2, 2]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let result =
                var_builtin(Value::GpuTensor(handle), vec![Value::from("omitnan")]).expect("var");
            let gathered = test_support::gather(result).expect("gather");
            let cpu = var_host(
                Value::Tensor(tensor),
                &ParsedArguments {
                    axes: VarAxes::Default,
                    normalization: VarNormalization::Sample,
                    nan_mode: ReductionNaN::Omit,
                },
            )
            .expect("cpu var");
            match cpu {
                Value::Tensor(expected) => {
                    assert_eq!(gathered.shape(), expected.shape());
                    for (a, b) in values(&gathered).iter().zip(values(&expected).iter()) {
                        if b.is_nan() {
                            assert!(a.is_nan());
                        } else {
                            assert!((a - b).abs() < 1e-12);
                        }
                    }
                }
                _ => panic!("expected tensor output from host path"),
            }
        });
    }
}
