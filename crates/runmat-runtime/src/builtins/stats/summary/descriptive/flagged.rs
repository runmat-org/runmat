//! Flagged deviation and standardized moment reductions.
use super::{descriptive_error, gathered, is_real_typed_integer_value};
use crate::builtins::common::{random_args::keyword_of, tensor};
use crate::BuiltinResult;
use runmat_builtins::{
    moment_flag_from_f64, BuiltinExtensionDescriptor, MomentControl, ReductionNaN as NanFlag,
    KURTOSIS_GPU_MULTI_AXIS_EXTENSION, KURTOSIS_INTEGER_CONTROL_EXTENSION,
    KURTOSIS_INTEGER_DATA_EXTENSION, MAD_INTEGER_DATA_EXTENSION,
    MAD_TYPED_INTEGER_CONTROL_EXTENSION, SKEWNESS_GPU_MULTI_AXIS_EXTENSION,
    SKEWNESS_INTEGER_CONTROL_EXTENSION, SKEWNESS_INTEGER_DATA_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, Tensor, Value};
use std::cmp::Ordering;

#[derive(Clone)]
enum Axes {
    Default,
    All,
    Dims(Vec<usize>),
}

#[derive(Clone)]
struct ReduceOptions {
    axes: Axes,
    nanflag: NanFlag,
}

async fn value_to_tensor(name: &str, value: Value) -> BuiltinResult<Tensor> {
    let gathered = gathered(value, name).await?;
    let tensor = tensor::value_into_tensor_for(name, gathered)
        .map_err(|err| descriptive_error(name, format!("{name}: {err}")))?;
    tensor::integer_tensor_to_f64(tensor)
        .map_err(|err| descriptive_error(name, format!("{name}: {err}")))
}

async fn gather_rest(
    name: &str,
    rest: crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<crate::call::BuiltinArguments> {
    rest.try_map_preserving_roots_async(rest.len(), runtime, |index| {
        gathered(rest[index].clone(), name)
    })
    .await
}

fn first_non_singleton(shape: &[usize]) -> usize {
    shape
        .iter()
        .position(|dim| *dim > 1)
        .map(|idx| idx + 1)
        .unwrap_or(1)
}

pub(super) fn scalar_flag(name: &str, value: &Value) -> BuiltinResult<Option<usize>> {
    let raw = match value {
        Value::Num(n) => *n,
        Value::Int(i) => i.to_f64(),
        Value::Bool(b) => {
            return Ok(Some(if *b { 1 } else { 0 }));
        }
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => tensor::tensor_value_f64(t, 0),
        _ => return Ok(None),
    };
    moment_flag_from_f64(raw)
        .map(|flag| Some(usize::from(flag)))
        .ok_or_else(|| descriptive_error(name, format!("{name}: flag must be 0 or 1")))
}

async fn dims_from_value(name: &str, value: &Value) -> BuiltinResult<Vec<usize>> {
    let Some(dims) = tensor::dims_from_value_async(value)
        .await
        .map_err(|err| descriptive_error(name, format!("{name}: {err}")))?
    else {
        return Err(descriptive_error(
            name,
            format!("{name}: expected dimension or dimension vector"),
        ));
    };
    if dims.is_empty() || dims.contains(&0) {
        return Err(descriptive_error(
            name,
            format!("{name}: dimensions must be positive integers"),
        ));
    }
    Ok(dims)
}

async fn parse_reduce_options(
    name: &str,
    rest: crate::call::FunctionArgumentView<'_>,
    default_nanflag: NanFlag,
) -> BuiltinResult<ReduceOptions> {
    let mut axes = Axes::Default;
    let mut nanflag = default_nanflag;
    for arg in rest.iter() {
        if let Some(keyword) = keyword_of(arg)? {
            match MomentControl::from_public_option(&keyword) {
                Some(MomentControl::All) => axes = Axes::All,
                Some(MomentControl::Missing(policy)) => nanflag = policy,
                None => {
                    return Err(descriptive_error(
                        name,
                        format!("{name}: unsupported option '{keyword}'"),
                    ));
                }
            }
        } else {
            axes = Axes::Dims(dims_from_value(name, arg).await?);
        }
    }
    Ok(ReduceOptions { axes, nanflag })
}

async fn parse_flagged_options(
    name: &str,
    rest: crate::call::BuiltinArguments,
    default_flag: usize,
    default_nanflag: NanFlag,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<(usize, ReduceOptions)> {
    let rest = gather_rest(name, rest, runtime).await?;
    let mut flag = default_flag;
    let mut start = 0usize;
    if let Some(first) = rest.first() {
        if matches!(first, Value::Tensor(tensor) if tensor.is_empty()) {
            // An empty first argument explicitly requests the default flag,
            // leaving the following argument available as a dimension.
            start = 1;
        } else if keyword_of(first)?.is_none() {
            if let Some(parsed) = scalar_flag(name, first)? {
                flag = parsed;
                start = 1;
            }
        }
    }
    let options = parse_reduce_options(name, rest.view().range(start..)?, default_nanflag).await?;
    Ok((flag, options))
}

fn normalize_shape(input: &Tensor) -> Vec<usize> {
    tensor::default_shape_for(input.shape(), tensor::tensor_element_len(input))
}

fn resolved_axes(shape: &[usize], axes: Axes) -> Vec<usize> {
    match axes {
        Axes::Default => vec![first_non_singleton(shape) - 1],
        Axes::All => (0..shape.len().max(2)).collect(),
        Axes::Dims(dims) => {
            let mut out: Vec<usize> = dims.into_iter().map(|dim| dim - 1).collect();
            out.sort_unstable();
            out.dedup();
            out
        }
    }
}

fn strides_for(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for idx in 1..shape.len() {
        strides[idx] = strides[idx - 1] * shape[idx - 1];
    }
    strides
}

fn reduce_tensor<F>(
    name: &str,
    input: Tensor,
    options: ReduceOptions,
    op: F,
) -> BuiltinResult<Value>
where
    F: Fn(&[f64]) -> f64,
{
    let shape = normalize_shape(&input);
    let output_dtype = match input.numeric_dtype() {
        NumericDType::F32 => NumericDType::F32,
        _ => NumericDType::F64,
    };
    // Dimensions beyond the physical rank are singleton reductions. Keep the
    // existing shape and evaluate the statistic on each singleton instead of
    // materializing rank-proportional metadata for a large dimension value.
    let axes: Vec<_> = resolved_axes(&shape, options.axes)
        .into_iter()
        .filter(|axis| *axis < shape.len())
        .collect();
    let out_shape = {
        let mut out_shape = shape.clone();
        for &axis in &axes {
            out_shape[axis] = 1;
        }
        out_shape
    };
    let out_len = tensor::element_count(&out_shape);
    let in_strides = strides_for(&shape);
    let out_strides = strides_for(&out_shape);
    let mut buckets = vec![Vec::<f64>::new(); out_len];
    for (linear, value) in tensor::tensor_into_values_f64(input)
        .into_iter()
        .enumerate()
    {
        let mut dst = 0usize;
        for dim in 0..shape.len() {
            let coord = (linear / in_strides[dim]) % shape[dim];
            if axes.binary_search(&dim).is_err() {
                dst += coord * out_strides[dim];
            }
        }
        buckets[dst].push(value);
    }
    let mut out = Vec::with_capacity(out_len);
    for mut values in buckets {
        if options.nanflag == NanFlag::Omit {
            values.retain(|value| !value.is_nan());
        }
        out.push(if values.is_empty() {
            f64::NAN
        } else {
            op(&values)
        });
    }
    Tensor::new_with_dtype(out, out_shape, output_dtype)
        .map(tensor::tensor_into_value)
        .map_err(|err| descriptive_error(name, format!("{name}: {err}")))
}

fn arithmetic_mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Greater));
    let n = values.len();
    if n == 0 {
        f64::NAN
    } else if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

fn mad_slice(values: &[f64], flag: usize) -> f64 {
    if values.iter().any(|value| value.is_nan()) {
        return f64::NAN;
    }
    if flag == 0 {
        let center = arithmetic_mean(values);
        arithmetic_mean(
            &values
                .iter()
                .map(|value| (value - center).abs())
                .collect::<Vec<_>>(),
        )
    } else {
        let center = median(values.to_vec());
        median(
            values
                .iter()
                .map(|value| (value - center).abs())
                .collect::<Vec<_>>(),
        )
    }
}

fn is_typed_integer_numeric_value(value: &Value) -> bool {
    is_real_typed_integer_value(value)
        || matches!(value, Value::ComplexTensor(tensor) if tensor.integer_storage().is_some())
}

fn ensure_floating_stat_integer_extensions(
    name: &str,
    value: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
    data_extension: &BuiltinExtensionDescriptor,
    control_extension: &BuiltinExtensionDescriptor,
) -> BuiltinResult<()> {
    if is_typed_integer_numeric_value(value) {
        crate::compatibility::ensure_builtin_extension_enabled(data_extension, name)?;
    }
    if rest.iter().any(is_real_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(control_extension, name)?;
    }
    Ok(())
}

fn rest_requests_multi_axis(rest: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<bool> {
    for value in rest.iter() {
        if keyword_of(value)?.is_some_and(|keyword| keyword == "all")
            || matches!(value, Value::Tensor(tensor) if tensor::tensor_element_len(tensor) > 1)
            || matches!(value, Value::GpuTensor(handle) if handle.shape().iter().any(|extent| *extent > 1))
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn ensure_gpu_multi_axis_extension(
    name: &str,
    value: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
    extension: &BuiltinExtensionDescriptor,
) -> BuiltinResult<()> {
    if matches!(value, Value::GpuTensor(_)) && rest_requests_multi_axis(rest)? {
        crate::compatibility::ensure_builtin_extension_enabled(extension, name)?;
    }
    Ok(())
}

fn ensure_mad_integer_extensions(
    value: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    if is_real_typed_integer_value(value) {
        crate::compatibility::ensure_builtin_extension_enabled(&MAD_INTEGER_DATA_EXTENSION, "mad")?;
    }
    if rest.iter().any(is_real_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &MAD_TYPED_INTEGER_CONTROL_EXTENSION,
            "mad",
        )?;
    }
    Ok(())
}

fn skewness_slice(values: &[f64], flag: usize) -> f64 {
    if values.iter().any(|value| value.is_nan()) {
        return f64::NAN;
    }
    let n = values.len();
    let mean = arithmetic_mean(values);
    let m2 = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / n as f64;
    if m2 == 0.0 {
        return f64::NAN;
    }
    let m3 = values
        .iter()
        .map(|value| (value - mean).powi(3))
        .sum::<f64>()
        / n as f64;
    let g1 = m3 / m2.powf(1.5);
    if flag == 1 {
        g1
    } else if n > 2 {
        ((n * (n - 1)) as f64).sqrt() / (n - 2) as f64 * g1
    } else {
        f64::NAN
    }
}

fn kurtosis_slice(values: &[f64], flag: usize) -> f64 {
    if values.iter().any(|value| value.is_nan()) {
        return f64::NAN;
    }
    let n = values.len();
    let mean = arithmetic_mean(values);
    let m2 = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / n as f64;
    if m2 == 0.0 {
        return f64::NAN;
    }
    let m4 = values
        .iter()
        .map(|value| (value - mean).powi(4))
        .sum::<f64>()
        / n as f64;
    let biased = m4 / (m2 * m2);
    if flag == 1 {
        biased
    } else if n > 3 {
        let excess = biased - 3.0;
        ((n - 1) as f64 / ((n - 2) * (n - 3)) as f64) * (((n + 1) as f64) * excess + 6.0) + 3.0
    } else {
        f64::NAN
    }
}

async fn flagged_reduce_builtin<F>(
    name: &str,
    value: Value,
    rest: crate::call::BuiltinArguments,
    default_flag: usize,
    default_nanflag: NanFlag,
    op: F,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Value>
where
    F: Fn(&[f64], usize) -> f64,
{
    let input = value_to_tensor(name, value).await?;
    let (flag, options) =
        parse_flagged_options(name, rest, default_flag, default_nanflag, runtime).await?;
    reduce_tensor(name, input, options, |slice| op(slice, flag))
}

pub mod mad {
    use super::*;
    #[cfg(test)]
    pub(crate) async fn mad_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let rest = crate::execution::RootedValues::new(rest)
            .map_err(|error| error.into_runtime_error())?;
        mad_builtin_registered(None, value, rest.into()).await
    }

    #[runtime_builtin(
        name = "mad",
        binding_variant = "default",
        context = true,
        builtin_path = "crate::builtins::stats::summary::descriptive::flagged::mad"
    )]
    pub(crate) async fn mad_builtin_registered(
        runtime: Option<crate::context::RuntimeContext>,
        value: Value,
        rest: crate::call::BuiltinArguments,
    ) -> BuiltinResult<Value> {
        super::ensure_mad_integer_extensions(&value, rest.view())?;
        super::flagged_reduce_builtin(
            "mad",
            value,
            rest,
            0,
            super::NanFlag::Omit,
            super::mad_slice,
            runtime.as_ref(),
        )
        .await
    }
}

pub mod skewness {
    use super::*;
    #[cfg(test)]
    pub(crate) async fn skewness_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let rest = crate::execution::RootedValues::new(rest)
            .map_err(|error| error.into_runtime_error())?;
        skewness_builtin_registered(None, value, rest.into()).await
    }

    #[runtime_builtin(
        name = "skewness",
        binding_variant = "default",
        context = true,
        builtin_path = "crate::builtins::stats::summary::descriptive::flagged::skewness"
    )]
    pub(crate) async fn skewness_builtin_registered(
        runtime: Option<crate::context::RuntimeContext>,
        value: Value,
        rest: crate::call::BuiltinArguments,
    ) -> BuiltinResult<Value> {
        super::ensure_floating_stat_integer_extensions(
            "skewness",
            &value,
            rest.view(),
            &SKEWNESS_INTEGER_DATA_EXTENSION,
            &SKEWNESS_INTEGER_CONTROL_EXTENSION,
        )?;
        super::ensure_gpu_multi_axis_extension(
            "skewness",
            &value,
            rest.view(),
            &SKEWNESS_GPU_MULTI_AXIS_EXTENSION,
        )?;
        super::flagged_reduce_builtin(
            "skewness",
            value,
            rest,
            1,
            super::NanFlag::Omit,
            super::skewness_slice,
            runtime.as_ref(),
        )
        .await
    }
}

pub mod kurtosis {
    use super::*;
    #[cfg(test)]
    pub(crate) async fn kurtosis_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let rest = crate::execution::RootedValues::new(rest)
            .map_err(|error| error.into_runtime_error())?;
        kurtosis_builtin_registered(None, value, rest.into()).await
    }

    #[runtime_builtin(
        name = "kurtosis",
        binding_variant = "default",
        context = true,
        builtin_path = "crate::builtins::stats::summary::descriptive::flagged::kurtosis"
    )]
    pub(crate) async fn kurtosis_builtin_registered(
        runtime: Option<crate::context::RuntimeContext>,
        value: Value,
        rest: crate::call::BuiltinArguments,
    ) -> BuiltinResult<Value> {
        super::ensure_floating_stat_integer_extensions(
            "kurtosis",
            &value,
            rest.view(),
            &KURTOSIS_INTEGER_DATA_EXTENSION,
            &KURTOSIS_INTEGER_CONTROL_EXTENSION,
        )?;
        super::ensure_gpu_multi_axis_extension(
            "kurtosis",
            &value,
            rest.view(),
            &KURTOSIS_GPU_MULTI_AXIS_EXTENSION,
        )?;
        super::flagged_reduce_builtin(
            "kurtosis",
            value,
            rest,
            1,
            super::NanFlag::Omit,
            super::kurtosis_slice,
            runtime.as_ref(),
        )
        .await
    }
}
