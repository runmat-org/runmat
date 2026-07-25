//! Descriptive statistics compatibility helpers.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, ResolveContext, StringArray, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::broadcast::BroadcastPlan;
use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const PARAM_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input data array.",
};

const PARAM_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Second input data array.",
};

const PARAM_FLAG: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "flag",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Normalization or deviation mode flag.",
};

const PARAM_DIM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "dim",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Dimension, dimension vector, or \"all\" selector.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Optional dimension and missing-value arguments.",
};

const PARAM_WEIGHT_OPTION: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "optionName",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"Weights\""),
    description: "Name-value option name.",
};

const PARAM_WEIGHTS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "W",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Observation weights.",
};

const OUTPUT_Y: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Computed statistic.",
}];

const OUTPUT_TABLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tbl",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Frequency table.",
}];

const INPUTS_X: [BuiltinParamDescriptor; 1] = [PARAM_X];
const INPUTS_X_DIM: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_DIM];
const INPUTS_X_OPTIONS: [BuiltinParamDescriptor; 3] = [PARAM_X, PARAM_DIM, PARAM_OPTIONS];
const INPUTS_X_FLAG: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_FLAG];
const INPUTS_X_FLAG_OPTIONS: [BuiltinParamDescriptor; 4] =
    [PARAM_X, PARAM_FLAG, PARAM_DIM, PARAM_OPTIONS];
const INPUTS_X_Y: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_Y];
const INPUTS_X_Y_DIM: [BuiltinParamDescriptor; 3] = [PARAM_X, PARAM_Y, PARAM_DIM];
const INPUTS_X_Y_OPTIONS: [BuiltinParamDescriptor; 4] =
    [PARAM_X, PARAM_Y, PARAM_DIM, PARAM_OPTIONS];
const INPUTS_X_Y_WEIGHTS: [BuiltinParamDescriptor; 4] =
    [PARAM_X, PARAM_Y, PARAM_WEIGHT_OPTION, PARAM_WEIGHTS];
const INPUTS_X_Y_DIM_WEIGHTS: [BuiltinParamDescriptor; 6] = [
    PARAM_X,
    PARAM_Y,
    PARAM_DIM,
    PARAM_OPTIONS,
    PARAM_WEIGHT_OPTION,
    PARAM_WEIGHTS,
];

macro_rules! reduction_signatures {
    ($const_name:ident, $name:literal) => {
        const $const_name: [BuiltinSignatureDescriptor; 3] = [
            BuiltinSignatureDescriptor {
                label: concat!("Y = ", $name, "(X)"),
                inputs: &INPUTS_X,
                outputs: &OUTPUT_Y,
            },
            BuiltinSignatureDescriptor {
                label: concat!("Y = ", $name, "(X, dim)"),
                inputs: &INPUTS_X_DIM,
                outputs: &OUTPUT_Y,
            },
            BuiltinSignatureDescriptor {
                label: concat!("Y = ", $name, "(X, dim, nanflag)"),
                inputs: &INPUTS_X_OPTIONS,
                outputs: &OUTPUT_Y,
            },
        ];
    };
}

macro_rules! flagged_signatures {
    ($const_name:ident, $name:literal) => {
        const $const_name: [BuiltinSignatureDescriptor; 3] = [
            BuiltinSignatureDescriptor {
                label: concat!("Y = ", $name, "(X)"),
                inputs: &INPUTS_X,
                outputs: &OUTPUT_Y,
            },
            BuiltinSignatureDescriptor {
                label: concat!("Y = ", $name, "(X, flag)"),
                inputs: &INPUTS_X_FLAG,
                outputs: &OUTPUT_Y,
            },
            BuiltinSignatureDescriptor {
                label: concat!("Y = ", $name, "(X, flag, dim)"),
                inputs: &INPUTS_X_FLAG_OPTIONS,
                outputs: &OUTPUT_Y,
            },
        ];
    };
}

reduction_signatures!(GEOMEAN_SIGNATURES, "geomean");
reduction_signatures!(HARMMEAN_SIGNATURES, "harmmean");
reduction_signatures!(RMS_SIGNATURES, "rms");
flagged_signatures!(MAD_SIGNATURES, "mad");
flagged_signatures!(SKEWNESS_SIGNATURES, "skewness");
flagged_signatures!(KURTOSIS_SIGNATURES, "kurtosis");

const RMSE_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "E = rmse(X, Y)",
        inputs: &INPUTS_X_Y,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "E = rmse(X, Y, \"Weights\", W)",
        inputs: &INPUTS_X_Y_WEIGHTS,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "E = rmse(X, Y, dim)",
        inputs: &INPUTS_X_Y_DIM,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "E = rmse(X, Y, dim, nanflag)",
        inputs: &INPUTS_X_Y_OPTIONS,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "E = rmse(X, Y, dim, nanflag, \"Weights\", W)",
        inputs: &INPUTS_X_Y_DIM_WEIGHTS,
        outputs: &OUTPUT_Y,
    },
];

const TABULATE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tbl = tabulate(X)",
    inputs: &INPUTS_X,
    outputs: &OUTPUT_TABLE,
}];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DESCRIPTIVE.INVALID_ARGUMENT",
    identifier: None,
    when: "Inputs, flags, dimensions, or options are malformed.",
    message: "descriptive statistics: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DESCRIPTIVE.INTERNAL",
    identifier: None,
    when: "Internal tensor conversion or allocation fails.",
    message: "descriptive statistics: internal error",
};

macro_rules! descriptive_descriptor {
    ($name:literal, $signatures:expr) => {
        const ERRORS: [BuiltinErrorDescriptor; 2] = [
            BuiltinErrorDescriptor {
                code: concat!("RM.", $name, ".INVALID_ARGUMENT"),
                identifier: Some(concat!("RunMat:", $name, ":InvalidArgument")),
                when: ERROR_INVALID_ARGUMENT.when,
                message: ERROR_INVALID_ARGUMENT.message,
            },
            BuiltinErrorDescriptor {
                code: concat!("RM.", $name, ".INTERNAL"),
                identifier: Some(concat!("RunMat:", $name, ":Internal")),
                when: ERROR_INTERNAL.when,
                message: ERROR_INTERNAL.message,
            },
        ];

        pub const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
    };
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum NanFlag {
    Include,
    Omit,
}

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

fn descriptive_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

fn descriptive_error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

async fn gathered(value: Value, name: &str) -> BuiltinResult<Value> {
    gather_if_needed_async(&value)
        .await
        .map_err(|err| descriptive_error(name, format!("{name}: {err}")))
}

async fn value_to_tensor(name: &str, value: Value) -> BuiltinResult<Tensor> {
    let gathered = gathered(value, name).await?;
    tensor::value_into_tensor_for(name, gathered)
        .map_err(|err| descriptive_error(name, format!("{name}: {err}")))
}

async fn value_to_magnitude_tensor(name: &str, value: Value) -> BuiltinResult<Tensor> {
    let gathered = gathered(value, name).await?;
    match gathered {
        Value::Complex(re, im) => Tensor::new(vec![re.hypot(im)], vec![1, 1])
            .map_err(|err| descriptive_error(name, format!("{name}: {err}"))),
        Value::ComplexTensor(tensor) => {
            let data = tensor
                .data
                .into_iter()
                .map(|(re, im)| re.hypot(im))
                .collect::<Vec<_>>();
            Tensor::new(data, tensor.shape)
                .map_err(|err| descriptive_error(name, format!("{name}: {err}")))
        }
        other => tensor::value_into_tensor_for(name, other)
            .map_err(|err| descriptive_error(name, format!("{name}: {err}"))),
    }
}

async fn gather_rest(name: &str, rest: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(rest.len());
    for value in rest {
        out.push(gathered(value, name).await?);
    }
    Ok(out)
}

fn first_non_singleton(shape: &[usize]) -> usize {
    shape
        .iter()
        .position(|dim| *dim > 1)
        .map(|idx| idx + 1)
        .unwrap_or(1)
}

fn scalar_flag(name: &str, value: &Value) -> BuiltinResult<Option<usize>> {
    let raw = match value {
        Value::Num(n) => *n,
        Value::Int(i) => i.to_f64(),
        Value::Bool(b) => {
            return Ok(Some(if *b { 1 } else { 0 }));
        }
        Value::Tensor(t) if t.data.len() == 1 => t
            .integer_storage()
            .and_then(|storage| storage.value_at(0))
            .map_or(t.data[0], |int| int.to_f64()),
        _ => return Ok(None),
    };
    if !raw.is_finite() || raw.fract().abs() > 1e-12 || !(0.0..=1.0).contains(&raw) {
        return Err(descriptive_error(
            name,
            format!("{name}: flag must be 0 or 1"),
        ));
    }
    Ok(Some(raw as usize))
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
    rest: Vec<Value>,
    default_nanflag: NanFlag,
) -> BuiltinResult<ReduceOptions> {
    let rest = gather_rest(name, rest).await?;
    let mut axes = Axes::Default;
    let mut nanflag = default_nanflag;
    for arg in rest {
        if let Some(keyword) = keyword_of(&arg) {
            match keyword.as_str() {
                "all" => axes = Axes::All,
                "includenan" | "includemissing" => nanflag = NanFlag::Include,
                "omitnan" | "omitmissing" => nanflag = NanFlag::Omit,
                other => {
                    return Err(descriptive_error(
                        name,
                        format!("{name}: unsupported option '{other}'"),
                    ));
                }
            }
        } else {
            axes = Axes::Dims(dims_from_value(name, &arg).await?);
        }
    }
    Ok(ReduceOptions { axes, nanflag })
}

async fn parse_reduce_options_with_weights(
    name: &str,
    rest: Vec<Value>,
    default_nanflag: NanFlag,
) -> BuiltinResult<(ReduceOptions, Option<Tensor>)> {
    let rest = gather_rest(name, rest).await?;
    let mut reduce_args = Vec::new();
    let mut weights = None;
    let mut idx = 0usize;
    while idx < rest.len() {
        if let Some(keyword) = keyword_of(&rest[idx]) {
            if keyword == "weights" {
                idx += 1;
                if idx >= rest.len() {
                    return Err(descriptive_error(
                        name,
                        format!("{name}: Weights option requires a value"),
                    ));
                }
                let tensor = tensor::value_into_tensor_for(name, rest[idx].clone())
                    .map_err(|err| descriptive_error(name, format!("{name}: {err}")))?;
                weights = Some(tensor);
                idx += 1;
                continue;
            }
        }
        reduce_args.push(rest[idx].clone());
        idx += 1;
    }
    let options = parse_reduce_options(name, reduce_args, default_nanflag).await?;
    Ok((options, weights))
}

async fn parse_flagged_options(
    name: &str,
    rest: Vec<Value>,
    default_flag: usize,
    default_nanflag: NanFlag,
) -> BuiltinResult<(usize, ReduceOptions)> {
    let rest = gather_rest(name, rest).await?;
    let mut flag = default_flag;
    let mut start = 0usize;
    if let Some(first) = rest.first() {
        if keyword_of(first).is_none() {
            if let Some(parsed) = scalar_flag(name, first)? {
                flag = parsed;
                start = 1;
            }
        }
    }
    let options = parse_reduce_options(
        name,
        rest.into_iter().skip(start).collect(),
        default_nanflag,
    )
    .await?;
    Ok((flag, options))
}

fn normalize_shape(input: &Tensor) -> Vec<usize> {
    tensor::default_shape_for(&input.shape, input.data.len())
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
    let axes = resolved_axes(&shape, options.axes);
    if axes.is_empty() {
        return Tensor::new(input.data, shape)
            .map(tensor::tensor_into_value)
            .map_err(|err| descriptive_error(name, format!("{name}: {err}")));
    }
    let rank = shape.len().max(axes.iter().copied().max().unwrap_or(0) + 1);
    let mut padded_shape = shape;
    padded_shape.resize(rank, 1);
    let mut out_shape = padded_shape.clone();
    for &axis in &axes {
        out_shape[axis] = 1;
    }
    let out_len = tensor::element_count(&out_shape);
    let in_strides = strides_for(&padded_shape);
    let out_strides = strides_for(&out_shape);
    let mut buckets = vec![Vec::<f64>::new(); out_len.max(1)];
    for (linear, value) in input.data.into_iter().enumerate() {
        let mut dst = 0usize;
        for dim in 0..rank {
            let coord = (linear / in_strides[dim]) % padded_shape[dim];
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
    Tensor::new(out, out_shape)
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

fn geomean_slice(values: &[f64]) -> f64 {
    if values.iter().any(|value| value.is_nan() || *value < 0.0) {
        return f64::NAN;
    }
    if values.contains(&0.0) {
        return 0.0;
    }
    (values.iter().map(|value| value.ln()).sum::<f64>() / values.len() as f64).exp()
}

fn harmmean_slice(values: &[f64]) -> f64 {
    if values.iter().any(|value| value.is_nan()) {
        return f64::NAN;
    }
    if values.contains(&0.0) {
        return 0.0;
    }
    values.len() as f64 / values.iter().map(|value| 1.0 / value).sum::<f64>()
}

fn rms_slice(values: &[f64]) -> f64 {
    (values.iter().map(|value| value * value).sum::<f64>() / values.len() as f64).sqrt()
}

struct ComplexBuffer {
    data: Vec<(f64, f64)>,
    shape: Vec<usize>,
}

async fn value_to_complex_buffer(name: &str, value: Value) -> BuiltinResult<ComplexBuffer> {
    let gathered = gathered(value, name).await?;
    match gathered {
        Value::Complex(re, im) => Ok(ComplexBuffer {
            data: vec![(re, im)],
            shape: vec![1, 1],
        }),
        Value::ComplexTensor(tensor) => {
            let shape = tensor::default_shape_for(&tensor.shape, tensor.data.len());
            Ok(ComplexBuffer {
                data: tensor.data,
                shape,
            })
        }
        other => {
            let tensor = tensor::value_into_tensor_for(name, other)
                .map_err(|err| descriptive_error(name, format!("{name}: {err}")))?;
            Ok(ComplexBuffer {
                shape: tensor::default_shape_for(&tensor.shape, tensor.data.len()),
                data: tensor.data.into_iter().map(|value| (value, 0.0)).collect(),
            })
        }
    }
}

fn residual_magnitudes(
    name: &str,
    lhs: ComplexBuffer,
    rhs: ComplexBuffer,
) -> BuiltinResult<Tensor> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(|err| descriptive_error(name, format!("{name}: {err}")))?;
    let mut out = vec![0.0; plan.len()];
    for (dst, lhs_idx, rhs_idx) in plan.iter() {
        let (lhs_re, lhs_im) = lhs.data[lhs_idx];
        let (rhs_re, rhs_im) = rhs.data[rhs_idx];
        out[dst] = (lhs_re - rhs_re).hypot(lhs_im - rhs_im);
    }
    Tensor::new(out, plan.output_shape().to_vec())
        .map_err(|err| descriptive_error(name, format!("{name}: {err}")))
}

enum WeightSpec {
    Scalar(f64),
    Elementwise(Vec<f64>),
    Axis { axis: usize, weights: Vec<f64> },
}

fn weight_spec_for(
    name: &str,
    weights: Tensor,
    input_shape: &[usize],
    axes: &[usize],
) -> BuiltinResult<WeightSpec> {
    if weights.data.iter().any(|value| *value < 0.0) {
        return Err(descriptive_error(
            name,
            format!("{name}: weights must be nonnegative"),
        ));
    }
    if weights.data.len() == 1 {
        return Ok(WeightSpec::Scalar(weights.data[0]));
    }
    if weights.data.len() == tensor::element_count(input_shape) {
        return Ok(WeightSpec::Elementwise(weights.data));
    }
    if axes.len() == 1 && weights.data.len() == input_shape[axes[0]] {
        return Ok(WeightSpec::Axis {
            axis: axes[0],
            weights: weights.data,
        });
    }
    Err(descriptive_error(
        name,
        format!("{name}: weights must be scalar, match the input size, or match the reduction dimension"),
    ))
}

fn weighted_rmse_tensor(
    name: &str,
    input: Tensor,
    options: ReduceOptions,
    weights: Tensor,
) -> BuiltinResult<Value> {
    let shape = normalize_shape(&input);
    let axes = resolved_axes(&shape, options.axes.clone());
    match &options.axes {
        Axes::All => {
            return Err(descriptive_error(
                name,
                format!("{name}: weights are not supported with \"all\""),
            ));
        }
        Axes::Dims(dims) if dims.len() != 1 => {
            return Err(descriptive_error(
                name,
                format!("{name}: weights are not supported with vector dimensions"),
            ));
        }
        _ => {}
    }
    let rank = shape.len().max(axes.iter().copied().max().unwrap_or(0) + 1);
    let mut padded_shape = shape;
    padded_shape.resize(rank, 1);
    let mut out_shape = padded_shape.clone();
    for &axis in &axes {
        out_shape[axis] = 1;
    }
    let out_len = tensor::element_count(&out_shape);
    let in_strides = strides_for(&padded_shape);
    let out_strides = strides_for(&out_shape);
    let weight_spec = weight_spec_for(name, weights, &padded_shape, &axes)?;
    let mut buckets = vec![Vec::<(f64, f64)>::new(); out_len.max(1)];
    for (linear, value) in input.data.into_iter().enumerate() {
        let mut dst = 0usize;
        let mut axis_coord = 0usize;
        for dim in 0..rank {
            let coord = (linear / in_strides[dim]) % padded_shape[dim];
            if axes.binary_search(&dim).is_err() {
                dst += coord * out_strides[dim];
            }
            if let WeightSpec::Axis { axis, .. } = &weight_spec {
                if dim == *axis {
                    axis_coord = coord;
                }
            }
        }
        let weight = match &weight_spec {
            WeightSpec::Scalar(weight) => *weight,
            WeightSpec::Elementwise(weights) => weights[linear],
            WeightSpec::Axis { weights, .. } => weights[axis_coord],
        };
        buckets[dst].push((value, weight));
    }
    let mut out = Vec::with_capacity(out_len);
    for bucket in buckets {
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        let mut saw_nan = false;
        for (value, weight) in bucket {
            if value.is_nan() || weight.is_nan() {
                if options.nanflag == NanFlag::Omit {
                    continue;
                }
                saw_nan = true;
                break;
            }
            numerator += weight * value * value;
            denominator += weight;
        }
        out.push(if saw_nan || denominator == 0.0 {
            f64::NAN
        } else {
            (numerator / denominator).sqrt()
        });
    }
    Tensor::new(out, out_shape)
        .map(tensor::tensor_into_value)
        .map_err(|err| descriptive_error(name, format!("{name}: {err}")))
}

fn mad_slice(values: &[f64], flag: usize) -> f64 {
    if values.iter().any(|value| value.is_nan()) {
        return f64::NAN;
    }
    if flag == 0 {
        let center = median(values.to_vec());
        median(
            values
                .iter()
                .map(|value| (value - center).abs())
                .collect::<Vec<_>>(),
        )
    } else {
        let center = arithmetic_mean(values);
        arithmetic_mean(
            &values
                .iter()
                .map(|value| (value - center).abs())
                .collect::<Vec<_>>(),
        )
    }
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

async fn reduce_builtin<F>(
    name: &str,
    value: Value,
    rest: Vec<Value>,
    default_nanflag: NanFlag,
    op: F,
) -> BuiltinResult<Value>
where
    F: Fn(&[f64]) -> f64,
{
    let input = value_to_tensor(name, value).await?;
    let options = parse_reduce_options(name, rest, default_nanflag).await?;
    reduce_tensor(name, input, options, op)
}

async fn flagged_reduce_builtin<F>(
    name: &str,
    value: Value,
    rest: Vec<Value>,
    default_flag: usize,
    default_nanflag: NanFlag,
    op: F,
) -> BuiltinResult<Value>
where
    F: Fn(&[f64], usize) -> f64,
{
    let input = value_to_tensor(name, value).await?;
    let (flag, options) = parse_flagged_options(name, rest, default_flag, default_nanflag).await?;
    reduce_tensor(name, input, options, |slice| op(slice, flag))
}

fn numeric_tabulate(name: &str, tensor: Tensor) -> BuiltinResult<Value> {
    let mut values = tensor
        .data
        .into_iter()
        .filter(|value| !value.is_nan())
        .collect::<Vec<_>>();
    let n = values.len() as f64;
    values.sort_by(|a, b| match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
    });
    let positive_integers = !values.is_empty()
        && values
            .iter()
            .all(|value| value.is_finite() && *value >= 1.0 && value.fract().abs() <= 1e-12);
    let mut rows = Vec::<(f64, f64)>::new();
    if positive_integers {
        let max_value = values.last().copied().unwrap_or(0.0).round().max(0.0) as usize;
        let mut counts = vec![0usize; max_value + 1];
        for value in values {
            counts[value.round() as usize] += 1;
        }
        rows.extend((1..=max_value).map(|idx| (idx as f64, counts[idx] as f64)));
    } else {
        let mut idx = 0usize;
        while idx < values.len() {
            let value = values[idx];
            let mut end = idx + 1;
            while end < values.len()
                && ((value.is_nan() && values[end].is_nan()) || values[end] == value)
            {
                end += 1;
            }
            rows.push((value, (end - idx) as f64));
            idx = end;
        }
    }
    let rows_len = rows.len();
    let mut data = Vec::with_capacity(rows_len * 3);
    data.extend(rows.iter().map(|(value, _)| *value));
    data.extend(rows.iter().map(|(_, count)| *count));
    data.extend(rows.iter().map(|(_, count)| {
        if n == 0.0 {
            f64::NAN
        } else {
            *count * 100.0 / n
        }
    }));
    Tensor::new(data, vec![rows_len, 3])
        .map(tensor::tensor_into_value)
        .map_err(|err| descriptive_error(name, format!("{name}: {err}")))
}

fn logical_tabulate(name: &str, data: Vec<u8>) -> BuiltinResult<Value> {
    let false_count = data.iter().filter(|value| **value == 0).count();
    let true_count = data.len() - false_count;
    let total = data.len() as f64;
    let mut rows = Vec::new();
    if false_count > 0 {
        rows.push((false, false_count));
    }
    if true_count > 0 {
        rows.push((true, true_count));
    }
    let mut cell_data = Vec::with_capacity(rows.len() * 3);
    for (value, count) in &rows {
        cell_data.push(Value::Bool(*value));
        cell_data.push(Value::Num(*count as f64));
        cell_data.push(Value::Num(if total == 0.0 {
            f64::NAN
        } else {
            *count as f64 * 100.0 / total
        }));
    }
    CellArray::new(cell_data, rows.len(), 3)
        .map(Value::Cell)
        .map_err(|err| descriptive_error(name, format!("{name}: {err}")))
}

fn label_for_value(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        Value::CharArray(chars) if chars.rows == 1 => Some(chars.data.iter().collect()),
        Value::Num(num) => Some(num.to_string()),
        Value::Int(int) => Some(int.decimal_string()),
        Value::Bool(flag) => Some(flag.to_string()),
        _ => None,
    }
}

fn string_tabulate(name: &str, labels: impl IntoIterator<Item = String>) -> BuiltinResult<Value> {
    let mut counts = BTreeMap::<String, usize>::new();
    let mut total = 0usize;
    for label in labels {
        *counts.entry(label).or_insert(0) += 1;
        total += 1;
    }
    let rows = counts.len();
    let mut data = Vec::with_capacity(rows * 3);
    for (label, count) in counts {
        data.push(Value::StringArray(
            StringArray::new(vec![label], vec![1, 1])
                .map_err(|err| descriptive_error(name, format!("{name}: {err}")))?,
        ));
        data.push(Value::Num(count as f64));
        data.push(Value::Num(if total == 0 {
            f64::NAN
        } else {
            count as f64 * 100.0 / total as f64
        }));
    }
    CellArray::new(data, rows, 3)
        .map(Value::Cell)
        .map_err(|err| descriptive_error(name, format!("{name}: {err}")))
}

fn tabulate_value(name: &str, value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) => numeric_tabulate(name, tensor),
        Value::LogicalArray(logical) => logical_tabulate(name, logical.data),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            tensor::value_into_tensor_for(name, value)
                .map_err(|err| descriptive_error(name, format!("{name}: {err}")))
                .and_then(|tensor| numeric_tabulate(name, tensor))
        }
        Value::String(text) => string_tabulate(name, [text]),
        Value::StringArray(array) => string_tabulate(name, array.data),
        Value::CharArray(chars) if chars.rows == 1 => {
            string_tabulate(name, chars.data.into_iter().map(|ch| ch.to_string()))
        }
        Value::Cell(cell) => {
            let mut labels = Vec::with_capacity(cell.data.len());
            for entry in &cell.data {
                let Some(label) = label_for_value(entry) else {
                    return Err(descriptive_error(
                        name,
                        format!("{name}: cell entries must be scalar strings or scalars"),
                    ));
                };
                labels.push(label);
            }
            string_tabulate(name, labels)
        }
        other => Err(descriptive_error(
            name,
            format!("{name}: unsupported input type {other:?}"),
        )),
    }
}

pub mod geomean {
    use super::*;
    descriptive_descriptor!("geomean", GEOMEAN_SIGNATURES);

    #[runtime_builtin(
        name = "geomean",
        category = "stats/summary",
        summary = "Compute geometric mean.",
        keywords = "geomean,geometric mean,statistics,summary",
        type_resolver(super::descriptive_type),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::stats::summary::descriptive::geomean"
    )]
    pub(crate) async fn geomean_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        super::reduce_builtin(
            "geomean",
            value,
            rest,
            super::NanFlag::Include,
            super::geomean_slice,
        )
        .await
    }
}

pub mod harmmean {
    use super::*;
    descriptive_descriptor!("harmmean", HARMMEAN_SIGNATURES);

    #[runtime_builtin(
        name = "harmmean",
        category = "stats/summary",
        summary = "Compute harmonic mean.",
        keywords = "harmmean,harmonic mean,statistics,summary",
        type_resolver(super::descriptive_type),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::stats::summary::descriptive::harmmean"
    )]
    pub(crate) async fn harmmean_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        super::reduce_builtin(
            "harmmean",
            value,
            rest,
            super::NanFlag::Include,
            super::harmmean_slice,
        )
        .await
    }
}

pub mod rms {
    use super::*;
    descriptive_descriptor!("rms", RMS_SIGNATURES);

    #[runtime_builtin(
        name = "rms",
        category = "math/reduction",
        summary = "Compute root mean square.",
        keywords = "rms,root mean square,statistics,signal",
        type_resolver(super::descriptive_type),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::stats::summary::descriptive::rms"
    )]
    pub(crate) async fn rms_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let input = super::value_to_magnitude_tensor("rms", value).await?;
        let options = super::parse_reduce_options("rms", rest, super::NanFlag::Include).await?;
        super::reduce_tensor("rms", input, options, super::rms_slice)
    }
}

pub mod mad {
    use super::*;
    descriptive_descriptor!("mad", MAD_SIGNATURES);

    #[runtime_builtin(
        name = "mad",
        category = "stats/summary",
        summary = "Compute mean or median absolute deviation.",
        keywords = "mad,mean absolute deviation,median absolute deviation,statistics",
        type_resolver(super::descriptive_type),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::stats::summary::descriptive::mad"
    )]
    pub(crate) async fn mad_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        super::flagged_reduce_builtin(
            "mad",
            value,
            rest,
            0,
            super::NanFlag::Omit,
            super::mad_slice,
        )
        .await
    }
}

pub mod skewness {
    use super::*;
    descriptive_descriptor!("skewness", SKEWNESS_SIGNATURES);

    #[runtime_builtin(
        name = "skewness",
        category = "stats/summary",
        summary = "Compute sample skewness.",
        keywords = "skewness,third moment,statistics,summary",
        type_resolver(super::descriptive_type),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::stats::summary::descriptive::skewness"
    )]
    pub(crate) async fn skewness_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        super::flagged_reduce_builtin(
            "skewness",
            value,
            rest,
            1,
            super::NanFlag::Omit,
            super::skewness_slice,
        )
        .await
    }
}

pub mod kurtosis {
    use super::*;
    descriptive_descriptor!("kurtosis", KURTOSIS_SIGNATURES);

    #[runtime_builtin(
        name = "kurtosis",
        category = "stats/summary",
        summary = "Compute sample kurtosis.",
        keywords = "kurtosis,fourth moment,statistics,summary",
        type_resolver(super::descriptive_type),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::stats::summary::descriptive::kurtosis"
    )]
    pub(crate) async fn kurtosis_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        super::flagged_reduce_builtin(
            "kurtosis",
            value,
            rest,
            1,
            super::NanFlag::Omit,
            super::kurtosis_slice,
        )
        .await
    }
}

pub mod rmse {
    use super::*;
    descriptive_descriptor!("rmse", RMSE_SIGNATURES);

    #[runtime_builtin(
        name = "rmse",
        category = "stats/summary",
        summary = "Compute root mean squared error.",
        keywords = "rmse,root mean squared error,error,statistics",
        type_resolver(super::descriptive_type),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::stats::summary::descriptive::rmse"
    )]
    pub(crate) async fn rmse_builtin(
        lhs: Value,
        rhs: Value,
        rest: Vec<Value>,
    ) -> BuiltinResult<Value> {
        let lhs = super::value_to_complex_buffer("rmse", lhs).await?;
        let rhs = super::value_to_complex_buffer("rmse", rhs).await?;
        let input = super::residual_magnitudes("rmse", lhs, rhs)?;
        let (options, weights) =
            super::parse_reduce_options_with_weights("rmse", rest, super::NanFlag::Include).await?;
        if let Some(weights) = weights {
            super::weighted_rmse_tensor("rmse", input, options, weights)
        } else {
            super::reduce_tensor("rmse", input, options, super::rms_slice)
        }
    }
}

pub mod tabulate {
    use super::*;
    descriptive_descriptor!("tabulate", TABULATE_SIGNATURES);

    #[runtime_builtin(
        name = "tabulate",
        category = "stats/summary",
        summary = "Create a frequency table.",
        keywords = "tabulate,frequency,count,percent,statistics",
        type_resolver(super::descriptive_type),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::stats::summary::descriptive::tabulate"
    )]
    pub(crate) async fn tabulate_builtin(value: Value) -> BuiltinResult<Value> {
        let value = super::gathered(value, "tabulate").await?;
        super::tabulate_value("tabulate", value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[test]
    fn tabulate_labels_preserve_exact_uint64_text() {
        assert_eq!(
            label_for_value(&Value::Int(runmat_builtins::IntValue::U64(u64::MAX))),
            Some("18446744073709551615".to_string())
        );
    }

    #[test]
    fn scalar_flag_accepts_typed_integer_tensor_scalars() {
        let one = Tensor::new_integer(runmat_builtins::IntegerStorage::U64(vec![1]), vec![1, 1])
            .expect("flag");
        assert_eq!(scalar_flag("mad", &Value::Tensor(one)).unwrap(), Some(1));

        let two = Tensor::new_integer(runmat_builtins::IntegerStorage::U64(vec![2]), vec![1, 1])
            .expect("flag");
        assert!(scalar_flag("mad", &Value::Tensor(two)).is_err());
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-10,
            "expected {expected}, got {actual}"
        );
    }

    fn tensor_values(value: Value) -> (Vec<f64>, Vec<usize>) {
        match value {
            Value::Num(num) => (vec![num], vec![1, 1]),
            Value::Tensor(tensor) => (tensor.data, tensor.shape),
            other => panic!("expected numeric output, got {other:?}"),
        }
    }

    #[test]
    fn geomean_reduces_columns_and_omits_nans() {
        let x = Value::Tensor(Tensor::new(vec![1.0, 4.0, f64::NAN, 9.0], vec![2, 2]).unwrap());
        let out = block_on(geomean::geomean_builtin(
            x,
            vec![Value::CharArray(runmat_builtins::CharArray::new_row(
                "omitnan",
            ))],
        ))
        .unwrap();
        let (data, shape) = tensor_values(out);
        assert_eq!(shape, vec![1, 2]);
        assert_close(data[0], 2.0);
        assert_close(data[1], 9.0);
    }

    #[test]
    fn harmmean_and_rms_support_all_selector() {
        let x = Value::Tensor(Tensor::new(vec![1.0, 2.0, 4.0], vec![1, 3]).unwrap());
        let harmonic = block_on(harmmean::harmmean_builtin(
            x.clone(),
            vec![Value::CharArray(runmat_builtins::CharArray::new_row("all"))],
        ))
        .unwrap();
        assert_close(tensor_values(harmonic).0[0], 12.0 / 7.0);

        let rms = block_on(rms::rms_builtin(
            x,
            vec![Value::CharArray(runmat_builtins::CharArray::new_row("all"))],
        ))
        .unwrap();
        assert_close(tensor_values(rms).0[0], (21.0_f64 / 3.0).sqrt());
    }

    #[test]
    fn mad_supports_median_default_and_mean_flag_modes() {
        let x = Value::Tensor(Tensor::new(vec![1.0, 2.0, 10.0], vec![1, 3]).unwrap());
        let default_mad = block_on(mad::mad_builtin(
            x.clone(),
            vec![Value::CharArray(runmat_builtins::CharArray::new_row("all"))],
        ))
        .unwrap();
        assert_close(tensor_values(default_mad).0[0], 1.0);

        let median_mad = block_on(mad::mad_builtin(
            x.clone(),
            vec![
                Value::Num(0.0),
                Value::CharArray(runmat_builtins::CharArray::new_row("all")),
            ],
        ))
        .unwrap();
        assert_close(tensor_values(median_mad).0[0], 1.0);

        let mean_mad = block_on(mad::mad_builtin(
            x,
            vec![
                Value::Num(1.0),
                Value::CharArray(runmat_builtins::CharArray::new_row("all")),
            ],
        ))
        .unwrap();
        assert_close(tensor_values(mean_mad).0[0], 34.0 / 9.0);
    }

    #[test]
    fn skewness_and_kurtosis_match_biased_moments() {
        let x = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap());
        let skew = block_on(skewness::skewness_builtin(
            x.clone(),
            vec![Value::CharArray(runmat_builtins::CharArray::new_row("all"))],
        ))
        .unwrap();
        assert_close(tensor_values(skew).0[0], 0.0);

        let kurt = block_on(kurtosis::kurtosis_builtin(
            x,
            vec![Value::CharArray(runmat_builtins::CharArray::new_row("all"))],
        ))
        .unwrap();
        assert_close(tensor_values(kurt).0[0], 1.5);
    }

    #[test]
    fn rmse_broadcasts_and_reduces_residuals() {
        let x = Value::Tensor(Tensor::new(vec![2.0, 4.0, 6.0], vec![3, 1]).unwrap());
        let y = Value::Num(1.0);
        let out = block_on(rmse::rmse_builtin(
            x,
            y,
            vec![Value::CharArray(runmat_builtins::CharArray::new_row("all"))],
        ))
        .unwrap();
        assert_close(
            tensor_values(out).0[0],
            ((1.0 + 9.0 + 25.0) / 3.0_f64).sqrt(),
        );
    }

    #[test]
    fn rmse_supports_weights_and_omitnan() {
        let x = Value::Tensor(Tensor::new(vec![2.0, f64::NAN, 6.0], vec![3, 1]).unwrap());
        let y = Value::Num(1.0);
        let weights = Value::Tensor(Tensor::new(vec![1.0, 100.0, 3.0], vec![3, 1]).unwrap());
        let out = block_on(rmse::rmse_builtin(
            x,
            y,
            vec![
                Value::Num(1.0),
                Value::CharArray(runmat_builtins::CharArray::new_row("omitnan")),
                Value::CharArray(runmat_builtins::CharArray::new_row("Weights")),
                weights,
            ],
        ))
        .unwrap();
        let (data, shape) = tensor_values(out);
        assert_eq!(shape, vec![1, 1]);
        assert_close(data[0], ((1.0 + 75.0) / 4.0_f64).sqrt());
    }

    #[test]
    fn flagged_moments_omit_nans_by_default() {
        let x = Value::Tensor(Tensor::new(vec![1.0, 2.0, f64::NAN], vec![1, 3]).unwrap());
        let mad_out = block_on(mad::mad_builtin(
            x.clone(),
            vec![Value::CharArray(runmat_builtins::CharArray::new_row("all"))],
        ))
        .unwrap();
        assert_close(tensor_values(mad_out).0[0], 0.5);

        let skew_out = block_on(skewness::skewness_builtin(
            x.clone(),
            vec![Value::CharArray(runmat_builtins::CharArray::new_row("all"))],
        ))
        .unwrap();
        assert_close(tensor_values(skew_out).0[0], 0.0);

        let kurt_out = block_on(kurtosis::kurtosis_builtin(
            x,
            vec![Value::CharArray(runmat_builtins::CharArray::new_row("all"))],
        ))
        .unwrap();
        assert_close(tensor_values(kurt_out).0[0], 1.0);
    }

    #[test]
    fn rms_and_rmse_support_complex_magnitudes() {
        let x = Value::ComplexTensor(
            runmat_builtins::ComplexTensor::new(vec![(3.0, 4.0), (0.0, 12.0)], vec![2, 1]).unwrap(),
        );
        let out = block_on(rms::rms_builtin(
            x,
            vec![Value::CharArray(runmat_builtins::CharArray::new_row("all"))],
        ))
        .unwrap();
        assert_close(tensor_values(out).0[0], ((25.0 + 144.0) / 2.0_f64).sqrt());

        let out = block_on(rmse::rmse_builtin(
            Value::Complex(3.0, 4.0),
            Value::Num(0.0),
            vec![Value::CharArray(runmat_builtins::CharArray::new_row("all"))],
        ))
        .unwrap();
        assert_close(tensor_values(out).0[0], 5.0);
    }

    #[test]
    fn rmse_uses_implicit_expansion_and_rejects_weighted_all() {
        let lhs = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
        let rhs = Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap());
        let out = block_on(rmse::rmse_builtin(lhs, rhs, vec![Value::Num(1.0)])).unwrap();
        let (data, shape) = tensor_values(out);
        assert_eq!(shape, vec![1, 2]);
        assert_close(data[0], (0.5_f64).sqrt());
        assert_close(data[1], (0.5_f64).sqrt());

        let err = block_on(rmse::rmse_builtin(
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
            Value::Num(0.0),
            vec![
                Value::CharArray(runmat_builtins::CharArray::new_row("all")),
                Value::CharArray(runmat_builtins::CharArray::new_row("Weights")),
                Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![2, 1]).unwrap()),
            ],
        ))
        .unwrap_err();
        assert!(err.message.contains("weights are not supported"));
    }

    #[test]
    fn tabulate_expands_positive_integer_levels() {
        let x = Value::Tensor(Tensor::new(vec![1.0, 3.0, 3.0], vec![1, 3]).unwrap());
        let out = block_on(tabulate::tabulate_builtin(x)).unwrap();
        let (data, shape) = tensor_values(out);
        assert_eq!(shape, vec![3, 3]);
        assert_eq!(&data[0..3], &[1.0, 2.0, 3.0]);
        assert_eq!(&data[3..6], &[1.0, 0.0, 2.0]);
        assert_close(data[6], 100.0 / 3.0);
        assert_close(data[7], 0.0);
        assert_close(data[8], 200.0 / 3.0);
    }

    #[test]
    fn tabulate_numeric_omits_nans_from_counts_and_percent() {
        let x = Value::Tensor(Tensor::new(vec![1.0, 3.0, 3.0, f64::NAN], vec![1, 4]).unwrap());
        let out = block_on(tabulate::tabulate_builtin(x)).unwrap();
        let (data, shape) = tensor_values(out);
        assert_eq!(shape, vec![3, 3]);
        assert_eq!(&data[0..3], &[1.0, 2.0, 3.0]);
        assert_eq!(&data[3..6], &[1.0, 0.0, 2.0]);
        assert_close(data[6], 100.0 / 3.0);
        assert_close(data[7], 0.0);
        assert_close(data[8], 200.0 / 3.0);
    }

    #[test]
    fn tabulate_logical_returns_cell_table() {
        let x = Value::LogicalArray(
            runmat_builtins::LogicalArray::new(vec![1, 0, 1], vec![3, 1]).unwrap(),
        );
        let out = block_on(tabulate::tabulate_builtin(x)).unwrap();
        match out {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 2);
                assert_eq!(cell.cols, 3);
                assert_eq!(cell.get(0, 0).unwrap(), Value::Bool(false));
                assert_eq!(cell.get(0, 1).unwrap(), Value::Num(1.0));
                assert_eq!(cell.get(1, 0).unwrap(), Value::Bool(true));
                assert_eq!(cell.get(1, 1).unwrap(), Value::Num(2.0));
            }
            other => panic!("expected cell table, got {other:?}"),
        }
    }

    #[test]
    fn tabulate_string_array_returns_cell_table() {
        let x = Value::StringArray(
            StringArray::new(vec!["b".into(), "a".into(), "b".into()], vec![1, 3]).unwrap(),
        );
        let out = block_on(tabulate::tabulate_builtin(x)).unwrap();
        match out {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 2);
                assert_eq!(cell.cols, 3);
                assert_eq!(cell.get(0, 1).unwrap(), Value::Num(1.0));
                assert_eq!(cell.get(1, 1).unwrap(), Value::Num(2.0));
            }
            other => panic!("expected cell table, got {other:?}"),
        }
    }
}
