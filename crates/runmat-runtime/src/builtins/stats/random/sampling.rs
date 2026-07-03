//! Sampling utilities for Statistics and Machine Learning Toolbox compatibility.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, LogicalArray, ResolveContext, StringArray, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random;
use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAMPLING.INVALID_ARGUMENT",
    identifier: None,
    when: "Inputs, dimensions, replacement flags, or weights are malformed.",
    message: "sampling: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAMPLING.INTERNAL",
    identifier: None,
    when: "Internal conversion or allocation fails.",
    message: "sampling: internal error",
};

macro_rules! sampling_descriptor {
    ($name:literal, $signatures:expr, $output_mode:expr) => {
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
            output_mode: $output_mode,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
    };
}

const OUTPUT_Y: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Random sample.",
}];

const OUTPUT_Y_IDX: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Random sample.",
    },
    BuiltinParamDescriptor {
        name: "idx",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "One-based sampled indices along the sampled dimension.",
    },
];

const OUTPUT_R: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "r",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Discrete uniform random sample.",
}];

const PARAM_DATA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "data",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Population to sample from.",
};

const PARAM_N: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Population size or upper discrete uniform bound.",
};

const PARAM_K: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "k",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of samples.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Dimension, replacement, and weights options.",
};

const PARAM_SZ: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sz",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Output dimensions.",
};

const INPUTS_DATA_K: [BuiltinParamDescriptor; 2] = [PARAM_DATA, PARAM_K];
const INPUTS_DATA_K_OPTIONS: [BuiltinParamDescriptor; 3] = [PARAM_DATA, PARAM_K, PARAM_OPTIONS];
const INPUTS_N_K: [BuiltinParamDescriptor; 2] = [PARAM_N, PARAM_K];
const INPUTS_N_K_OPTIONS: [BuiltinParamDescriptor; 3] = [PARAM_N, PARAM_K, PARAM_OPTIONS];
const INPUTS_N: [BuiltinParamDescriptor; 1] = [PARAM_N];
const INPUTS_N_SZ: [BuiltinParamDescriptor; 2] = [PARAM_N, PARAM_SZ];

const DATASAMPLE_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "y = datasample(data, k)",
        inputs: &INPUTS_DATA_K,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "y = datasample(data, k, dim)",
        inputs: &INPUTS_DATA_K_OPTIONS,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "y = datasample(___, Name, Value)",
        inputs: &INPUTS_DATA_K_OPTIONS,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "[y, idx] = datasample(___)",
        inputs: &INPUTS_DATA_K_OPTIONS,
        outputs: &OUTPUT_Y_IDX,
    },
];

const RANDSAMPLE_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "y = randsample(n, k)",
        inputs: &INPUTS_N_K,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "y = randsample(population, k)",
        inputs: &INPUTS_DATA_K,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "y = randsample(___, replacement, w)",
        inputs: &INPUTS_N_K_OPTIONS,
        outputs: &OUTPUT_Y,
    },
];

const UNIDRND_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "r = unidrnd(n)",
        inputs: &INPUTS_N,
        outputs: &OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "r = unidrnd(n, sz)",
        inputs: &INPUTS_N_SZ,
        outputs: &OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "r = unidrnd(n, sz1, sz2, ...)",
        inputs: &INPUTS_N_SZ,
        outputs: &OUTPUT_R,
    },
];

fn sampling_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

fn numeric_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

fn sampling_error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

async fn gathered(value: Value, name: &str) -> BuiltinResult<Value> {
    gather_if_needed_async(&value)
        .await
        .map_err(|err| sampling_error(name, format!("{name}: {err}")))
}

fn parse_positive_usize(name: &str, value: &Value, label: &str) -> BuiltinResult<usize> {
    let raw = match value {
        Value::Num(v) => *v,
        Value::Int(i) => i.to_f64(),
        Value::Bool(v) => {
            if *v {
                1.0
            } else {
                0.0
            }
        }
        other => {
            return Err(sampling_error(
                name,
                format!("{name}: {label} must be a positive integer, got {other:?}"),
            ));
        }
    };
    if !raw.is_finite() || raw < 1.0 || raw.fract() != 0.0 || raw > usize::MAX as f64 {
        return Err(sampling_error(
            name,
            format!("{name}: {label} must be a positive integer"),
        ));
    }
    Ok(raw as usize)
}

fn parse_bool(name: &str, value: &Value, label: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(v) => Ok(*v),
        Value::Num(v) if *v == 0.0 || *v == 1.0 => Ok(*v != 0.0),
        Value::Int(i) if i.to_i64() == 0 || i.to_i64() == 1 => Ok(i.to_i64() != 0),
        other => Err(sampling_error(
            name,
            format!("{name}: {label} must be logical true or false, got {other:?}"),
        )),
    }
}

fn first_non_singleton(shape: &[usize]) -> usize {
    shape.iter().position(|dim| *dim > 1).unwrap_or(0)
}

fn normalize_shape(mut shape: Vec<usize>) -> Vec<usize> {
    if shape.is_empty() {
        shape = vec![1, 1];
    } else if shape.len() == 1 {
        shape.push(1);
    }
    while shape.len() > 2 && shape.last() == Some(&1) {
        shape.pop();
    }
    shape
}

fn parse_weights(name: &str, value: Value, expected: usize) -> BuiltinResult<Vec<f64>> {
    let tensor = tensor::value_into_tensor_for(name, value)
        .map_err(|err| sampling_error(name, format!("{name}: {err}")))?;
    if tensor.data.len() != expected {
        return Err(sampling_error(
            name,
            format!("{name}: weights length must match the sampled dimension"),
        ));
    }
    if tensor
        .data
        .iter()
        .any(|weight| weight.is_nan() || *weight < 0.0)
    {
        return Err(sampling_error(
            name,
            format!("{name}: weights must be nonnegative and cannot contain NaN"),
        ));
    }
    if tensor.data.iter().sum::<f64>() <= 0.0 {
        return Err(sampling_error(
            name,
            format!("{name}: weights must contain at least one positive value"),
        ));
    }
    Ok(tensor.data)
}

fn sample_indices(
    name: &str,
    population_len: usize,
    k: usize,
    replacement: bool,
    weights: Option<&[f64]>,
) -> BuiltinResult<Vec<usize>> {
    if population_len == 0 {
        return Err(sampling_error(name, format!("{name}: population is empty")));
    }
    if !replacement && k > population_len {
        return Err(sampling_error(
            name,
            format!("{name}: k cannot exceed population size without replacement"),
        ));
    }
    match (replacement, weights) {
        (true, Some(weights)) => weighted_with_replacement(name, k, weights),
        (false, Some(weights)) => weighted_without_replacement(name, k, weights),
        (true, None) => {
            let uniforms = random::generate_uniform(k, name)?;
            Ok(uniforms
                .into_iter()
                .map(|u| ((u * population_len as f64).floor() as usize).min(population_len - 1))
                .collect())
        }
        (false, None) => unweighted_without_replacement(name, population_len, k),
    }
}

fn unweighted_without_replacement(
    name: &str,
    population_len: usize,
    k: usize,
) -> BuiltinResult<Vec<usize>> {
    let uniforms = random::generate_uniform(k, name)?;
    let mut pool = (0..population_len).collect::<Vec<_>>();
    let mut out = Vec::with_capacity(k);
    for (draw, u) in uniforms.into_iter().enumerate() {
        let span = population_len - draw;
        let offset = ((u * span as f64).floor() as usize).min(span - 1);
        out.push(pool.swap_remove(offset));
    }
    Ok(out)
}

fn weighted_with_replacement(name: &str, k: usize, weights: &[f64]) -> BuiltinResult<Vec<usize>> {
    let total = weights.iter().sum::<f64>();
    let uniforms = random::generate_uniform(k, name)?;
    Ok(uniforms
        .into_iter()
        .map(|u| choose_weighted(weights, total, u))
        .collect())
}

fn weighted_without_replacement(
    name: &str,
    k: usize,
    weights: &[f64],
) -> BuiltinResult<Vec<usize>> {
    let uniforms = random::generate_uniform(k, name)?;
    let mut weights = weights.to_vec();
    let mut out = Vec::with_capacity(k);
    for u in uniforms {
        let total = weights.iter().sum::<f64>();
        if total <= 0.0 {
            return Err(sampling_error(
                name,
                format!("{name}: not enough positive weights to sample without replacement"),
            ));
        }
        let idx = choose_weighted(&weights, total, u);
        weights[idx] = 0.0;
        out.push(idx);
    }
    Ok(out)
}

fn choose_weighted(weights: &[f64], total: f64, u: f64) -> usize {
    let mut threshold = u * total;
    for (idx, weight) in weights.iter().enumerate() {
        if *weight <= 0.0 {
            continue;
        }
        if threshold < *weight {
            return idx;
        }
        threshold -= *weight;
    }
    weights
        .iter()
        .rposition(|weight| *weight > 0.0)
        .unwrap_or(0)
}

fn indices_value(indices: &[usize]) -> BuiltinResult<Value> {
    Tensor::new(
        indices.iter().map(|idx| (idx + 1) as f64).collect(),
        vec![indices.len(), 1],
    )
    .map(tensor::tensor_into_value)
    .map_err(|err| sampling_error("datasample", format!("datasample: {err}")))
}

fn sample_tensor_axis(
    data: &[f64],
    shape: &[usize],
    axis: usize,
    indices: &[usize],
    name: &str,
) -> BuiltinResult<Value> {
    let mut out_shape = shape.to_vec();
    out_shape[axis] = indices.len();
    let out_len = tensor::element_count(&out_shape);
    let mut out = vec![0.0; out_len];
    let pre: usize = shape[..axis].iter().product();
    let axis_len = shape[axis];
    let post: usize = shape[axis + 1..].iter().product();
    for prefix in 0..pre {
        for suffix in 0..post {
            for (dst_axis, src_axis) in indices.iter().enumerate() {
                if *src_axis >= axis_len {
                    return Err(sampling_error(
                        name,
                        format!("{name}: sample index out of range"),
                    ));
                }
                let src = prefix + src_axis * pre + suffix * pre * axis_len;
                let dst = prefix + dst_axis * pre + suffix * pre * indices.len();
                out[dst] = data[src];
            }
        }
    }
    Tensor::new(out, out_shape)
        .map(tensor::tensor_into_value)
        .map_err(|err| sampling_error(name, format!("{name}: {err}")))
}

fn sample_logical_axis(
    data: &[u8],
    shape: &[usize],
    axis: usize,
    indices: &[usize],
    name: &str,
) -> BuiltinResult<Value> {
    let mut out_shape = shape.to_vec();
    out_shape[axis] = indices.len();
    let out_len = tensor::element_count(&out_shape);
    let mut out = vec![0u8; out_len];
    let pre: usize = shape[..axis].iter().product();
    let axis_len = shape[axis];
    let post: usize = shape[axis + 1..].iter().product();
    for prefix in 0..pre {
        for suffix in 0..post {
            for (dst_axis, src_axis) in indices.iter().enumerate() {
                let src = prefix + src_axis * pre + suffix * pre * axis_len;
                let dst = prefix + dst_axis * pre + suffix * pre * indices.len();
                out[dst] = data[src];
            }
        }
    }
    LogicalArray::new(out, out_shape)
        .map(Value::LogicalArray)
        .map_err(|err| sampling_error(name, format!("{name}: {err}")))
}

fn sample_string_axis(
    array: &StringArray,
    axis: usize,
    indices: &[usize],
    _name: &str,
) -> BuiltinResult<Value> {
    let shape = normalize_shape(array.shape.clone());
    let mut out_shape = shape.clone();
    out_shape[axis] = indices.len();
    let out_len = tensor::element_count(&out_shape);
    let mut out = vec![String::new(); out_len];
    let pre: usize = shape[..axis].iter().product();
    let axis_len = shape[axis];
    let post: usize = shape[axis + 1..].iter().product();
    for prefix in 0..pre {
        for suffix in 0..post {
            for (dst_axis, src_axis) in indices.iter().enumerate() {
                let src = prefix + src_axis * pre + suffix * pre * axis_len;
                let dst = prefix + dst_axis * pre + suffix * pre * indices.len();
                out[dst] = array.data[src].clone();
            }
        }
    }
    let rows = *out_shape.first().unwrap_or(&1);
    let cols = *out_shape.get(1).unwrap_or(&1);
    Ok(Value::StringArray(StringArray {
        data: out,
        shape: out_shape,
        rows,
        cols,
    }))
}

fn sample_char_axis(
    array: &CharArray,
    axis: usize,
    indices: &[usize],
    name: &str,
) -> BuiltinResult<Value> {
    let shape = vec![array.rows, array.cols];
    let mut out_shape = shape.clone();
    out_shape[axis] = indices.len();
    let out_len = tensor::element_count(&out_shape);
    let mut out = vec![' '; out_len];
    let pre: usize = shape[..axis].iter().product();
    let axis_len = shape[axis];
    let post: usize = shape[axis + 1..].iter().product();
    for prefix in 0..pre {
        for suffix in 0..post {
            for (dst_axis, src_axis) in indices.iter().enumerate() {
                let src = prefix + src_axis * pre + suffix * pre * axis_len;
                let dst = prefix + dst_axis * pre + suffix * pre * indices.len();
                out[dst] = array.data[src];
            }
        }
    }
    CharArray::new(out, out_shape[0], out_shape[1])
        .map(Value::CharArray)
        .map_err(|err| sampling_error(name, format!("{name}: {err}")))
}

fn sample_value_axis(
    data: Value,
    axis: usize,
    indices: &[usize],
    name: &str,
) -> BuiltinResult<Value> {
    match data {
        Value::Tensor(t) => {
            sample_tensor_axis(&t.data, &normalize_shape(t.shape), axis, indices, name)
        }
        Value::Num(value) => sample_tensor_axis(&[value], &[1, 1], axis, indices, name),
        Value::Int(value) => sample_tensor_axis(&[value.to_f64()], &[1, 1], axis, indices, name),
        Value::Bool(value) => {
            let byte = if value { 1 } else { 0 };
            sample_logical_axis(&[byte], &[1, 1], axis, indices, name)
        }
        Value::LogicalArray(array) => sample_logical_axis(
            &array.data,
            &normalize_shape(array.shape),
            axis,
            indices,
            name,
        ),
        Value::String(value) => sample_string_axis(
            &StringArray {
                data: vec![value],
                shape: vec![1, 1],
                rows: 1,
                cols: 1,
            },
            axis,
            indices,
            name,
        ),
        Value::StringArray(array) => sample_string_axis(&array, axis, indices, name),
        Value::CharArray(array) => sample_char_axis(&array, axis, indices, name),
        other => Err(sampling_error(
            name,
            format!("{name}: unsupported population type {other:?}"),
        )),
    }
}

fn shape_of_sampled_value(value: &Value) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Tensor(t) => Ok(normalize_shape(t.shape.clone())),
        Value::LogicalArray(a) => Ok(normalize_shape(a.shape.clone())),
        Value::StringArray(a) => Ok(normalize_shape(a.shape.clone())),
        Value::CharArray(a) => Ok(vec![a.rows, a.cols]),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::String(_) => Ok(vec![1, 1]),
        other => Err(sampling_error(
            "datasample",
            format!("datasample: unsupported population type {other:?}"),
        )),
    }
}

#[derive(Clone)]
struct DatasampleArgs {
    data: Value,
    k: usize,
    dim: Option<usize>,
    replacement: bool,
    weights: Option<Vec<f64>>,
}

async fn parse_datasample_args(data: Value, rest: Vec<Value>) -> BuiltinResult<DatasampleArgs> {
    if rest.is_empty() {
        return Err(sampling_error("datasample", "datasample: k is required"));
    }
    let data = gathered(data, "datasample").await?;
    let k = parse_positive_usize("datasample", &rest[0], "k")?;
    let mut dim = None;
    let mut replacement = true;
    let mut weight_value = None;
    let mut idx = 1usize;
    while idx < rest.len() {
        if let Some(keyword) = keyword_of(&rest[idx]) {
            match keyword.as_str() {
                "replace" => {
                    let Some(value) = rest.get(idx + 1) else {
                        return Err(sampling_error(
                            "datasample",
                            "datasample: Replace requires a value",
                        ));
                    };
                    replacement = parse_bool("datasample", value, "Replace")?;
                    idx += 2;
                    continue;
                }
                "weights" => {
                    let Some(value) = rest.get(idx + 1) else {
                        return Err(sampling_error(
                            "datasample",
                            "datasample: Weights requires a value",
                        ));
                    };
                    weight_value = Some(gathered(value.clone(), "datasample").await?);
                    idx += 2;
                    continue;
                }
                other => {
                    return Err(sampling_error(
                        "datasample",
                        format!("datasample: unsupported option '{other}'"),
                    ));
                }
            }
        }
        if dim.is_some() {
            return Err(sampling_error(
                "datasample",
                "datasample: dimension can only be specified once",
            ));
        }
        dim = Some(parse_positive_usize("datasample", &rest[idx], "dim")?);
        idx += 1;
    }
    let shape = shape_of_sampled_value(&data)?;
    let axis = dim
        .map(|value| value - 1)
        .unwrap_or_else(|| first_non_singleton(&shape));
    if axis >= shape.len() {
        return Err(sampling_error(
            "datasample",
            "datasample: dimension exceeds input rank",
        ));
    }
    let weights = match weight_value {
        Some(value) => Some(parse_weights("datasample", value, shape[axis])?),
        None => None,
    };
    Ok(DatasampleArgs {
        data,
        k,
        dim: Some(axis),
        replacement,
        weights,
    })
}

fn datasample_compute(args: DatasampleArgs) -> BuiltinResult<(Value, Value)> {
    let shape = shape_of_sampled_value(&args.data)?;
    let axis = args.dim.unwrap_or_else(|| first_non_singleton(&shape));
    let indices = sample_indices(
        "datasample",
        shape[axis],
        args.k,
        args.replacement,
        args.weights.as_deref(),
    )?;
    let idx_value = indices_value(&indices)?;
    let sample = sample_value_axis(args.data, axis, &indices, "datasample")?;
    Ok((sample, idx_value))
}

pub mod datasample {
    use super::*;
    sampling_descriptor!(
        "datasample",
        DATASAMPLE_SIGNATURES,
        BuiltinOutputMode::ByRequestedOutputCount
    );

    #[runtime_builtin(
        name = "datasample",
        category = "stats/random",
        summary = "Randomly sample from data with or without replacement.",
        keywords = "datasample,random,sample,replacement,weights,statistics",
        type_resolver(super::sampling_type),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::stats::random::sampling::datasample"
    )]
    pub(crate) async fn datasample_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let args = super::parse_datasample_args(value, rest).await?;
        let (sample, idx) = super::datasample_compute(args)?;
        match crate::output_count::current_output_count() {
            Some(0) => Ok(Value::OutputList(Vec::new())),
            Some(1) => Ok(Value::OutputList(vec![sample])),
            Some(out_count) => Ok(crate::output_count::output_list_with_padding(
                out_count,
                vec![sample, idx],
            )),
            None => Ok(sample),
        }
    }
}

enum RandsamplePopulation {
    Range(usize),
    Values(Value, Vec<usize>, usize),
}

struct RandsampleArgs {
    population: RandsamplePopulation,
    k: usize,
    replacement: bool,
    weights: Option<Vec<f64>>,
}

async fn parse_randsample_args(args: Vec<Value>) -> BuiltinResult<RandsampleArgs> {
    if args.len() < 2 {
        return Err(sampling_error(
            "randsample",
            "randsample: population and k are required",
        ));
    }
    let first = gathered(args[0].clone(), "randsample").await?;
    let k = parse_positive_usize("randsample", &args[1], "k")?;
    let mut replacement = false;
    let mut weights_value = None;
    match args.len() {
        2 => {}
        3 => replacement = parse_bool("randsample", &args[2], "replacement")?,
        4 => {
            replacement = parse_bool("randsample", &args[2], "replacement")?;
            weights_value = Some(gathered(args[3].clone(), "randsample").await?);
        }
        _ => {
            return Err(sampling_error(
                "randsample",
                "randsample: too many arguments",
            ))
        }
    }
    let population = match &first {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            RandsamplePopulation::Range(parse_positive_usize("randsample", &first, "n")?)
        }
        _ => {
            let shape = shape_of_sampled_value(&first).map_err(|err| {
                sampling_error("randsample", format!("randsample: {}", err.message()))
            })?;
            if shape.iter().filter(|dim| **dim > 1).count() > 1 {
                return Err(sampling_error(
                    "randsample",
                    "randsample: population must be a vector",
                ));
            }
            let axis = first_non_singleton(&shape);
            RandsamplePopulation::Values(first, shape, axis)
        }
    };
    let pop_len = match &population {
        RandsamplePopulation::Range(n) => *n,
        RandsamplePopulation::Values(_, shape, axis) => shape[*axis],
    };
    let weights = match weights_value {
        Some(value) => {
            if !replacement {
                return Err(sampling_error(
                    "randsample",
                    "randsample: weights require sampling with replacement",
                ));
            }
            Some(parse_weights("randsample", value, pop_len)?)
        }
        None => None,
    };
    Ok(RandsampleArgs {
        population,
        k,
        replacement,
        weights,
    })
}

fn randsample_compute(args: RandsampleArgs) -> BuiltinResult<Value> {
    match args.population {
        RandsamplePopulation::Range(n) => {
            let indices = sample_indices(
                "randsample",
                n,
                args.k,
                args.replacement,
                args.weights.as_deref(),
            )?;
            Tensor::new(
                indices.into_iter().map(|idx| (idx + 1) as f64).collect(),
                if args.k == 1 {
                    vec![1, 1]
                } else {
                    vec![args.k, 1]
                },
            )
            .map(tensor::tensor_into_value)
            .map_err(|err| sampling_error("randsample", format!("randsample: {err}")))
        }
        RandsamplePopulation::Values(value, shape, axis) => {
            let indices = sample_indices(
                "randsample",
                shape[axis],
                args.k,
                args.replacement,
                args.weights.as_deref(),
            )?;
            sample_value_axis(value, axis, &indices, "randsample")
        }
    }
}

pub mod randsample {
    use super::*;
    sampling_descriptor!(
        "randsample",
        RANDSAMPLE_SIGNATURES,
        BuiltinOutputMode::Fixed
    );

    #[runtime_builtin(
        name = "randsample",
        category = "stats/random",
        summary = "Randomly sample from a range or population vector.",
        keywords = "randsample,random,sample,replacement,weights,statistics",
        type_resolver(super::sampling_type),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::stats::random::sampling::randsample"
    )]
    pub(crate) async fn randsample_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        let args = super::parse_randsample_args(args).await?;
        super::randsample_compute(args)
    }
}

async fn parse_shape_args(name: &str, rest: &[Value]) -> BuiltinResult<Vec<usize>> {
    if rest.is_empty() {
        return Ok(vec![1, 1]);
    }
    let mut dims = Vec::new();
    for arg in rest {
        match crate::builtins::common::random_args::extract_dims(arg, name).await {
            Ok(Some(values)) => dims.extend(values),
            Ok(None) => {
                return Err(sampling_error(
                    name,
                    format!("{name}: invalid size argument {arg:?}"),
                ));
            }
            Err(err) => return Err(sampling_error(name, err)),
        }
    }
    if dims.is_empty() {
        Ok(vec![0, 0])
    } else if dims.len() == 1 {
        Ok(vec![dims[0], dims[0]])
    } else {
        while dims.len() > 2 && dims.last() == Some(&1) {
            dims.pop();
        }
        Ok(dims)
    }
}

async fn parse_unidrnd_args(args: Vec<Value>) -> BuiltinResult<(Tensor, Vec<usize>)> {
    if args.is_empty() {
        return Err(sampling_error("unidrnd", "unidrnd: n is required"));
    }
    let n = tensor::value_into_tensor_for("unidrnd", gathered(args[0].clone(), "unidrnd").await?)
        .map_err(|err| sampling_error("unidrnd", format!("unidrnd: {err}")))?;
    if n.data
        .iter()
        .any(|value| !value.is_finite() || *value < 1.0 || value.fract() != 0.0)
    {
        return Err(sampling_error(
            "unidrnd",
            "unidrnd: n must contain positive integers",
        ));
    }
    let shape = if args.len() == 1 {
        normalize_shape(n.shape.clone())
    } else {
        parse_shape_args("unidrnd", &args[1..]).await?
    };
    if n.data.len() != 1 && normalize_shape(n.shape.clone()) != shape {
        return Err(sampling_error(
            "unidrnd",
            "unidrnd: requested size must match non-scalar n",
        ));
    }
    Ok((n, shape))
}

pub mod unidrnd {
    use super::*;
    sampling_descriptor!("unidrnd", UNIDRND_SIGNATURES, BuiltinOutputMode::Fixed);

    #[runtime_builtin(
        name = "unidrnd",
        category = "stats/random",
        summary = "Generate random integers from a discrete uniform distribution.",
        keywords = "unidrnd,uniform,discrete,random,integer,statistics",
        type_resolver(super::numeric_type),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::stats::random::sampling::unidrnd"
    )]
    pub(crate) async fn unidrnd_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        let (n, shape) = super::parse_unidrnd_args(args).await?;
        let len = tensor::element_count(&shape);
        let uniforms = random::generate_uniform(len, "unidrnd")?;
        let data = uniforms
            .into_iter()
            .enumerate()
            .map(|(idx, u)| {
                let upper = if n.data.len() == 1 {
                    n.data[0]
                } else {
                    n.data[idx]
                };
                (u * upper).floor() + 1.0
            })
            .collect();
        Tensor::new(data, shape)
            .map(tensor::tensor_into_value)
            .map_err(|err| sampling_error("unidrnd", format!("unidrnd: {err}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[test]
    fn datasample_samples_rows_and_returns_indices() {
        let _lock = random::test_lock().lock().unwrap();
        random::reset_rng();
        let data =
            Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0], vec![3, 2]).unwrap());
        let _guard = crate::output_count::push_output_count(Some(2));
        let out = block_on(datasample::datasample_builtin(
            data,
            vec![Value::Num(2.0), Value::from("Replace"), Value::Bool(false)],
        ))
        .unwrap();
        match out {
            Value::OutputList(values) => {
                assert_eq!(values.len(), 2);
                match &values[0] {
                    Value::Tensor(t) => {
                        assert_eq!(t.shape, vec![2, 2]);
                        assert_eq!(t.data.len(), 4);
                    }
                    other => panic!("expected tensor sample, got {other:?}"),
                }
                match &values[1] {
                    Value::Tensor(t) => {
                        assert_eq!(t.shape, vec![2, 1]);
                        assert!(t.data.iter().all(|idx| (1.0..=3.0).contains(idx)));
                    }
                    other => panic!("expected tensor indices, got {other:?}"),
                }
            }
            other => panic!("expected output list, got {other:?}"),
        }
    }

    #[test]
    fn datasample_supports_char_weights() {
        let _lock = random::test_lock().lock().unwrap();
        random::reset_rng();
        let data = Value::CharArray(CharArray::new_row("ACGT"));
        let out = block_on(datasample::datasample_builtin(
            data,
            vec![
                Value::Num(5.0),
                Value::from("Weights"),
                Value::Tensor(Tensor::new(vec![0.0, 0.0, 1.0, 0.0], vec![1, 4]).unwrap()),
            ],
        ))
        .unwrap();
        match out {
            Value::CharArray(chars) => {
                assert_eq!(chars.rows, 1);
                assert_eq!(chars.cols, 5);
                assert_eq!(chars.data, vec!['G'; 5]);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[test]
    fn randsample_range_and_population_vector() {
        let _lock = random::test_lock().lock().unwrap();
        random::reset_rng();
        let range = block_on(randsample::randsample_builtin(vec![
            Value::Num(5.0),
            Value::Num(3.0),
            Value::Bool(false),
        ]))
        .unwrap();
        match range {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 1]);
                assert!(t.data.iter().all(|value| (1.0..=5.0).contains(value)));
            }
            other => panic!("expected tensor, got {other:?}"),
        }

        random::reset_rng();
        let population = Value::Tensor(Tensor::new(vec![10.0, 20.0, 30.0], vec![1, 3]).unwrap());
        let out = block_on(randsample::randsample_builtin(vec![
            population,
            Value::Num(4.0),
            Value::Bool(true),
        ]))
        .unwrap();
        match out {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 4]);
                assert!(t
                    .data
                    .iter()
                    .all(|value| [10.0, 20.0, 30.0].contains(value)));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn unidrnd_generates_with_scalar_or_array_upper_bound() {
        let _lock = random::test_lock().lock().unwrap();
        random::reset_rng();
        let out = block_on(unidrnd::unidrnd_builtin(vec![
            Value::Num(3.0),
            Value::Num(2.0),
            Value::Num(2.0),
        ]))
        .unwrap();
        match out {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert!(t.data.iter().all(|value| (1.0..=3.0).contains(value)));
            }
            other => panic!("expected tensor, got {other:?}"),
        }

        random::reset_rng();
        let n = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap());
        let out = block_on(unidrnd::unidrnd_builtin(vec![n])).unwrap();
        match out {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 1]);
                assert_eq!(t.data[0], 1.0);
                assert!((1.0..=2.0).contains(&t.data[1]));
                assert!((1.0..=3.0).contains(&t.data[2]));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }
}
