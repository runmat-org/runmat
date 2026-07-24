//! Sampling utilities for Statistics and Machine Learning Toolbox compatibility.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, CharArray, LogicalArray, ResolveContext, StringArray, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random;
use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const MAX_DIVIDERAND_Q: usize = 10_000_000;

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

const OUTPUT_DIVIDERAND: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "trainInd",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based training-set indices.",
    },
    BuiltinParamDescriptor {
        name: "valInd",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "One-based validation-set indices.",
    },
    BuiltinParamDescriptor {
        name: "testInd",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "One-based test-set indices.",
    },
];

const OUTPUT_BOOTSTAT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "bootstat",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Bootstrap statistics, one bootstrap replicate per row.",
}];

const OUTPUT_BOOTSTAT_BOOTSAM: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "bootstat",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Bootstrap statistics, one bootstrap replicate per row.",
    },
    BuiltinParamDescriptor {
        name: "bootsam",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "One-based bootstrap sample indices.",
    },
];

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

const PARAM_Q: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Q",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of targets to divide.",
};

const PARAM_TRAIN_RATIO: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "trainRatio",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("0.7"),
    description: "Training-set allocation ratio.",
};

const PARAM_VAL_RATIO: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "valRatio",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("0.15"),
    description: "Validation-set allocation ratio.",
};

const PARAM_TEST_RATIO: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "testRatio",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("0.15"),
    description: "Test-set allocation ratio.",
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

const PARAM_NBOOT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nboot",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of bootstrap samples.",
};

const PARAM_BOOTFUN: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "bootfun",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Function handle applied to each bootstrap sample.",
};

const PARAM_BOOT_DATA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "data",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Data arrays and name-value options.",
};

const INPUTS_DATA_K: [BuiltinParamDescriptor; 2] = [PARAM_DATA, PARAM_K];
const INPUTS_DATA_K_OPTIONS: [BuiltinParamDescriptor; 3] = [PARAM_DATA, PARAM_K, PARAM_OPTIONS];
const INPUTS_N_K: [BuiltinParamDescriptor; 2] = [PARAM_N, PARAM_K];
const INPUTS_N_K_OPTIONS: [BuiltinParamDescriptor; 3] = [PARAM_N, PARAM_K, PARAM_OPTIONS];
const INPUTS_N: [BuiltinParamDescriptor; 1] = [PARAM_N];
const INPUTS_N_SZ: [BuiltinParamDescriptor; 2] = [PARAM_N, PARAM_SZ];
const INPUTS_BOOTSTRP: [BuiltinParamDescriptor; 3] = [PARAM_NBOOT, PARAM_BOOTFUN, PARAM_BOOT_DATA];
const INPUTS_DIVIDERAND_Q: [BuiltinParamDescriptor; 1] = [PARAM_Q];
const INPUTS_DIVIDERAND_RATIOS: [BuiltinParamDescriptor; 4] = [
    PARAM_Q,
    PARAM_TRAIN_RATIO,
    PARAM_VAL_RATIO,
    PARAM_TEST_RATIO,
];

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

const BOOTSTRP_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "bootstat = bootstrp(nboot, bootfun, d)",
        inputs: &INPUTS_BOOTSTRP,
        outputs: &OUTPUT_BOOTSTAT,
    },
    BuiltinSignatureDescriptor {
        label: "bootstat = bootstrp(nboot, bootfun, d1, ..., dN)",
        inputs: &INPUTS_BOOTSTRP,
        outputs: &OUTPUT_BOOTSTAT,
    },
    BuiltinSignatureDescriptor {
        label: "[bootstat, bootsam] = bootstrp(___)",
        inputs: &INPUTS_BOOTSTRP,
        outputs: &OUTPUT_BOOTSTAT_BOOTSAM,
    },
];

const DIVIDERAND_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "[trainInd, valInd, testInd] = dividerand(Q)",
        inputs: &INPUTS_DIVIDERAND_Q,
        outputs: &OUTPUT_DIVIDERAND,
    },
    BuiltinSignatureDescriptor {
        label: "[trainInd, valInd, testInd] = dividerand(Q, trainRatio, valRatio, testRatio)",
        inputs: &INPUTS_DIVIDERAND_RATIOS,
        outputs: &OUTPUT_DIVIDERAND,
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
    if let Value::Int(value) = value {
        return value
            .try_to_usize()
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                sampling_error(name, format!("{name}: {label} must be a positive integer"))
            });
    }
    let raw = match value {
        Value::Num(v) => *v,
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
    if !raw.is_finite()
        || raw < 1.0
        || raw.fract() != 0.0
        || raw > usize::MAX as f64
        || (usize::BITS == 64 && raw == usize::MAX as f64)
    {
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

fn sample_cell_axis(
    array: &CellArray,
    axis: usize,
    indices: &[usize],
    name: &str,
) -> BuiltinResult<Value> {
    let shape = normalize_shape(array.shape.clone());
    let mut out_shape = shape.clone();
    out_shape[axis] = indices.len();
    let out_len = tensor::element_count(&out_shape);
    let mut out = vec![Value::Num(0.0); out_len];
    let source_strides = row_major_strides(&shape);
    let output_strides = row_major_strides(&out_shape);
    for output_linear in 0..out_len {
        let mut rem = output_linear;
        let mut coords = vec![0usize; out_shape.len()];
        for (dim, stride) in output_strides.iter().enumerate() {
            coords[dim] = rem / *stride;
            rem %= *stride;
        }
        let source_axis = indices[coords[axis]];
        if source_axis >= shape[axis] {
            return Err(sampling_error(
                name,
                format!("{name}: sample index out of range"),
            ));
        }
        coords[axis] = source_axis;
        let source_linear = coords
            .iter()
            .zip(source_strides.iter())
            .map(|(coord, stride)| coord * stride)
            .sum::<usize>();
        out[output_linear] = array.data[source_linear].clone();
    }
    CellArray::new_with_shape(out, out_shape)
        .map(Value::Cell)
        .map_err(|err| sampling_error(name, format!("{name}: {err}")))
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    let mut acc = 1usize;
    for idx in (0..shape.len()).rev() {
        strides[idx] = acc;
        acc = acc.saturating_mul(shape[idx]);
    }
    strides
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
        Value::Cell(array) => sample_cell_axis(&array, axis, indices, name),
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
        Value::Cell(a) => Ok(normalize_shape(a.shape.clone())),
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

#[derive(Clone, Copy)]
struct DividerandArgs {
    q: usize,
    ratios: [f64; 3],
}

async fn parse_dividerand_args(args: Vec<Value>) -> BuiltinResult<DividerandArgs> {
    if args.len() != 1 && args.len() != 4 {
        return Err(sampling_error(
            "dividerand",
            "dividerand: expected Q or Q, trainRatio, valRatio, testRatio",
        ));
    }
    let q = parse_nonnegative_usize(
        "dividerand",
        gathered(args[0].clone(), "dividerand").await?,
        "Q",
    )?;
    let ratios = if args.len() == 1 {
        [0.7, 0.15, 0.15]
    } else {
        [
            parse_nonnegative_scalar_ratio(
                gathered(args[1].clone(), "dividerand").await?,
                "trainRatio",
            )?,
            parse_nonnegative_scalar_ratio(
                gathered(args[2].clone(), "dividerand").await?,
                "valRatio",
            )?,
            parse_nonnegative_scalar_ratio(
                gathered(args[3].clone(), "dividerand").await?,
                "testRatio",
            )?,
        ]
    };
    if ratios.iter().sum::<f64>() <= 0.0 {
        return Err(sampling_error(
            "dividerand",
            "dividerand: at least one ratio must be positive",
        ));
    }
    Ok(DividerandArgs { q, ratios })
}

fn parse_nonnegative_usize(name: &str, value: Value, label: &str) -> BuiltinResult<usize> {
    if let Value::Int(value) = &value {
        return value.try_to_usize().ok_or_else(|| {
            sampling_error(
                name,
                format!("{name}: {label} must be a nonnegative integer"),
            )
        });
    }
    let tensor = tensor::value_into_tensor_for(name, value)
        .map_err(|err| sampling_error(name, format!("{name}: {err}")))?;
    if tensor.data.len() != 1 {
        return Err(sampling_error(
            name,
            format!("{name}: {label} must be a scalar"),
        ));
    }
    let raw = tensor.data[0];
    if !raw.is_finite()
        || raw < 0.0
        || raw.fract() != 0.0
        || raw > usize::MAX as f64
        || (usize::BITS == 64 && raw == usize::MAX as f64)
    {
        return Err(sampling_error(
            name,
            format!("{name}: {label} must be a nonnegative integer"),
        ));
    }
    Ok(raw as usize)
}

fn parse_nonnegative_scalar_ratio(value: Value, label: &str) -> BuiltinResult<f64> {
    let tensor = tensor::value_into_tensor_for("dividerand", value)
        .map_err(|err| sampling_error("dividerand", format!("dividerand: {err}")))?;
    if tensor.data.len() != 1 {
        return Err(sampling_error(
            "dividerand",
            format!("dividerand: {label} must be a scalar"),
        ));
    }
    let raw = tensor.data[0];
    if !raw.is_finite() || raw < 0.0 {
        return Err(sampling_error(
            "dividerand",
            format!("dividerand: {label} must be a finite nonnegative scalar"),
        ));
    }
    Ok(raw)
}

fn dividerand_counts(q: usize, ratios: [f64; 3]) -> [usize; 3] {
    if q == 0 {
        return [0, 0, 0];
    }
    let max_ratio = ratios.iter().copied().fold(0.0, f64::max);
    let normalized = ratios.map(|ratio| ratio / max_ratio);
    let total = normalized.iter().sum::<f64>();
    let mut counts = [0usize; 3];
    let mut remainders = [(0usize, 0.0f64); 3];
    let mut assigned = 0usize;
    for (idx, ratio) in normalized.iter().enumerate() {
        let exact = (*ratio / total) * q as f64;
        let count = exact.floor().min(q as f64) as usize;
        counts[idx] = count;
        assigned += count;
        remainders[idx] = (idx, exact - count as f64);
    }
    remainders.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    for (idx, _) in remainders.iter().take(q.saturating_sub(assigned)) {
        counts[*idx] += 1;
    }
    counts
}

fn dividerand_permutation(q: usize) -> BuiltinResult<Vec<usize>> {
    if q > MAX_DIVIDERAND_Q {
        return Err(sampling_error(
            "dividerand",
            format!("dividerand: Q exceeds the maximum supported value of {MAX_DIVIDERAND_Q}"),
        ));
    }
    let uniforms = random::generate_uniform(q, "dividerand")?;
    let mut pool = Vec::new();
    pool.try_reserve_exact(q)
        .map_err(|_| sampling_error("dividerand", "dividerand: requested output is too large"))?;
    pool.extend(0..q);
    for (draw, u) in uniforms.into_iter().enumerate() {
        let span = q - draw;
        let offset = ((u * span as f64).floor() as usize).min(span - 1);
        pool.swap(draw, draw + offset);
    }
    Ok(pool)
}

fn row_index_value(indices: &[usize]) -> BuiltinResult<Value> {
    let mut data = Vec::new();
    data.try_reserve_exact(indices.len())
        .map_err(|_| sampling_error("dividerand", "dividerand: requested output is too large"))?;
    data.extend(indices.iter().map(|idx| (idx + 1) as f64));
    Tensor::new(data, vec![1, indices.len()])
        .map(Value::Tensor)
        .map_err(|err| sampling_error("dividerand", format!("dividerand: {err}")))
}

fn dividerand_compute(args: DividerandArgs) -> BuiltinResult<[Value; 3]> {
    let counts = dividerand_counts(args.q, args.ratios);
    let permutation = if args.q == 0 {
        Vec::new()
    } else {
        dividerand_permutation(args.q)?
    };
    let train_end = counts[0];
    let val_end = train_end + counts[1];
    Ok([
        row_index_value(&permutation[..train_end])?,
        row_index_value(&permutation[train_end..val_end])?,
        row_index_value(&permutation[val_end..])?,
    ])
}

pub mod dividerand {
    use super::*;
    sampling_descriptor!(
        "dividerand",
        DIVIDERAND_SIGNATURES,
        BuiltinOutputMode::ByRequestedOutputCount
    );

    #[runtime_builtin(
        name = "dividerand",
        category = "stats/random",
        summary = "Divide target indices randomly into training, validation, and test sets.",
        keywords = "dividerand,random,partition,train,validation,test,statistics,machine-learning",
        type_resolver(super::sampling_type),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::stats::random::sampling::dividerand"
    )]
    pub(crate) async fn dividerand_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        let parsed = super::parse_dividerand_args(args).await?;
        let [train, val, test] = super::dividerand_compute(parsed)?;
        match crate::output_count::current_output_count() {
            Some(0) => Ok(Value::OutputList(Vec::new())),
            Some(1) => Ok(Value::OutputList(vec![train])),
            Some(2) => Ok(Value::OutputList(vec![train, val])),
            Some(3) => Ok(Value::OutputList(vec![train, val, test])),
            None => Ok(train),
            Some(_) => Err(super::sampling_error(
                "dividerand",
                "dividerand: too many output arguments",
            )),
        }
    }
}

struct BootstrpArgs {
    nboot: usize,
    bootfun: Value,
    data: Vec<Value>,
    sample_axis: usize,
    sample_len: usize,
    weights: Option<Vec<f64>>,
}

struct BootstrpEval {
    bootstat: Value,
    bootsam: Option<Value>,
}

fn is_empty_function(value: &Value) -> bool {
    match value {
        Value::Tensor(t) => t.data.is_empty(),
        Value::LogicalArray(a) => a.data.is_empty(),
        Value::Cell(c) => c.data.is_empty(),
        Value::String(s) => s.is_empty(),
        Value::StringArray(a) => a.data.is_empty(),
        Value::CharArray(a) => a.data.is_empty(),
        _ => false,
    }
}

fn is_scalar_boot_arg(value: &Value) -> bool {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::String(_) => true,
        Value::Tensor(t) => t.data.len() == 1,
        Value::LogicalArray(a) => a.data.len() == 1,
        Value::StringArray(a) => a.data.len() == 1,
        Value::CharArray(a) => a.data.len() == 1,
        Value::Cell(a) => a.data.len() == 1,
        _ => false,
    }
}

fn bootstrp_data_axis(
    value: &Value,
    single_data_arg: bool,
) -> BuiltinResult<Option<(usize, usize)>> {
    if is_scalar_boot_arg(value) {
        return Ok(None);
    }
    let shape = shape_of_sampled_value(value)
        .map_err(|err| sampling_error("bootstrp", format!("bootstrp: {}", err.message())))?;
    if single_data_arg && shape.iter().filter(|dim| **dim > 1).count() == 1 {
        let axis = first_non_singleton(&shape);
        Ok(Some((axis, shape[axis])))
    } else {
        Ok(Some((0, shape[0])))
    }
}

async fn parse_bootstrp_args(args: Vec<Value>) -> BuiltinResult<BootstrpArgs> {
    if args.len() < 3 {
        return Err(sampling_error(
            "bootstrp",
            "bootstrp: expected nboot, bootfun, and at least one data argument",
        ));
    }
    let nboot = parse_positive_usize("bootstrp", &args[0], "nboot")?;
    let bootfun = gathered(args[1].clone(), "bootstrp").await?;
    let mut data = Vec::new();
    let mut weight_value = None;
    let mut idx = 2usize;
    while idx < args.len() {
        if let Some(keyword) = keyword_of(&args[idx]) {
            match keyword.as_str() {
                "weights" => {
                    let Some(value) = args.get(idx + 1) else {
                        return Err(sampling_error(
                            "bootstrp",
                            "bootstrp: weights requires a value",
                        ));
                    };
                    weight_value = Some(gathered(value.clone(), "bootstrp").await?);
                    idx += 2;
                    continue;
                }
                "options" => {
                    let Some(value) = args.get(idx + 1) else {
                        return Err(sampling_error(
                            "bootstrp",
                            "bootstrp: options requires a value",
                        ));
                    };
                    let options = gathered(value.clone(), "bootstrp").await?;
                    if !matches!(options, Value::Struct(_)) {
                        return Err(sampling_error(
                            "bootstrp",
                            "bootstrp: Options must be a statset/options struct",
                        ));
                    }
                    idx += 2;
                    continue;
                }
                _ => {}
            }
        }
        data.push(gathered(args[idx].clone(), "bootstrp").await?);
        idx += 1;
    }
    if data.is_empty() {
        return Err(sampling_error(
            "bootstrp",
            "bootstrp: at least one data argument is required",
        ));
    }
    let single_nonscalar_arg = data
        .iter()
        .filter(|value| !is_scalar_boot_arg(value))
        .count()
        == 1;
    let mut sample_axis = 0usize;
    let mut sample_len = None;
    for value in &data {
        let Some((axis, len)) = bootstrp_data_axis(value, single_nonscalar_arg)? else {
            continue;
        };
        if let Some(expected) = sample_len {
            if len != expected {
                return Err(sampling_error(
                    "bootstrp",
                    "bootstrp: nonscalar data arguments must have the same number of rows",
                ));
            }
        } else {
            sample_axis = axis;
            sample_len = Some(len);
        }
    }
    let sample_len = sample_len.unwrap_or(1);
    if sample_len == 0 {
        return Err(sampling_error("bootstrp", "bootstrp: data cannot be empty"));
    }
    let weights = match weight_value {
        Some(value) => Some(parse_weights("bootstrp", value, sample_len)?),
        None => None,
    };
    Ok(BootstrpArgs {
        nboot,
        bootfun,
        data,
        sample_axis,
        sample_len,
        weights,
    })
}

async fn bootstat_row(value: Value) -> BuiltinResult<Vec<f64>> {
    let mut value = gather_if_needed_async(&value)
        .await
        .map_err(|err| sampling_error("bootstrp", format!("bootstrp: {err}")))?;
    if let Value::OutputList(values) = value {
        if values.len() != 1 {
            return Err(sampling_error(
                "bootstrp",
                "bootstrp: bootfun must return exactly one output",
            ));
        }
        value = values.into_iter().next().unwrap_or(Value::Num(0.0));
        value = gather_if_needed_async(&value)
            .await
            .map_err(|err| sampling_error("bootstrp", format!("bootstrp: {err}")))?;
    }
    match value {
        Value::OutputList(_) => Err(sampling_error(
            "bootstrp",
            "bootstrp: bootfun must return exactly one output",
        )),
        Value::Num(v) => Ok(vec![v]),
        Value::Int(v) => Ok(vec![v.to_f64()]),
        Value::Bool(v) => Ok(vec![if v { 1.0 } else { 0.0 }]),
        Value::Tensor(t) => Ok(t.data),
        Value::LogicalArray(a) => Ok(a
            .data
            .into_iter()
            .map(|value| if value != 0 { 1.0 } else { 0.0 })
            .collect()),
        other => Err(sampling_error(
            "bootstrp",
            format!("bootstrp: bootfun must return numeric or logical values, got {other:?}"),
        )),
    }
}

fn bootsam_value(samples: &[Vec<usize>], n: usize, nboot: usize) -> BuiltinResult<Value> {
    let len = n.checked_mul(nboot).ok_or_else(|| {
        sampling_error(
            "bootstrp",
            "bootstrp: bootstrap sample index output is too large",
        )
    })?;
    let mut data = Vec::with_capacity(len);
    for sample in samples {
        data.extend(sample.iter().map(|idx| (*idx + 1) as f64));
    }
    Tensor::new(data, vec![n, nboot])
        .map(Value::Tensor)
        .map_err(|err| sampling_error("bootstrp", format!("bootstrp: {err}")))
}

fn empty_bootstat(nboot: usize) -> BuiltinResult<Value> {
    Tensor::new(Vec::new(), vec![nboot, 0])
        .map(Value::Tensor)
        .map_err(|err| sampling_error("bootstrp", format!("bootstrp: {err}")))
}

async fn bootstrp_compute(
    args: BootstrpArgs,
    include_bootsam: bool,
) -> BuiltinResult<BootstrpEval> {
    let mut samples = if include_bootsam {
        Some(Vec::with_capacity(args.nboot))
    } else {
        None
    };
    let mut rows = Vec::with_capacity(args.nboot);
    for _ in 0..args.nboot {
        let indices = sample_indices(
            "bootstrp",
            args.sample_len,
            args.sample_len,
            true,
            args.weights.as_deref(),
        )?;
        if !is_empty_function(&args.bootfun) {
            let mut callback_args = Vec::with_capacity(args.data.len());
            for value in args.data.iter().cloned() {
                if is_scalar_boot_arg(&value) {
                    callback_args.push(value);
                } else {
                    callback_args.push(sample_value_axis(
                        value,
                        args.sample_axis,
                        &indices,
                        "bootstrp",
                    )?);
                }
            }
            let result =
                crate::call_feval_async_with_outputs(args.bootfun.clone(), &callback_args, 1)
                    .await
                    .map_err(|err| {
                        sampling_error(
                            "bootstrp",
                            format!("bootstrp: bootfun failed: {}", err.message()),
                        )
                    })?;
            rows.push(bootstat_row(result).await?);
        }
        if let Some(samples) = samples.as_mut() {
            samples.push(indices);
        }
    }
    let bootsam = match samples {
        Some(samples) => Some(bootsam_value(&samples, args.sample_len, args.nboot)?),
        None => None,
    };
    let bootstat = if is_empty_function(&args.bootfun) {
        empty_bootstat(args.nboot)?
    } else {
        let width = rows.first().map(|row| row.len()).unwrap_or(0);
        if rows.iter().any(|row| row.len() != width) {
            return Err(sampling_error(
                "bootstrp",
                "bootstrp: bootfun output size must be consistent across bootstrap samples",
            ));
        }
        let len = args.nboot.checked_mul(width).ok_or_else(|| {
            sampling_error(
                "bootstrp",
                "bootstrp: bootstrap statistic output is too large",
            )
        })?;
        let mut data = vec![0.0; len];
        for (boot_idx, row) in rows.iter().enumerate() {
            for (col, value) in row.iter().enumerate() {
                data[boot_idx + col * args.nboot] = *value;
            }
        }
        Tensor::new(data, vec![args.nboot, width])
            .map(tensor::tensor_into_value)
            .map_err(|err| sampling_error("bootstrp", format!("bootstrp: {err}")))?
    };
    Ok(BootstrpEval { bootstat, bootsam })
}

pub mod bootstrp {
    use super::*;
    sampling_descriptor!(
        "bootstrp",
        BOOTSTRP_SIGNATURES,
        BuiltinOutputMode::ByRequestedOutputCount
    );

    #[runtime_builtin(
        name = "bootstrp",
        category = "stats/random",
        summary = "Bootstrap samples and evaluate a statistic.",
        keywords = "bootstrp,bootstrap,resampling,statistics,weights",
        type_resolver(super::sampling_type),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::stats::random::sampling::bootstrp"
    )]
    pub(crate) async fn bootstrp_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        let requested_outputs = crate::output_count::current_output_count();
        match requested_outputs {
            Some(0) => return Ok(Value::OutputList(Vec::new())),
            Some(count) if count > 2 => {
                return Err(super::sampling_error(
                    "bootstrp",
                    "bootstrp: too many output arguments; maximum is 2",
                ));
            }
            _ => {}
        }
        let include_bootsam = matches!(requested_outputs, Some(2));
        let args = super::parse_bootstrp_args(args).await?;
        let eval = super::bootstrp_compute(args, include_bootsam).await?;
        match requested_outputs {
            Some(1) => Ok(Value::OutputList(vec![eval.bootstat])),
            Some(2) => Ok(Value::OutputList(vec![
                eval.bootstat,
                eval.bootsam.unwrap_or(Value::Num(0.0)),
            ])),
            None => Ok(eval.bootstat),
            Some(0) | Some(_) => unreachable!("validated output count before evaluation"),
        }
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
    fn datasample_supports_cell_arrays() {
        let _lock = random::test_lock().lock().unwrap();
        random::reset_rng();
        let data = Value::Cell(
            CellArray::new(
                vec![
                    Value::from("a"),
                    Value::from("b"),
                    Value::from("c"),
                    Value::from("d"),
                ],
                2,
                2,
            )
            .unwrap(),
        );
        let out = block_on(datasample::datasample_builtin(
            data,
            vec![
                Value::Num(3.0),
                Value::from("Weights"),
                Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap()),
            ],
        ))
        .unwrap();
        match out {
            Value::Cell(cell) => {
                assert_eq!(cell.shape, vec![3, 2]);
                assert_eq!(
                    cell.data,
                    vec![
                        Value::from("c"),
                        Value::from("d"),
                        Value::from("c"),
                        Value::from("d"),
                        Value::from("c"),
                        Value::from("d"),
                    ]
                );
            }
            other => panic!("expected cell array, got {other:?}"),
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

    #[test]
    fn dividerand_partitions_indices_into_row_vectors() {
        let _lock = random::test_lock().lock().unwrap();
        random::reset_rng();
        let _guard = crate::output_count::push_output_count(Some(3));
        let out = block_on(dividerand::dividerand_builtin(vec![
            Value::Num(10.0),
            Value::Num(0.6),
            Value::Num(0.2),
            Value::Num(0.2),
        ]))
        .unwrap();
        match out {
            Value::OutputList(values) => {
                assert_eq!(values.len(), 3);
                let mut seen = Vec::new();
                for (value, expected_len) in values.iter().zip([6usize, 2, 2]) {
                    match value {
                        Value::Tensor(t) => {
                            assert_eq!(t.shape, vec![1, expected_len]);
                            seen.extend_from_slice(&t.data);
                        }
                        other => panic!("expected tensor indices, got {other:?}"),
                    }
                }
                seen.sort_by(|a, b| a.partial_cmp(b).unwrap());
                assert_eq!(seen, (1..=10).map(|idx| idx as f64).collect::<Vec<_>>());
            }
            other => panic!("expected output list, got {other:?}"),
        }
    }

    #[test]
    fn dividerand_supports_defaults_and_empty_partitions() {
        let _lock = random::test_lock().lock().unwrap();
        random::reset_rng();
        {
            let _guard = crate::output_count::push_output_count(Some(3));
            let out = block_on(dividerand::dividerand_builtin(vec![Value::Num(4.0)])).unwrap();
            match out {
                Value::OutputList(values) => {
                    let shapes = values
                        .iter()
                        .map(|value| match value {
                            Value::Tensor(t) => t.shape.clone(),
                            other => panic!("expected tensor indices, got {other:?}"),
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(shapes, vec![vec![1, 3], vec![1, 1], vec![1, 0]]);
                }
                other => panic!("expected output list, got {other:?}"),
            }
        }

        random::reset_rng();
        {
            let _guard = crate::output_count::push_output_count(Some(3));
            let out = block_on(dividerand::dividerand_builtin(vec![
                Value::Num(3.0),
                Value::Num(1.0),
                Value::Num(0.0),
                Value::Num(0.0),
            ]))
            .unwrap();
            match out {
                Value::OutputList(values) => {
                    assert!(matches!(&values[0], Value::Tensor(t) if t.shape == vec![1, 3]));
                    assert!(matches!(&values[1], Value::Tensor(t) if t.shape == vec![1, 0]));
                    assert!(matches!(&values[2], Value::Tensor(t) if t.shape == vec![1, 0]));
                }
                other => panic!("expected output list, got {other:?}"),
            }
        }
    }

    #[test]
    fn dividerand_rejects_bad_arguments() {
        let err = block_on(dividerand::dividerand_builtin(vec![
            Value::Num(3.5),
            Value::Num(0.7),
            Value::Num(0.2),
            Value::Num(0.1),
        ]))
        .unwrap_err();
        assert!(err.message().contains("nonnegative integer"));

        let err = block_on(dividerand::dividerand_builtin(vec![
            Value::Num(3.0),
            Value::Num(0.0),
            Value::Num(0.0),
            Value::Num(0.0),
        ]))
        .unwrap_err();
        assert!(err.message().contains("at least one ratio"));

        let err = block_on(dividerand::dividerand_builtin(vec![
            Value::Num(3.0),
            Value::Num(1.0),
        ]))
        .unwrap_err();
        assert!(err.message().contains("expected Q"));
    }

    #[test]
    fn dividerand_handles_extreme_ratios_and_rejects_excessive_q() {
        assert_eq!(
            dividerand_counts(10, [f64::MAX, f64::MAX, f64::MAX]),
            [4, 3, 3]
        );

        let err = block_on(dividerand::dividerand_builtin(vec![Value::Num(
            (MAX_DIVIDERAND_Q + 1) as f64,
        )]))
        .unwrap_err();
        assert!(err.message().contains("maximum supported value"));
    }

    #[test]
    fn bootstrp_weighted_mean_returns_stats_and_samples() {
        let _lock = random::test_lock().lock().unwrap();
        random::reset_rng();
        let data = Value::Tensor(Tensor::new(vec![10.0, 20.0, 30.0], vec![3, 1]).unwrap());
        let _guard = crate::output_count::push_output_count(Some(2));
        let out = block_on(bootstrp::bootstrp_builtin(vec![
            Value::Num(4.0),
            Value::FunctionHandle("mean".to_string()),
            data,
            Value::from("Weights"),
            Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0], vec![3, 1]).unwrap()),
        ]))
        .unwrap();
        match out {
            Value::OutputList(values) => {
                assert_eq!(values.len(), 2);
                match &values[0] {
                    Value::Tensor(t) => {
                        assert_eq!(t.shape, vec![4, 1]);
                        assert_eq!(t.data, vec![10.0; 4]);
                    }
                    Value::Num(value) => assert_eq!(*value, 10.0),
                    other => panic!("expected tensor bootstat, got {other:?}"),
                }
                match &values[1] {
                    Value::Tensor(t) => {
                        assert_eq!(t.shape, vec![3, 4]);
                        assert_eq!(t.data, vec![1.0; 12]);
                    }
                    other => panic!("expected tensor bootsam, got {other:?}"),
                }
            }
            other => panic!("expected output list, got {other:?}"),
        }
    }

    #[test]
    fn bootstrp_empty_function_returns_indices_without_evaluating() {
        let _lock = random::test_lock().lock().unwrap();
        random::reset_rng();
        let data = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap());
        let _guard = crate::output_count::push_output_count(Some(2));
        let out = block_on(bootstrp::bootstrp_builtin(vec![
            Value::Num(2.0),
            Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
            data,
        ]))
        .unwrap();
        match out {
            Value::OutputList(values) => {
                assert_eq!(values.len(), 2);
                match &values[0] {
                    Value::Tensor(t) => assert_eq!(t.shape, vec![2, 0]),
                    other => panic!("expected empty tensor bootstat, got {other:?}"),
                }
                match &values[1] {
                    Value::Tensor(t) => {
                        assert_eq!(t.shape, vec![3, 2]);
                        assert!(t.data.iter().all(|idx| (1.0..=3.0).contains(idx)));
                    }
                    other => panic!("expected tensor bootsam, got {other:?}"),
                }
            }
            other => panic!("expected output list, got {other:?}"),
        }
    }

    #[test]
    fn bootstrp_multiple_data_arguments_sample_rows_together() {
        let _lock = random::test_lock().lock().unwrap();
        random::reset_rng();
        let x = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap());
        let y = Value::Tensor(Tensor::new(vec![2.0, 4.0, 6.0, 8.0], vec![4, 1]).unwrap());
        let out = block_on(bootstrp::bootstrp_builtin(vec![
            Value::Num(3.0),
            Value::FunctionHandle("corr".to_string()),
            x,
            y,
        ]))
        .unwrap();
        match out {
            Value::Tensor(t) => assert_eq!(t.shape, vec![3, 1]),
            other => panic!("expected tensor bootstat, got {other:?}"),
        }
    }

    #[test]
    fn bootstrp_scalar_data_arguments_are_passed_through() {
        let _lock = random::test_lock().lock().unwrap();
        random::reset_rng();
        let out = block_on(bootstrp::bootstrp_builtin(vec![
            Value::Num(3.0),
            Value::FunctionHandle("mean".to_string()),
            Value::Num(42.0),
        ]))
        .unwrap();
        match out {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 1]);
                assert_eq!(t.data, vec![42.0; 3]);
            }
            other => panic!("expected tensor bootstat, got {other:?}"),
        }
    }

    #[test]
    fn bootstrp_row_vector_sampling_ignores_scalar_passthrough_for_axis_choice() {
        let _lock = random::test_lock().lock().unwrap();
        random::reset_rng();
        let row = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap());
        let _guard = crate::output_count::push_output_count(Some(2));
        let out = block_on(bootstrp::bootstrp_builtin(vec![
            Value::Num(2.0),
            Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
            row,
            Value::from("tag"),
        ]))
        .unwrap();
        match out {
            Value::OutputList(values) => match &values[1] {
                Value::Tensor(t) => {
                    assert_eq!(t.shape, vec![4, 2]);
                    assert!(t.data.iter().all(|idx| (1.0..=4.0).contains(idx)));
                }
                other => panic!("expected tensor bootsam, got {other:?}"),
            },
            other => panic!("expected output list, got {other:?}"),
        }
    }

    #[test]
    fn bootstrp_rejects_mismatched_data_and_extra_outputs() {
        let x = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap());
        let y = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap());
        let err = block_on(bootstrp::bootstrp_builtin(vec![
            Value::Num(2.0),
            Value::FunctionHandle("corr".to_string()),
            x,
            y,
        ]))
        .unwrap_err();
        assert!(err.message().contains("same number of rows"));

        let _guard = crate::output_count::push_output_count(Some(3));
        let data = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap());
        let err = block_on(bootstrp::bootstrp_builtin(vec![
            Value::Num(2.0),
            Value::FunctionHandle("definitely_missing_bootstrap_callback".to_string()),
            data,
        ]))
        .unwrap_err();
        assert!(err.message().contains("too many output"));
    }

    #[test]
    fn sampling_count_parsers_preserve_typed_integers_and_reject_lossy_f64() {
        assert_eq!(
            parse_positive_usize("test", &Value::Int(runmat_builtins::IntValue::U16(3)), "k",)
                .unwrap(),
            3
        );
        assert_eq!(
            parse_nonnegative_usize("test", Value::Int(runmat_builtins::IntValue::U16(0)), "Q",)
                .unwrap(),
            0
        );
        for value in [
            Value::Int(runmat_builtins::IntValue::I8(-1)),
            Value::Num(1.5),
            Value::Num(usize::MAX as f64 + 1.0),
        ] {
            assert!(parse_positive_usize("test", &value, "k").is_err());
        }
        for value in [
            Value::Int(runmat_builtins::IntValue::I8(-1)),
            Value::Num(-0.5),
            Value::Num(usize::MAX as f64 + 1.0),
        ] {
            assert!(parse_nonnegative_usize("test", value, "Q").is_err());
        }
    }
}
