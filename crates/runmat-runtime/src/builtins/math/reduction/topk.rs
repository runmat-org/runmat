//! MATLAB-compatible `maxk` and `mink` builtins.

use std::cmp::Ordering;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::arg_tokens::{tokens_from_context, ArgToken};
use crate::builtins::common::random_args::{complex_tensor_into_value, keyword_of};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

fn topk_type(args: &[Type], ctx: &ResolveContext) -> Type {
    match args.first() {
        Some(Type::Tensor { shape }) | Some(Type::Logical { shape }) => Type::Tensor {
            shape: topk_output_shape(shape.clone(), ctx),
        },
        Some(Type::Num | Type::Int | Type::Bool) => Type::Num,
        Some(Type::Unknown) => Type::Unknown,
        _ => Type::Unknown,
    }
}

fn topk_output_shape(
    shape: Option<Vec<Option<usize>>>,
    ctx: &ResolveContext,
) -> Option<Vec<Option<usize>>> {
    let mut out = shape?;
    let tokens = tokens_from_context(ctx);
    let k = tokens.get(1).and_then(token_to_nonnegative_usize);
    let dim = explicit_dim_from_tokens(&tokens).or_else(|| first_nonsingleton_dim(&out));
    let (Some(k), Some(dim)) = (k, dim) else {
        return Some(out);
    };
    let axis = dim.saturating_sub(1);
    if axis < out.len() {
        out[axis] = Some(match out[axis] {
            Some(len) => k.min(len),
            None => k,
        });
    }
    Some(out)
}

fn token_to_nonnegative_usize(token: &ArgToken) -> Option<usize> {
    let ArgToken::Number(raw) = token else {
        return None;
    };
    if !raw.is_finite() || *raw < 0.0 {
        return None;
    }
    let rounded = raw.round();
    if (rounded - raw).abs() > f64::EPSILON {
        return None;
    }
    Some(rounded as usize)
}

fn explicit_dim_from_tokens(tokens: &[ArgToken]) -> Option<usize> {
    let mut idx = 2usize;
    while idx < tokens.len() {
        match &tokens[idx] {
            ArgToken::String(text) if text == "comparisonmethod" => {
                idx += 2;
            }
            token => return token_to_nonnegative_usize(token).filter(|dim| *dim >= 1),
        }
    }
    None
}

fn first_nonsingleton_dim(shape: &[Option<usize>]) -> Option<usize> {
    shape
        .iter()
        .position(|dim| !matches!(dim, Some(1)))
        .map(|idx| idx + 1)
        .or(Some(1))
}

const VALUE_OUTPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Selected values.",
};

const INDEX_OUTPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "I",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One-based indices along the selected dimension.",
};

const OUTPUT_VALUE: [BuiltinParamDescriptor; 1] = [VALUE_OUTPUT];
const OUTPUT_VALUE_INDEX: [BuiltinParamDescriptor; 2] = [VALUE_OUTPUT, INDEX_OUTPUT];

const INPUT_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array.",
};

const INPUT_K: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "k",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of elements to select from each slice.",
};

const INPUT_DIM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "dim",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Dimension to operate along.",
};

const INPUT_OPTION_NAME: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "optionName",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"ComparisonMethod\""),
    description: "Name-value option name.",
};

const INPUT_OPTION_VALUE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "optionValue",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"auto\""),
    description: "Name-value option value.",
};

const INPUTS_A_K: [BuiltinParamDescriptor; 2] = [INPUT_A, INPUT_K];
const INPUTS_A_K_DIM: [BuiltinParamDescriptor; 3] = [INPUT_A, INPUT_K, INPUT_DIM];
const INPUTS_A_K_OPTIONS: [BuiltinParamDescriptor; 4] =
    [INPUT_A, INPUT_K, INPUT_OPTION_NAME, INPUT_OPTION_VALUE];
const INPUTS_A_K_DIM_OPTIONS: [BuiltinParamDescriptor; 5] = [
    INPUT_A,
    INPUT_K,
    INPUT_DIM,
    INPUT_OPTION_NAME,
    INPUT_OPTION_VALUE,
];

const MAXK_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "B = maxk(A, k)",
        inputs: &INPUTS_A_K,
        outputs: &OUTPUT_VALUE,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = maxk(A, k)",
        inputs: &INPUTS_A_K,
        outputs: &OUTPUT_VALUE_INDEX,
    },
    BuiltinSignatureDescriptor {
        label: "B = maxk(A, k, dim)",
        inputs: &INPUTS_A_K_DIM,
        outputs: &OUTPUT_VALUE,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = maxk(A, k, dim)",
        inputs: &INPUTS_A_K_DIM,
        outputs: &OUTPUT_VALUE_INDEX,
    },
    BuiltinSignatureDescriptor {
        label: "B = maxk(A, k, \"ComparisonMethod\", method)",
        inputs: &INPUTS_A_K_OPTIONS,
        outputs: &OUTPUT_VALUE,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = maxk(A, k, \"ComparisonMethod\", method)",
        inputs: &INPUTS_A_K_OPTIONS,
        outputs: &OUTPUT_VALUE_INDEX,
    },
    BuiltinSignatureDescriptor {
        label: "B = maxk(A, k, dim, \"ComparisonMethod\", method)",
        inputs: &INPUTS_A_K_DIM_OPTIONS,
        outputs: &OUTPUT_VALUE,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = maxk(A, k, dim, \"ComparisonMethod\", method)",
        inputs: &INPUTS_A_K_DIM_OPTIONS,
        outputs: &OUTPUT_VALUE_INDEX,
    },
];

const MINK_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "B = mink(A, k)",
        inputs: &INPUTS_A_K,
        outputs: &OUTPUT_VALUE,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = mink(A, k)",
        inputs: &INPUTS_A_K,
        outputs: &OUTPUT_VALUE_INDEX,
    },
    BuiltinSignatureDescriptor {
        label: "B = mink(A, k, dim)",
        inputs: &INPUTS_A_K_DIM,
        outputs: &OUTPUT_VALUE,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = mink(A, k, dim)",
        inputs: &INPUTS_A_K_DIM,
        outputs: &OUTPUT_VALUE_INDEX,
    },
    BuiltinSignatureDescriptor {
        label: "B = mink(A, k, \"ComparisonMethod\", method)",
        inputs: &INPUTS_A_K_OPTIONS,
        outputs: &OUTPUT_VALUE,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = mink(A, k, \"ComparisonMethod\", method)",
        inputs: &INPUTS_A_K_OPTIONS,
        outputs: &OUTPUT_VALUE_INDEX,
    },
    BuiltinSignatureDescriptor {
        label: "B = mink(A, k, dim, \"ComparisonMethod\", method)",
        inputs: &INPUTS_A_K_DIM_OPTIONS,
        outputs: &OUTPUT_VALUE,
    },
    BuiltinSignatureDescriptor {
        label: "[B, I] = mink(A, k, dim, \"ComparisonMethod\", method)",
        inputs: &INPUTS_A_K_DIM_OPTIONS,
        outputs: &OUTPUT_VALUE_INDEX,
    },
];

const TOPK_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TOPK.INVALID_ARGUMENT",
    identifier: Some("RunMat:topk:InvalidArgument"),
    when: "Argument count, k, dimension, or option values are invalid.",
    message: "topk: invalid argument",
};

const TOPK_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TOPK.INVALID_INPUT",
    identifier: Some("RunMat:topk:InvalidInput"),
    when: "Input values cannot be converted to supported top-k domains.",
    message: "topk: invalid input",
};

const TOPK_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TOPK.INTERNAL",
    identifier: Some("RunMat:topk:Internal"),
    when: "Top-k selection fails due to gather or allocation internals.",
    message: "topk: internal failure",
};

const TOPK_ERRORS: [BuiltinErrorDescriptor; 3] = [
    TOPK_ERROR_INVALID_ARGUMENT,
    TOPK_ERROR_INVALID_INPUT,
    TOPK_ERROR_INTERNAL,
];

pub const MAXK_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MAXK_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TOPK_ERRORS,
};

pub const MINK_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MINK_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TOPK_ERRORS,
};

#[runtime_builtin(
    name = "maxk",
    category = "math/reduction",
    summary = "Return the k largest elements along a dimension.",
    keywords = "maxk,top-k,maximum,reduction,indices",
    type_resolver(topk_type),
    descriptor(crate::builtins::math::reduction::topk::MAXK_DESCRIPTOR),
    builtin_path = "crate::builtins::math::reduction::topk"
)]
async fn maxk_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    evaluate_topk(TopKKind::Max, value, &rest)
        .await?
        .into_value()
}

#[runtime_builtin(
    name = "mink",
    category = "math/reduction",
    summary = "Return the k smallest elements along a dimension.",
    keywords = "mink,top-k,minimum,reduction,indices",
    type_resolver(topk_type),
    descriptor(crate::builtins::math::reduction::topk::MINK_DESCRIPTOR),
    builtin_path = "crate::builtins::math::reduction::topk"
)]
async fn mink_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    evaluate_topk(TopKKind::Min, value, &rest)
        .await?
        .into_value()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TopKKind {
    Max,
    Min,
}

impl TopKKind {
    fn name(self) -> &'static str {
        match self {
            TopKKind::Max => "maxk",
            TopKKind::Min => "mink",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ComparisonMethod {
    Auto,
    Abs,
    Real,
}

#[derive(Debug)]
pub struct TopKEvaluation {
    values: Value,
    indices: Value,
}

impl TopKEvaluation {
    fn into_value(self) -> BuiltinResult<Value> {
        if let Some(out_count) = crate::output_count::current_output_count() {
            if out_count == 0 {
                return Ok(Value::OutputList(Vec::new()));
            }
            if out_count == 1 {
                return Ok(Value::OutputList(vec![self.values]));
            }
            return Ok(crate::output_count::output_list_with_padding(
                out_count,
                vec![self.values, self.indices],
            ));
        }
        Ok(self.values)
    }
}

#[derive(Clone, Debug)]
struct TopKArgs {
    k: usize,
    dim: Option<usize>,
    comparison: ComparisonMethod,
}

async fn evaluate_topk(
    kind: TopKKind,
    value: Value,
    rest: &[Value],
) -> BuiltinResult<TopKEvaluation> {
    crate::builtins::common::validation::reject_typed_complex_integer(&value, kind.name())?;
    for argument in rest {
        crate::builtins::common::validation::reject_typed_complex_integer(argument, kind.name())?;
    }
    let args = parse_topk_args(kind, rest).await?;
    let input = gather_topk_input(kind, value).await?;
    match input {
        TopKInput::Real(tensor) => evaluate_real(kind, tensor, &args),
        TopKInput::Complex(tensor) => evaluate_complex(kind, tensor, &args),
    }
}

async fn parse_topk_args(kind: TopKKind, rest: &[Value]) -> BuiltinResult<TopKArgs> {
    if rest.is_empty() {
        return Err(topk_invalid_argument(kind, "k is required"));
    }
    let k = parse_k(kind, &rest[0]).await?;
    let mut dim = None;
    let mut comparison = ComparisonMethod::Auto;
    let mut idx = 1usize;
    while idx < rest.len() {
        if let Some(keyword) = keyword_of(&rest[idx]) {
            if keyword == "comparisonmethod" {
                let Some(value) = rest.get(idx + 1) else {
                    return Err(topk_invalid_argument(
                        kind,
                        "expected a value after 'ComparisonMethod'",
                    ));
                };
                comparison = parse_comparison(kind, value)?;
                idx += 2;
                continue;
            }
        }
        if dim.is_none() {
            if let Some(parsed) = tensor::dimension_from_value_async(&rest[idx], kind.name(), false)
                .await
                .map_err(|message| topk_invalid_argument(kind, message))?
            {
                dim = Some(parsed);
                idx += 1;
                continue;
            }
        }
        return Err(topk_invalid_argument(kind, "unrecognized argument"));
    }
    Ok(TopKArgs { k, dim, comparison })
}

async fn parse_k(kind: TopKKind, value: &Value) -> BuiltinResult<usize> {
    let Some(raw) = tensor::scalar_f64_from_value_async(value)
        .await
        .map_err(|message| topk_invalid_argument(kind, message))?
    else {
        return Err(topk_invalid_argument(kind, "k must be a numeric scalar"));
    };
    if !raw.is_finite() {
        return Err(topk_invalid_argument(kind, "k must be finite"));
    }
    let rounded = raw.round();
    if (rounded - raw).abs() > 1e-6 {
        return Err(topk_invalid_argument(kind, "k must be an integer"));
    }
    if rounded < 0.0 {
        return Err(topk_invalid_argument(kind, "k must be nonnegative"));
    }
    Ok(rounded as usize)
}

fn parse_comparison(kind: TopKKind, value: &Value) -> BuiltinResult<ComparisonMethod> {
    let Some(keyword) = keyword_of(value) else {
        return Err(topk_invalid_argument(
            kind,
            "'ComparisonMethod' expects a string value",
        ));
    };
    match keyword.as_str() {
        "auto" => Ok(ComparisonMethod::Auto),
        "abs" | "magnitude" => Ok(ComparisonMethod::Abs),
        "real" => Ok(ComparisonMethod::Real),
        other => Err(topk_invalid_argument(
            kind,
            format!("unsupported ComparisonMethod '{other}'"),
        )),
    }
}

enum TopKInput {
    Real(Tensor),
    Complex(ComplexTensor),
}

async fn gather_topk_input(kind: TopKKind, value: Value) -> BuiltinResult<TopKInput> {
    let host = match value {
        Value::GpuTensor(handle) => Value::Tensor(
            gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|err| topk_internal(kind, err.message()))?,
        ),
        other => other,
    };
    match host {
        Value::Tensor(tensor) => Ok(TopKInput::Real(tensor)),
        Value::LogicalArray(logical) => Ok(TopKInput::Real(
            tensor::logical_to_tensor(&logical).map_err(|message| topk_internal(kind, message))?,
        )),
        Value::Num(value) => Ok(TopKInput::Real(
            Tensor::new(vec![value], vec![1, 1]).map_err(|message| topk_internal(kind, message))?,
        )),
        Value::Int(value) => Ok(TopKInput::Real(
            Tensor::new(vec![value.to_f64()], vec![1, 1])
                .map_err(|message| topk_internal(kind, message))?,
        )),
        Value::Bool(value) => Ok(TopKInput::Real(
            Tensor::new(vec![if value { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|message| topk_internal(kind, message))?,
        )),
        Value::Complex(re, im) => Ok(TopKInput::Complex(
            ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|message| topk_internal(kind, message))?,
        )),
        Value::ComplexTensor(tensor) => Ok(TopKInput::Complex(tensor)),
        _ => Err(topk_invalid_input(
            kind,
            "expected numeric, logical, or complex input",
        )),
    }
}

fn evaluate_real(kind: TopKKind, tensor: Tensor, args: &TopKArgs) -> BuiltinResult<TopKEvaluation> {
    let shape = normalize_shape(tensor.shape.clone());
    let dim = selected_dim(&shape, args.dim);
    let axis = dim.saturating_sub(1);
    let axis_len = shape.get(axis).copied().unwrap_or(1);
    let take = args.k.min(axis_len);
    if axis >= shape.len() {
        let indices = Tensor::new(vec![1.0; tensor.data.len()], shape)
            .map_err(|message| topk_internal(kind, message))?;
        return Ok(TopKEvaluation {
            values: tensor::tensor_into_value(tensor),
            indices: tensor::tensor_into_value(indices),
        });
    }
    let output_shape = output_shape_for_topk(&shape, axis, take);
    if tensor.data.is_empty() || take == 0 {
        let values = Tensor::new(Vec::new(), output_shape.clone())
            .map_err(|message| topk_internal(kind, message))?;
        let indices = Tensor::new(Vec::new(), output_shape)
            .map_err(|message| topk_internal(kind, message))?;
        return Ok(TopKEvaluation {
            values: tensor::tensor_into_value(values),
            indices: tensor::tensor_into_value(indices),
        });
    }

    let input_strides = compute_strides(&shape);
    let output_strides = compute_strides(&output_shape);
    let output_len = checked_element_count(kind, &output_shape)?;
    let mut values = vec![0.0; output_len];
    let mut indices = vec![0.0; output_len];
    let mut coords = vec![0usize; output_shape.len()];
    for out_base in 0..output_len {
        if coords.get(axis).copied().unwrap_or(0) != 0 {
            increment_coords(&mut coords, &output_shape);
            continue;
        }
        let mut entries = Vec::with_capacity(axis_len);
        for reduce_idx in 0..axis_len {
            let mut input_coords = coords.clone();
            if axis >= input_coords.len() {
                input_coords.resize(axis + 1, 0);
            }
            input_coords[axis] = reduce_idx;
            let input_index = map_linear_index(&input_coords, &input_strides);
            entries.push(RealEntry {
                value: tensor.data[input_index],
                index: reduce_idx,
            });
        }
        entries.sort_by(|a, b| compare_real_entries(kind, args.comparison, a, b));
        for (rank, entry) in entries.iter().take(take).enumerate() {
            let mut out_coords = coords.clone();
            out_coords[axis] = rank;
            let out_idx = map_linear_index(&out_coords, &output_strides);
            values[out_idx] = entry.value;
            indices[out_idx] = (entry.index + 1) as f64;
        }
        let _ = out_base;
        increment_coords(&mut coords, &output_shape);
    }

    let values = Tensor::new(values, output_shape.clone())
        .map_err(|message| topk_internal(kind, message))?;
    let indices =
        Tensor::new(indices, output_shape).map_err(|message| topk_internal(kind, message))?;
    Ok(TopKEvaluation {
        values: tensor::tensor_into_value(values),
        indices: tensor::tensor_into_value(indices),
    })
}

fn evaluate_complex(
    kind: TopKKind,
    tensor: ComplexTensor,
    args: &TopKArgs,
) -> BuiltinResult<TopKEvaluation> {
    let shape = normalize_shape(tensor.shape.clone());
    let dim = selected_dim(&shape, args.dim);
    let axis = dim.saturating_sub(1);
    let axis_len = shape.get(axis).copied().unwrap_or(1);
    let take = args.k.min(axis_len);
    if axis >= shape.len() {
        let indices = Tensor::new(vec![1.0; tensor.data.len()], shape)
            .map_err(|message| topk_internal(kind, message))?;
        return Ok(TopKEvaluation {
            values: complex_tensor_into_value(tensor),
            indices: tensor::tensor_into_value(indices),
        });
    }
    let output_shape = output_shape_for_topk(&shape, axis, take);
    if tensor.data.is_empty() || take == 0 {
        let values = ComplexTensor::new(Vec::new(), output_shape.clone())
            .map_err(|message| topk_internal(kind, message))?;
        let indices = Tensor::new(Vec::new(), output_shape)
            .map_err(|message| topk_internal(kind, message))?;
        return Ok(TopKEvaluation {
            values: complex_tensor_into_value(values),
            indices: tensor::tensor_into_value(indices),
        });
    }

    let input_strides = compute_strides(&shape);
    let output_strides = compute_strides(&output_shape);
    let output_len = checked_element_count(kind, &output_shape)?;
    let mut values = vec![(0.0, 0.0); output_len];
    let mut indices = vec![0.0; output_len];
    let mut coords = vec![0usize; output_shape.len()];
    for out_base in 0..output_len {
        if coords.get(axis).copied().unwrap_or(0) != 0 {
            increment_coords(&mut coords, &output_shape);
            continue;
        }
        let mut entries = Vec::with_capacity(axis_len);
        for reduce_idx in 0..axis_len {
            let mut input_coords = coords.clone();
            if axis >= input_coords.len() {
                input_coords.resize(axis + 1, 0);
            }
            input_coords[axis] = reduce_idx;
            let input_index = map_linear_index(&input_coords, &input_strides);
            entries.push(ComplexEntry {
                value: tensor.data[input_index],
                index: reduce_idx,
            });
        }
        entries.sort_by(|a, b| compare_complex_entries(kind, args.comparison, a, b));
        for (rank, entry) in entries.iter().take(take).enumerate() {
            let mut out_coords = coords.clone();
            out_coords[axis] = rank;
            let out_idx = map_linear_index(&out_coords, &output_strides);
            values[out_idx] = entry.value;
            indices[out_idx] = (entry.index + 1) as f64;
        }
        let _ = out_base;
        increment_coords(&mut coords, &output_shape);
    }

    let values = ComplexTensor::new(values, output_shape.clone())
        .map_err(|message| topk_internal(kind, message))?;
    let indices =
        Tensor::new(indices, output_shape).map_err(|message| topk_internal(kind, message))?;
    Ok(TopKEvaluation {
        values: complex_tensor_into_value(values),
        indices: tensor::tensor_into_value(indices),
    })
}

#[derive(Clone, Copy)]
struct RealEntry {
    value: f64,
    index: usize,
}

#[derive(Clone, Copy)]
struct ComplexEntry {
    value: (f64, f64),
    index: usize,
}

fn compare_real_entries(
    kind: TopKKind,
    method: ComparisonMethod,
    a: &RealEntry,
    b: &RealEntry,
) -> Ordering {
    let ordering = compare_real_values(method, a.value, b.value);
    let ordering = match kind {
        TopKKind::Max => ordering.reverse(),
        TopKKind::Min => ordering,
    };
    ordering.then_with(|| a.index.cmp(&b.index))
}

fn compare_real_values(method: ComparisonMethod, a: f64, b: f64) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => match method {
            ComparisonMethod::Auto | ComparisonMethod::Real => {
                a.partial_cmp(&b).unwrap_or(Ordering::Equal)
            }
            ComparisonMethod::Abs => a
                .abs()
                .partial_cmp(&b.abs())
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.partial_cmp(&b).unwrap_or(Ordering::Equal)),
        },
    }
}

fn compare_complex_entries(
    kind: TopKKind,
    method: ComparisonMethod,
    a: &ComplexEntry,
    b: &ComplexEntry,
) -> Ordering {
    let ordering = compare_complex_values(method, a.value, b.value);
    let ordering = match kind {
        TopKKind::Max => ordering.reverse(),
        TopKKind::Min => ordering,
    };
    ordering.then_with(|| a.index.cmp(&b.index))
}

fn compare_complex_values(method: ComparisonMethod, a: (f64, f64), b: (f64, f64)) -> Ordering {
    let a_nan = a.0.is_nan() || a.1.is_nan();
    let b_nan = b.0.is_nan() || b.1.is_nan();
    match (a_nan, b_nan) {
        (true, true) => return Ordering::Equal,
        (true, false) => return Ordering::Greater,
        (false, true) => return Ordering::Less,
        (false, false) => {}
    }
    match method {
        ComparisonMethod::Real => {
            a.0.partial_cmp(&b.0)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
        }
        ComparisonMethod::Auto | ComparisonMethod::Abs => {
            let amag = a.0.hypot(a.1);
            let bmag = b.0.hypot(b.1);
            amag.partial_cmp(&bmag)
                .unwrap_or(Ordering::Equal)
                .then_with(|| {
                    a.1.atan2(a.0)
                        .partial_cmp(&b.1.atan2(b.0))
                        .unwrap_or(Ordering::Equal)
                })
        }
    }
}

fn normalize_shape(mut shape: Vec<usize>) -> Vec<usize> {
    if shape.is_empty() {
        shape.push(1);
        shape.push(1);
    }
    if shape.len() == 1 {
        shape.push(1);
    }
    shape
}

fn selected_dim(shape: &[usize], requested: Option<usize>) -> usize {
    if let Some(dim) = requested {
        return dim;
    }
    shape
        .iter()
        .position(|&len| len > 1)
        .map(|idx| idx + 1)
        .unwrap_or(1)
}

fn output_shape_for_topk(shape: &[usize], axis: usize, take: usize) -> Vec<usize> {
    let mut output = shape.to_vec();
    if axis < output.len() {
        output[axis] = take;
    }
    output
}

fn checked_element_count(kind: TopKKind, shape: &[usize]) -> BuiltinResult<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| topk_internal(kind, "shape element count overflow"))
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for idx in 1..shape.len() {
        strides[idx] = strides[idx - 1].saturating_mul(shape[idx - 1]);
    }
    strides
}

fn map_linear_index(coords: &[usize], strides: &[usize]) -> usize {
    coords
        .iter()
        .zip(strides.iter())
        .fold(0usize, |acc, (&coord, &stride)| {
            acc.saturating_add(coord.saturating_mul(stride))
        })
}

fn increment_coords(coords: &mut [usize], shape: &[usize]) {
    for dim in 0..coords.len() {
        coords[dim] += 1;
        if coords[dim] < shape[dim] {
            break;
        }
        coords[dim] = 0;
    }
}

fn topk_error(
    kind: TopKKind,
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!(
        "{}: {}: {}",
        kind.name(),
        descriptor.message,
        detail.as_ref()
    ))
    .with_builtin(kind.name());
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn topk_invalid_argument(kind: TopKKind, detail: impl AsRef<str>) -> RuntimeError {
    topk_error(kind, &TOPK_ERROR_INVALID_ARGUMENT, detail)
}

fn topk_invalid_input(kind: TopKKind, detail: impl AsRef<str>) -> RuntimeError {
    topk_error(kind, &TOPK_ERROR_INVALID_INPUT, detail)
}

fn topk_internal(kind: TopKKind, detail: impl AsRef<str>) -> RuntimeError {
    topk_error(kind, &TOPK_ERROR_INTERNAL, detail)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::{IntegerComplexStorage, IntegerStorage};

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(data, shape).unwrap())
    }

    #[tokio::test]
    async fn topk_rejects_typed_complex_integer_inputs() {
        let input = Value::ComplexTensor(
            ComplexTensor::new_integer(
                IntegerComplexStorage::new(
                    IntegerStorage::I64(vec![i64::MAX]),
                    IntegerStorage::I64(vec![-1]),
                )
                .expect("storage"),
                vec![1, 1],
            )
            .expect("tensor"),
        );
        let err = evaluate_topk(TopKKind::Max, input, &[Value::Num(1.0)])
            .await
            .expect_err("typed complex integer input must reject");
        assert!(err.message().contains("complex numbers with integer types"));
    }

    fn outputs(value: Value) -> Vec<Value> {
        match value {
            Value::OutputList(values) => values,
            other => vec![other],
        }
    }

    #[tokio::test]
    async fn maxk_defaults_to_first_nonsingleton_dimension() {
        let input = tensor(vec![1.0, 4.0, 3.0, 2.0, 6.0, 5.0], vec![3, 2]);
        let eval = evaluate_topk(TopKKind::Max, input, &[Value::Num(2.0)])
            .await
            .unwrap();
        let Value::Tensor(values) = eval.values else {
            panic!("expected tensor");
        };
        assert_eq!(values.shape, vec![2, 2]);
        assert_eq!(values.data, vec![4.0, 3.0, 6.0, 5.0]);
        let Value::Tensor(indices) = eval.indices else {
            panic!("expected indices");
        };
        assert_eq!(indices.data, vec![2.0, 3.0, 2.0, 3.0]);
    }

    #[tokio::test]
    async fn mink_supports_explicit_row_dimension() {
        let input = tensor(vec![3.0, 4.0, 1.0, 2.0, 5.0, 6.0], vec![2, 3]);
        let eval = evaluate_topk(TopKKind::Min, input, &[Value::Num(2.0), Value::Num(2.0)])
            .await
            .unwrap();
        let Value::Tensor(values) = eval.values else {
            panic!("expected tensor");
        };
        assert_eq!(values.shape, vec![2, 2]);
        assert_eq!(values.data, vec![1.0, 2.0, 3.0, 4.0]);
        let Value::Tensor(indices) = eval.indices else {
            panic!("expected indices");
        };
        assert_eq!(indices.data, vec![2.0, 2.0, 1.0, 1.0]);
    }

    #[tokio::test]
    async fn maxk_clamps_k_to_dimension_length() {
        let input = tensor(vec![2.0, 1.0, 3.0], vec![3, 1]);
        let eval = evaluate_topk(TopKKind::Max, input, &[Value::Num(10.0)])
            .await
            .unwrap();
        let Value::Tensor(values) = eval.values else {
            panic!("expected tensor");
        };
        assert_eq!(values.shape, vec![3, 1]);
        assert_eq!(values.data, vec![3.0, 2.0, 1.0]);
    }

    #[tokio::test]
    async fn topk_dimension_greater_than_rank_returns_input_and_one_indices() {
        let input = tensor(vec![2.0, 1.0, 3.0], vec![3, 1]);
        let eval = evaluate_topk(TopKKind::Max, input, &[Value::Num(2.0), Value::Num(5.0)])
            .await
            .unwrap();
        let Value::Tensor(values) = eval.values else {
            panic!("expected tensor");
        };
        assert_eq!(values.shape, vec![3, 1]);
        assert_eq!(values.data, vec![2.0, 1.0, 3.0]);
        let Value::Tensor(indices) = eval.indices else {
            panic!("expected indices");
        };
        assert_eq!(indices.data, vec![1.0, 1.0, 1.0]);
    }

    #[tokio::test]
    async fn topk_rejects_invalid_k() {
        let input = tensor(vec![1.0, 2.0], vec![2, 1]);
        let err = evaluate_topk(TopKKind::Max, input, &[Value::Num(-1.0)])
            .await
            .unwrap_err();
        assert!(err.message().contains("k must be nonnegative"));
    }

    #[tokio::test]
    async fn topk_allows_zero_k() {
        let input = tensor(vec![1.0, 2.0], vec![2, 1]);
        let eval = evaluate_topk(TopKKind::Max, input, &[Value::Num(0.0)])
            .await
            .unwrap();
        let Value::Tensor(values) = eval.values else {
            panic!("expected tensor");
        };
        assert_eq!(values.shape, vec![0, 1]);
        assert!(values.data.is_empty());
    }

    #[tokio::test]
    async fn maxk_real_abs_comparison_uses_magnitude() {
        let input = tensor(vec![-3.0, 2.0], vec![1, 2]);
        let eval = evaluate_topk(
            TopKKind::Max,
            input,
            &[
                Value::Num(1.0),
                Value::from("ComparisonMethod"),
                Value::from("abs"),
            ],
        )
        .await
        .unwrap();
        assert_eq!(eval.values, Value::Num(-3.0));
        assert_eq!(eval.indices, Value::Num(1.0));
    }

    #[tokio::test]
    async fn maxk_supports_complex_comparison_method_real() {
        let input = Value::ComplexTensor(
            ComplexTensor::new(vec![(1.0, 10.0), (3.0, -1.0), (2.0, 0.0)], vec![3, 1]).unwrap(),
        );
        let eval = evaluate_topk(
            TopKKind::Max,
            input,
            &[
                Value::Num(2.0),
                Value::from("ComparisonMethod"),
                Value::from("real"),
            ],
        )
        .await
        .unwrap();
        let Value::ComplexTensor(values) = eval.values else {
            panic!("expected complex tensor");
        };
        assert_eq!(values.data, vec![(3.0, -1.0), (2.0, 0.0)]);
    }

    #[tokio::test]
    async fn builtin_wraps_multiple_outputs() {
        let input = tensor(vec![1.0, 3.0, 2.0], vec![1, 3]);
        let _guard = crate::output_count::push_output_count(Some(2));
        let result = maxk_builtin(input, vec![Value::Num(2.0)]).await.unwrap();
        let values = outputs(result);
        assert_eq!(values.len(), 2);
        let Value::Tensor(selected) = &values[0] else {
            panic!("expected tensor");
        };
        assert_eq!(selected.data, vec![3.0, 2.0]);
        let Value::Tensor(indices) = &values[1] else {
            panic!("expected indices");
        };
        assert_eq!(indices.data, vec![2.0, 3.0]);
    }
}
