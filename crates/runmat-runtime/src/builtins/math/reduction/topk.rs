//! MATLAB-compatible `maxk` and `mink` builtins.

use std::cmp::Ordering;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, IntValue, IntegerStorage, LogicalArray, NumericScalar, ResolveContext, Tensor,
    Type, Value,
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
        Some(Type::Num) => Type::Num,
        Some(Type::Int) => Type::Int,
        Some(Type::Bool) => Type::Bool,
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

const MAXK_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "maxk-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "maxk with a resident GPU input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:MaxkGpuInputExtension"),
};

const MINK_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "mink-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "mink with a resident GPU input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:MinkGpuInputExtension"),
};

const MAXK_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [MAXK_GPU_INPUT_EXTENSION];
const MINK_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [MINK_GPU_INPUT_EXTENSION];

const TOPK_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented input-array domain includes all eight real integer classes.",
    },
    BuiltinIntegerInputCapability {
        name: "k",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented nonnegative integer-scalar selector is parsed exactly from every integer class; integer-valued scalar double is also accepted.",
    },
    BuiltinIntegerInputCapability {
        name: "dim",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The optional documented positive integer-scalar dimension is parsed exactly from every integer class; integer-valued scalar double is also accepted.",
    },
];

const MAXK_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[B, I] = maxk(integer_A, integer_k, integer_dim)",
        inputs: &TOPK_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "B preserves A's exact integer class and stable equal-value order; optional I is one-based double. Resident execution is a separately gated RunMat extension.",
    }];

const MINK_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[B, I] = mink(integer_A, integer_k, integer_dim)",
        inputs: &TOPK_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "B preserves A's exact integer class and stable equal-value order; optional I is one-based double. Resident execution is a separately gated RunMat extension.",
    }];

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
    accel = "sink",
    sink = true,
    type_resolver(topk_type),
    descriptor(crate::builtins::math::reduction::topk::MAXK_DESCRIPTOR),
    extensions(MAXK_EXTENSIONS),
    integer_capabilities(MAXK_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::reduction::topk"
)]
async fn maxk_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    evaluate_topk(TopKKind::Max, value, &rest)
        .await?
        .into_value(TopKKind::Max)
}

#[runtime_builtin(
    name = "mink",
    category = "math/reduction",
    summary = "Return the k smallest elements along a dimension.",
    keywords = "mink,top-k,minimum,reduction,indices",
    accel = "sink",
    sink = true,
    type_resolver(topk_type),
    descriptor(crate::builtins::math::reduction::topk::MINK_DESCRIPTOR),
    extensions(MINK_EXTENSIONS),
    integer_capabilities(MINK_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::reduction::topk"
)]
async fn mink_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    evaluate_topk(TopKKind::Min, value, &rest)
        .await?
        .into_value(TopKKind::Min)
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

    fn gpu_extension(self) -> &'static BuiltinExtensionDescriptor {
        match self {
            TopKKind::Max => &MAXK_GPU_INPUT_EXTENSION,
            TopKKind::Min => &MINK_GPU_INPUT_EXTENSION,
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
    fn into_value(self, kind: TopKKind) -> BuiltinResult<Value> {
        if let Some(out_count) = crate::output_count::current_output_count() {
            if out_count == 0 {
                return Ok(Value::OutputList(Vec::new()));
            }
            if out_count == 1 {
                return Ok(Value::OutputList(vec![self.values]));
            }
            if out_count == 2 {
                return Ok(Value::OutputList(vec![self.values, self.indices]));
            }
            return Err(topk_invalid_argument(
                kind,
                "too many output arguments; maximum is 2",
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
    let gpu_provider = match &value {
        Value::GpuTensor(handle) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                kind.gpu_extension(),
                kind.name(),
            )?;
            runmat_accelerate_api::provider_for_handle(handle)
                .or_else(runmat_accelerate_api::provider)
        }
        _ => None,
    };
    let args = parse_topk_args(kind, rest).await?;
    let input = gather_topk_input(kind, value).await?;
    let evaluation = match input {
        TopKInput::Real(tensor) => evaluate_real(kind, tensor, &args),
        TopKInput::Complex(tensor) => evaluate_complex(kind, tensor, &args),
        TopKInput::Logical(logical) => evaluate_logical(kind, logical, &args),
    }?;
    match gpu_provider {
        Some(provider) => upload_topk_evaluation(kind, provider, evaluation),
        None => Ok(evaluation),
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
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return parse_integer_k(kind, &integer);
    }
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
    if rounded > usize::MAX as f64 || (usize::BITS == 64 && rounded == usize::MAX as f64) {
        return Err(topk_invalid_argument(
            kind,
            "k is outside the supported range",
        ));
    }
    Ok(rounded as usize)
}

fn parse_integer_k(kind: TopKKind, value: &IntValue) -> BuiltinResult<usize> {
    value.try_to_usize().ok_or_else(|| {
        topk_invalid_argument(
            kind,
            "k must be a nonnegative integer in the supported range",
        )
    })
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
    Logical(LogicalArray),
}

async fn gather_topk_input(kind: TopKKind, value: Value) -> BuiltinResult<TopKInput> {
    let host = gpu_helpers::gather_value_async(&value)
        .await
        .map_err(|err| topk_internal(kind, err.message()))?;
    match host {
        Value::Tensor(tensor) => Ok(TopKInput::Real(tensor)),
        Value::LogicalArray(logical) => Ok(TopKInput::Logical(logical)),
        Value::Num(value) => Ok(TopKInput::Real(
            Tensor::new(vec![value], vec![1, 1]).map_err(|message| topk_internal(kind, message))?,
        )),
        Value::Int(value) => Ok(TopKInput::Real(
            Tensor::new_integer(IntegerStorage::from_scalar(value), vec![1, 1])
                .map_err(|message| topk_internal(kind, message))?,
        )),
        Value::Bool(value) => Ok(TopKInput::Logical(
            LogicalArray::new(vec![u8::from(value)], vec![1, 1])
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

fn evaluate_logical(
    kind: TopKKind,
    logical: LogicalArray,
    args: &TopKArgs,
) -> BuiltinResult<TopKEvaluation> {
    let shape = logical.shape.clone();
    let tensor = Tensor::new_integer(IntegerStorage::U8(logical.data), shape)
        .map_err(|message| topk_internal(kind, message))?;
    let evaluation = evaluate_real(kind, tensor, args)?;
    let values = match evaluation.values {
        Value::Int(IntValue::U8(value)) => Value::Bool(value != 0),
        Value::Tensor(tensor) => {
            let shape = tensor.shape.clone();
            let IntegerStorage::U8(values) = tensor
                .into_numeric_storage()
                .map_err(|error| topk_internal(kind, error))?
                .into_integer_storage()
                .map_err(|_| topk_internal(kind, "logical selection lost integer storage"))?
            else {
                return Err(topk_internal(
                    kind,
                    "logical selection changed its storage class",
                ));
            };
            Value::LogicalArray(
                LogicalArray::new(values, shape).map_err(|message| topk_internal(kind, message))?,
            )
        }
        other => {
            return Err(topk_internal(
                kind,
                format!("logical selection produced unexpected value {other:?}"),
            ))
        }
    };
    Ok(TopKEvaluation {
        values,
        indices: evaluation.indices,
    })
}

fn upload_topk_evaluation(
    kind: TopKKind,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    evaluation: TopKEvaluation,
) -> BuiltinResult<TopKEvaluation> {
    Ok(TopKEvaluation {
        values: upload_topk_value(kind, provider, evaluation.values)?,
        indices: upload_topk_value(kind, provider, evaluation.indices)?,
    })
}

fn upload_topk_value(
    kind: TopKKind,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    value: Value,
) -> BuiltinResult<Value> {
    let upload_tensor = |tensor: Tensor, logical: bool| -> BuiltinResult<Value> {
        let handle = gpu_helpers::upload_tensor(provider, &tensor)
            .map_err(|error| topk_internal(kind, format!("GPU upload failed: {error}")))?;
        Ok(if logical {
            gpu_helpers::logical_gpu_value(handle)
        } else {
            gpu_helpers::resident_gpu_value(handle)
        })
    };
    match value {
        Value::Tensor(tensor) => upload_tensor(tensor, false),
        Value::Num(number) => upload_tensor(
            Tensor::new(vec![number], vec![1, 1]).map_err(|error| topk_internal(kind, error))?,
            false,
        ),
        Value::Int(integer) => upload_tensor(
            Tensor::new_integer(IntegerStorage::from_scalar(integer), vec![1, 1])
                .map_err(|error| topk_internal(kind, error))?,
            false,
        ),
        Value::Bool(logical) => upload_tensor(
            Tensor::new(vec![if logical { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|error| topk_internal(kind, error))?,
            true,
        ),
        Value::LogicalArray(logical) => {
            let tensor =
                tensor::logical_to_tensor(&logical).map_err(|error| topk_internal(kind, error))?;
            upload_tensor(tensor, true)
        }
        Value::Complex(real, imag) => {
            let tensor = ComplexTensor::new(vec![(real, imag)], vec![1, 1])
                .map_err(|error| topk_internal(kind, error))?;
            let handle = gpu_helpers::upload_complex_tensor(provider, &tensor)
                .map_err(|error| topk_internal(kind, error.message()))?;
            Ok(gpu_helpers::complex_gpu_value(handle))
        }
        Value::ComplexTensor(tensor) => {
            let handle = gpu_helpers::upload_complex_tensor(provider, &tensor)
                .map_err(|error| topk_internal(kind, error.message()))?;
            Ok(gpu_helpers::complex_gpu_value(handle))
        }
        other => Err(topk_internal(
            kind,
            format!("cannot upload unexpected output {other:?}"),
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
        let indices = Tensor::new(vec![1.0; tensor.len()], shape)
            .map_err(|message| topk_internal(kind, message))?;
        return Ok(TopKEvaluation {
            values: tensor::tensor_into_value(tensor),
            indices: tensor::tensor_into_value(indices),
        });
    }
    let output_shape = output_shape_for_topk(&shape, axis, take);
    let storage = tensor
        .into_numeric_storage()
        .map_err(|message| topk_internal(kind, message))?;
    if storage.is_empty() || take == 0 {
        let values = Tensor::from_numeric_storage(storage.zeros_like(0), output_shape.clone())
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
    let mut selected = vec![0usize; output_len];
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
                value: storage.value_at(input_index).ok_or_else(|| {
                    topk_internal(kind, format!("input index {input_index} is out of bounds"))
                })?,
                index: reduce_idx,
                source_index: input_index,
            });
        }
        entries.sort_by(|a, b| compare_real_entries(kind, args.comparison, a, b));
        for (rank, entry) in entries.iter().take(take).enumerate() {
            let mut out_coords = coords.clone();
            out_coords[axis] = rank;
            let out_idx = map_linear_index(&out_coords, &output_strides);
            selected[out_idx] = entry.source_index;
            indices[out_idx] = (entry.index + 1) as f64;
        }
        let _ = out_base;
        increment_coords(&mut coords, &output_shape);
    }

    let values = storage
        .gather(&selected)
        .and_then(|values| Tensor::from_numeric_storage(values, output_shape.clone()))
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
    let storage = tensor.into_complex_storage();
    let comparison_values = storage.materialize_f64();
    let dim = selected_dim(&shape, args.dim);
    let axis = dim.saturating_sub(1);
    let axis_len = shape.get(axis).copied().unwrap_or(1);
    let take = args.k.min(axis_len);
    if axis >= shape.len() {
        let indices = Tensor::new(vec![1.0; storage.len()], shape.clone())
            .map_err(|message| topk_internal(kind, message))?;
        let values = ComplexTensor::from_complex_storage(storage, shape)
            .map_err(|message| topk_internal(kind, message))?;
        return Ok(TopKEvaluation {
            values: complex_tensor_into_value(values),
            indices: tensor::tensor_into_value(indices),
        });
    }
    let output_shape = output_shape_for_topk(&shape, axis, take);
    if storage.is_empty() || take == 0 {
        let values = storage
            .gather(&[])
            .and_then(|storage| ComplexTensor::from_complex_storage(storage, output_shape.clone()))
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
    let mut selected = vec![0usize; output_len];
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
                value: comparison_values[input_index],
                index: reduce_idx,
                source_index: input_index,
            });
        }
        entries.sort_by(|a, b| compare_complex_entries(kind, args.comparison, a, b));
        for (rank, entry) in entries.iter().take(take).enumerate() {
            let mut out_coords = coords.clone();
            out_coords[axis] = rank;
            let out_idx = map_linear_index(&out_coords, &output_strides);
            selected[out_idx] = entry.source_index;
            indices[out_idx] = (entry.index + 1) as f64;
        }
        let _ = out_base;
        increment_coords(&mut coords, &output_shape);
    }

    let values = storage
        .gather(&selected)
        .and_then(|storage| ComplexTensor::from_complex_storage(storage, output_shape.clone()))
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
    value: NumericScalar,
    index: usize,
    source_index: usize,
}

#[derive(Clone, Copy)]
struct ComplexEntry {
    value: (f64, f64),
    index: usize,
    source_index: usize,
}

fn compare_real_entries(
    kind: TopKKind,
    method: ComparisonMethod,
    a: &RealEntry,
    b: &RealEntry,
) -> Ordering {
    let ordering = compare_numeric_scalars(method, a.value, b.value);
    let ordering = match kind {
        TopKKind::Max => ordering.reverse(),
        TopKKind::Min => ordering,
    };
    ordering.then_with(|| a.index.cmp(&b.index))
}

fn compare_numeric_scalars(
    method: ComparisonMethod,
    a: NumericScalar,
    b: NumericScalar,
) -> Ordering {
    macro_rules! compare_signed {
        ($left:expr, $right:expr) => {{
            match method {
                ComparisonMethod::Auto | ComparisonMethod::Real => $left.cmp(&$right),
                ComparisonMethod::Abs => u128::from($left.unsigned_abs())
                    .cmp(&u128::from($right.unsigned_abs()))
                    .then_with(|| $left.cmp(&$right).reverse()),
            }
        }};
    }
    macro_rules! compare_unsigned {
        ($left:expr, $right:expr) => {{
            match method {
                ComparisonMethod::Auto | ComparisonMethod::Real => $left.cmp(&$right),
                ComparisonMethod::Abs => $left.cmp(&$right),
            }
        }};
    }
    match (a, b) {
        (NumericScalar::F64(left), NumericScalar::F64(right)) => {
            compare_f64_values(method, left, right)
        }
        (NumericScalar::F32(left), NumericScalar::F32(right)) => {
            compare_f32_values(method, left, right)
        }
        (NumericScalar::I8(left), NumericScalar::I8(right)) => compare_signed!(left, right),
        (NumericScalar::I16(left), NumericScalar::I16(right)) => compare_signed!(left, right),
        (NumericScalar::I32(left), NumericScalar::I32(right)) => compare_signed!(left, right),
        (NumericScalar::I64(left), NumericScalar::I64(right)) => compare_signed!(left, right),
        (NumericScalar::U8(left), NumericScalar::U8(right)) => compare_unsigned!(left, right),
        (NumericScalar::U16(left), NumericScalar::U16(right)) => compare_unsigned!(left, right),
        (NumericScalar::U32(left), NumericScalar::U32(right)) => compare_unsigned!(left, right),
        (NumericScalar::U64(left), NumericScalar::U64(right)) => compare_unsigned!(left, right),
        (left, right) => left
            .numeric_dtype()
            .class_name()
            .cmp(right.numeric_dtype().class_name()),
    }
}

fn compare_f64_values(method: ComparisonMethod, a: f64, b: f64) -> Ordering {
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
                .then_with(|| a.partial_cmp(&b).unwrap_or(Ordering::Equal).reverse()),
        },
    }
}

fn compare_f32_values(method: ComparisonMethod, a: f32, b: f32) -> Ordering {
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
                .then_with(|| a.partial_cmp(&b).unwrap_or(Ordering::Equal).reverse()),
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
    use runmat_builtins::{IntegerComplexStorage, IntegerStorage, NumericStorage};

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(data, shape).unwrap())
    }

    fn values(tensor: &Tensor) -> Vec<f64> {
        tensor.materialize_f64()
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
        assert_eq!(self::values(&values), vec![4.0, 3.0, 6.0, 5.0]);
        let Value::Tensor(indices) = eval.indices else {
            panic!("expected indices");
        };
        assert_eq!(self::values(&indices), vec![2.0, 3.0, 2.0, 3.0]);
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
        assert_eq!(self::values(&values), vec![1.0, 2.0, 3.0, 4.0]);
        let Value::Tensor(indices) = eval.indices else {
            panic!("expected indices");
        };
        assert_eq!(self::values(&indices), vec![2.0, 2.0, 1.0, 1.0]);
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
        assert_eq!(self::values(&values), vec![3.0, 2.0, 1.0]);
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
        assert_eq!(self::values(&values), vec![2.0, 1.0, 3.0]);
        let Value::Tensor(indices) = eval.indices else {
            panic!("expected indices");
        };
        assert_eq!(self::values(&indices), vec![1.0, 1.0, 1.0]);
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
    async fn topk_k_uses_exact_storage_for_every_integer_class_and_rejects_wide_values() {
        let storages = vec![
            IntegerStorage::I8(vec![1]),
            IntegerStorage::I16(vec![1]),
            IntegerStorage::I32(vec![1]),
            IntegerStorage::I64(vec![1]),
            IntegerStorage::U8(vec![1]),
            IntegerStorage::U16(vec![1]),
            IntegerStorage::U32(vec![1]),
            IntegerStorage::U64(vec![1]),
        ];
        for storage in storages {
            let k = Tensor::new_integer(storage, vec![1, 1]).expect("k");
            assert_eq!(parse_k(TopKKind::Max, &Value::Tensor(k)).await.unwrap(), 1);
        }
        let wide =
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).expect("wide k");
        if usize::BITS == 64 {
            assert_eq!(
                parse_k(TopKKind::Max, &Value::Tensor(wide)).await.unwrap(),
                usize::MAX
            );
        } else {
            assert!(parse_k(TopKKind::Max, &Value::Tensor(wide)).await.is_err());
        }
    }

    #[tokio::test]
    async fn topk_dimension_uses_exact_storage_for_every_integer_class() {
        let storages = vec![
            IntegerStorage::I8(vec![2]),
            IntegerStorage::I16(vec![2]),
            IntegerStorage::I32(vec![2]),
            IntegerStorage::I64(vec![2]),
            IntegerStorage::U8(vec![2]),
            IntegerStorage::U16(vec![2]),
            IntegerStorage::U32(vec![2]),
            IntegerStorage::U64(vec![2]),
        ];
        for storage in storages {
            let dim = Value::Tensor(Tensor::new_integer(storage, vec![1, 1]).expect("dim"));
            let input = tensor(vec![3.0, 4.0, 1.0, 2.0], vec![2, 2]);
            let evaluation = evaluate_topk(TopKKind::Min, input, &[Value::Num(1.0), dim])
                .await
                .expect("typed dimension");
            let Value::Tensor(values) = evaluation.values else {
                panic!("expected tensor");
            };
            assert_eq!(values.shape, vec![2, 1]);
            assert_eq!(self::values(&values), vec![1.0, 2.0]);
        }
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
        assert!(values.is_empty());
    }

    #[tokio::test]
    async fn topk_preserves_every_integer_class_and_returns_double_indices() {
        let cases = vec![
            (
                IntegerStorage::I8(vec![i8::MIN, 0, i8::MAX]),
                NumericStorage::I8(vec![i8::MAX, 0]),
                NumericStorage::I8(vec![i8::MIN, 0]),
            ),
            (
                IntegerStorage::I16(vec![i16::MIN, 0, i16::MAX]),
                NumericStorage::I16(vec![i16::MAX, 0]),
                NumericStorage::I16(vec![i16::MIN, 0]),
            ),
            (
                IntegerStorage::I32(vec![i32::MIN, 0, i32::MAX]),
                NumericStorage::I32(vec![i32::MAX, 0]),
                NumericStorage::I32(vec![i32::MIN, 0]),
            ),
            (
                IntegerStorage::I64(vec![i64::MIN, 0, i64::MAX]),
                NumericStorage::I64(vec![i64::MAX, 0]),
                NumericStorage::I64(vec![i64::MIN, 0]),
            ),
            (
                IntegerStorage::U8(vec![0, u8::MAX - 1, u8::MAX]),
                NumericStorage::U8(vec![u8::MAX, u8::MAX - 1]),
                NumericStorage::U8(vec![0, u8::MAX - 1]),
            ),
            (
                IntegerStorage::U16(vec![0, u16::MAX - 1, u16::MAX]),
                NumericStorage::U16(vec![u16::MAX, u16::MAX - 1]),
                NumericStorage::U16(vec![0, u16::MAX - 1]),
            ),
            (
                IntegerStorage::U32(vec![0, u32::MAX - 1, u32::MAX]),
                NumericStorage::U32(vec![u32::MAX, u32::MAX - 1]),
                NumericStorage::U32(vec![0, u32::MAX - 1]),
            ),
            (
                IntegerStorage::U64(vec![0, u64::MAX - 1, u64::MAX]),
                NumericStorage::U64(vec![u64::MAX, u64::MAX - 1]),
                NumericStorage::U64(vec![0, u64::MAX - 1]),
            ),
        ];
        for (input, expected_max, expected_min) in cases {
            for (kind, expected_values, expected_indices) in [
                (TopKKind::Max, expected_max, vec![3.0, 2.0]),
                (TopKKind::Min, expected_min, vec![1.0, 2.0]),
            ] {
                let input = Value::Tensor(
                    Tensor::new_integer(input.clone(), vec![3, 1]).expect("integer input"),
                );
                let evaluation = evaluate_topk(kind, input, &[Value::Num(2.0)])
                    .await
                    .expect("topk");
                let Value::Tensor(output) = evaluation.values else {
                    panic!("expected typed tensor output");
                };
                assert_eq!(output.into_numeric_storage().unwrap(), expected_values);
                let Value::Tensor(indices) = evaluation.indices else {
                    panic!("expected double index tensor");
                };
                assert_eq!(values(&indices), expected_indices);
                assert!(matches!(
                    indices.into_numeric_storage().unwrap(),
                    NumericStorage::F64(_)
                ));
            }
        }
    }

    #[tokio::test]
    async fn topk_preserves_native_single_storage() {
        let input =
            Value::Tensor(Tensor::from_f32(vec![1.0, 3.0, 2.0], vec![3, 1]).expect("single input"));
        let evaluation = evaluate_topk(TopKKind::Max, input, &[Value::Num(2.0)])
            .await
            .expect("topk");
        let Value::Tensor(output) = evaluation.values else {
            panic!("expected single tensor");
        };
        assert_eq!(
            output.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![3.0, 2.0])
        );
    }

    #[tokio::test]
    async fn topk_preserves_native_complex_single_and_logical_storage() {
        let complex = Value::ComplexTensor(
            ComplexTensor::from_f32(vec![(1.0, 1.0), (3.0, -1.0), (2.0, 0.0)], vec![3, 1])
                .expect("complex single"),
        );
        let evaluation = evaluate_topk(
            TopKKind::Max,
            complex,
            &[
                Value::Num(2.0),
                Value::from("ComparisonMethod"),
                Value::from("real"),
            ],
        )
        .await
        .expect("complex topk");
        let Value::ComplexTensor(values) = evaluation.values else {
            panic!("expected complex tensor");
        };
        assert_eq!(
            values.into_complex_storage(),
            runmat_builtins::ComplexStorage::F32(vec![(3.0, -1.0), (2.0, 0.0)])
        );

        let logical = Value::LogicalArray(
            LogicalArray::new(vec![0, 1, 0], vec![3, 1]).expect("logical input"),
        );
        let evaluation = evaluate_topk(TopKKind::Max, logical, &[Value::Num(2.0)])
            .await
            .expect("logical topk");
        let Value::LogicalArray(values) = evaluation.values else {
            panic!("expected logical output");
        };
        assert_eq!(values.shape, vec![2, 1]);
        assert_eq!(values.data, vec![1, 0]);
        let Value::Tensor(indices) = evaluation.indices else {
            panic!("expected indices");
        };
        assert_eq!(self::values(&indices), vec![2.0, 1.0]);
    }

    #[tokio::test]
    async fn topk_preserves_wide_integer_scalar_exactly() {
        let evaluation = evaluate_topk(
            TopKKind::Max,
            Value::Int(IntValue::U64(u64::MAX)),
            &[Value::Num(1.0)],
        )
        .await
        .expect("topk");
        assert_eq!(evaluation.values, Value::Int(IntValue::U64(u64::MAX)));
        assert_eq!(evaluation.indices, Value::Num(1.0));
    }

    #[tokio::test]
    async fn topk_abs_comparison_handles_signed_min_without_overflow() {
        let input = Value::Tensor(
            Tensor::new_integer(IntegerStorage::I64(vec![i64::MAX, i64::MIN]), vec![2, 1])
                .expect("integer input"),
        );
        let evaluation = evaluate_topk(
            TopKKind::Max,
            input,
            &[
                Value::Num(1.0),
                Value::from("ComparisonMethod"),
                Value::from("abs"),
            ],
        )
        .await
        .expect("topk");
        assert_eq!(evaluation.values, Value::Int(IntValue::I64(i64::MIN)));
        assert_eq!(evaluation.indices, Value::Num(2.0));
    }

    #[tokio::test]
    async fn topk_abs_comparison_uses_phase_for_equal_real_magnitudes() {
        for (kind, expected_values, expected_indices) in [
            (TopKKind::Max, vec![-3.0, 3.0], vec![1.0, 2.0]),
            (TopKKind::Min, vec![3.0, -3.0], vec![2.0, 1.0]),
        ] {
            let evaluation = evaluate_topk(
                kind,
                tensor(vec![-3.0, 3.0], vec![2, 1]),
                &[
                    Value::Num(2.0),
                    Value::from("ComparisonMethod"),
                    Value::from("abs"),
                ],
            )
            .await
            .expect("absolute topk");
            let Value::Tensor(values) = evaluation.values else {
                panic!("expected values");
            };
            assert_eq!(self::values(&values), expected_values);
            let Value::Tensor(indices) = evaluation.indices else {
                panic!("expected indices");
            };
            assert_eq!(self::values(&indices), expected_indices);
        }
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
        assert_eq!(values.materialize_f64(), vec![(3.0, -1.0), (2.0, 0.0)]);
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
        assert_eq!(self::values(selected), vec![3.0, 2.0]);
        let Value::Tensor(indices) = &values[1] else {
            panic!("expected indices");
        };
        assert_eq!(self::values(indices), vec![2.0, 3.0]);
    }

    #[tokio::test]
    async fn topk_rejects_more_than_two_outputs() {
        let _guard = crate::output_count::push_output_count(Some(3));
        let err = maxk_builtin(
            tensor(vec![1.0, 3.0, 2.0], vec![1, 3]),
            vec![Value::Num(2.0)],
        )
        .await
        .expect_err("too many outputs");
        assert!(err.message().contains("maximum is 2"));
    }

    #[test]
    fn topk_gpu_extension_gates_before_gather_and_preserves_residency() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let input = Tensor::new_integer(
                IntegerStorage::U64(vec![u64::MAX - 1, u64::MAX, 0]),
                vec![3, 1],
            )
            .expect("integer input");
            let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload");
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
                let err = futures::executor::block_on(maxk_builtin(
                    Value::GpuTensor(handle.clone()),
                    vec![Value::Num(2.0)],
                ))
                .expect_err("strict GPU gate");
                assert_eq!(
                    err.identifier(),
                    Some("RunMat:compatibility:MaxkGpuInputExtension")
                );
            }
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
                let _outputs = crate::output_count::push_output_count(Some(2));
                let Value::OutputList(outputs) = futures::executor::block_on(maxk_builtin(
                    Value::GpuTensor(handle),
                    vec![Value::Num(2.0)],
                ))
                .expect("GPU extension") else {
                    panic!("expected outputs");
                };
                assert!(matches!(outputs[0], Value::GpuTensor(_)));
                assert!(matches!(outputs[1], Value::GpuTensor(_)));
                let values = crate::builtins::common::test_support::gather(outputs[0].clone())
                    .expect("gather values");
                assert_eq!(
                    values.into_numeric_storage().unwrap(),
                    NumericStorage::U64(vec![u64::MAX, u64::MAX - 1])
                );
                let indices = crate::builtins::common::test_support::gather(outputs[1].clone())
                    .expect("gather indices");
                assert_eq!(self::values(&indices), vec![2.0, 1.0]);
            }

            let logical = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &[0.0, 1.0, 0.0],
                    shape: &[3, 1],
                })
                .expect("logical upload");
            runmat_accelerate_api::set_handle_logical(&logical, true);
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let err = futures::executor::block_on(mink_builtin(
                Value::GpuTensor(logical),
                vec![Value::Num(1.0)],
            ))
            .expect_err("strict logical GPU gate");
            assert_eq!(
                err.identifier(),
                Some("RunMat:compatibility:MinkGpuInputExtension")
            );
        });
    }
}
