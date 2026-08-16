//! Quantiles, percentiles, and ranks.

use std::cmp::Ordering;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    IntValue, ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

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

const PARAM_P: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Requested quantile probabilities or percentiles.",
};

const PARAM_DIM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "dim",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Dimension to operate along.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Optional nanflag or method arguments.",
};

const OUTPUT_Q: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Q",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Quantile or percentile values.",
}];

const OUTPUT_R: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "R",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Average ranks for tied observations.",
}];

const OUTPUT_R_TIEADJ: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "R",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Average ranks for tied observations.",
    },
    BuiltinParamDescriptor {
        name: "tieadj",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Tie adjustment terms for rank-based statistics.",
    },
];

const INPUTS_X_P: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_P];
const INPUTS_X_P_DIM: [BuiltinParamDescriptor; 3] = [PARAM_X, PARAM_P, PARAM_DIM];
const INPUTS_X_P_OPTIONS: [BuiltinParamDescriptor; 4] =
    [PARAM_X, PARAM_P, PARAM_DIM, PARAM_OPTIONS];
const INPUTS_X: [BuiltinParamDescriptor; 1] = [PARAM_X];

const QUANTILE_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "Q = quantile(X, p)",
        inputs: &INPUTS_X_P,
        outputs: &OUTPUT_Q,
    },
    BuiltinSignatureDescriptor {
        label: "Q = quantile(X, p, dim)",
        inputs: &INPUTS_X_P_DIM,
        outputs: &OUTPUT_Q,
    },
    BuiltinSignatureDescriptor {
        label: "Q = quantile(X, p, dim, nanflag)",
        inputs: &INPUTS_X_P_OPTIONS,
        outputs: &OUTPUT_Q,
    },
];

const PRCTILE_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "Y = prctile(X, p)",
        inputs: &INPUTS_X_P,
        outputs: &OUTPUT_Q,
    },
    BuiltinSignatureDescriptor {
        label: "Y = prctile(X, p, dim)",
        inputs: &INPUTS_X_P_DIM,
        outputs: &OUTPUT_Q,
    },
    BuiltinSignatureDescriptor {
        label: "Y = prctile(X, p, dim, nanflag)",
        inputs: &INPUTS_X_P_OPTIONS,
        outputs: &OUTPUT_Q,
    },
];

const TIEDRANK_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "R = tiedrank(X)",
        inputs: &INPUTS_X,
        outputs: &OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "[R, tieadj] = tiedrank(X)",
        inputs: &INPUTS_X,
        outputs: &OUTPUT_R_TIEADJ,
    },
];

const TIEDRANK_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tiedrank-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tiedrank with typed-integer input data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TiedrankIntegerDataExtension"),
};

const TIEDRANK_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [TIEDRANK_INTEGER_DATA_EXTENSION];

const TIEDRANK_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented data domain is single or double; RunMat mode additionally accepts all eight real integer classes.",
    }];

const TIEDRANK_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[R, tieadj] = tiedrank(integer_X)",
        inputs: &TIEDRANK_INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat's integer-data extension compares and groups same-class integers exactly, including wide values above flintmax, then returns double ranks and tie adjustments.",
    }];

macro_rules! quantile_integer_metadata {
    ($data_id:literal, $control_id:literal, $data_description:literal, $control_description:literal, $data_identifier:literal, $control_identifier:literal) => {
        pub(super) const INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor =
            BuiltinExtensionDescriptor {
                id: $data_id,
                mode: BuiltinExtensionMode::RunMatOnly,
                description: $data_description,
                error_identifier: Some($data_identifier),
            };
        pub(super) const INTEGER_PROBABILITY_EXTENSION: BuiltinExtensionDescriptor =
            BuiltinExtensionDescriptor {
                id: $control_id,
                mode: BuiltinExtensionMode::RunMatOnly,
                description: $control_description,
                error_identifier: Some($control_identifier),
            };
        pub const EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
            [INTEGER_DATA_EXTENSION, INTEGER_PROBABILITY_EXTENSION];
        const INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
            [BuiltinIntegerInputCapability {
                name: "A",
                classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
                availability: BuiltinIntegerInputAvailability::RunMatOnly,
                scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
                notes: "The compatibility target's documented data domain is double, single, duration, or datetime; RunMat mode additionally accepts all eight real integer classes.",
            }];
        const INTEGER_PROBABILITY_INPUT: [BuiltinIntegerInputCapability; 1] =
            [BuiltinIntegerInputCapability {
                name: "p_or_pct",
                classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
                availability: BuiltinIntegerInputAvailability::RunMatOnly,
                scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
                notes: "The documented probability/percentage domain is single or double; RunMat mode additionally accepts exact typed-integer request values.",
            }];
        pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
            BuiltinIntegerCapabilityDescriptor {
                form: "Q = quantile_or_prctile(A, p, dim_or_vecdim_or_all, Method=method) with integer A",
                inputs: &INTEGER_DATA_INPUT,
                computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
                output_class: BuiltinIntegerOutputClassRule::Double,
                overflow: BuiltinIntegerOverflowRule::NotApplicable,
                backend: BuiltinIntegerBackendRule::GatherFallback,
                overload: BuiltinIntegerOverloadKind::Multiple,
                notes: "RunMat's integer-data extension orders same-class integers exactly before the selected interpolation method materializes the double result; wide ordering does not pass through binary64.",
            },
            BuiltinIntegerCapabilityDescriptor {
                form: "Q = quantile_or_prctile(A, integer_p_or_pct, ...) with floating A",
                inputs: &INTEGER_PROBABILITY_INPUT,
                computation_domain: BuiltinIntegerComputationDomain::Structural,
                output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput,
                overflow: BuiltinIntegerOverflowRule::NotApplicable,
                backend: BuiltinIntegerBackendRule::HostOnly,
                overload: BuiltinIntegerOverloadKind::StructuralParameter,
                notes: "Typed-integer probability or percentage controls are decoded exactly and independently gated; typed integer dimension controls remain documented.",
            },
        ];
    };
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

fn ensure_quantile_integer_extensions(
    name: &str,
    input: &Value,
    rest: &[Value],
    data_extension: &BuiltinExtensionDescriptor,
    probability_extension: &BuiltinExtensionDescriptor,
) -> BuiltinResult<()> {
    if is_real_typed_integer_value(input) {
        crate::compatibility::ensure_builtin_extension_enabled(data_extension, name)?;
    }
    if rest.first().is_some_and(is_real_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(probability_extension, name)?;
    }
    Ok(())
}

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ORDER_STATS.INVALID_ARGUMENT",
    identifier: None,
    when: "Inputs, dimensions, probabilities, or options are malformed.",
    message: "order statistics: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ORDER_STATS.INTERNAL",
    identifier: None,
    when: "Internal tensor conversion or allocation fails.",
    message: "order statistics: internal error",
};

macro_rules! order_descriptor {
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

macro_rules! order_descriptor_by_output_count {
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
            output_mode: BuiltinOutputMode::ByRequestedOutputCount,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
    };
}

fn same_shape_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    match args.first() {
        Some(Type::Tensor { shape }) | Some(Type::Logical { shape }) => Type::Tensor {
            shape: shape.clone(),
        },
        Some(Type::Num | Type::Int | Type::Bool) => Type::Num,
        Some(Type::Unknown) | None => Type::Unknown,
        _ => Type::Unknown,
    }
}

fn reduced_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

fn order_error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

async fn value_to_tensor(name: &str, value: Value) -> BuiltinResult<Tensor> {
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(|err| order_error(name, format!("{name}: {err}")))?;
    tensor::value_into_tensor_for(name, gathered)
        .map_err(|err| order_error(name, format!("{name}: {err}")))
}

fn first_non_singleton(shape: &[usize]) -> usize {
    shape
        .iter()
        .position(|dim| *dim > 1)
        .map(|idx| idx + 1)
        .unwrap_or(1)
}

fn parse_dim(name: &str, value: &Value) -> BuiltinResult<usize> {
    tensor::parse_dimension(value, name).map_err(|err| order_error(name, err))
}

fn parse_probabilities(name: &str, value: Value, scale: f64) -> BuiltinResult<Vec<f64>> {
    let tensor = tensor::value_into_tensor_for(name, value)
        .map_err(|err| order_error(name, format!("{name}: {err}")))?;
    let values = tensor::tensor_into_values_f64(tensor);
    let mut out = Vec::with_capacity(values.len());
    for raw in values {
        let p = raw / scale;
        if p.is_nan() || !(0.0..=1.0).contains(&p) {
            return Err(order_error(
                name,
                format!("{name}: probabilities must be in the closed interval [0, 1]"),
            ));
        }
        out.push(p);
    }
    Ok(out)
}

#[derive(Clone, Copy)]
enum NanFlag {
    Include,
    Omit,
}

/// A value used only while ordering observations.  Integer order statistics
/// return doubles, but their input ordering and tie detection must not pass
/// through the lossy f64 mirror first (notably for uint64 values above 2^53).
#[derive(Clone)]
enum OrderedValue {
    Float(f64),
    Integer(IntValue),
}

impl OrderedValue {
    fn is_nan(&self) -> bool {
        matches!(self, Self::Float(value) if value.is_nan())
    }

    fn as_f64(&self) -> f64 {
        match self {
            Self::Float(value) => *value,
            Self::Integer(value) => value.to_f64(),
        }
    }
}

fn ordered_input_values(input: &Tensor) -> Vec<OrderedValue> {
    if let Some(storage) = input.integer_storage() {
        return storage
            .exact_values()
            .iter()
            .cloned()
            .map(OrderedValue::Integer)
            .collect();
    }
    tensor::tensor_values_f64_cow(input)
        .iter()
        .copied()
        .map(OrderedValue::Float)
        .collect()
}

fn compare_ordered_values(left: &OrderedValue, right: &OrderedValue) -> Ordering {
    match (left, right) {
        (OrderedValue::Float(left), OrderedValue::Float(right)) => {
            left.partial_cmp(right).unwrap_or(Ordering::Greater)
        }
        (OrderedValue::Integer(left), OrderedValue::Integer(right)) => match (left, right) {
            (IntValue::I8(left), IntValue::I8(right)) => left.cmp(right),
            (IntValue::I16(left), IntValue::I16(right)) => left.cmp(right),
            (IntValue::I32(left), IntValue::I32(right)) => left.cmp(right),
            (IntValue::I64(left), IntValue::I64(right)) => left.cmp(right),
            (IntValue::U8(left), IntValue::U8(right)) => left.cmp(right),
            (IntValue::U16(left), IntValue::U16(right)) => left.cmp(right),
            (IntValue::U32(left), IntValue::U32(right)) => left.cmp(right),
            (IntValue::U64(left), IntValue::U64(right)) => left.cmp(right),
            _ => unreachable!("integer tensor storage is homogeneous"),
        },
        _ => unreachable!("an order-statistics input has one numeric representation"),
    }
}

fn ordered_values_equal(left: &OrderedValue, right: &OrderedValue) -> bool {
    match (left, right) {
        (OrderedValue::Float(left), OrderedValue::Float(right)) => left == right,
        (OrderedValue::Integer(left), OrderedValue::Integer(right)) => left == right,
        _ => false,
    }
}

struct QuantileArgs {
    input: Tensor,
    probabilities: Vec<f64>,
    dim: usize,
    nanflag: NanFlag,
    method: QuantileMethod,
}

#[derive(Clone, Copy)]
enum QuantileMethod {
    Midpoint,
    Inclusive,
    Exclusive,
}

async fn parse_quantile_args(
    name: &str,
    input: Value,
    rest: Vec<Value>,
    scale: f64,
) -> BuiltinResult<QuantileArgs> {
    if rest.is_empty() {
        return Err(order_error(
            name,
            format!("{name}: probability vector is required"),
        ));
    }
    let input = value_to_tensor(name, input).await?;
    let p_value = gather_if_needed_async(&rest[0])
        .await
        .map_err(|err| order_error(name, format!("{name}: {err}")))?;
    let probabilities = parse_probabilities(name, p_value, scale)?;
    let shape = tensor::default_shape_for(&input.shape, tensor::tensor_element_len(&input));
    let mut dim = first_non_singleton(&shape);
    let mut nanflag = NanFlag::Omit;
    let mut method = QuantileMethod::Midpoint;
    let mut idx = 1usize;
    while idx < rest.len() {
        let arg = &rest[idx];
        if let Some(keyword) = keyword_of(arg) {
            match keyword.to_ascii_lowercase().as_str() {
                "all" => {
                    dim = 0;
                }
                "includenan" => nanflag = NanFlag::Include,
                "omitnan" => nanflag = NanFlag::Omit,
                "midpoint" | "exact" => method = QuantileMethod::Midpoint,
                "inclusive" | "linear" => method = QuantileMethod::Inclusive,
                "exclusive" => method = QuantileMethod::Exclusive,
                "approximate" => {
                    return Err(order_error(
                        name,
                        format!(
                            "{name}: approximate quantile method is only supported for tall arrays"
                        ),
                    ));
                }
                "method" => {
                    idx += 1;
                    if idx >= rest.len() {
                        return Err(order_error(
                            name,
                            format!("{name}: Method option requires a value"),
                        ));
                    }
                    let method_name = keyword_of(&rest[idx]).ok_or_else(|| {
                        order_error(
                            name,
                            format!("{name}: Method value must be a string scalar"),
                        )
                    })?;
                    match method_name.to_ascii_lowercase().as_str() {
                        "midpoint" | "exact" => method = QuantileMethod::Midpoint,
                        "inclusive" | "linear" => method = QuantileMethod::Inclusive,
                        "exclusive" => method = QuantileMethod::Exclusive,
                        "approximate" => {
                            return Err(order_error(
                                name,
                                format!("{name}: approximate quantile method is only supported for tall arrays"),
                            ));
                        }
                        other => {
                            return Err(order_error(
                                name,
                                format!("{name}: unsupported Method '{other}'"),
                            ));
                        }
                    }
                }
                other => {
                    return Err(order_error(
                        name,
                        format!("{name}: unsupported option '{other}'"),
                    ));
                }
            }
        } else {
            dim = parse_dim(name, arg)?;
        }
        idx += 1;
    }
    Ok(QuantileArgs {
        input,
        probabilities,
        dim,
        nanflag,
        method,
    })
}

fn sorted_slice(mut values: Vec<OrderedValue>, nanflag: NanFlag) -> Vec<OrderedValue> {
    if values.is_empty() {
        return values;
    }
    match nanflag {
        NanFlag::Include if values.iter().any(OrderedValue::is_nan) => {
            return vec![OrderedValue::Float(f64::NAN)]
        }
        NanFlag::Include => {}
        NanFlag::Omit => values.retain(|value| !value.is_nan()),
    }
    values.sort_by(compare_ordered_values);
    values
}

fn quantile_from_sorted(values: &[OrderedValue], p: f64, method: QuantileMethod) -> f64 {
    if values.is_empty() || values.iter().any(OrderedValue::is_nan) {
        return f64::NAN;
    }
    if values.len() == 1 {
        return values[0].as_f64();
    }
    let n = values.len() as f64;
    let position = match method {
        QuantileMethod::Midpoint => p * n - 0.5,
        QuantileMethod::Inclusive => p * (n - 1.0),
        QuantileMethod::Exclusive => p * (n + 1.0) - 1.0,
    }
    .clamp(0.0, n - 1.0);
    let lo = position.floor() as usize;
    let hi = position.ceil() as usize;
    if lo == hi {
        values[lo].as_f64()
    } else {
        let weight = position - lo as f64;
        values[lo].as_f64() * (1.0 - weight) + values[hi].as_f64() * weight
    }
}

fn quantile_tensor(args: QuantileArgs, name: &str) -> BuiltinResult<Value> {
    let input_values = ordered_input_values(&args.input);
    let shape = tensor::default_shape_for(&args.input.shape, input_values.len());
    if args.dim == 0 {
        let values = sorted_slice(input_values, args.nanflag);
        let data = args
            .probabilities
            .iter()
            .map(|p| quantile_from_sorted(&values, *p, args.method))
            .collect::<Vec<_>>();
        let out_shape = if args.probabilities.len() == 1 {
            vec![1, 1]
        } else {
            vec![args.probabilities.len(), 1]
        };
        return Tensor::new(data, out_shape)
            .map(tensor::tensor_into_value)
            .map_err(|err| order_error(name, format!("{name}: {err}")));
    }
    let axis = args.dim - 1;
    let rank = shape.len().max(axis + 1);
    let mut padded_shape = shape.clone();
    padded_shape.resize(rank, 1);
    let axis_len = padded_shape[axis];
    let p_len = args.probabilities.len();
    let mut out_shape = padded_shape.clone();
    out_shape[axis] = p_len;
    let out_len = tensor::element_count(&out_shape);
    let mut out = vec![0.0; out_len];
    let pre: usize = padded_shape[..axis].iter().product();
    let post: usize = padded_shape[axis + 1..].iter().product();
    for prefix in 0..pre {
        for suffix in 0..post {
            let mut slice = Vec::with_capacity(axis_len);
            for idx in 0..axis_len {
                let src = prefix + idx * pre + suffix * pre * axis_len;
                slice.push(input_values[src].clone());
            }
            let slice = sorted_slice(slice, args.nanflag);
            for (p_idx, p) in args.probabilities.iter().enumerate() {
                let dst = prefix + p_idx * pre + suffix * pre * p_len;
                out[dst] = quantile_from_sorted(&slice, *p, args.method);
            }
        }
    }
    Tensor::new(out, out_shape)
        .map(tensor::tensor_into_value)
        .map_err(|err| order_error(name, format!("{name}: {err}")))
}

fn tiedrank_slice(values: &[OrderedValue]) -> (Vec<f64>, f64) {
    let mut indexed = values
        .iter()
        .cloned()
        .enumerate()
        .filter(|(_, value)| !value.is_nan())
        .collect::<Vec<_>>();
    indexed.sort_by(|(_, a), (_, b)| compare_ordered_values(a, b));
    let mut ranks = vec![f64::NAN; values.len()];
    let mut tieadj = 0.0;
    let mut start = 0usize;
    while start < indexed.len() {
        let mut end = start + 1;
        while end < indexed.len() && ordered_values_equal(&indexed[end].1, &indexed[start].1) {
            end += 1;
        }
        let average_rank = (start + 1 + end) as f64 / 2.0;
        let tie_len = end - start;
        if tie_len > 1 {
            tieadj += (tie_len * tie_len * tie_len - tie_len) as f64;
        }
        for (original, _) in &indexed[start..end] {
            ranks[*original] = average_rank;
        }
        start = end;
    }
    (ranks, tieadj)
}

fn is_vector_shape(shape: &[usize]) -> bool {
    shape.iter().filter(|dim| **dim > 1).count() <= 1
}

fn tiedrank_tensor(input: Tensor) -> BuiltinResult<(Value, Value)> {
    let input_values = ordered_input_values(&input);
    let shape = tensor::default_shape_for(&input.shape, input_values.len());
    if is_vector_shape(&shape) {
        let (ranks, tieadj) = tiedrank_slice(&input_values);
        let ranks = Tensor::new(ranks, shape)
            .map(tensor::tensor_into_value)
            .map_err(|err| order_error("tiedrank", format!("tiedrank: {err}")))?;
        return Ok((ranks, Value::Num(tieadj)));
    }

    let axis = if shape.len() <= 2 { 0 } else { 1 };
    let rank = shape.len().max(axis + 1);
    let mut padded_shape = shape.clone();
    padded_shape.resize(rank, 1);
    let axis_len = padded_shape[axis];
    let mut tieadj_shape = padded_shape.clone();
    tieadj_shape[axis] = 1;
    let tieadj_len = tensor::element_count(&tieadj_shape);
    let mut ranks = vec![f64::NAN; input_values.len()];
    let mut tieadj = vec![0.0; tieadj_len];
    let pre: usize = padded_shape[..axis].iter().product();
    let post: usize = padded_shape[axis + 1..].iter().product();
    for prefix in 0..pre {
        for suffix in 0..post {
            let mut slice = Vec::with_capacity(axis_len);
            for idx in 0..axis_len {
                let src = prefix + idx * pre + suffix * pre * axis_len;
                slice.push(input_values[src].clone());
            }
            let (slice_ranks, slice_tieadj) = tiedrank_slice(&slice);
            let tie_dst = prefix + suffix * pre;
            tieadj[tie_dst] = slice_tieadj;
            for (idx, rank_value) in slice_ranks.into_iter().enumerate() {
                let dst = prefix + idx * pre + suffix * pre * axis_len;
                ranks[dst] = rank_value;
            }
        }
    }
    let ranks = Tensor::new(ranks, padded_shape)
        .map(tensor::tensor_into_value)
        .map_err(|err| order_error("tiedrank", format!("tiedrank: {err}")))?;
    let tieadj = Tensor::new(tieadj, tieadj_shape)
        .map(tensor::tensor_into_value)
        .map_err(|err| order_error("tiedrank", format!("tiedrank: {err}")))?;
    Ok((ranks, tieadj))
}

pub mod quantile {
    use super::*;
    order_descriptor!("quantile", QUANTILE_SIGNATURES);
    quantile_integer_metadata!(
        "quantile-integer-data",
        "quantile-typed-integer-probability",
        "quantile with typed-integer input data is a RunMat extension",
        "quantile with typed-integer probabilities is a RunMat extension",
        "RunMat:compatibility:QuantileIntegerDataExtension",
        "RunMat:compatibility:QuantileTypedIntegerProbabilityExtension"
    );

    #[runtime_builtin(
        name = "quantile",
        category = "stats/summary",
        summary = "Compute sample quantiles using linear interpolation.",
        keywords = "quantile,percentile,statistics,order",
        type_resolver(super::reduced_type),
        descriptor(self::DESCRIPTOR),
        extensions(self::EXTENSIONS),
        integer_capabilities(self::INTEGER_CAPABILITIES),
        builtin_path = "crate::builtins::stats::summary::order_stats::quantile"
    )]
    pub(crate) async fn quantile_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        super::ensure_quantile_integer_extensions(
            "quantile",
            &value,
            &rest,
            &INTEGER_DATA_EXTENSION,
            &INTEGER_PROBABILITY_EXTENSION,
        )?;
        let args = super::parse_quantile_args("quantile", value, rest, 1.0).await?;
        super::quantile_tensor(args, "quantile")
    }
}

pub mod prctile {
    use super::*;
    order_descriptor!("prctile", PRCTILE_SIGNATURES);
    quantile_integer_metadata!(
        "prctile-integer-data",
        "prctile-typed-integer-percentage",
        "prctile with typed-integer input data is a RunMat extension",
        "prctile with typed-integer percentages is a RunMat extension",
        "RunMat:compatibility:PrctileIntegerDataExtension",
        "RunMat:compatibility:PrctileTypedIntegerPercentageExtension"
    );

    #[runtime_builtin(
        name = "prctile",
        category = "stats/summary",
        summary = "Compute sample percentiles using linear interpolation.",
        keywords = "prctile,percentile,quantile,statistics,order",
        type_resolver(super::reduced_type),
        descriptor(self::DESCRIPTOR),
        extensions(self::EXTENSIONS),
        integer_capabilities(self::INTEGER_CAPABILITIES),
        builtin_path = "crate::builtins::stats::summary::order_stats::prctile"
    )]
    pub(crate) async fn prctile_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        super::ensure_quantile_integer_extensions(
            "prctile",
            &value,
            &rest,
            &INTEGER_DATA_EXTENSION,
            &INTEGER_PROBABILITY_EXTENSION,
        )?;
        let args = super::parse_quantile_args("prctile", value, rest, 100.0).await?;
        super::quantile_tensor(args, "prctile")
    }
}

pub mod tiedrank {
    use super::*;
    order_descriptor_by_output_count!("tiedrank", TIEDRANK_SIGNATURES);

    #[runtime_builtin(
        name = "tiedrank",
        category = "stats/summary",
        summary = "Rank observations using average ranks for ties.",
        keywords = "tiedrank,rank,ties,statistics",
        type_resolver(super::same_shape_type),
        descriptor(self::DESCRIPTOR),
        extensions(super::TIEDRANK_EXTENSIONS),
        integer_capabilities(super::TIEDRANK_INTEGER_CAPABILITIES),
        builtin_path = "crate::builtins::stats::summary::order_stats::tiedrank"
    )]
    pub(crate) async fn tiedrank_builtin(value: Value) -> BuiltinResult<Value> {
        if super::is_real_typed_integer_value(&value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &TIEDRANK_INTEGER_DATA_EXTENSION,
                "tiedrank",
            )?;
        }
        let input = super::value_to_tensor("tiedrank", value).await?;
        let (ranks, tieadj) = super::tiedrank_tensor(input)?;
        match crate::output_count::current_output_count() {
            Some(0) => Ok(Value::OutputList(Vec::new())),
            Some(1) => Ok(Value::OutputList(vec![ranks])),
            Some(out_count) => Ok(crate::output_count::output_list_with_padding(
                out_count,
                vec![ranks, tieadj],
            )),
            None => Ok(ranks),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::IntegerStorage;

    #[test]
    fn quantile_vector_uses_midpoint_interpolation() {
        let x = Value::Tensor(Tensor::new(vec![1.0, 2.0, 4.0, 8.0], vec![4, 1]).unwrap());
        let out = block_on(quantile::quantile_builtin(
            x,
            vec![Value::Tensor(
                Tensor::new(vec![0.25, 0.5, 0.75], vec![1, 3]).unwrap(),
            )],
        ))
        .unwrap();
        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![3, 1]);
                assert_eq!(tensor.materialize_f64(), vec![1.5, 3.0, 6.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn quantile_supports_compatibility_target_midpoint_inclusive_and_exclusive_methods() {
        let input =
            || Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 6.0, 10.0], vec![5, 1]).unwrap());
        let midpoint =
            block_on(quantile::quantile_builtin(input(), vec![Value::Num(0.4)])).unwrap();
        let inclusive = block_on(quantile::quantile_builtin(
            input(),
            vec![
                Value::Num(0.4),
                Value::from("Method"),
                Value::from("inclusive"),
            ],
        ))
        .unwrap();
        let exclusive = block_on(quantile::quantile_builtin(
            input(),
            vec![
                Value::Num(0.4),
                Value::from("Method"),
                Value::from("exclusive"),
            ],
        ))
        .unwrap();
        assert!(matches!(midpoint, Value::Num(value) if value == 2.5));
        assert!(matches!(inclusive, Value::Num(value) if (value - 2.6).abs() < 1.0e-12));
        assert!(matches!(exclusive, Value::Num(value) if (value - 2.4).abs() < 1.0e-12));
    }

    #[test]
    fn quantile_and_prctile_integer_extensions_follow_compatibility_mode() {
        let integer_data = || {
            Value::Tensor(Tensor::new_integer(IntegerStorage::I16(vec![1, 3]), vec![2, 1]).unwrap())
        };
        let floating = || Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap());
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(quantile::quantile_builtin(
                integer_data(),
                vec![Value::Num(0.5)],
            ))
            .unwrap_err();
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:QuantileIntegerDataExtension")
            );
            let error = block_on(prctile::prctile_builtin(
                floating(),
                vec![Value::Int(IntValue::U8(50))],
            ))
            .unwrap_err();
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:PrctileTypedIntegerPercentageExtension")
            );
        }
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let out = block_on(quantile::quantile_builtin(
                integer_data(),
                vec![Value::Num(0.5)],
            ))
            .unwrap();
            assert!(matches!(out, Value::Num(value) if value == 2.0));
        }
    }

    #[test]
    fn tiedrank_integer_extension_follows_compatibility_mode() {
        let input = || {
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![1, 2, 1]), vec![3, 1]).unwrap(),
            )
        };
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(tiedrank::tiedrank_builtin(input())).unwrap_err();
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:TiedrankIntegerDataExtension")
            );
        }
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let out = block_on(tiedrank::tiedrank_builtin(input())).unwrap();
            assert!(
                matches!(out, Value::Tensor(tensor) if tensor.materialize_f64() == vec![1.5, 3.0, 1.5])
            );
        }
    }

    #[test]
    fn prctile_reduces_columns_by_default() {
        let x = Value::Tensor(Tensor::new(vec![1.0, 3.0, 10.0, 20.0], vec![2, 2]).unwrap());
        let out = block_on(prctile::prctile_builtin(x, vec![Value::Num(50.0)])).unwrap();
        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 2]);
                assert_eq!(tensor.materialize_f64(), vec![2.0, 15.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn tiedrank_averages_ties_and_preserves_nan() {
        let x = Value::Tensor(Tensor::new(vec![10.0, 20.0, 20.0, f64::NAN], vec![4, 1]).unwrap());
        let out = block_on(tiedrank::tiedrank_builtin(x)).unwrap();
        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![4, 1]);
                assert_eq!(tensor.materialize_f64()[0], 1.0);
                assert_eq!(tensor.materialize_f64()[1], 2.5);
                assert_eq!(tensor.materialize_f64()[2], 2.5);
                assert!(tensor.materialize_f64()[3].is_nan());
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn quantile_omits_nan_by_default_and_rejects_approximate_method() {
        let x = Value::Tensor(Tensor::new(vec![1.0, f64::NAN, 3.0], vec![3, 1]).unwrap());
        let out = block_on(quantile::quantile_builtin(x, vec![Value::Num(0.5)])).unwrap();
        match out {
            Value::Num(value) => assert_eq!(value, 2.0),
            other => panic!("expected scalar, got {other:?}"),
        }

        let x = Value::Tensor(Tensor::new(vec![1.0, f64::NAN, 3.0], vec![3, 1]).unwrap());
        let out = block_on(quantile::quantile_builtin(
            x,
            vec![Value::Num(0.5), Value::from("includenan")],
        ))
        .unwrap();
        assert!(matches!(out, Value::Num(value) if value.is_nan()));

        let x = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap());
        let err = block_on(quantile::quantile_builtin(
            x,
            vec![
                Value::Num(0.5),
                Value::from("Method"),
                Value::from("approximate"),
            ],
        ))
        .unwrap_err();
        assert!(err.message().contains("approximate quantile method"));
    }

    #[test]
    fn tiedrank_ranks_matrix_columns_and_returns_tieadj() {
        let x = Value::Tensor(Tensor::new(vec![3.0, 1.0, 1.0, 2.0, 2.0, 5.0], vec![3, 2]).unwrap());
        let _guard = crate::output_count::push_output_count(Some(2));
        let out = block_on(tiedrank::tiedrank_builtin(x)).unwrap();
        match out {
            Value::OutputList(values) => {
                assert_eq!(values.len(), 2);
                match &values[0] {
                    Value::Tensor(tensor) => {
                        assert_eq!(tensor.shape, vec![3, 2]);
                        assert_eq!(tensor.materialize_f64(), vec![3.0, 1.5, 1.5, 1.5, 1.5, 3.0]);
                    }
                    other => panic!("expected rank tensor, got {other:?}"),
                }
                match &values[1] {
                    Value::Tensor(tensor) => {
                        assert_eq!(tensor.shape, vec![1, 2]);
                        assert_eq!(tensor.materialize_f64(), vec![6.0, 6.0]);
                    }
                    other => panic!("expected tieadj tensor, got {other:?}"),
                }
            }
            other => panic!("expected output list, got {other:?}"),
        }
    }

    #[test]
    fn quantile_typed_integer_input_and_probability_read_exact_storage() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let x = Tensor::new_integer(IntegerStorage::I16(vec![1, 3, 9, 27]), vec![4, 1]).unwrap();
        let p = Tensor::new_integer(IntegerStorage::U8(vec![0, 1]), vec![1, 2]).unwrap();

        let out = block_on(quantile::quantile_builtin(
            Value::Tensor(x),
            vec![Value::Tensor(p), Value::from("all")],
        ))
        .unwrap();
        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 1]);
                assert_eq!(tensor.materialize_f64(), vec![1.0, 27.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn prctile_typed_integer_input_and_percentiles_read_exact_storage() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let x = Tensor::new_integer(
            IntegerStorage::I16(vec![10, 30, 50, 70, 20, 40, 60, 80]),
            vec![4, 2],
        )
        .unwrap();
        let p = Tensor::new_integer(IntegerStorage::U8(vec![25, 50, 75]), vec![1, 3]).unwrap();

        let out = block_on(prctile::prctile_builtin(
            Value::Tensor(x),
            vec![Value::Tensor(p)],
        ))
        .unwrap();
        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![3, 2]);
                let expected = [20.0, 40.0, 60.0, 30.0, 50.0, 70.0];
                for (actual, expect) in tensor.materialize_f64().iter().zip(expected.iter()) {
                    assert!((actual - expect).abs() < 1.0e-12, "{actual} vs {expect}");
                }
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn tiedrank_typed_integer_matrix_reads_exact_storage() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let x =
            Tensor::new_integer(IntegerStorage::I16(vec![3, 1, 1, 2, 2, 5]), vec![3, 2]).unwrap();

        let _guard = crate::output_count::push_output_count(Some(2));
        let out = block_on(tiedrank::tiedrank_builtin(Value::Tensor(x))).unwrap();
        match out {
            Value::OutputList(values) => {
                assert_eq!(values.len(), 2);
                match &values[0] {
                    Value::Tensor(tensor) => {
                        assert_eq!(tensor.shape, vec![3, 2]);
                        assert_eq!(tensor.materialize_f64(), vec![3.0, 1.5, 1.5, 1.5, 1.5, 3.0]);
                    }
                    other => panic!("expected rank tensor, got {other:?}"),
                }
                match &values[1] {
                    Value::Tensor(tensor) => {
                        assert_eq!(tensor.shape, vec![1, 2]);
                        assert_eq!(tensor.materialize_f64(), vec![6.0, 6.0]);
                    }
                    other => panic!("expected tieadj tensor, got {other:?}"),
                }
            }
            other => panic!("expected output list, got {other:?}"),
        }
    }

    #[test]
    fn tiedrank_uint64_distinguishes_values_beyond_f64_precision() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let base = 1_u64 << 53;
        let x = Tensor::new_integer(IntegerStorage::U64(vec![base, base + 1, base]), vec![3, 1])
            .unwrap();

        let out = block_on(tiedrank::tiedrank_builtin(Value::Tensor(x))).unwrap();
        match out {
            Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![1.5, 3.0, 1.5]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn ordered_integer_comparison_covers_all_storage_classes() {
        macro_rules! assert_ordered {
            ($variant:ident, $lower:expr, $upper:expr) => {
                assert_eq!(
                    compare_ordered_values(
                        &OrderedValue::Integer(IntValue::$variant($lower)),
                        &OrderedValue::Integer(IntValue::$variant($upper)),
                    ),
                    Ordering::Less,
                );
            };
        }

        assert_ordered!(I8, -1, 1);
        assert_ordered!(I16, -1, 1);
        assert_ordered!(I32, -1, 1);
        assert_ordered!(I64, -1, 1);
        assert_ordered!(U8, 1, 2);
        assert_ordered!(U16, 1, 2);
        assert_ordered!(U32, 1, 2);
        assert_ordered!(U64, 1, 2);
    }

    #[test]
    fn tiedrank_registered_extension_covers_all_integer_classes() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let storages = vec![
            IntegerStorage::I8(vec![1, 2, 1]),
            IntegerStorage::I16(vec![1, 2, 1]),
            IntegerStorage::I32(vec![1, 2, 1]),
            IntegerStorage::I64(vec![1, 2, 1]),
            IntegerStorage::U8(vec![1, 2, 1]),
            IntegerStorage::U16(vec![1, 2, 1]),
            IntegerStorage::U32(vec![1, 2, 1]),
            IntegerStorage::U64(vec![1, 2, 1]),
        ];
        for storage in storages {
            let input = Tensor::new_integer(storage, vec![3, 1]).unwrap();
            let out = block_on(tiedrank::tiedrank_builtin(Value::Tensor(input))).unwrap();
            assert!(
                matches!(out, Value::Tensor(tensor) if tensor.materialize_f64() == vec![1.5, 3.0, 1.5])
            );
        }
    }
}
