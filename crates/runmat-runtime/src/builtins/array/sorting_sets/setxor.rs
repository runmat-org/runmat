//! MATLAB-compatible `setxor` builtin with host-authoritative set semantics.
//!
//! Supports element-wise and row-wise symmetric differences with sorted or stable
//! ordering and index outputs. GPU tensors use typed host fallback and their
//! public outputs are restored to the owning provider.

use std::cmp::Ordering;
use std::collections::{hash_map::Entry, HashMap};

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CharArray, ComplexStorage, ComplexTensor, IntValue, IntegerStorage, NumericDType,
    NumericScalar, NumericStorage, StringArray, Tensor, Value,
};

use super::{float_order::SetFloat, integer_order, type_resolvers::set_values_output_type};
use crate::build_runtime_error;
use crate::builtins::common::arg_tokens::tokens_from_values;
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::math::elementwise::integer_cast::IntegerTarget;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::sorting_sets::setxor")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "setxor",
    op_kind: GpuOpKind::Custom("setxor"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: true,
    notes: "`setxor` gathers through exact typed fallback and restores symmetric-difference values plus double indices to the input owner.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::array::sorting_sets::setxor"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "setxor",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "`setxor` terminates fusion chains and materialises results on the host; upstream tensors are gathered when necessary.",
};

const BUILTIN_NAME: &str = "setxor";

const SETXOR_OUTPUT_C: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Values or rows that appear in exactly one input.",
}];

const SETXOR_OUTPUT_C_IA_IB: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Values or rows that appear in exactly one input.",
    },
    BuiltinParamDescriptor {
        name: "ia",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indices selecting values or rows from A.",
    },
    BuiltinParamDescriptor {
        name: "ib",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indices selecting values or rows from B.",
    },
];

const SETXOR_INPUTS_A_B: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First input array.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second input array.",
    },
];

const SETXOR_INPUTS_A_B_OPTIONS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First input array.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second input array.",
    },
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option tokens: 'rows'|'sorted'|'stable'.",
    },
];

const SETXOR_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "C = setxor(A, B)",
        inputs: &SETXOR_INPUTS_A_B,
        outputs: &SETXOR_OUTPUT_C,
    },
    BuiltinSignatureDescriptor {
        label: "C = setxor(A, B, option...)",
        inputs: &SETXOR_INPUTS_A_B_OPTIONS,
        outputs: &SETXOR_OUTPUT_C,
    },
    BuiltinSignatureDescriptor {
        label: "[C, ia, ib] = setxor(A, B)",
        inputs: &SETXOR_INPUTS_A_B,
        outputs: &SETXOR_OUTPUT_C_IA_IB,
    },
    BuiltinSignatureDescriptor {
        label: "[C, ia, ib] = setxor(A, B, option...)",
        inputs: &SETXOR_INPUTS_A_B_OPTIONS,
        outputs: &SETXOR_OUTPUT_C_IA_IB,
    },
];

const SETXOR_ERROR_LEGACY_OPTION_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETXOR.LEGACY_OPTION_UNSUPPORTED",
    identifier: Some("RunMat:setxor:LegacyOptionUnsupported"),
    when: "Legacy compatibility options are requested.",
    message: "setxor: the 'legacy' behaviour is not supported",
};

const SETXOR_ERROR_CONFLICTING_ORDER_OPTIONS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETXOR.CONFLICTING_ORDER_OPTIONS",
    identifier: Some("RunMat:setxor:ConflictingOrderOptions"),
    when: "Both 'sorted' and 'stable' options are provided.",
    message: "setxor: cannot combine 'sorted' with 'stable'",
};

const SETXOR_ERROR_UNKNOWN_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETXOR.UNKNOWN_OPTION",
    identifier: Some("RunMat:setxor:UnknownOption"),
    when: "An unsupported option token is provided.",
    message: "setxor: unrecognised option",
};

const SETXOR_ERROR_ROWS_COLUMN_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETXOR.ROWS_COLUMN_MISMATCH",
    identifier: Some("RunMat:setxor:RowsColumnMismatch"),
    when: "'rows' mode is used and column counts differ.",
    message: "setxor: inputs must have the same number of columns when using 'rows'",
};

const SETXOR_ERROR_UNSUPPORTED_INPUT_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETXOR.UNSUPPORTED_INPUT_TYPE",
    identifier: Some("RunMat:setxor:UnsupportedInputType"),
    when: "Input values cannot be converted into supported setxor domains.",
    message: "setxor: unsupported input type",
};

const SETXOR_ERROR_NUMERIC_CLASS_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETXOR.NUMERIC_CLASS_MISMATCH",
    identifier: Some("RunMat:setxor:NumericClassMismatch"),
    when: "Numeric inputs have incompatible nondouble classes.",
    message: "setxor: numeric inputs must have the same class, except double may be combined with one nondouble class",
};

const SETXOR_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETXOR.INVALID_ARGUMENT",
    identifier: Some("RunMat:setxor:InvalidArgument"),
    when: "Option arguments are not string-like where required.",
    message: "setxor: expected string option arguments",
};

const SETXOR_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETXOR.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:setxor:TooManyOutputs"),
    when: "More than three output arguments are requested.",
    message: "setxor: too many output arguments",
};

const SETXOR_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETXOR.INTERNAL",
    identifier: Some("RunMat:setxor:Internal"),
    when: "Internal conversion, allocation, or provider decode fails.",
    message: "setxor: internal operation failed",
};

const SETXOR_ERRORS: [BuiltinErrorDescriptor; 9] = [
    SETXOR_ERROR_LEGACY_OPTION_UNSUPPORTED,
    SETXOR_ERROR_CONFLICTING_ORDER_OPTIONS,
    SETXOR_ERROR_UNKNOWN_OPTION,
    SETXOR_ERROR_ROWS_COLUMN_MISMATCH,
    SETXOR_ERROR_UNSUPPORTED_INPUT_TYPE,
    SETXOR_ERROR_NUMERIC_CLASS_MISMATCH,
    SETXOR_ERROR_INVALID_ARGUMENT,
    SETXOR_ERROR_TOO_MANY_OUTPUTS,
    SETXOR_ERROR_INTERNAL,
];

const SETXOR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[C, ia, ib] = setxor(integer_A, integer_B, options)",
        inputs: &super::BINARY_SET_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "C preserves the common nondouble integer class, including when paired with double; ia and ib are one-based double. GPU supports integer classes through 32 bits and restores outputs after typed fallback.",
    }];

pub const SETXOR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SETXOR_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SETXOR_ERRORS,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SetxorOrder {
    Sorted,
    Stable,
}

#[derive(Debug, Clone)]
struct SetxorOptions {
    rows: bool,
    order: SetxorOrder,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Origin {
    A,
    B,
}

#[derive(Debug)]
struct SymEntry<T> {
    value: T,
    a_index: Option<usize>,
    b_index: Option<usize>,
    order_rank: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum NumericKey {
    Value(u64),
    UniqueNan(Origin, usize),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum NumericRowKey {
    Values(Vec<u64>),
    UniqueNan(Origin, usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ComplexKey {
    re: u64,
    im: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ComplexElementKey {
    Value(ComplexKey),
    UniqueNan(Origin, usize),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ComplexRowKey {
    Values(Vec<ComplexKey>),
    UniqueNan(Origin, usize),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RowCharKey(Vec<u32>);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RowStringKey(Vec<String>);

fn setxor_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn setxor_error(error: &'static BuiltinErrorDescriptor) -> crate::RuntimeError {
    setxor_error_with(error, error.message)
}

fn setxor_internal_error(message: impl Into<String>) -> crate::RuntimeError {
    setxor_error_with(&SETXOR_ERROR_INTERNAL, message)
}

#[runtime_builtin(
    name = "setxor",
    category = "array/sorting_sets",
    summary = "Return the symmetric difference of two arrays or row sets.",
    keywords = "setxor,symmetric difference,exclusive or,stable,rows,indices,gpu",
    accel = "array_construct",
    sink = true,
    type_resolver(set_values_output_type),
    descriptor(crate::builtins::array::sorting_sets::setxor::SETXOR_DESCRIPTOR),
    integer_capabilities(SETXOR_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::array::sorting_sets::setxor"
)]
async fn setxor_builtin(a: Value, b: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 3) {
        return Err(setxor_error_with(
            &SETXOR_ERROR_TOO_MANY_OUTPUTS,
            "setxor: too many output arguments; maximum is 3",
        ));
    }
    let provider = super::set_output_provider(&a, &b);
    let eval = evaluate(a, b, &rest).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count == 1 {
            let outputs = super::restore_set_outputs(
                provider,
                BUILTIN_NAME,
                vec![eval.into_values_value()],
                setxor_internal_error,
            )?;
            return Ok(Value::OutputList(outputs));
        }
        let (values, ia, ib) = eval.into_triple();
        let mut host_outputs = vec![values, ia, ib];
        host_outputs.truncate(out_count);
        let outputs = super::restore_set_outputs(
            provider,
            BUILTIN_NAME,
            host_outputs,
            setxor_internal_error,
        )?;
        return Ok(Value::OutputList(outputs));
    }
    let mut outputs = super::restore_set_outputs(
        provider,
        BUILTIN_NAME,
        vec![eval.into_values_value()],
        setxor_internal_error,
    )?;
    Ok(outputs.pop().expect("setxor output"))
}

pub async fn evaluate(
    a: Value,
    b: Value,
    rest: &[Value],
) -> crate::BuiltinResult<SetxorEvaluation> {
    crate::builtins::common::validation::reject_typed_complex_integer(&a, "setxor")?;
    crate::builtins::common::validation::reject_typed_complex_integer(&b, "setxor")?;
    let opts = parse_options(rest)?;
    for value in [&a, &b] {
        if let Value::GpuTensor(handle) = value {
            if super::is_unsupported_set_gpu_integer(handle) {
                return Err(setxor_error_with(
                    &SETXOR_ERROR_UNSUPPORTED_INPUT_TYPE,
                    "setxor: resident 64-bit integer inputs are not supported",
                ));
            }
        }
    }
    match (a, b) {
        (Value::GpuTensor(handle_a), Value::GpuTensor(handle_b)) => {
            setxor_gpu_pair(handle_a, handle_b, &opts).await
        }
        (Value::GpuTensor(handle_a), other) => setxor_gpu_mixed(handle_a, other, &opts, true).await,
        (other, Value::GpuTensor(handle_b)) => {
            setxor_gpu_mixed(handle_b, other, &opts, false).await
        }
        (left, right) => setxor_host(left, right, &opts),
    }
}

fn parse_options(rest: &[Value]) -> crate::BuiltinResult<SetxorOptions> {
    let mut opts = SetxorOptions {
        rows: false,
        order: SetxorOrder::Sorted,
    };
    let mut seen_order: Option<SetxorOrder> = None;

    let tokens = tokens_from_values(rest);
    for (arg, token) in rest.iter().zip(tokens.iter()) {
        let text = match token {
            crate::builtins::common::arg_tokens::ArgToken::String(text) => text.as_str(),
            _ => {
                let text = tensor::value_to_string(arg)
                    .ok_or_else(|| setxor_error(&SETXOR_ERROR_INVALID_ARGUMENT))?;
                let lowered = text.trim().to_ascii_lowercase();
                parse_setxor_option(&mut opts, &mut seen_order, &lowered)?;
                continue;
            }
        };
        parse_setxor_option(&mut opts, &mut seen_order, text)?;
    }

    Ok(opts)
}

fn parse_setxor_option(
    opts: &mut SetxorOptions,
    seen_order: &mut Option<SetxorOrder>,
    lowered: &str,
) -> crate::BuiltinResult<()> {
    match lowered {
        "rows" => opts.rows = true,
        "sorted" => {
            if let Some(prev) = seen_order {
                if *prev != SetxorOrder::Sorted {
                    return Err(setxor_error(&SETXOR_ERROR_CONFLICTING_ORDER_OPTIONS));
                }
            }
            *seen_order = Some(SetxorOrder::Sorted);
            opts.order = SetxorOrder::Sorted;
        }
        "stable" => {
            if let Some(prev) = seen_order {
                if *prev != SetxorOrder::Stable {
                    return Err(setxor_error(&SETXOR_ERROR_CONFLICTING_ORDER_OPTIONS));
                }
            }
            *seen_order = Some(SetxorOrder::Stable);
            opts.order = SetxorOrder::Stable;
        }
        "legacy" | "r2012a" => {
            return Err(setxor_error(&SETXOR_ERROR_LEGACY_OPTION_UNSUPPORTED));
        }
        other => {
            return Err(setxor_error_with(
                &SETXOR_ERROR_UNKNOWN_OPTION,
                format!("setxor: unrecognised option '{other}'"),
            ))
        }
    }
    Ok(())
}

async fn setxor_gpu_pair(
    handle_a: GpuTensorHandle,
    handle_b: GpuTensorHandle,
    opts: &SetxorOptions,
) -> crate::BuiltinResult<SetxorEvaluation> {
    let tensor_a = gpu_helpers::gather_tensor_async(&handle_a).await?;
    let tensor_b = gpu_helpers::gather_tensor_async(&handle_b).await?;
    setxor_numeric(tensor_a, tensor_b, opts)
}

async fn setxor_gpu_mixed(
    handle_gpu: GpuTensorHandle,
    other: Value,
    opts: &SetxorOptions,
    gpu_is_a: bool,
) -> crate::BuiltinResult<SetxorEvaluation> {
    let tensor_gpu = gpu_helpers::gather_tensor_async(&handle_gpu).await?;
    if matches!(other, Value::ComplexTensor(_) | Value::Complex(_, _)) {
        let complex_gpu = tensor_to_complex(tensor_gpu)?;
        let complex_other = value_into_complex_tensor(other)?;
        return if gpu_is_a {
            setxor_complex(complex_gpu, complex_other, opts)
        } else {
            setxor_complex(complex_other, complex_gpu, opts)
        };
    }
    let tensor_other =
        tensor::value_into_tensor_for("setxor", other).map_err(setxor_internal_error)?;
    if gpu_is_a {
        setxor_numeric(tensor_gpu, tensor_other, opts)
    } else {
        setxor_numeric(tensor_other, tensor_gpu, opts)
    }
}

fn setxor_host(a: Value, b: Value, opts: &SetxorOptions) -> crate::BuiltinResult<SetxorEvaluation> {
    match (a, b) {
        (Value::ComplexTensor(at), right) => {
            let bt = value_into_complex_tensor(right)?;
            setxor_complex(at, bt, opts)
        }
        (left, Value::ComplexTensor(bt)) => {
            let at = value_into_complex_tensor(left)?;
            setxor_complex(at, bt, opts)
        }
        (Value::Complex(re, im), right) => {
            let at = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| setxor_internal_error(format!("setxor: {e}")))?;
            let bt = value_into_complex_tensor(right)?;
            setxor_complex(at, bt, opts)
        }
        (left, Value::Complex(re, im)) => {
            let at = value_into_complex_tensor(left)?;
            let bt = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| setxor_internal_error(format!("setxor: {e}")))?;
            setxor_complex(at, bt, opts)
        }
        (Value::CharArray(ac), Value::CharArray(bc)) => setxor_char(ac, bc, opts),
        (Value::StringArray(astring), right) if value_is_string_compatible(&right) => {
            let bstring = value_into_string_array(right)?;
            setxor_string(astring, bstring, opts)
        }
        (left, Value::StringArray(bstring)) if value_is_string_compatible(&left) => {
            let astring = value_into_string_array(left)?;
            setxor_string(astring, bstring, opts)
        }
        (Value::String(a), right) if value_is_string_compatible(&right) => {
            let astring = StringArray::new(vec![a], vec![1, 1])
                .map_err(|e| setxor_internal_error(format!("setxor: {e}")))?;
            let bstring = value_into_string_array(right)?;
            setxor_string(astring, bstring, opts)
        }
        (left, Value::String(b)) if value_is_string_compatible(&left) => {
            let astring = value_into_string_array(left)?;
            let bstring = StringArray::new(vec![b], vec![1, 1])
                .map_err(|e| setxor_internal_error(format!("setxor: {e}")))?;
            setxor_string(astring, bstring, opts)
        }
        (Value::StringArray(astring), Value::StringArray(bstring)) => {
            setxor_string(astring, bstring, opts)
        }
        (Value::StringArray(astring), Value::String(b)) => {
            let bstring = StringArray::new(vec![b], vec![1, 1])
                .map_err(|e| setxor_internal_error(format!("setxor: {e}")))?;
            setxor_string(astring, bstring, opts)
        }
        (Value::String(a), Value::StringArray(bstring)) => {
            let astring = StringArray::new(vec![a], vec![1, 1])
                .map_err(|e| setxor_internal_error(format!("setxor: {e}")))?;
            setxor_string(astring, bstring, opts)
        }
        (Value::String(a), Value::String(b)) => {
            let astring = StringArray::new(vec![a], vec![1, 1])
                .map_err(|e| setxor_internal_error(format!("setxor: {e}")))?;
            let bstring = StringArray::new(vec![b], vec![1, 1])
                .map_err(|e| setxor_internal_error(format!("setxor: {e}")))?;
            setxor_string(astring, bstring, opts)
        }
        (Value::CharArray(ac), right) if value_is_char_numeric_compatible(&right) => {
            let bc = value_into_char_array(right)?;
            setxor_char(ac, bc, opts)
        }
        (left, Value::CharArray(bc)) if value_is_char_numeric_compatible(&left) => {
            let ac = value_into_char_array(left)?;
            setxor_char(ac, bc, opts)
        }
        (left, right) => {
            let tensor_a = tensor::value_into_tensor_for("setxor", left)
                .map_err(|e| setxor_error_with(&SETXOR_ERROR_UNSUPPORTED_INPUT_TYPE, e))?;
            let tensor_b = tensor::value_into_tensor_for("setxor", right)
                .map_err(|e| setxor_error_with(&SETXOR_ERROR_UNSUPPORTED_INPUT_TYPE, e))?;
            setxor_numeric(tensor_a, tensor_b, opts)
        }
    }
}

fn value_into_complex_tensor(value: Value) -> crate::BuiltinResult<ComplexTensor> {
    match value {
        Value::ComplexTensor(tensor) => Ok(tensor),
        Value::Complex(re, im) => ComplexTensor::new(vec![(re, im)], vec![1, 1])
            .map_err(|e| setxor_internal_error(format!("setxor: {e}"))),
        other => {
            let tensor = tensor::value_into_tensor_for("setxor", other)
                .map_err(|e| setxor_error_with(&SETXOR_ERROR_UNSUPPORTED_INPUT_TYPE, e))?;
            tensor_to_complex(tensor)
        }
    }
}

fn tensor_to_complex(tensor: Tensor) -> crate::BuiltinResult<ComplexTensor> {
    let shape = tensor.shape.clone();
    let data = tensor
        .into_numeric_storage()
        .map_err(setxor_internal_error)?
        .materialize_f64()
        .into_iter()
        .map(|real| (real, 0.0))
        .collect::<Vec<_>>();
    ComplexTensor::new(data, shape).map_err(|e| setxor_internal_error(format!("setxor: {e}")))
}

fn value_is_string_compatible(value: &Value) -> bool {
    matches!(
        value,
        Value::StringArray(_) | Value::String(_) | Value::CharArray(_)
    )
}

fn value_into_string_array(value: Value) -> crate::BuiltinResult<StringArray> {
    match value {
        Value::StringArray(array) => Ok(array),
        Value::String(value) => StringArray::new(vec![value], vec![1, 1])
            .map_err(|e| setxor_internal_error(format!("setxor: {e}"))),
        Value::CharArray(chars) => char_array_to_string_array(chars),
        other => Err(setxor_error_with(
            &SETXOR_ERROR_UNSUPPORTED_INPUT_TYPE,
            format!("setxor: cannot convert {other:?} to string array"),
        )),
    }
}

fn char_array_to_string_array(chars: CharArray) -> crate::BuiltinResult<StringArray> {
    let values = (0..chars.rows)
        .map(|row| {
            chars.data[row * chars.cols..row * chars.cols + chars.cols]
                .iter()
                .collect()
        })
        .collect::<Vec<String>>();
    let shape = if chars.rows == 0 {
        vec![0, 1]
    } else if chars.rows == 1 {
        vec![1, 1]
    } else {
        vec![chars.rows, 1]
    };
    StringArray::new(values, shape).map_err(|e| setxor_internal_error(format!("setxor: {e}")))
}

fn value_is_char_numeric_compatible(value: &Value) -> bool {
    matches!(
        value,
        Value::Tensor(_) | Value::LogicalArray(_) | Value::Num(_) | Value::Int(_) | Value::Bool(_)
    )
}

fn value_into_char_array(value: Value) -> crate::BuiltinResult<CharArray> {
    let tensor = tensor::value_into_tensor_for("setxor", value)
        .map_err(|e| setxor_error_with(&SETXOR_ERROR_UNSUPPORTED_INPUT_TYPE, e))?;
    tensor_into_char_array(tensor)
}

fn tensor_into_char_array(tensor: Tensor) -> crate::BuiltinResult<CharArray> {
    let rows = tensor.rows;
    let cols = tensor.cols;
    let mut values = vec!['\0'; rows * cols];
    for col in 0..cols {
        for row in 0..rows {
            let value = tensor.numeric_value_at(row + col * rows).ok_or_else(|| {
                setxor_internal_error("setxor: numeric character source length mismatch")
            })?;
            values[row * cols + col] = numeric_to_char(value)?;
        }
    }
    CharArray::new(values, rows, cols).map_err(|e| setxor_internal_error(format!("setxor: {e}")))
}

fn numeric_to_char(value: NumericScalar) -> crate::BuiltinResult<char> {
    let code = match value {
        NumericScalar::F64(value) => float_to_char_code(value)?,
        NumericScalar::F32(value) => float_to_char_code(f64::from(value))?,
        value => value
            .into_int_value()
            .and_then(|value| value.try_to_u64())
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| {
                setxor_error_with(
                    &SETXOR_ERROR_UNSUPPORTED_INPUT_TYPE,
                    "setxor: numeric values mixed with char inputs must be finite character codes",
                )
            })?,
    };
    char::from_u32(code).ok_or_else(|| {
        setxor_error_with(
            &SETXOR_ERROR_UNSUPPORTED_INPUT_TYPE,
            "setxor: numeric values mixed with char inputs must be valid character codes",
        )
    })
}

fn float_to_char_code(value: f64) -> crate::BuiltinResult<u32> {
    if !value.is_finite() || value.fract() != 0.0 || value < 0.0 || value > u32::MAX as f64 {
        return Err(setxor_error_with(
            &SETXOR_ERROR_UNSUPPORTED_INPUT_TYPE,
            "setxor: numeric values mixed with char inputs must be finite character codes",
        ));
    }
    Ok(value as u32)
}

fn setxor_numeric(
    a: Tensor,
    b: Tensor,
    opts: &SetxorOptions,
) -> crate::BuiltinResult<SetxorEvaluation> {
    let a_dtype = a.numeric_dtype();
    let b_dtype = b.numeric_dtype();
    if let (Some(a_storage), Some(b_storage)) = (a.integer_storage(), b.integer_storage()) {
        if a_storage.class_name() == b_storage.class_name() {
            return if opts.rows {
                setxor_integer_rows(a_storage, a.shape.clone(), b_storage, b.shape.clone(), opts)
            } else {
                setxor_integer_elements(
                    a_storage,
                    a.shape.clone(),
                    b_storage,
                    b.shape.clone(),
                    opts,
                )
            };
        }
    }
    match (a.integer_storage(), b.integer_storage()) {
        (Some(storage), None) if b_dtype == NumericDType::F64 => {
            let target = IntegerTarget::from_storage(storage);
            let b = target.cast_tensor(b).map_err(setxor_internal_error)?;
            return setxor_numeric(a, b, opts);
        }
        (None, Some(storage)) if a_dtype == NumericDType::F64 => {
            let target = IntegerTarget::from_storage(storage);
            let a = target.cast_tensor(a).map_err(setxor_internal_error)?;
            return setxor_numeric(a, b, opts);
        }
        _ => {}
    }
    numeric_output_dtype(a_dtype, b_dtype)?;
    let a_shape = a.shape.clone();
    let b_shape = b.shape.clone();
    let a_storage = a.into_numeric_storage().map_err(setxor_internal_error)?;
    let b_storage = b.into_numeric_storage().map_err(setxor_internal_error)?;
    match (a_storage, b_storage) {
        (NumericStorage::F64(a), NumericStorage::F64(b)) => {
            setxor_floating(a, a_shape, b, b_shape, opts)
        }
        (NumericStorage::F32(a), NumericStorage::F32(b)) => {
            setxor_floating(a, a_shape, b, b_shape, opts)
        }
        (NumericStorage::F64(a), NumericStorage::F32(b)) => {
            setxor_promoted_left_f64_to_f32(a, a_shape, b, b_shape, opts)
        }
        (NumericStorage::F32(a), NumericStorage::F64(b)) => {
            setxor_promoted_right_f64_to_f32(a, a_shape, b, b_shape, opts)
        }
        _ => Err(setxor_error(&SETXOR_ERROR_NUMERIC_CLASS_MISMATCH)),
    }
}

fn setxor_promoted_left_f64_to_f32(
    double_values: Vec<f64>,
    double_shape: Vec<usize>,
    single_values: Vec<f32>,
    single_shape: Vec<usize>,
    opts: &SetxorOptions,
) -> crate::BuiltinResult<SetxorEvaluation> {
    setxor_floating(
        double_values
            .into_iter()
            .map(|value| value as f32)
            .collect(),
        double_shape,
        single_values,
        single_shape,
        opts,
    )
}

fn setxor_promoted_right_f64_to_f32(
    single_values: Vec<f32>,
    single_shape: Vec<usize>,
    double_values: Vec<f64>,
    double_shape: Vec<usize>,
    opts: &SetxorOptions,
) -> crate::BuiltinResult<SetxorEvaluation> {
    setxor_floating(
        single_values,
        single_shape,
        double_values
            .into_iter()
            .map(|value| value as f32)
            .collect(),
        double_shape,
        opts,
    )
}

fn setxor_floating<T: SetFloat>(
    a_values: Vec<T>,
    a_shape: Vec<usize>,
    b_values: Vec<T>,
    b_shape: Vec<usize>,
    opts: &SetxorOptions,
) -> crate::BuiltinResult<SetxorEvaluation> {
    if opts.rows {
        setxor_floating_rows(a_values, a_shape, b_values, b_shape, opts)
    } else {
        let row_output = element_row_output(&a_shape, &b_shape);
        let mut entries = Vec::<SymEntry<T>>::new();
        let mut map: HashMap<NumericKey, usize> = HashMap::new();
        let mut order_counter = 0usize;
        for (idx, &value) in a_values.iter().enumerate() {
            add_sym_entry(
                &mut entries,
                &mut map,
                numeric_key(value, Origin::A, idx),
                value,
                Origin::A,
                idx,
                &mut order_counter,
            );
        }
        for (idx, &value) in b_values.iter().enumerate() {
            add_sym_entry(
                &mut entries,
                &mut map,
                numeric_key(value, Origin::B, idx),
                value,
                Origin::B,
                idx,
                &mut order_counter,
            );
        }
        assemble_floating(entries, opts, row_output)
    }
}

fn setxor_integer_elements(
    a_storage: &IntegerStorage,
    a_shape: Vec<usize>,
    b_storage: &IntegerStorage,
    b_shape: Vec<usize>,
    opts: &SetxorOptions,
) -> crate::BuiltinResult<SetxorEvaluation> {
    let mut entries = Vec::<SymEntry<IntValue>>::new();
    let mut map = HashMap::<IntValue, usize>::new();
    let mut order_counter = 0usize;
    for (index, value) in a_storage.exact_values().into_iter().enumerate() {
        add_sym_entry(
            &mut entries,
            &mut map,
            value.clone(),
            value,
            Origin::A,
            index,
            &mut order_counter,
        );
    }
    for (index, value) in b_storage.exact_values().into_iter().enumerate() {
        add_sym_entry(
            &mut entries,
            &mut map,
            value.clone(),
            value,
            Origin::B,
            index,
            &mut order_counter,
        );
    }
    assemble_integer(
        entries,
        a_storage,
        opts,
        element_row_output(&a_shape, &b_shape),
    )
}

fn setxor_integer_rows(
    a_storage: &IntegerStorage,
    a_shape: Vec<usize>,
    b_storage: &IntegerStorage,
    b_shape: Vec<usize>,
    opts: &SetxorOptions,
) -> crate::BuiltinResult<SetxorEvaluation> {
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(setxor_internal_error(
            "setxor: 'rows' option requires 2-D numeric matrices",
        ));
    }
    if a_shape[1] != b_shape[1] {
        return Err(setxor_error(&SETXOR_ERROR_ROWS_COLUMN_MISMATCH));
    }
    let (rows_a, rows_b, cols) = (a_shape[0], b_shape[0], a_shape[1]);
    let a_values = a_storage.exact_values();
    let b_values = b_storage.exact_values();
    let mut entries = Vec::<SymEntry<Vec<IntValue>>>::new();
    let mut map = HashMap::<Vec<IntValue>, usize>::new();
    let mut order_counter = 0usize;
    for row in 0..rows_a {
        let values: Vec<_> = (0..cols)
            .map(|col| a_values[row + col * rows_a].clone())
            .collect();
        add_sym_entry(
            &mut entries,
            &mut map,
            values.clone(),
            values,
            Origin::A,
            row,
            &mut order_counter,
        );
    }
    for row in 0..rows_b {
        let values: Vec<_> = (0..cols)
            .map(|col| b_values[row + col * rows_b].clone())
            .collect();
        add_sym_entry(
            &mut entries,
            &mut map,
            values.clone(),
            values,
            Origin::B,
            row,
            &mut order_counter,
        );
    }
    assemble_integer_rows(entries, a_storage, opts, cols)
}

fn setxor_floating_rows<T: SetFloat>(
    a_values: Vec<T>,
    a_shape: Vec<usize>,
    b_values: Vec<T>,
    b_shape: Vec<usize>,
    opts: &SetxorOptions,
) -> crate::BuiltinResult<SetxorEvaluation> {
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(setxor_internal_error(
            "setxor: 'rows' option requires 2-D numeric matrices",
        ));
    }
    if a_shape[1] != b_shape[1] {
        return Err(setxor_error(&SETXOR_ERROR_ROWS_COLUMN_MISMATCH));
    }
    let rows_a = a_shape[0];
    let rows_b = b_shape[0];
    let cols = a_shape[1];
    let mut entries = Vec::<SymEntry<Vec<T>>>::new();
    let mut map: HashMap<NumericRowKey, usize> = HashMap::new();
    let mut order_counter = 0usize;
    for row in 0..rows_a {
        let values = numeric_row_from_values(&a_values, row, rows_a, cols);
        let key = numeric_row_key(&values, Origin::A, row);
        add_sym_entry(
            &mut entries,
            &mut map,
            key,
            values,
            Origin::A,
            row,
            &mut order_counter,
        );
    }
    for row in 0..rows_b {
        let values = numeric_row_from_values(&b_values, row, rows_b, cols);
        let key = numeric_row_key(&values, Origin::B, row);
        add_sym_entry(
            &mut entries,
            &mut map,
            key,
            values,
            Origin::B,
            row,
            &mut order_counter,
        );
    }
    assemble_floating_rows(entries, opts, cols)
}

fn setxor_complex(
    a: ComplexTensor,
    b: ComplexTensor,
    opts: &SetxorOptions,
) -> crate::BuiltinResult<SetxorEvaluation> {
    let a_shape = a.shape.clone();
    let b_shape = b.shape.clone();
    match (a.into_complex_storage(), b.into_complex_storage()) {
        (ComplexStorage::F64(a), ComplexStorage::F64(b)) => {
            setxor_floating_complex(a, a_shape, b, b_shape, opts)
        }
        (ComplexStorage::F32(a), ComplexStorage::F32(b)) => {
            setxor_floating_complex(a, a_shape, b, b_shape, opts)
        }
        (a, b) => setxor_promoted_complex_f64(a, a_shape, b, b_shape, opts),
    }
}

fn setxor_promoted_complex_f64(
    a: ComplexStorage,
    a_shape: Vec<usize>,
    b: ComplexStorage,
    b_shape: Vec<usize>,
    opts: &SetxorOptions,
) -> crate::BuiltinResult<SetxorEvaluation> {
    setxor_floating_complex(
        a.materialize_f64(),
        a_shape,
        b.materialize_f64(),
        b_shape,
        opts,
    )
}

fn setxor_floating_complex<T: SetFloat>(
    a: Vec<(T, T)>,
    a_shape: Vec<usize>,
    b: Vec<(T, T)>,
    b_shape: Vec<usize>,
    opts: &SetxorOptions,
) -> crate::BuiltinResult<SetxorEvaluation> {
    if opts.rows {
        setxor_complex_rows(a, a_shape, b, b_shape, opts)
    } else {
        let row_output = element_row_output(&a_shape, &b_shape);
        let mut entries = Vec::<SymEntry<(T, T)>>::new();
        let mut map: HashMap<ComplexElementKey, usize> = HashMap::new();
        let mut order_counter = 0usize;
        for (idx, &value) in a.iter().enumerate() {
            add_sym_entry(
                &mut entries,
                &mut map,
                complex_element_key(value, Origin::A, idx),
                value,
                Origin::A,
                idx,
                &mut order_counter,
            );
        }
        for (idx, &value) in b.iter().enumerate() {
            add_sym_entry(
                &mut entries,
                &mut map,
                complex_element_key(value, Origin::B, idx),
                value,
                Origin::B,
                idx,
                &mut order_counter,
            );
        }
        assemble_complex(entries, opts, row_output)
    }
}

fn setxor_complex_rows<T: SetFloat>(
    a: Vec<(T, T)>,
    a_shape: Vec<usize>,
    b: Vec<(T, T)>,
    b_shape: Vec<usize>,
    opts: &SetxorOptions,
) -> crate::BuiltinResult<SetxorEvaluation> {
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(setxor_internal_error(
            "setxor: 'rows' option requires 2-D complex matrices",
        ));
    }
    if a_shape[1] != b_shape[1] {
        return Err(setxor_error(&SETXOR_ERROR_ROWS_COLUMN_MISMATCH));
    }
    let rows_a = a_shape[0];
    let rows_b = b_shape[0];
    let cols = a_shape[1];
    let mut entries = Vec::<SymEntry<Vec<(T, T)>>>::new();
    let mut map: HashMap<ComplexRowKey, usize> = HashMap::new();
    let mut order_counter = 0usize;
    for row in 0..rows_a {
        let values = complex_row(&a, row, rows_a, cols);
        let key = complex_row_key(&values, Origin::A, row);
        add_sym_entry(
            &mut entries,
            &mut map,
            key,
            values,
            Origin::A,
            row,
            &mut order_counter,
        );
    }
    for row in 0..rows_b {
        let values = complex_row(&b, row, rows_b, cols);
        let key = complex_row_key(&values, Origin::B, row);
        add_sym_entry(
            &mut entries,
            &mut map,
            key,
            values,
            Origin::B,
            row,
            &mut order_counter,
        );
    }
    assemble_complex_rows(entries, opts, cols)
}

fn setxor_char(
    a: CharArray,
    b: CharArray,
    opts: &SetxorOptions,
) -> crate::BuiltinResult<SetxorEvaluation> {
    if opts.rows {
        setxor_char_rows(a, b, opts)
    } else {
        let row_output = a.rows == 1 && b.rows == 1;
        let mut entries = Vec::<SymEntry<char>>::new();
        let mut map: HashMap<u32, usize> = HashMap::new();
        let mut order_counter = 0usize;
        for col in 0..a.cols {
            for row in 0..a.rows {
                let linear_idx = row + col * a.rows;
                let data_idx = row * a.cols + col;
                let ch = a.data[data_idx];
                add_sym_entry(
                    &mut entries,
                    &mut map,
                    ch as u32,
                    ch,
                    Origin::A,
                    linear_idx,
                    &mut order_counter,
                );
            }
        }
        for col in 0..b.cols {
            for row in 0..b.rows {
                let linear_idx = row + col * b.rows;
                let data_idx = row * b.cols + col;
                let ch = b.data[data_idx];
                add_sym_entry(
                    &mut entries,
                    &mut map,
                    ch as u32,
                    ch,
                    Origin::B,
                    linear_idx,
                    &mut order_counter,
                );
            }
        }
        assemble_char(entries, opts, row_output)
    }
}

fn setxor_char_rows(
    a: CharArray,
    b: CharArray,
    opts: &SetxorOptions,
) -> crate::BuiltinResult<SetxorEvaluation> {
    if a.cols != b.cols {
        return Err(setxor_error(&SETXOR_ERROR_ROWS_COLUMN_MISMATCH));
    }
    let mut entries = Vec::<SymEntry<Vec<char>>>::new();
    let mut map: HashMap<RowCharKey, usize> = HashMap::new();
    let mut order_counter = 0usize;
    for row in 0..a.rows {
        let values = char_row(&a, row);
        add_sym_entry(
            &mut entries,
            &mut map,
            RowCharKey(values.iter().map(|&ch| ch as u32).collect()),
            values,
            Origin::A,
            row,
            &mut order_counter,
        );
    }
    for row in 0..b.rows {
        let values = char_row(&b, row);
        add_sym_entry(
            &mut entries,
            &mut map,
            RowCharKey(values.iter().map(|&ch| ch as u32).collect()),
            values,
            Origin::B,
            row,
            &mut order_counter,
        );
    }
    assemble_char_rows(entries, opts, a.cols)
}

fn setxor_string(
    a: StringArray,
    b: StringArray,
    opts: &SetxorOptions,
) -> crate::BuiltinResult<SetxorEvaluation> {
    if opts.rows {
        setxor_string_rows(a, b, opts)
    } else {
        let row_output = element_row_output(&a.shape, &b.shape);
        let mut entries = Vec::<SymEntry<String>>::new();
        let mut map: HashMap<String, usize> = HashMap::new();
        let mut order_counter = 0usize;
        for (idx, value) in a.data.iter().enumerate() {
            add_sym_entry(
                &mut entries,
                &mut map,
                value.clone(),
                value.clone(),
                Origin::A,
                idx,
                &mut order_counter,
            );
        }
        for (idx, value) in b.data.iter().enumerate() {
            add_sym_entry(
                &mut entries,
                &mut map,
                value.clone(),
                value.clone(),
                Origin::B,
                idx,
                &mut order_counter,
            );
        }
        assemble_string(entries, opts, row_output)
    }
}

fn setxor_string_rows(
    a: StringArray,
    b: StringArray,
    opts: &SetxorOptions,
) -> crate::BuiltinResult<SetxorEvaluation> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(setxor_internal_error(
            "setxor: 'rows' option requires 2-D string arrays",
        ));
    }
    if a.shape[1] != b.shape[1] {
        return Err(setxor_error(&SETXOR_ERROR_ROWS_COLUMN_MISMATCH));
    }
    let rows_a = a.shape[0];
    let rows_b = b.shape[0];
    let cols = a.shape[1];
    let mut entries = Vec::<SymEntry<Vec<String>>>::new();
    let mut map: HashMap<RowStringKey, usize> = HashMap::new();
    let mut order_counter = 0usize;
    for row in 0..rows_a {
        let values = string_row(&a, row, cols);
        add_sym_entry(
            &mut entries,
            &mut map,
            RowStringKey(values.clone()),
            values,
            Origin::A,
            row,
            &mut order_counter,
        );
    }
    for row in 0..rows_b {
        let values = string_row(&b, row, cols);
        add_sym_entry(
            &mut entries,
            &mut map,
            RowStringKey(values.clone()),
            values,
            Origin::B,
            row,
            &mut order_counter,
        );
    }
    assemble_string_rows(entries, opts, cols)
}

fn add_sym_entry<K, T>(
    entries: &mut Vec<SymEntry<T>>,
    map: &mut HashMap<K, usize>,
    key: K,
    value: T,
    origin: Origin,
    index: usize,
    order_counter: &mut usize,
) where
    K: Eq + std::hash::Hash,
{
    match map.entry(key) {
        Entry::Occupied(occ) => {
            let entry = &mut entries[*occ.get()];
            match origin {
                Origin::A => {
                    if entry.a_index.is_none() {
                        entry.a_index = Some(index);
                    }
                }
                Origin::B => {
                    if entry.b_index.is_none() {
                        entry.b_index = Some(index);
                    }
                }
            }
        }
        Entry::Vacant(v) => {
            let entry_idx = entries.len();
            let (a_index, b_index) = match origin {
                Origin::A => (Some(index), None),
                Origin::B => (None, Some(index)),
            };
            entries.push(SymEntry {
                value,
                a_index,
                b_index,
                order_rank: *order_counter,
            });
            v.insert(entry_idx);
            *order_counter += 1;
        }
    }
}

fn symmetric_order<T>(
    entries: &[SymEntry<T>],
    opts: &SetxorOptions,
    compare: impl Fn(&T, &T) -> Ordering,
) -> Vec<usize> {
    let mut order = entries
        .iter()
        .enumerate()
        .filter_map(|(idx, entry)| {
            if entry.a_index.is_some() ^ entry.b_index.is_some() {
                Some(idx)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    match opts.order {
        SetxorOrder::Sorted => {
            order.sort_by(|&lhs, &rhs| compare(&entries[lhs].value, &entries[rhs].value))
        }
        SetxorOrder::Stable => order.sort_by_key(|&idx| entries[idx].order_rank),
    }
    order
}

fn collect_indices<T>(entries: &[SymEntry<T>], order: &[usize]) -> (Vec<f64>, Vec<f64>) {
    let mut ia = Vec::new();
    let mut ib = Vec::new();
    for &idx in order {
        let entry = &entries[idx];
        if let Some(a_idx) = entry.a_index {
            ia.push((a_idx + 1) as f64);
        } else if let Some(b_idx) = entry.b_index {
            ib.push((b_idx + 1) as f64);
        }
    }
    (ia, ib)
}

fn index_tensors(ia: Vec<f64>, ib: Vec<f64>) -> crate::BuiltinResult<(Tensor, Tensor)> {
    let ia_len = ia.len();
    let ib_len = ib.len();
    let ia_tensor = Tensor::new(ia, vec![ia_len, 1])
        .map_err(|e| setxor_internal_error(format!("setxor: {e}")))?;
    let ib_tensor = Tensor::new(ib, vec![ib_len, 1])
        .map_err(|e| setxor_internal_error(format!("setxor: {e}")))?;
    Ok((ia_tensor, ib_tensor))
}

fn is_row_vector_shape(shape: &[usize]) -> bool {
    match shape {
        [] => false,
        [_] => true,
        [rows, ..] if *rows != 1 => false,
        [_, _, rest @ ..] => rest.iter().all(|&dim| dim == 1),
    }
}

fn element_row_output(a_shape: &[usize], b_shape: &[usize]) -> bool {
    is_row_vector_shape(a_shape) && is_row_vector_shape(b_shape)
}

fn element_shape(row_output: bool, len: usize) -> Vec<usize> {
    if row_output {
        vec![1, len]
    } else {
        vec![len, 1]
    }
}

fn numeric_output_dtype(
    a_dtype: NumericDType,
    b_dtype: NumericDType,
) -> crate::BuiltinResult<NumericDType> {
    match (a_dtype, b_dtype) {
        (lhs, rhs) if lhs == rhs => Ok(lhs),
        (NumericDType::F64, rhs) => Ok(rhs),
        (lhs, NumericDType::F64) => Ok(lhs),
        _ => Err(setxor_error(&SETXOR_ERROR_NUMERIC_CLASS_MISMATCH)),
    }
}

fn assemble_floating<T: SetFloat>(
    entries: Vec<SymEntry<T>>,
    opts: &SetxorOptions,
    row_output: bool,
) -> crate::BuiltinResult<SetxorEvaluation> {
    let order = symmetric_order(&entries, opts, |lhs, rhs| lhs.compare(*rhs));
    let values = order
        .iter()
        .map(|&idx| entries[idx].value)
        .collect::<Vec<_>>();
    let (ia, ib) = collect_indices(&entries, &order);
    let value_tensor = Tensor::from_numeric_storage(
        T::numeric_storage(values),
        element_shape(row_output, order.len()),
    )
    .map_err(|e| setxor_internal_error(format!("setxor: {e}")))?;
    let (ia_tensor, ib_tensor) = index_tensors(ia, ib)?;
    let value = if value_tensor.numeric_dtype() == NumericDType::F32 {
        Value::Tensor(value_tensor)
    } else {
        tensor::tensor_into_value(value_tensor)
    };
    Ok(SetxorEvaluation::new(value, ia_tensor, ib_tensor))
}

fn assemble_integer(
    entries: Vec<SymEntry<IntValue>>,
    storage: &IntegerStorage,
    opts: &SetxorOptions,
    row_output: bool,
) -> crate::BuiltinResult<SetxorEvaluation> {
    let order = symmetric_order(&entries, opts, |lhs, rhs| {
        integer_order::compare(lhs, rhs, false, false)
    });
    let values = order
        .iter()
        .map(|&index| entries[index].value.clone())
        .collect::<Vec<_>>();
    let (ia, ib) = collect_indices(&entries, &order);
    let values = Tensor::new_integer(
        storage
            .from_exact_values_like(values)
            .map_err(|e| setxor_internal_error(format!("setxor: {e}")))?,
        element_shape(row_output, order.len()),
    )
    .map_err(|e| setxor_internal_error(format!("setxor: {e}")))?;
    let (ia, ib) = index_tensors(ia, ib)?;
    Ok(SetxorEvaluation::new(Value::Tensor(values), ia, ib))
}

fn assemble_floating_rows<T: SetFloat>(
    entries: Vec<SymEntry<Vec<T>>>,
    opts: &SetxorOptions,
    cols: usize,
) -> crate::BuiltinResult<SetxorEvaluation> {
    let order = symmetric_order(&entries, opts, |lhs, rhs| compare_floating_rows(lhs, rhs));
    let rows = order.len();
    let mut values = vec![T::default(); rows * cols];
    for (row_pos, &entry_idx) in order.iter().enumerate() {
        for col in 0..cols {
            values[row_pos + col * rows] = entries[entry_idx].value[col];
        }
    }
    let (ia, ib) = collect_indices(&entries, &order);
    let value_tensor = Tensor::from_numeric_storage(T::numeric_storage(values), vec![rows, cols])
        .map_err(|e| setxor_internal_error(format!("setxor: {e}")))?;
    let (ia_tensor, ib_tensor) = index_tensors(ia, ib)?;
    let value = if value_tensor.numeric_dtype() == NumericDType::F32 {
        Value::Tensor(value_tensor)
    } else {
        tensor::tensor_into_value(value_tensor)
    };
    Ok(SetxorEvaluation::new(value, ia_tensor, ib_tensor))
}

fn assemble_integer_rows(
    entries: Vec<SymEntry<Vec<IntValue>>>,
    storage: &IntegerStorage,
    opts: &SetxorOptions,
    cols: usize,
) -> crate::BuiltinResult<SetxorEvaluation> {
    let order = symmetric_order(&entries, opts, |lhs, rhs| {
        for (left, right) in lhs.iter().zip(rhs) {
            let ordering = integer_order::compare(left, right, false, false);
            if ordering != Ordering::Equal {
                return ordering;
            }
        }
        Ordering::Equal
    });
    let rows = order.len();
    let mut values = Vec::with_capacity(rows * cols);
    for col in 0..cols {
        for &index in &order {
            values.push(entries[index].value[col].clone());
        }
    }
    let (ia, ib) = collect_indices(&entries, &order);
    let values = Tensor::new_integer(
        storage
            .from_exact_values_like(values)
            .map_err(|e| setxor_internal_error(format!("setxor: {e}")))?,
        vec![rows, cols],
    )
    .map_err(|e| setxor_internal_error(format!("setxor: {e}")))?;
    let (ia, ib) = index_tensors(ia, ib)?;
    Ok(SetxorEvaluation::new(Value::Tensor(values), ia, ib))
}

fn assemble_complex<T: SetFloat>(
    entries: Vec<SymEntry<(T, T)>>,
    opts: &SetxorOptions,
    row_output: bool,
) -> crate::BuiltinResult<SetxorEvaluation> {
    let order = symmetric_order(&entries, opts, |lhs, rhs| compare_complex(*lhs, *rhs));
    let values = order
        .iter()
        .map(|&idx| entries[idx].value)
        .collect::<Vec<_>>();
    let (ia, ib) = collect_indices(&entries, &order);
    let value_tensor = ComplexTensor::from_complex_storage(
        T::complex_storage(values),
        element_shape(row_output, order.len()),
    )
    .map_err(|e| setxor_internal_error(format!("setxor: {e}")))?;
    let (ia_tensor, ib_tensor) = index_tensors(ia, ib)?;
    let value = if value_tensor.as_f32_slice().is_some() {
        Value::ComplexTensor(value_tensor)
    } else {
        complex_tensor_into_value(value_tensor)
    };
    Ok(SetxorEvaluation::new(value, ia_tensor, ib_tensor))
}

fn assemble_complex_rows<T: SetFloat>(
    entries: Vec<SymEntry<Vec<(T, T)>>>,
    opts: &SetxorOptions,
    cols: usize,
) -> crate::BuiltinResult<SetxorEvaluation> {
    let order = symmetric_order(&entries, opts, |lhs, rhs| compare_complex_rows(lhs, rhs));
    let rows = order.len();
    let mut values = vec![(T::default(), T::default()); rows * cols];
    for (row_pos, &entry_idx) in order.iter().enumerate() {
        for col in 0..cols {
            values[row_pos + col * rows] = entries[entry_idx].value[col];
        }
    }
    let (ia, ib) = collect_indices(&entries, &order);
    let value_tensor =
        ComplexTensor::from_complex_storage(T::complex_storage(values), vec![rows, cols])
            .map_err(|e| setxor_internal_error(format!("setxor: {e}")))?;
    let (ia_tensor, ib_tensor) = index_tensors(ia, ib)?;
    let value = if value_tensor.as_f32_slice().is_some() {
        Value::ComplexTensor(value_tensor)
    } else {
        complex_tensor_into_value(value_tensor)
    };
    Ok(SetxorEvaluation::new(value, ia_tensor, ib_tensor))
}

fn assemble_char(
    entries: Vec<SymEntry<char>>,
    opts: &SetxorOptions,
    row_output: bool,
) -> crate::BuiltinResult<SetxorEvaluation> {
    let order = symmetric_order(&entries, opts, |lhs, rhs| lhs.cmp(rhs));
    let values = order
        .iter()
        .map(|&idx| entries[idx].value)
        .collect::<Vec<_>>();
    let (ia, ib) = collect_indices(&entries, &order);
    let (rows, cols) = if row_output {
        (1, order.len())
    } else {
        (order.len(), 1)
    };
    let value_array = CharArray::new(values, rows, cols)
        .map_err(|e| setxor_internal_error(format!("setxor: {e}")))?;
    let (ia_tensor, ib_tensor) = index_tensors(ia, ib)?;
    Ok(SetxorEvaluation::new(
        Value::CharArray(value_array),
        ia_tensor,
        ib_tensor,
    ))
}

fn assemble_char_rows(
    entries: Vec<SymEntry<Vec<char>>>,
    opts: &SetxorOptions,
    cols: usize,
) -> crate::BuiltinResult<SetxorEvaluation> {
    let order = symmetric_order(&entries, opts, |lhs, rhs| compare_char_rows(lhs, rhs));
    let rows = order.len();
    let mut values = vec!['\0'; rows * cols];
    for (row_pos, &entry_idx) in order.iter().enumerate() {
        for col in 0..cols {
            values[row_pos * cols + col] = entries[entry_idx].value[col];
        }
    }
    let (ia, ib) = collect_indices(&entries, &order);
    let value_array = CharArray::new(values, rows, cols)
        .map_err(|e| setxor_internal_error(format!("setxor: {e}")))?;
    let (ia_tensor, ib_tensor) = index_tensors(ia, ib)?;
    Ok(SetxorEvaluation::new(
        Value::CharArray(value_array),
        ia_tensor,
        ib_tensor,
    ))
}

fn assemble_string(
    entries: Vec<SymEntry<String>>,
    opts: &SetxorOptions,
    row_output: bool,
) -> crate::BuiltinResult<SetxorEvaluation> {
    let order = symmetric_order(&entries, opts, |lhs, rhs| lhs.cmp(rhs));
    let values = order
        .iter()
        .map(|&idx| entries[idx].value.clone())
        .collect::<Vec<_>>();
    let (ia, ib) = collect_indices(&entries, &order);
    let value_array = StringArray::new(values, element_shape(row_output, order.len()))
        .map_err(|e| setxor_internal_error(format!("setxor: {e}")))?;
    let (ia_tensor, ib_tensor) = index_tensors(ia, ib)?;
    Ok(SetxorEvaluation::new(
        Value::StringArray(value_array),
        ia_tensor,
        ib_tensor,
    ))
}

fn assemble_string_rows(
    entries: Vec<SymEntry<Vec<String>>>,
    opts: &SetxorOptions,
    cols: usize,
) -> crate::BuiltinResult<SetxorEvaluation> {
    let order = symmetric_order(&entries, opts, |lhs, rhs| compare_string_rows(lhs, rhs));
    let rows = order.len();
    let mut values = vec![String::new(); rows * cols];
    for (row_pos, &entry_idx) in order.iter().enumerate() {
        for col in 0..cols {
            values[row_pos + col * rows] = entries[entry_idx].value[col].clone();
        }
    }
    let (ia, ib) = collect_indices(&entries, &order);
    let value_array = StringArray::new(values, vec![rows, cols])
        .map_err(|e| setxor_internal_error(format!("setxor: {e}")))?;
    let (ia_tensor, ib_tensor) = index_tensors(ia, ib)?;
    Ok(SetxorEvaluation::new(
        Value::StringArray(value_array),
        ia_tensor,
        ib_tensor,
    ))
}

#[derive(Debug, Clone)]
pub struct SetxorEvaluation {
    values: Value,
    ia: Tensor,
    ib: Tensor,
}

impl SetxorEvaluation {
    fn new(values: Value, ia: Tensor, ib: Tensor) -> Self {
        Self { values, ia, ib }
    }

    pub fn into_values_value(self) -> Value {
        self.values
    }

    pub fn into_triple(self) -> (Value, Value, Value) {
        (
            self.values,
            tensor::tensor_into_value(self.ia),
            tensor::tensor_into_value(self.ib),
        )
    }

    pub fn values_value(&self) -> Value {
        self.values.clone()
    }

    pub fn ia_value(&self) -> Value {
        tensor::tensor_into_value(self.ia.clone())
    }

    pub fn ib_value(&self) -> Value {
        tensor::tensor_into_value(self.ib.clone())
    }
}

fn numeric_key<T: SetFloat>(value: T, origin: Origin, index: usize) -> NumericKey {
    if value.is_nan() {
        NumericKey::UniqueNan(origin, index)
    } else {
        NumericKey::Value(value.canonical_key())
    }
}

fn numeric_row_key<T: SetFloat>(values: &[T], origin: Origin, row: usize) -> NumericRowKey {
    if values.iter().any(|value| value.is_nan()) {
        NumericRowKey::UniqueNan(origin, row)
    } else {
        NumericRowKey::Values(values.iter().map(|&value| value.canonical_key()).collect())
    }
}

fn complex_element_key<T: SetFloat>(
    value: (T, T),
    origin: Origin,
    index: usize,
) -> ComplexElementKey {
    if complex_is_nan(value) {
        ComplexElementKey::UniqueNan(origin, index)
    } else {
        ComplexElementKey::Value(ComplexKey::new(value))
    }
}

fn complex_row_key<T: SetFloat>(values: &[(T, T)], origin: Origin, row: usize) -> ComplexRowKey {
    if values.iter().any(|&value| complex_is_nan(value)) {
        ComplexRowKey::UniqueNan(origin, row)
    } else {
        ComplexRowKey::Values(values.iter().map(|&value| ComplexKey::new(value)).collect())
    }
}

fn numeric_row_from_values<T: Copy>(values: &[T], row: usize, rows: usize, cols: usize) -> Vec<T> {
    (0..cols).map(|col| values[row + col * rows]).collect()
}

fn complex_row<T: Copy>(values: &[(T, T)], row: usize, rows: usize, cols: usize) -> Vec<(T, T)> {
    (0..cols).map(|col| values[row + col * rows]).collect()
}

fn char_row(array: &CharArray, row: usize) -> Vec<char> {
    (0..array.cols)
        .map(|col| array.data[row * array.cols + col])
        .collect()
}

fn string_row(array: &StringArray, row: usize, cols: usize) -> Vec<String> {
    (0..cols)
        .map(|col| array.data[row + col * array.shape[0]].clone())
        .collect()
}

fn compare_floating_rows<T: SetFloat>(a: &[T], b: &[T]) -> Ordering {
    for (lhs, rhs) in a.iter().zip(b.iter()) {
        let ord = lhs.compare(*rhs);
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

impl ComplexKey {
    fn new<T: SetFloat>(value: (T, T)) -> Self {
        Self {
            re: value.0.canonical_key(),
            im: value.1.canonical_key(),
        }
    }
}

fn complex_is_nan<T: SetFloat>(value: (T, T)) -> bool {
    value.0.is_nan() || value.1.is_nan()
}

fn compare_complex<T: SetFloat>(a: (T, T), b: (T, T)) -> Ordering {
    match (complex_is_nan(a), complex_is_nan(b)) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => {
            let mag_cmp = a.0.hypot(a.1).compare(b.0.hypot(b.1));
            if mag_cmp != Ordering::Equal {
                return mag_cmp;
            }
            let phase_cmp = a.1.atan2(a.0).compare(b.1.atan2(b.0));
            if phase_cmp != Ordering::Equal {
                return phase_cmp;
            }
            let re_cmp = a.0.compare(b.0);
            if re_cmp != Ordering::Equal {
                re_cmp
            } else {
                a.1.compare(b.1)
            }
        }
    }
}

fn compare_complex_rows<T: SetFloat>(a: &[(T, T)], b: &[(T, T)]) -> Ordering {
    for (lhs, rhs) in a.iter().zip(b.iter()) {
        let ord = compare_complex(*lhs, *rhs);
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

fn compare_char_rows(a: &[char], b: &[char]) -> Ordering {
    for (lhs, rhs) in a.iter().zip(b.iter()) {
        let ord = lhs.cmp(rhs);
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

fn compare_string_rows(a: &[String], b: &[String]) -> Ordering {
    for (lhs, rhs) in a.iter().zip(b.iter()) {
        let ord = lhs.cmp(rhs);
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{ResolveContext, Type};
    use runmat_value::IntValue;

    fn evaluate_sync(a: Value, b: Value, rest: &[Value]) -> crate::BuiltinResult<SetxorEvaluation> {
        futures::executor::block_on(evaluate(a, b, rest))
    }

    fn builtin_sync(a: Value, b: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        futures::executor::block_on(setxor_builtin(a, b, rest))
    }

    fn assert_double(tensor: &Tensor, expected: &[f64]) {
        assert_eq!(tensor.as_f64_slice().expect("double tensor"), expected);
    }

    #[test]
    fn registered_builtin_restores_resident_outputs() {
        test_support::with_test_provider(|provider| {
            let left = Tensor::new_integer(IntegerStorage::I32(vec![7, 2, 9]), vec![3, 1]).unwrap();
            let right = Tensor::new_integer(IntegerStorage::I32(vec![2, 7]), vec![2, 1]).unwrap();
            let left =
                Value::GpuTensor(gpu_helpers::upload_tensor(provider, &left).expect("upload left"));
            let right = Value::GpuTensor(
                gpu_helpers::upload_tensor(provider, &right).expect("upload right"),
            );
            let _guard = crate::output_count::push_output_count(Some(3));
            let Value::OutputList(outputs) =
                builtin_sync(left, right, Vec::new()).expect("resident setxor")
            else {
                panic!("expected output list");
            };
            assert_eq!(outputs.len(), 3);
            assert!(outputs
                .iter()
                .all(|output| matches!(output, Value::GpuTensor(_))));
            assert_eq!(
                test_support::gather(outputs[0].clone())
                    .expect("gather values")
                    .integer_storage(),
                Some(&IntegerStorage::I32(vec![9]))
            );
        });
    }

    #[test]
    fn setxor_type_resolver_numeric() {
        assert_eq!(
            set_values_output_type(
                &[Type::tensor(), Type::tensor()],
                &ResolveContext::new(Vec::new()),
            ),
            Type::tensor()
        );
    }

    #[test]
    fn setxor_numeric_sorted_default_with_indices() {
        let a = Tensor::new(vec![5.0, 1.0, 3.0, 3.0, 3.0], vec![5, 1]).unwrap();
        let b = Tensor::new(vec![4.0, 1.0, 2.0], vec![3, 1]).unwrap();
        let eval = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[]).expect("setxor");
        let values = tensor::value_into_tensor_for("setxor", eval.values_value()).unwrap();
        assert_double(&values, &[2.0, 3.0, 4.0, 5.0]);
        assert_eq!(values.shape, vec![4, 1]);
        let ia = tensor::value_into_tensor_for("setxor", eval.ia_value()).unwrap();
        assert_double(&ia, &[3.0, 1.0]);
        let ib = tensor::value_into_tensor_for("setxor", eval.ib_value()).unwrap();
        assert_double(&ib, &[3.0, 1.0]);
    }

    #[test]
    fn setxor_preserves_exact_integer_elements_and_rows() {
        let a = Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![u64::MAX, 0, 9_007_199_254_740_993]),
            vec![3, 1],
        )
        .expect("input");
        let b = Tensor::new_integer(runmat_value::IntegerStorage::U64(vec![0, 7]), vec![2, 1])
            .expect("input");
        let (values, ia, ib) = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[])
            .expect("setxor")
            .into_triple();
        let Value::Tensor(values) = values else {
            panic!("exact values");
        };
        assert_eq!(
            values.integer_storage(),
            Some(&runmat_value::IntegerStorage::U64(vec![
                7,
                9_007_199_254_740_993,
                u64::MAX
            ]))
        );
        let ia = tensor::value_into_tensor_for("setxor", ia).expect("indices");
        assert_double(&ia, &[3.0, 1.0]);
        let ib = tensor::value_into_tensor_for("setxor", ib).expect("indices");
        assert_double(&ib, &[2.0]);

        let a = Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993, 0, 1]),
            vec![2, 2],
        )
        .expect("rows input");
        let b = Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![9_007_199_254_740_993, 4, 1, 2]),
            vec![2, 2],
        )
        .expect("rows input");
        let (values, ia, ib) =
            evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[Value::from("rows")])
                .expect("setxor rows")
                .into_triple();
        let Value::Tensor(values) = values else {
            panic!("exact row values");
        };
        assert_eq!(
            values.integer_storage(),
            Some(&runmat_value::IntegerStorage::U64(vec![4, u64::MAX, 2, 0]))
        );
        let ia = tensor::value_into_tensor_for("setxor", ia).expect("row indices");
        assert_double(&ia, &[1.0]);
        let ib = tensor::value_into_tensor_for("setxor", ib).expect("row indices");
        assert_double(&ib, &[2.0]);
    }

    #[test]
    fn setxor_numeric_integer_and_double_preserve_exact_target_storage() {
        let a = Tensor::new_integer(runmat_value::IntegerStorage::U16(vec![7, 2, 9]), vec![3, 1])
            .expect("input");
        let b = Tensor::new(vec![2.0, 5.0], vec![2, 1]).expect("input");
        let (values, ia, ib) = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[])
            .expect("setxor")
            .into_triple();
        let values = tensor::value_into_tensor_for("setxor", values).expect("values");
        assert_eq!(
            values.integer_storage(),
            Some(&IntegerStorage::U16(vec![5, 7, 9]))
        );
        assert_eq!(values.shape, vec![3, 1]);
        let ia = tensor::value_into_tensor_for("setxor", ia).expect("indices");
        assert_double(&ia, &[1.0, 3.0]);
        let ib = tensor::value_into_tensor_for("setxor", ib).expect("indices");
        assert_double(&ib, &[2.0]);

        let a = Tensor::new_integer(
            runmat_value::IntegerStorage::U16(vec![1, 3, 1, 2, 4, 2]),
            vec![3, 2],
        )
        .expect("rows input");
        let b = Tensor::new(vec![3.0, 5.0, 4.0, 6.0], vec![2, 2]).expect("rows input");
        let (values, ia, ib) =
            evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[Value::from("rows")])
                .expect("setxor rows")
                .into_triple();
        let values = tensor::value_into_tensor_for("setxor", values).expect("row values");
        assert_eq!(values.shape, vec![2, 2]);
        assert_eq!(
            values.integer_storage(),
            Some(&IntegerStorage::U16(vec![1, 5, 2, 6]))
        );
        let ia = tensor::value_into_tensor_for("setxor", ia).expect("row indices");
        assert_double(&ia, &[1.0]);
        let ib = tensor::value_into_tensor_for("setxor", ib).expect("row indices");
        assert_double(&ib, &[2.0]);

        let wide =
            Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
                .unwrap();
        let double = Tensor::new(vec![7.0], vec![1, 1]).unwrap();
        let values = evaluate_sync(Value::Tensor(wide), Value::Tensor(double), &[])
            .unwrap()
            .into_values_value();
        let Value::Tensor(values) = values else {
            panic!("integer result");
        };
        assert_eq!(
            values.integer_storage(),
            Some(&IntegerStorage::U64(vec![7, 9_007_199_254_740_993]))
        );
    }

    #[test]
    fn setxor_numeric_preserves_row_vector_shape_when_both_inputs_are_rows() {
        let a = Tensor::new(vec![5.0, 1.0, 3.0], vec![1, 3]).unwrap();
        let b = Tensor::new(vec![4.0, 1.0, 2.0], vec![1, 3]).unwrap();
        let eval = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[]).expect("setxor");
        let values = tensor::value_into_tensor_for("setxor", eval.values_value()).unwrap();
        assert_double(&values, &[2.0, 3.0, 4.0, 5.0]);
        assert_eq!(values.shape, vec![1, 4]);
        let ia = tensor::value_into_tensor_for("setxor", eval.ia_value()).unwrap();
        assert_eq!(ia.shape, vec![2, 1]);
        let ib = tensor::value_into_tensor_for("setxor", eval.ib_value()).unwrap();
        assert_eq!(ib.shape, vec![2, 1]);
    }

    #[test]
    fn setxor_numeric_preserves_matching_dtype() {
        let a = Tensor::new_with_dtype(vec![5.0, 1.0, 3.0], vec![1, 3], NumericDType::U32).unwrap();
        let b = Tensor::new_with_dtype(vec![5.0, 2.0], vec![1, 2], NumericDType::U32).unwrap();
        let eval = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[]).expect("setxor");
        let values = tensor::value_into_tensor_for("setxor", eval.values_value()).unwrap();
        assert_eq!(
            values.integer_storage(),
            Some(&IntegerStorage::U32(vec![1, 2, 3]))
        );
        assert_eq!(values.shape, vec![1, 3]);
        assert_eq!(values.numeric_dtype(), NumericDType::U32);
    }

    #[test]
    fn setxor_preserves_native_single_elements_and_rows() {
        let a = Tensor::from_f32(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let b = Tensor::from_f32(vec![2.0], vec![1, 1]).unwrap();
        let values = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[])
            .expect("single setxor")
            .into_values_value();
        let Value::Tensor(values) = values else {
            panic!("expected native single values");
        };
        assert_eq!(
            values.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![1.0])
        );

        let a = Tensor::from_f32(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_f32(vec![3.0, 5.0, 4.0, 6.0], vec![2, 2]).unwrap();
        let values = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[Value::from("rows")])
            .expect("single row setxor")
            .into_values_value();
        let Value::Tensor(values) = values else {
            panic!("expected native single rows");
        };
        assert_eq!(values.shape, vec![2, 2]);
        assert_eq!(
            values.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![1.0, 5.0, 2.0, 6.0])
        );
    }

    #[test]
    fn setxor_double_single_promotion_preserves_origin_and_single_class() {
        let a = Tensor::from_f32(vec![3.0, 2.0], vec![2, 1]).unwrap();
        let b = Tensor::new(vec![2.0, 1.0], vec![2, 1]).unwrap();
        let (values, ia, ib) =
            evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[Value::from("stable")])
                .expect("single-double setxor")
                .into_triple();
        let Value::Tensor(values) = values else {
            panic!("expected native single values");
        };
        assert_eq!(
            values.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![3.0, 1.0])
        );
        let ia = tensor::value_into_tensor_for("setxor", ia).unwrap();
        let ib = tensor::value_into_tensor_for("setxor", ib).unwrap();
        assert_double(&ia, &[1.0]);
        assert_double(&ib, &[2.0]);

        let a = Tensor::new(vec![3.0, 2.0], vec![2, 1]).unwrap();
        let b = Tensor::from_f32(vec![2.0, 1.0], vec![2, 1]).unwrap();
        let (values, ia, ib) =
            evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[Value::from("stable")])
                .expect("double-single setxor")
                .into_triple();
        let Value::Tensor(values) = values else {
            panic!("expected native single values");
        };
        assert_eq!(
            values.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![3.0, 1.0])
        );
        let ia = tensor::value_into_tensor_for("setxor", ia).unwrap();
        let ib = tensor::value_into_tensor_for("setxor", ib).unwrap();
        assert_double(&ia, &[1.0]);
        assert_double(&ib, &[2.0]);
    }

    #[test]
    fn setxor_preserves_native_complex_single_elements_and_rows() {
        let a = ComplexTensor::from_f32(vec![(1.0, 1.0), (2.0, 0.0)], vec![2, 1]).unwrap();
        let b = ComplexTensor::from_f32(vec![(2.0, 0.0)], vec![1, 1]).unwrap();
        let values = evaluate_sync(Value::ComplexTensor(a), Value::ComplexTensor(b), &[])
            .expect("complex single setxor")
            .into_values_value();
        let Value::ComplexTensor(values) = values else {
            panic!("expected native complex single value");
        };
        assert_eq!(values.as_f32_slice(), Some(&[(1.0, 1.0)][..]));

        let a = ComplexTensor::from_f32(
            vec![(1.0, 0.0), (3.0, 0.0), (2.0, 1.0), (4.0, 1.0)],
            vec![2, 2],
        )
        .unwrap();
        let b = ComplexTensor::from_f32(
            vec![(3.0, 0.0), (5.0, 0.0), (4.0, 1.0), (6.0, 1.0)],
            vec![2, 2],
        )
        .unwrap();
        let values = evaluate_sync(
            Value::ComplexTensor(a),
            Value::ComplexTensor(b),
            &[Value::from("rows")],
        )
        .expect("complex single row setxor")
        .into_values_value();
        let Value::ComplexTensor(values) = values else {
            panic!("expected native complex single rows");
        };
        assert_eq!(values.shape, vec![2, 2]);
        assert_eq!(
            values.as_f32_slice(),
            Some(&[(1.0, 0.0), (5.0, 0.0), (2.0, 1.0), (6.0, 1.0),][..])
        );
    }

    #[test]
    fn setxor_numeric_double_and_nondouble_returns_nondouble_dtype() {
        let a = Tensor::new_with_dtype(vec![5.0, 1.0, 3.0], vec![1, 3], NumericDType::U32).unwrap();
        let b = Tensor::new(vec![5.0, 2.0], vec![1, 2]).unwrap();
        let eval = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[]).expect("setxor");
        let values = tensor::value_into_tensor_for("setxor", eval.values_value()).unwrap();
        assert_eq!(
            values.integer_storage(),
            Some(&IntegerStorage::U32(vec![1, 2, 3]))
        );
        assert_eq!(values.numeric_dtype(), NumericDType::U32);
    }

    #[test]
    fn setxor_numeric_rejects_incompatible_nondouble_classes() {
        let a = Tensor::new_with_dtype(vec![1.0, 2.0], vec![1, 2], NumericDType::U8).unwrap();
        let b = Tensor::new_with_dtype(vec![2.0, 3.0], vec![1, 2], NumericDType::U32).unwrap();
        let err = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[]).unwrap_err();
        assert_eq!(
            err.identifier(),
            SETXOR_ERROR_NUMERIC_CLASS_MISMATCH.identifier
        );
    }

    #[test]
    fn setxor_numeric_stable_order() {
        let a = Tensor::new(vec![5.0, 1.0, 3.0, 3.0, 3.0], vec![5, 1]).unwrap();
        let b = Tensor::new(vec![4.0, 1.0, 2.0], vec![3, 1]).unwrap();
        let eval =
            evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[Value::from("stable")]).unwrap();
        let values = tensor::value_into_tensor_for("setxor", eval.values_value()).unwrap();
        assert_double(&values, &[5.0, 3.0, 4.0, 2.0]);
        let ia = tensor::value_into_tensor_for("setxor", eval.ia_value()).unwrap();
        assert_double(&ia, &[1.0, 3.0]);
        let ib = tensor::value_into_tensor_for("setxor", eval.ib_value()).unwrap();
        assert_double(&ib, &[1.0, 3.0]);
    }

    #[test]
    fn setxor_treats_nan_values_as_distinct() {
        let a = Tensor::new(vec![5.0, f64::NAN, f64::NAN], vec![3, 1]).unwrap();
        let b = Tensor::new(vec![5.0, f64::NAN, f64::NAN], vec![3, 1]).unwrap();
        let eval = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[]).expect("setxor");
        let values = tensor::value_into_tensor_for("setxor", eval.values_value()).unwrap();
        assert_eq!(values.shape, vec![4, 1]);
        assert!(values
            .as_f64_slice()
            .expect("double values")
            .iter()
            .all(|value| value.is_nan()));
        let ia = tensor::value_into_tensor_for("setxor", eval.ia_value()).unwrap();
        assert_double(&ia, &[2.0, 3.0]);
        let ib = tensor::value_into_tensor_for("setxor", eval.ib_value()).unwrap();
        assert_double(&ib, &[2.0, 3.0]);
    }

    #[test]
    fn setxor_numeric_rows_sorted() {
        let a = Tensor::new(
            vec![
                7.0, 7.0, 7.0, 1.0, 4.0, 8.0, 7.0, 7.0, 2.0, 5.0, 9.0, 1.0, 1.0, 3.0, 6.0,
            ],
            vec![5, 3],
        )
        .unwrap();
        let b = Tensor::new(
            vec![1.0, 4.0, 7.0, 2.0, 5.0, 7.0, 3.0, 6.0, 2.0],
            vec![3, 3],
        )
        .unwrap();
        let eval =
            evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[Value::from("rows")]).unwrap();
        let values = tensor::value_into_tensor_for("setxor", eval.values_value()).unwrap();
        assert_eq!(values.shape, vec![3, 3]);
        assert_double(&values, &[7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 1.0, 2.0, 9.0]);
        let ia = tensor::value_into_tensor_for("setxor", eval.ia_value()).unwrap();
        assert_double(&ia, &[2.0, 1.0]);
        let ib = tensor::value_into_tensor_for("setxor", eval.ib_value()).unwrap();
        assert_double(&ib, &[3.0]);
    }

    #[test]
    fn setxor_complex_values() {
        let a = ComplexTensor::new(vec![(1.0, 1.0), (2.0, 0.0)], vec![2, 1]).unwrap();
        let b = ComplexTensor::new(vec![(2.0, 0.0), (3.0, 0.0)], vec![2, 1]).unwrap();
        let eval =
            evaluate_sync(Value::ComplexTensor(a), Value::ComplexTensor(b), &[]).expect("setxor");
        let Value::ComplexTensor(values) = eval.values_value() else {
            panic!("expected complex tensor");
        };
        assert_eq!(values.materialize_f64(), vec![(1.0, 1.0), (3.0, 0.0)]);
        let ia = tensor::value_into_tensor_for("setxor", eval.ia_value()).unwrap();
        assert_double(&ia, &[1.0]);
        let ib = tensor::value_into_tensor_for("setxor", eval.ib_value()).unwrap();
        assert_double(&ib, &[2.0]);
    }

    #[test]
    fn setxor_promotes_real_input_to_complex_domain() {
        let a = ComplexTensor::new(vec![(1.0, 1.0), (2.0, 0.0)], vec![1, 2]).unwrap();
        let b = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
        let eval = evaluate_sync(Value::ComplexTensor(a), Value::Tensor(b), &[]).expect("setxor");
        let Value::ComplexTensor(values) = eval.values_value() else {
            panic!("expected complex tensor");
        };
        assert_eq!(values.materialize_f64(), vec![(1.0, 1.0), (3.0, 0.0)]);
        assert_eq!(values.shape, vec![1, 2]);
    }

    #[test]
    fn setxor_complex_sorted_uses_phase_after_magnitude() {
        let a = ComplexTensor::new(vec![(0.0, 1.0)], vec![1, 1]).unwrap();
        let b = ComplexTensor::new(vec![(1.0, 0.0)], vec![1, 1]).unwrap();
        let eval =
            evaluate_sync(Value::ComplexTensor(a), Value::ComplexTensor(b), &[]).expect("setxor");
        let Value::ComplexTensor(values) = eval.values_value() else {
            panic!("expected complex tensor");
        };
        assert_eq!(values.materialize_f64(), vec![(1.0, 0.0), (0.0, 1.0)]);
        assert_eq!(values.shape, vec![1, 2]);
    }

    #[test]
    fn setxor_char_elements() {
        let a = CharArray::new(vec!['d', 'o', 'g'], 1, 3).unwrap();
        let b = CharArray::new(vec!['d', 'i', 'g'], 1, 3).unwrap();
        let eval = evaluate_sync(Value::CharArray(a), Value::CharArray(b), &[]).expect("setxor");
        let Value::CharArray(values) = eval.values_value() else {
            panic!("expected char array");
        };
        assert_eq!(values.data, vec!['i', 'o']);
        assert_eq!((values.rows, values.cols), (1, 2));
        let ia = tensor::value_into_tensor_for("setxor", eval.ia_value()).unwrap();
        assert_double(&ia, &[2.0]);
        let ib = tensor::value_into_tensor_for("setxor", eval.ib_value()).unwrap();
        assert_double(&ib, &[2.0]);
    }

    #[test]
    fn setxor_char_and_numeric_compare_character_codes() {
        let a = CharArray::new_row("abc");
        let b = Tensor::new(vec![98.0, 100.0], vec![1, 2]).unwrap();
        let eval = evaluate_sync(Value::CharArray(a), Value::Tensor(b), &[]).expect("setxor");
        let Value::CharArray(values) = eval.values_value() else {
            panic!("expected char array");
        };
        assert_eq!(values.data, vec!['a', 'c', 'd']);
        assert_eq!((values.rows, values.cols), (1, 3));
        let ia = tensor::value_into_tensor_for("setxor", eval.ia_value()).unwrap();
        assert_double(&ia, &[1.0, 3.0]);
        let ib = tensor::value_into_tensor_for("setxor", eval.ib_value()).unwrap();
        assert_double(&ib, &[2.0]);
    }

    #[test]
    fn setxor_string_and_char_vector_compare_strings() {
        let a =
            StringArray::new(vec!["alpha".to_string(), "beta".to_string()], vec![1, 2]).unwrap();
        let b = CharArray::new_row("beta");
        let eval = evaluate_sync(Value::StringArray(a), Value::CharArray(b), &[]).expect("setxor");
        let Value::StringArray(values) = eval.values_value() else {
            panic!("expected string array");
        };
        assert_eq!(values.data, vec!["alpha".to_string()]);
        assert_eq!(values.shape, vec![1, 1]);
    }

    #[test]
    fn setxor_string_rows_stable() {
        let a = StringArray::new(
            vec![
                "alpha".to_string(),
                "gamma".to_string(),
                "beta".to_string(),
                "beta".to_string(),
            ],
            vec![2, 2],
        )
        .unwrap();
        let b = StringArray::new(
            vec![
                "gamma".to_string(),
                "delta".to_string(),
                "beta".to_string(),
                "beta".to_string(),
            ],
            vec![2, 2],
        )
        .unwrap();
        let eval = evaluate_sync(
            Value::StringArray(a),
            Value::StringArray(b),
            &[Value::from("rows"), Value::from("stable")],
        )
        .unwrap();
        let Value::StringArray(values) = eval.values_value() else {
            panic!("expected string array");
        };
        assert_eq!(values.shape, vec![2, 2]);
        assert_eq!(
            values.data,
            vec![
                "alpha".to_string(),
                "delta".to_string(),
                "beta".to_string(),
                "beta".to_string()
            ]
        );
        let ia = tensor::value_into_tensor_for("setxor", eval.ia_value()).unwrap();
        assert_double(&ia, &[1.0]);
        let ib = tensor::value_into_tensor_for("setxor", eval.ib_value()).unwrap();
        assert_double(&ib, &[2.0]);
    }

    #[test]
    fn setxor_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let a = Tensor::new(vec![4.0, 1.0, 2.0], vec![3, 1]).unwrap();
            let b = Tensor::new(vec![2.0, 5.0], vec![2, 1]).unwrap();
            let view_a = HostTensorView {
                data: a.as_f64_slice().expect("double A"),
                shape: &a.shape,
            };
            let view_b = HostTensorView {
                data: b.as_f64_slice().expect("double B"),
                shape: &b.shape,
            };
            let handle_a = provider.upload(&view_a).expect("upload A");
            let handle_b = provider.upload(&view_b).expect("upload B");
            let eval = evaluate_sync(
                Value::GpuTensor(handle_a),
                Value::GpuTensor(handle_b),
                &[Value::from("stable")],
            )
            .expect("setxor");
            let values = tensor::value_into_tensor_for("setxor", eval.values_value()).unwrap();
            assert_double(&values, &[4.0, 1.0, 5.0]);
            let ia = tensor::value_into_tensor_for("setxor", eval.ia_value()).unwrap();
            assert_double(&ia, &[1.0, 2.0]);
            let ib = tensor::value_into_tensor_for("setxor", eval.ib_value()).unwrap();
            assert_double(&ib, &[2.0]);
        });
    }

    #[test]
    fn setxor_gpu_real_and_host_complex_match_host_promotion() {
        test_support::with_test_provider(|provider| {
            let a = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
            let view_a = HostTensorView {
                data: a.as_f64_slice().expect("double A"),
                shape: &a.shape,
            };
            let handle_a = provider.upload(&view_a).expect("upload A");
            let b = ComplexTensor::new(vec![(1.0, 1.0), (2.0, 0.0)], vec![1, 2]).unwrap();
            let eval = evaluate_sync(Value::GpuTensor(handle_a), Value::ComplexTensor(b), &[])
                .expect("setxor");
            let Value::ComplexTensor(values) = eval.values_value() else {
                panic!("expected complex tensor");
            };
            assert_eq!(values.materialize_f64(), vec![(1.0, 1.0), (3.0, 0.0)]);
            assert_eq!(values.shape, vec![1, 2]);
            let ia = tensor::value_into_tensor_for("setxor", eval.ia_value()).unwrap();
            assert_double(&ia, &[2.0]);
            let ib = tensor::value_into_tensor_for("setxor", eval.ib_value()).unwrap();
            assert_double(&ib, &[1.0]);
        });
    }

    #[test]
    fn setxor_rejects_legacy_option() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = evaluate_sync(
            Value::Tensor(tensor.clone()),
            Value::Tensor(tensor),
            &[Value::from("legacy")],
        )
        .unwrap_err();
        assert_eq!(
            err.identifier(),
            SETXOR_ERROR_LEGACY_OPTION_UNSUPPORTED.identifier
        );
    }

    #[test]
    fn setxor_rejects_conflicting_order_options() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = evaluate_sync(
            Value::Tensor(tensor.clone()),
            Value::Tensor(tensor),
            &[Value::from("stable"), Value::from("sorted")],
        )
        .unwrap_err();
        assert_eq!(
            err.identifier(),
            SETXOR_ERROR_CONFLICTING_ORDER_OPTIONS.identifier
        );
    }

    #[test]
    fn setxor_rows_dimension_mismatch() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err =
            evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[Value::from("rows")]).unwrap_err();
        assert_eq!(
            err.identifier(),
            SETXOR_ERROR_ROWS_COLUMN_MISMATCH.identifier
        );
    }

    #[test]
    fn setxor_accepts_scalar_inputs() {
        let eval =
            evaluate_sync(Value::Int(IntValue::I32(1)), Value::Num(3.0), &[]).expect("setxor");
        let values = tensor::value_into_tensor_for("setxor", eval.values_value()).unwrap();
        assert_eq!(
            values.integer_storage(),
            Some(&IntegerStorage::I32(vec![1, 3]))
        );
        let ia = tensor::value_into_tensor_for("setxor", eval.ia_value()).unwrap();
        assert_double(&ia, &[1.0]);
        let ib = tensor::value_into_tensor_for("setxor", eval.ib_value()).unwrap();
        assert_double(&ib, &[1.0]);
    }

    #[test]
    fn setxor_rejects_more_than_three_outputs() {
        let _guard = crate::output_count::push_output_count(Some(4));
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let err = builtin_sync(
            Value::Tensor(tensor.clone()),
            Value::Tensor(tensor),
            Vec::new(),
        )
        .expect_err("too many outputs should fail");
        assert_eq!(err.identifier(), SETXOR_ERROR_TOO_MANY_OUTPUTS.identifier);
    }
}
