//! MATLAB-compatible `union` builtin with GPU-aware semantics for RunMat.
//!
//! Handles element-wise and row-wise unions with optional stable ordering and
//! index outputs that mirror MathWorks MATLAB semantics. GPU tensors use a
//! provider hook or typed host fallback, then public outputs are restored to the
//! owning provider.

use std::cmp::Ordering;
use std::collections::{hash_map::Entry, HashMap};

use runmat_accelerate_api::{GpuTensorHandle, UnionOptions, UnionOrder, UnionResult};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, CharArray, ComplexStorage, ComplexTensor, IntValue, IntegerStorage,
    NumericDType, NumericStorage, StringArray, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use super::{float_order::SetFloat, integer_order, type_resolvers::set_values_output_type};
use crate::build_runtime_error;
use crate::builtins::common::arg_tokens::tokens_from_values;
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::math::elementwise::integer_cast::IntegerTarget;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::sorting_sets::union")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "union",
    op_kind: GpuOpKind::Custom("union"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("union")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: true,
    notes: "Providers may expose a dedicated union hook; exact typed fallback gathers when needed and restores union values plus double indices to the input owner.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::sorting_sets::union")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "union",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "`union` terminates fusion chains and materialises results on the host; upstream tensors are gathered when necessary.",
};

const BUILTIN_NAME: &str = "union";

const UNION_OUTPUT_C: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Union values or rows.",
}];

const UNION_OUTPUT_C_IA: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Union values or rows.",
    },
    BuiltinParamDescriptor {
        name: "ia",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indices selecting contributions from A.",
    },
];

const UNION_OUTPUT_C_IA_IB: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Union values or rows.",
    },
    BuiltinParamDescriptor {
        name: "ia",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indices selecting contributions from A.",
    },
    BuiltinParamDescriptor {
        name: "ib",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indices selecting contributions from B.",
    },
];

const UNION_INPUTS_A_B: [BuiltinParamDescriptor; 2] = [
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

const UNION_INPUTS_A_B_OPTIONS: [BuiltinParamDescriptor; 3] = [
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

const UNION_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "C = union(A, B)",
        inputs: &UNION_INPUTS_A_B,
        outputs: &UNION_OUTPUT_C,
    },
    BuiltinSignatureDescriptor {
        label: "C = union(A, B, option...)",
        inputs: &UNION_INPUTS_A_B_OPTIONS,
        outputs: &UNION_OUTPUT_C,
    },
    BuiltinSignatureDescriptor {
        label: "[C, ia] = union(A, B)",
        inputs: &UNION_INPUTS_A_B,
        outputs: &UNION_OUTPUT_C_IA,
    },
    BuiltinSignatureDescriptor {
        label: "[C, ia] = union(A, B, option...)",
        inputs: &UNION_INPUTS_A_B_OPTIONS,
        outputs: &UNION_OUTPUT_C_IA,
    },
    BuiltinSignatureDescriptor {
        label: "[C, ia, ib] = union(A, B)",
        inputs: &UNION_INPUTS_A_B,
        outputs: &UNION_OUTPUT_C_IA_IB,
    },
    BuiltinSignatureDescriptor {
        label: "[C, ia, ib] = union(A, B, option...)",
        inputs: &UNION_INPUTS_A_B_OPTIONS,
        outputs: &UNION_OUTPUT_C_IA_IB,
    },
];

const UNION_ERROR_LEGACY_OPTION_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNION.LEGACY_OPTION_UNSUPPORTED",
    identifier: Some("RunMat:union:LegacyOptionUnsupported"),
    when: "Legacy compatibility options are requested.",
    message: "union: the 'legacy' behaviour is not supported",
};

const UNION_ERROR_CONFLICTING_ORDER_OPTIONS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNION.CONFLICTING_ORDER_OPTIONS",
    identifier: Some("RunMat:union:ConflictingOrderOptions"),
    when: "Both 'sorted' and 'stable' options are provided.",
    message: "union: cannot combine 'sorted' with 'stable'",
};

const UNION_ERROR_UNKNOWN_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNION.UNKNOWN_OPTION",
    identifier: Some("RunMat:union:UnknownOption"),
    when: "An unsupported option token is provided.",
    message: "union: unrecognised option",
};

const UNION_ERROR_ROWS_COLUMN_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNION.ROWS_COLUMN_MISMATCH",
    identifier: Some("RunMat:union:RowsColumnMismatch"),
    when: "'rows' mode is used and column counts differ.",
    message: "union: inputs must have the same number of columns when using 'rows'",
};

const UNION_ERROR_UNSUPPORTED_INPUT_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNION.UNSUPPORTED_INPUT_TYPE",
    identifier: Some("RunMat:union:UnsupportedInputType"),
    when: "Input values cannot be converted into supported union domains.",
    message: "union: unsupported input type",
};

const UNION_ERROR_NUMERIC_CLASS_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNION.NUMERIC_CLASS_MISMATCH",
    identifier: Some("RunMat:union:NumericClassMismatch"),
    when: "Numeric inputs have incompatible nondouble classes.",
    message: "union: numeric inputs must have the same class, except double may be combined with one nondouble class",
};

const UNION_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNION.INVALID_ARGUMENT",
    identifier: Some("RunMat:union:InvalidArgument"),
    when: "Option arguments are not string-like where required.",
    message: "union: expected string option arguments",
};

const UNION_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNION.INTERNAL",
    identifier: Some("RunMat:union:Internal"),
    when: "Internal conversion/allocation/provider decode fails.",
    message: "union: internal operation failed",
};

const UNION_ERRORS: [BuiltinErrorDescriptor; 8] = [
    UNION_ERROR_LEGACY_OPTION_UNSUPPORTED,
    UNION_ERROR_CONFLICTING_ORDER_OPTIONS,
    UNION_ERROR_UNKNOWN_OPTION,
    UNION_ERROR_ROWS_COLUMN_MISMATCH,
    UNION_ERROR_UNSUPPORTED_INPUT_TYPE,
    UNION_ERROR_NUMERIC_CLASS_MISMATCH,
    UNION_ERROR_INVALID_ARGUMENT,
    UNION_ERROR_INTERNAL,
];

const UNION_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[C, ia, ib] = union(integer_A, integer_B, options)",
        inputs: &super::BINARY_SET_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "C preserves the common nondouble integer class, including when paired with double; ia and ib are one-based double. GPU supports integer classes through 32 bits and restores outputs after typed fallback.",
    }];

pub const UNION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &UNION_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &UNION_ERRORS,
};

fn union_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn union_error(error: &'static BuiltinErrorDescriptor) -> crate::RuntimeError {
    union_error_with(error, error.message)
}

fn union_internal_error(message: impl Into<String>) -> crate::RuntimeError {
    union_error_with(&UNION_ERROR_INTERNAL, message)
}

#[runtime_builtin(
    name = "union",
    category = "array/sorting_sets",
    summary = "Return unions of input arrays with ordering and index-output controls.",
    keywords = "union,set,stable,rows,indices,gpu",
    accel = "array_construct",
    sink = true,
    type_resolver(set_values_output_type),
    descriptor(crate::builtins::array::sorting_sets::union::UNION_DESCRIPTOR),
    integer_capabilities(UNION_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::array::sorting_sets::union"
)]
async fn union_builtin(a: Value, b: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 3) {
        return Err(union_error_with(
            &UNION_ERROR_INVALID_ARGUMENT,
            "union: too many output arguments; maximum is 3",
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
                union_internal_error,
            )?;
            return Ok(Value::OutputList(outputs));
        }
        if out_count == 2 {
            let (values, ia) = eval.into_pair();
            let outputs = super::restore_set_outputs(
                provider,
                BUILTIN_NAME,
                vec![values, ia],
                union_internal_error,
            )?;
            return Ok(Value::OutputList(outputs));
        }
        let (values, ia, ib) = eval.into_triple();
        let outputs = super::restore_set_outputs(
            provider,
            BUILTIN_NAME,
            vec![values, ia, ib],
            union_internal_error,
        )?;
        return Ok(Value::OutputList(outputs));
    }
    let mut outputs = super::restore_set_outputs(
        provider,
        BUILTIN_NAME,
        vec![eval.into_values_value()],
        union_internal_error,
    )?;
    Ok(outputs.pop().expect("union output"))
}

/// Evaluate the `union` builtin once and expose all outputs.
pub async fn evaluate(a: Value, b: Value, rest: &[Value]) -> crate::BuiltinResult<UnionEvaluation> {
    crate::builtins::common::validation::reject_typed_complex_integer(&a, "union")?;
    crate::builtins::common::validation::reject_typed_complex_integer(&b, "union")?;
    let opts = parse_options(rest)?;
    for value in [&a, &b] {
        if let Value::GpuTensor(handle) = value {
            if super::is_unsupported_set_gpu_integer(handle) {
                return Err(union_error_with(
                    &UNION_ERROR_UNSUPPORTED_INPUT_TYPE,
                    "union: resident 64-bit integer inputs are not supported",
                ));
            }
        }
    }
    match (a, b) {
        (Value::GpuTensor(handle_a), Value::GpuTensor(handle_b)) => {
            union_gpu_pair(handle_a, handle_b, &opts).await
        }
        (Value::GpuTensor(handle_a), other) => union_gpu_mixed(handle_a, other, &opts, true).await,
        (other, Value::GpuTensor(handle_b)) => union_gpu_mixed(handle_b, other, &opts, false).await,
        (left, right) => union_host(left, right, &opts),
    }
}

fn parse_options(rest: &[Value]) -> crate::BuiltinResult<UnionOptions> {
    let mut opts = UnionOptions {
        rows: false,
        order: UnionOrder::Sorted,
    };
    let mut seen_order: Option<UnionOrder> = None;

    let tokens = tokens_from_values(rest);
    for (arg, token) in rest.iter().zip(tokens.iter()) {
        let text = match token {
            crate::builtins::common::arg_tokens::ArgToken::String(text) => text.as_str(),
            _ => {
                let text = tensor::value_to_string(arg)
                    .ok_or_else(|| union_error(&UNION_ERROR_INVALID_ARGUMENT))?;
                let lowered = text.trim().to_ascii_lowercase();
                parse_union_option(&mut opts, &mut seen_order, &lowered)?;
                continue;
            }
        };
        parse_union_option(&mut opts, &mut seen_order, text)?;
    }

    Ok(opts)
}

fn parse_union_option(
    opts: &mut UnionOptions,
    seen_order: &mut Option<UnionOrder>,
    lowered: &str,
) -> crate::BuiltinResult<()> {
    match lowered {
        "rows" => opts.rows = true,
        "sorted" => {
            if let Some(prev) = seen_order {
                if *prev != UnionOrder::Sorted {
                    return Err(union_error(&UNION_ERROR_CONFLICTING_ORDER_OPTIONS));
                }
            }
            *seen_order = Some(UnionOrder::Sorted);
            opts.order = UnionOrder::Sorted;
        }
        "stable" => {
            if let Some(prev) = seen_order {
                if *prev != UnionOrder::Stable {
                    return Err(union_error(&UNION_ERROR_CONFLICTING_ORDER_OPTIONS));
                }
            }
            *seen_order = Some(UnionOrder::Stable);
            opts.order = UnionOrder::Stable;
        }
        "legacy" | "r2012a" => {
            return Err(union_error(&UNION_ERROR_LEGACY_OPTION_UNSUPPORTED));
        }
        other => {
            return Err(union_error_with(
                &UNION_ERROR_UNKNOWN_OPTION,
                format!("union: unrecognised option '{other}'"),
            ))
        }
    }
    Ok(())
}

async fn union_gpu_pair(
    handle_a: GpuTensorHandle,
    handle_b: GpuTensorHandle,
    opts: &UnionOptions,
) -> crate::BuiltinResult<UnionEvaluation> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle_a)
        .or_else(runmat_accelerate_api::provider)
    {
        match provider.union(&handle_a, &handle_b, opts).await {
            Ok(result) => return UnionEvaluation::from_union_result(result),
            Err(_) => {
                // Fall back to host gather when provider union is unavailable.
            }
        }
    }
    let tensor_a = gpu_helpers::gather_tensor_async(&handle_a).await?;
    let tensor_b = gpu_helpers::gather_tensor_async(&handle_b).await?;
    union_numeric(tensor_a, tensor_b, opts)
}

async fn union_gpu_mixed(
    handle_gpu: GpuTensorHandle,
    other: Value,
    opts: &UnionOptions,
    gpu_is_a: bool,
) -> crate::BuiltinResult<UnionEvaluation> {
    let tensor_gpu = gpu_helpers::gather_tensor_async(&handle_gpu).await?;
    let tensor_other =
        tensor::value_into_tensor_for("union", other).map_err(|e| union_internal_error(e))?;
    if gpu_is_a {
        union_numeric(tensor_gpu, tensor_other, opts)
    } else {
        union_numeric(tensor_other, tensor_gpu, opts)
    }
}

fn union_host(a: Value, b: Value, opts: &UnionOptions) -> crate::BuiltinResult<UnionEvaluation> {
    match (a, b) {
        // Complex cases
        (Value::ComplexTensor(at), Value::ComplexTensor(bt)) => union_complex(at, bt, opts),
        (Value::ComplexTensor(at), Value::Complex(re, im)) => {
            let bt = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| union_internal_error(format!("union: {e}")))?;
            union_complex(at, bt, opts)
        }
        (Value::Complex(re, im), Value::ComplexTensor(bt)) => {
            let at = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| union_internal_error(format!("union: {e}")))?;
            union_complex(at, bt, opts)
        }
        (Value::Complex(a_re, a_im), Value::Complex(b_re, b_im)) => {
            let at = ComplexTensor::new(vec![(a_re, a_im)], vec![1, 1])
                .map_err(|e| union_internal_error(format!("union: {e}")))?;
            let bt = ComplexTensor::new(vec![(b_re, b_im)], vec![1, 1])
                .map_err(|e| union_internal_error(format!("union: {e}")))?;
            union_complex(at, bt, opts)
        }

        // Character arrays
        (Value::CharArray(ac), Value::CharArray(bc)) => union_char(ac, bc, opts),

        // String arrays / scalars
        (Value::StringArray(astring), Value::StringArray(bstring)) => {
            union_string(astring, bstring, opts)
        }
        (Value::StringArray(astring), Value::String(b)) => {
            let bstring = StringArray::new(vec![b], vec![1, 1])
                .map_err(|e| union_internal_error(format!("union: {e}")))?;
            union_string(astring, bstring, opts)
        }
        (Value::String(a), Value::StringArray(bstring)) => {
            let astring = StringArray::new(vec![a], vec![1, 1])
                .map_err(|e| union_internal_error(format!("union: {e}")))?;
            union_string(astring, bstring, opts)
        }
        (Value::String(a), Value::String(b)) => {
            let astring = StringArray::new(vec![a], vec![1, 1])
                .map_err(|e| union_internal_error(format!("union: {e}")))?;
            let bstring = StringArray::new(vec![b], vec![1, 1])
                .map_err(|e| union_internal_error(format!("union: {e}")))?;
            union_string(astring, bstring, opts)
        }

        // Fallback to numeric (includes tensors, logical arrays, ints, bools, doubles)
        (left, right) => {
            let tensor_a = tensor::value_into_tensor_for("union", left)
                .map_err(|e| union_error_with(&UNION_ERROR_UNSUPPORTED_INPUT_TYPE, e))?;
            let tensor_b = tensor::value_into_tensor_for("union", right)
                .map_err(|e| union_error_with(&UNION_ERROR_UNSUPPORTED_INPUT_TYPE, e))?;
            union_numeric(tensor_a, tensor_b, opts)
        }
    }
}

fn union_numeric(
    a: Tensor,
    b: Tensor,
    opts: &UnionOptions,
) -> crate::BuiltinResult<UnionEvaluation> {
    let a_dtype = a.numeric_dtype();
    let b_dtype = b.numeric_dtype();
    if let (Some(a_storage), Some(b_storage)) = (a.integer_storage(), b.integer_storage()) {
        if a_storage.class_name() == b_storage.class_name() {
            return if opts.rows {
                union_integer_rows(a_storage, a.shape.clone(), b_storage, b.shape.clone(), opts)
            } else {
                union_integer_elements(a_storage, b_storage, opts)
            };
        }
        return Err(union_error(&UNION_ERROR_NUMERIC_CLASS_MISMATCH));
    }
    match (a.integer_storage(), b.integer_storage()) {
        (Some(storage), None) if b_dtype == NumericDType::F64 => {
            let target = IntegerTarget::from_storage(storage);
            let b = target.cast_tensor(b).map_err(union_internal_error)?;
            return union_numeric(a, b, opts);
        }
        (None, Some(storage)) if a_dtype == NumericDType::F64 => {
            let target = IntegerTarget::from_storage(storage);
            let a = target.cast_tensor(a).map_err(union_internal_error)?;
            return union_numeric(a, b, opts);
        }
        _ => {}
    }
    if a_dtype != b_dtype && a_dtype != NumericDType::F64 && b_dtype != NumericDType::F64 {
        return Err(union_error(&UNION_ERROR_NUMERIC_CLASS_MISMATCH));
    }
    let a_shape = a.shape.clone();
    let b_shape = b.shape.clone();
    let a_storage = a
        .into_numeric_storage()
        .map_err(|e| union_internal_error(format!("union: {e}")))?;
    let b_storage = b
        .into_numeric_storage()
        .map_err(|e| union_internal_error(format!("union: {e}")))?;
    match (a_storage, b_storage) {
        (NumericStorage::F64(a), NumericStorage::F64(b)) => {
            union_floating(a, a_shape, b, b_shape, opts)
        }
        (NumericStorage::F32(a), NumericStorage::F32(b)) => {
            union_floating(a, a_shape, b, b_shape, opts)
        }
        (a, b) => union_promoted_f64(a, a_shape, b, b_shape, opts),
    }
}

fn union_promoted_f64(
    a: NumericStorage,
    a_shape: Vec<usize>,
    b: NumericStorage,
    b_shape: Vec<usize>,
    opts: &UnionOptions,
) -> crate::BuiltinResult<UnionEvaluation> {
    union_floating(
        a.materialize_f64(),
        a_shape,
        b.materialize_f64(),
        b_shape,
        opts,
    )
}

fn union_floating<T: SetFloat>(
    a: Vec<T>,
    a_shape: Vec<usize>,
    b: Vec<T>,
    b_shape: Vec<usize>,
    opts: &UnionOptions,
) -> crate::BuiltinResult<UnionEvaluation> {
    if opts.rows {
        union_floating_rows(a, a_shape, b, b_shape, opts)
    } else {
        union_floating_elements(a, b, opts)
    }
}

/// Helper exposed for acceleration providers handling numeric tensors entirely on the host.
pub fn union_numeric_from_tensors(
    a: Tensor,
    b: Tensor,
    opts: &UnionOptions,
) -> crate::BuiltinResult<UnionEvaluation> {
    union_numeric(a, b, opts)
}

fn union_integer_elements(
    a: &IntegerStorage,
    b: &IntegerStorage,
    opts: &UnionOptions,
) -> crate::BuiltinResult<UnionEvaluation> {
    let mut entries = Vec::<IntegerUnionEntry>::new();
    let mut map = HashMap::<IntValue, usize>::new();
    for (index, value) in a.exact_values().into_iter().enumerate() {
        if map.contains_key(&value) {
            continue;
        }
        let entry_index = entries.len();
        entries.push(IntegerUnionEntry {
            value: value.clone(),
            a_index: Some(index),
            b_index: None,
            order_rank: entry_index,
        });
        map.insert(value, entry_index);
    }
    for (index, value) in b.exact_values().into_iter().enumerate() {
        if map.contains_key(&value) {
            continue;
        }
        let entry_index = entries.len();
        entries.push(IntegerUnionEntry {
            value: value.clone(),
            a_index: None,
            b_index: Some(index),
            order_rank: entry_index,
        });
        map.insert(value, entry_index);
    }
    assemble_integer_union(entries, a, opts)
}

fn union_integer_rows(
    a_storage: &IntegerStorage,
    a_shape: Vec<usize>,
    b_storage: &IntegerStorage,
    b_shape: Vec<usize>,
    opts: &UnionOptions,
) -> crate::BuiltinResult<UnionEvaluation> {
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(union_internal_error(
            "union: 'rows' option requires 2-D numeric matrices",
        ));
    }
    if a_shape[1] != b_shape[1] {
        return Err(union_error(&UNION_ERROR_ROWS_COLUMN_MISMATCH));
    }
    let (rows_a, rows_b, cols) = (a_shape[0], b_shape[0], a_shape[1]);
    let a_values = a_storage.exact_values();
    let b_values = b_storage.exact_values();
    let mut entries = Vec::<IntegerRowUnionEntry>::new();
    let mut map = HashMap::<Vec<IntValue>, usize>::new();
    for row in 0..rows_a {
        let row_data: Vec<_> = (0..cols)
            .map(|col| a_values[row + col * rows_a].clone())
            .collect();
        if map.contains_key(&row_data) {
            continue;
        }
        let entry_index = entries.len();
        entries.push(IntegerRowUnionEntry {
            row_data: row_data.clone(),
            a_row: Some(row),
            b_row: None,
            order_rank: entry_index,
        });
        map.insert(row_data, entry_index);
    }
    for row in 0..rows_b {
        let row_data: Vec<_> = (0..cols)
            .map(|col| b_values[row + col * rows_b].clone())
            .collect();
        if map.contains_key(&row_data) {
            continue;
        }
        let entry_index = entries.len();
        entries.push(IntegerRowUnionEntry {
            row_data: row_data.clone(),
            a_row: None,
            b_row: Some(row),
            order_rank: entry_index,
        });
        map.insert(row_data, entry_index);
    }
    assemble_integer_row_union(entries, a_storage, opts, cols)
}

fn union_floating_elements<T: SetFloat>(
    a_values: Vec<T>,
    b_values: Vec<T>,
    opts: &UnionOptions,
) -> crate::BuiltinResult<UnionEvaluation> {
    let mut entries = Vec::<FloatingUnionEntry<T>>::new();
    let mut map: HashMap<u64, usize> = HashMap::new();
    let mut order_counter = 0usize;

    for (idx, &value) in a_values.iter().enumerate() {
        let key = value.canonical_key();
        match map.entry(key) {
            Entry::Occupied(_) => {
                // Already recorded from A; keep first occurrence only.
            }
            Entry::Vacant(v) => {
                let entry_idx = entries.len();
                entries.push(FloatingUnionEntry {
                    value,
                    a_index: Some(idx),
                    b_index: None,
                    order_rank: order_counter,
                });
                v.insert(entry_idx);
                order_counter += 1;
            }
        }
    }

    for (idx, &value) in b_values.iter().enumerate() {
        let key = value.canonical_key();
        match map.entry(key) {
            Entry::Occupied(occ) => {
                let entry = &mut entries[*occ.get()];
                if entry.a_index.is_none() && entry.b_index.is_none() {
                    entry.b_index = Some(idx);
                }
            }
            Entry::Vacant(v) => {
                let entry_idx = entries.len();
                entries.push(FloatingUnionEntry {
                    value,
                    a_index: None,
                    b_index: Some(idx),
                    order_rank: order_counter,
                });
                v.insert(entry_idx);
                order_counter += 1;
            }
        }
    }

    assemble_floating_union(entries, opts)
}

fn union_floating_rows<T: SetFloat>(
    a_values: Vec<T>,
    a_shape: Vec<usize>,
    b_values: Vec<T>,
    b_shape: Vec<usize>,
    opts: &UnionOptions,
) -> crate::BuiltinResult<UnionEvaluation> {
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(union_internal_error(
            "union: 'rows' option requires 2-D numeric matrices",
        ));
    }
    if a_shape[1] != b_shape[1] {
        return Err(union_error_with(
            &UNION_ERROR_ROWS_COLUMN_MISMATCH,
            UNION_ERROR_ROWS_COLUMN_MISMATCH.message,
        ));
    }
    let rows_a = a_shape[0];
    let cols = a_shape[1];
    let rows_b = b_shape[0];

    let mut entries = Vec::<FloatingRowUnionEntry<T>>::new();
    let mut map: HashMap<FloatingRowKey, usize> = HashMap::new();
    let mut order_counter = 0usize;

    for r in 0..rows_a {
        let mut row_values = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r + c * rows_a;
            row_values.push(a_values[idx]);
        }
        let key = FloatingRowKey::from_slice(&row_values);
        match map.entry(key) {
            Entry::Occupied(_) => {}
            Entry::Vacant(v) => {
                let entry_idx = entries.len();
                entries.push(FloatingRowUnionEntry {
                    row_data: row_values,
                    a_row: Some(r),
                    b_row: None,
                    order_rank: order_counter,
                });
                v.insert(entry_idx);
                order_counter += 1;
            }
        }
    }

    for r in 0..rows_b {
        let mut row_values = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r + c * rows_b;
            row_values.push(b_values[idx]);
        }
        let key = FloatingRowKey::from_slice(&row_values);
        match map.entry(key) {
            Entry::Occupied(occ) => {
                let entry = &mut entries[*occ.get()];
                if entry.a_row.is_none() && entry.b_row.is_none() {
                    entry.b_row = Some(r);
                }
            }
            Entry::Vacant(v) => {
                let entry_idx = entries.len();
                entries.push(FloatingRowUnionEntry {
                    row_data: row_values,
                    a_row: None,
                    b_row: Some(r),
                    order_rank: order_counter,
                });
                v.insert(entry_idx);
                order_counter += 1;
            }
        }
    }

    assemble_floating_row_union(entries, opts, cols)
}

fn union_complex(
    a: ComplexTensor,
    b: ComplexTensor,
    opts: &UnionOptions,
) -> crate::BuiltinResult<UnionEvaluation> {
    let a_shape = a.shape.clone();
    let b_shape = b.shape.clone();
    match (a.into_complex_storage(), b.into_complex_storage()) {
        (ComplexStorage::F64(a), ComplexStorage::F64(b)) => {
            union_floating_complex(a, a_shape, b, b_shape, opts)
        }
        (ComplexStorage::F32(a), ComplexStorage::F32(b)) => {
            union_floating_complex(a, a_shape, b, b_shape, opts)
        }
        (a, b) => union_promoted_complex_f64(a, a_shape, b, b_shape, opts),
    }
}

fn union_promoted_complex_f64(
    a: ComplexStorage,
    a_shape: Vec<usize>,
    b: ComplexStorage,
    b_shape: Vec<usize>,
    opts: &UnionOptions,
) -> crate::BuiltinResult<UnionEvaluation> {
    union_floating_complex(
        a.materialize_f64(),
        a_shape,
        b.materialize_f64(),
        b_shape,
        opts,
    )
}

fn union_floating_complex<T: SetFloat>(
    a: Vec<(T, T)>,
    a_shape: Vec<usize>,
    b: Vec<(T, T)>,
    b_shape: Vec<usize>,
    opts: &UnionOptions,
) -> crate::BuiltinResult<UnionEvaluation> {
    if opts.rows {
        union_complex_rows(a, a_shape, b, b_shape, opts)
    } else {
        union_complex_elements(a, b, opts)
    }
}

fn union_complex_elements<T: SetFloat>(
    a: Vec<(T, T)>,
    b: Vec<(T, T)>,
    opts: &UnionOptions,
) -> crate::BuiltinResult<UnionEvaluation> {
    let mut entries = Vec::<ComplexUnionEntry<T>>::new();
    let mut map: HashMap<ComplexKey, usize> = HashMap::new();
    let mut order_counter = 0usize;

    for (idx, &value) in a.iter().enumerate() {
        let key = ComplexKey::new(value);
        match map.entry(key) {
            Entry::Occupied(_) => {}
            Entry::Vacant(v) => {
                let entry_idx = entries.len();
                entries.push(ComplexUnionEntry {
                    value,
                    a_index: Some(idx),
                    b_index: None,
                    order_rank: order_counter,
                });
                v.insert(entry_idx);
                order_counter += 1;
            }
        }
    }

    for (idx, &value) in b.iter().enumerate() {
        let key = ComplexKey::new(value);
        match map.entry(key) {
            Entry::Occupied(occ) => {
                let entry = &mut entries[*occ.get()];
                if entry.a_index.is_none() && entry.b_index.is_none() {
                    entry.b_index = Some(idx);
                }
            }
            Entry::Vacant(v) => {
                let entry_idx = entries.len();
                entries.push(ComplexUnionEntry {
                    value,
                    a_index: None,
                    b_index: Some(idx),
                    order_rank: order_counter,
                });
                v.insert(entry_idx);
                order_counter += 1;
            }
        }
    }

    assemble_complex_union(entries, opts)
}

fn union_complex_rows<T: SetFloat>(
    a: Vec<(T, T)>,
    a_shape: Vec<usize>,
    b: Vec<(T, T)>,
    b_shape: Vec<usize>,
    opts: &UnionOptions,
) -> crate::BuiltinResult<UnionEvaluation> {
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(union_internal_error(
            "union: 'rows' option requires 2-D complex matrices",
        ));
    }
    if a_shape[1] != b_shape[1] {
        return Err(union_error_with(
            &UNION_ERROR_ROWS_COLUMN_MISMATCH,
            UNION_ERROR_ROWS_COLUMN_MISMATCH.message,
        ));
    }
    let rows_a = a_shape[0];
    let cols = a_shape[1];
    let rows_b = b_shape[0];

    let mut entries = Vec::<ComplexRowUnionEntry<T>>::new();
    let mut map: HashMap<Vec<ComplexKey>, usize> = HashMap::new();
    let mut order_counter = 0usize;

    for r in 0..rows_a {
        let mut row_values = Vec::with_capacity(cols);
        let mut key_row = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r + c * rows_a;
            let value = a[idx];
            row_values.push(value);
            key_row.push(ComplexKey::new(value));
        }
        match map.entry(key_row) {
            Entry::Occupied(_) => {}
            Entry::Vacant(v) => {
                let entry_idx = entries.len();
                entries.push(ComplexRowUnionEntry {
                    row_data: row_values,
                    a_row: Some(r),
                    b_row: None,
                    order_rank: order_counter,
                });
                v.insert(entry_idx);
                order_counter += 1;
            }
        }
    }

    for r in 0..rows_b {
        let mut row_values = Vec::with_capacity(cols);
        let mut key_row = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r + c * rows_b;
            let value = b[idx];
            row_values.push(value);
            key_row.push(ComplexKey::new(value));
        }
        match map.entry(key_row) {
            Entry::Occupied(occ) => {
                let entry = &mut entries[*occ.get()];
                if entry.a_row.is_none() && entry.b_row.is_none() {
                    entry.b_row = Some(r);
                }
            }
            Entry::Vacant(v) => {
                let entry_idx = entries.len();
                entries.push(ComplexRowUnionEntry {
                    row_data: row_values,
                    a_row: None,
                    b_row: Some(r),
                    order_rank: order_counter,
                });
                v.insert(entry_idx);
                order_counter += 1;
            }
        }
    }

    assemble_complex_row_union(entries, opts, cols)
}

fn union_char(
    a: CharArray,
    b: CharArray,
    opts: &UnionOptions,
) -> crate::BuiltinResult<UnionEvaluation> {
    if opts.rows {
        union_char_rows(a, b, opts)
    } else {
        union_char_elements(a, b, opts)
    }
}

fn union_char_elements(
    a: CharArray,
    b: CharArray,
    opts: &UnionOptions,
) -> crate::BuiltinResult<UnionEvaluation> {
    let mut entries = Vec::<CharUnionEntry>::new();
    let mut map: HashMap<u32, usize> = HashMap::new();
    let mut order_counter = 0usize;

    for col in 0..a.cols {
        for row in 0..a.rows {
            let linear_idx = row + col * a.rows;
            let data_idx = row * a.cols + col;
            let ch = a.data[data_idx];
            let key = ch as u32;
            match map.entry(key) {
                Entry::Occupied(_) => {}
                Entry::Vacant(v) => {
                    let entry_idx = entries.len();
                    entries.push(CharUnionEntry {
                        ch,
                        a_index: Some(linear_idx),
                        b_index: None,
                        order_rank: order_counter,
                    });
                    v.insert(entry_idx);
                    order_counter += 1;
                }
            }
        }
    }

    for col in 0..b.cols {
        for row in 0..b.rows {
            let linear_idx = row + col * b.rows;
            let data_idx = row * b.cols + col;
            let ch = b.data[data_idx];
            let key = ch as u32;
            match map.entry(key) {
                Entry::Occupied(occ) => {
                    let entry = &mut entries[*occ.get()];
                    if entry.a_index.is_none() && entry.b_index.is_none() {
                        entry.b_index = Some(linear_idx);
                    }
                }
                Entry::Vacant(v) => {
                    let entry_idx = entries.len();
                    entries.push(CharUnionEntry {
                        ch,
                        a_index: None,
                        b_index: Some(linear_idx),
                        order_rank: order_counter,
                    });
                    v.insert(entry_idx);
                    order_counter += 1;
                }
            }
        }
    }

    assemble_char_union(entries, opts)
}

fn union_char_rows(
    a: CharArray,
    b: CharArray,
    opts: &UnionOptions,
) -> crate::BuiltinResult<UnionEvaluation> {
    if a.cols != b.cols {
        return Err(union_error_with(
            &UNION_ERROR_ROWS_COLUMN_MISMATCH,
            UNION_ERROR_ROWS_COLUMN_MISMATCH.message,
        ));
    }
    let rows_a = a.rows;
    let rows_b = b.rows;
    let cols = a.cols;

    let mut entries = Vec::<CharRowUnionEntry>::new();
    let mut map: HashMap<RowCharKey, usize> = HashMap::new();
    let mut order_counter = 0usize;

    for r in 0..rows_a {
        let mut row_values = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r * cols + c;
            row_values.push(a.data[idx]);
        }
        let key = RowCharKey::from_slice(&row_values);
        match map.entry(key) {
            Entry::Occupied(_) => {}
            Entry::Vacant(v) => {
                let entry_idx = entries.len();
                entries.push(CharRowUnionEntry {
                    row_data: row_values,
                    a_row: Some(r),
                    b_row: None,
                    order_rank: order_counter,
                });
                v.insert(entry_idx);
                order_counter += 1;
            }
        }
    }

    for r in 0..rows_b {
        let mut row_values = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r * cols + c;
            row_values.push(b.data[idx]);
        }
        let key = RowCharKey::from_slice(&row_values);
        match map.entry(key) {
            Entry::Occupied(occ) => {
                let entry = &mut entries[*occ.get()];
                if entry.a_row.is_none() && entry.b_row.is_none() {
                    entry.b_row = Some(r);
                }
            }
            Entry::Vacant(v) => {
                let entry_idx = entries.len();
                entries.push(CharRowUnionEntry {
                    row_data: row_values,
                    a_row: None,
                    b_row: Some(r),
                    order_rank: order_counter,
                });
                v.insert(entry_idx);
                order_counter += 1;
            }
        }
    }

    assemble_char_row_union(entries, opts, cols)
}

fn union_string(
    a: StringArray,
    b: StringArray,
    opts: &UnionOptions,
) -> crate::BuiltinResult<UnionEvaluation> {
    if opts.rows {
        union_string_rows(a, b, opts)
    } else {
        union_string_elements(a, b, opts)
    }
}

fn union_string_elements(
    a: StringArray,
    b: StringArray,
    opts: &UnionOptions,
) -> crate::BuiltinResult<UnionEvaluation> {
    let mut entries = Vec::<StringUnionEntry>::new();
    let mut map: HashMap<String, usize> = HashMap::new();
    let mut order_counter = 0usize;

    for (idx, value) in a.data.iter().enumerate() {
        match map.entry(value.clone()) {
            Entry::Occupied(_) => {}
            Entry::Vacant(v) => {
                let entry_idx = entries.len();
                entries.push(StringUnionEntry {
                    value: value.clone(),
                    a_index: Some(idx),
                    b_index: None,
                    order_rank: order_counter,
                });
                v.insert(entry_idx);
                order_counter += 1;
            }
        }
    }

    for (idx, value) in b.data.iter().enumerate() {
        match map.entry(value.clone()) {
            Entry::Occupied(occ) => {
                let entry = &mut entries[*occ.get()];
                if entry.a_index.is_none() && entry.b_index.is_none() {
                    entry.b_index = Some(idx);
                }
            }
            Entry::Vacant(v) => {
                let entry_idx = entries.len();
                entries.push(StringUnionEntry {
                    value: value.clone(),
                    a_index: None,
                    b_index: Some(idx),
                    order_rank: order_counter,
                });
                v.insert(entry_idx);
                order_counter += 1;
            }
        }
    }

    assemble_string_union(entries, opts)
}

fn union_string_rows(
    a: StringArray,
    b: StringArray,
    opts: &UnionOptions,
) -> crate::BuiltinResult<UnionEvaluation> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(union_internal_error(
            "union: 'rows' option requires 2-D string arrays",
        ));
    }
    if a.shape[1] != b.shape[1] {
        return Err(union_error_with(
            &UNION_ERROR_ROWS_COLUMN_MISMATCH,
            UNION_ERROR_ROWS_COLUMN_MISMATCH.message,
        ));
    }
    let rows_a = a.shape[0];
    let cols = a.shape[1];
    let rows_b = b.shape[0];

    let mut entries = Vec::<StringRowUnionEntry>::new();
    let mut map: HashMap<RowStringKey, usize> = HashMap::new();
    let mut order_counter = 0usize;

    for r in 0..rows_a {
        let mut row_values = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r + c * rows_a;
            row_values.push(a.data[idx].clone());
        }
        let key = RowStringKey(row_values.clone());
        match map.entry(key) {
            Entry::Occupied(_) => {}
            Entry::Vacant(v) => {
                let entry_idx = entries.len();
                entries.push(StringRowUnionEntry {
                    row_data: row_values,
                    a_row: Some(r),
                    b_row: None,
                    order_rank: order_counter,
                });
                v.insert(entry_idx);
                order_counter += 1;
            }
        }
    }

    for r in 0..rows_b {
        let mut row_values = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r + c * rows_b;
            row_values.push(b.data[idx].clone());
        }
        let key = RowStringKey(row_values.clone());
        match map.entry(key) {
            Entry::Occupied(occ) => {
                let entry = &mut entries[*occ.get()];
                if entry.a_row.is_none() && entry.b_row.is_none() {
                    entry.b_row = Some(r);
                }
            }
            Entry::Vacant(v) => {
                let entry_idx = entries.len();
                entries.push(StringRowUnionEntry {
                    row_data: row_values,
                    a_row: None,
                    b_row: Some(r),
                    order_rank: order_counter,
                });
                v.insert(entry_idx);
                order_counter += 1;
            }
        }
    }

    assemble_string_row_union(entries, opts, cols)
}

#[derive(Debug, Clone)]
pub struct UnionEvaluation {
    values: Value,
    ia: Tensor,
    ib: Tensor,
}

impl UnionEvaluation {
    fn new(values: Value, ia: Tensor, ib: Tensor) -> Self {
        Self { values, ia, ib }
    }

    pub fn from_union_result(result: UnionResult) -> crate::BuiltinResult<Self> {
        let UnionResult { values, ia, ib } = result;
        let values_tensor = Tensor::new(values.data, values.shape)
            .map_err(|e| union_internal_error(format!("union: {e}")))?;
        let ia_tensor = Tensor::new(ia.data, ia.shape)
            .map_err(|e| union_internal_error(format!("union: {e}")))?;
        let ib_tensor = Tensor::new(ib.data, ib.shape)
            .map_err(|e| union_internal_error(format!("union: {e}")))?;
        Ok(UnionEvaluation::new(
            tensor::tensor_into_value(values_tensor),
            ia_tensor,
            ib_tensor,
        ))
    }

    pub fn into_numeric_union_result(self) -> crate::BuiltinResult<UnionResult> {
        let UnionEvaluation { values, ia, ib } = self;
        let values_tensor =
            tensor::value_into_tensor_for("union", values).map_err(|e| union_internal_error(e))?;
        Ok(UnionResult {
            values: tensor::tensor_into_host_f64_owned(values_tensor),
            ia: tensor::tensor_into_host_f64_owned(ia),
            ib: tensor::tensor_into_host_f64_owned(ib),
        })
    }

    pub fn into_values_value(self) -> Value {
        self.values
    }

    pub fn into_pair(self) -> (Value, Value) {
        let ia = tensor::tensor_into_value(self.ia);
        (self.values, ia)
    }

    pub fn into_triple(self) -> (Value, Value, Value) {
        let ia = tensor::tensor_into_value(self.ia);
        let ib = tensor::tensor_into_value(self.ib);
        (self.values, ia, ib)
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

#[derive(Debug)]
struct FloatingUnionEntry<T> {
    value: T,
    a_index: Option<usize>,
    b_index: Option<usize>,
    order_rank: usize,
}

#[derive(Debug)]
struct IntegerUnionEntry {
    value: IntValue,
    a_index: Option<usize>,
    b_index: Option<usize>,
    order_rank: usize,
}

#[derive(Debug)]
struct FloatingRowUnionEntry<T> {
    row_data: Vec<T>,
    a_row: Option<usize>,
    b_row: Option<usize>,
    order_rank: usize,
}

#[derive(Debug)]
struct IntegerRowUnionEntry {
    row_data: Vec<IntValue>,
    a_row: Option<usize>,
    b_row: Option<usize>,
    order_rank: usize,
}

#[derive(Debug)]
struct ComplexUnionEntry<T> {
    value: (T, T),
    a_index: Option<usize>,
    b_index: Option<usize>,
    order_rank: usize,
}

#[derive(Debug)]
struct ComplexRowUnionEntry<T> {
    row_data: Vec<(T, T)>,
    a_row: Option<usize>,
    b_row: Option<usize>,
    order_rank: usize,
}

#[derive(Debug)]
struct CharUnionEntry {
    ch: char,
    a_index: Option<usize>,
    b_index: Option<usize>,
    order_rank: usize,
}

#[derive(Debug)]
struct CharRowUnionEntry {
    row_data: Vec<char>,
    a_row: Option<usize>,
    b_row: Option<usize>,
    order_rank: usize,
}

#[derive(Debug)]
struct StringUnionEntry {
    value: String,
    a_index: Option<usize>,
    b_index: Option<usize>,
    order_rank: usize,
}

#[derive(Debug)]
struct StringRowUnionEntry {
    row_data: Vec<String>,
    a_row: Option<usize>,
    b_row: Option<usize>,
    order_rank: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FloatingRowKey(Vec<u64>);

impl FloatingRowKey {
    fn from_slice<T: SetFloat>(values: &[T]) -> Self {
        Self(values.iter().map(|&value| value.canonical_key()).collect())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ComplexKey {
    re: u64,
    im: u64,
}

impl ComplexKey {
    fn new<T: SetFloat>(value: (T, T)) -> Self {
        Self {
            re: value.0.canonical_key(),
            im: value.1.canonical_key(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RowCharKey(Vec<u32>);

impl RowCharKey {
    fn from_slice(values: &[char]) -> Self {
        RowCharKey(values.iter().map(|&ch| ch as u32).collect())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RowStringKey(Vec<String>);

fn assemble_floating_union<T: SetFloat>(
    entries: Vec<FloatingUnionEntry<T>>,
    opts: &UnionOptions,
) -> crate::BuiltinResult<UnionEvaluation> {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    match opts.order {
        UnionOrder::Sorted => {
            order.sort_by(|&lhs, &rhs| entries[lhs].value.compare(entries[rhs].value));
        }
        UnionOrder::Stable => {
            order.sort_by_key(|&idx| entries[idx].order_rank);
        }
    }

    let mut values = Vec::with_capacity(order.len());
    let mut ia = Vec::new();
    let mut ib = Vec::new();
    for &idx in &order {
        let entry = &entries[idx];
        values.push(entry.value);
        if let Some(a_idx) = entry.a_index {
            ia.push((a_idx + 1) as f64);
        } else if let Some(b_idx) = entry.b_index {
            ib.push((b_idx + 1) as f64);
        }
    }

    let value_tensor =
        Tensor::from_numeric_storage(T::numeric_storage(values), vec![order.len(), 1])
            .map_err(|e| union_internal_error(format!("union: {e}")))?;
    let ia_len = ia.len();
    let ib_len = ib.len();
    let ia_tensor = Tensor::new(ia, vec![ia_len, 1])
        .map_err(|e| union_internal_error(format!("union: {e}")))?;
    let ib_tensor = Tensor::new(ib, vec![ib_len, 1])
        .map_err(|e| union_internal_error(format!("union: {e}")))?;

    Ok(UnionEvaluation::new(
        tensor::tensor_into_value(value_tensor),
        ia_tensor,
        ib_tensor,
    ))
}

fn assemble_integer_union(
    entries: Vec<IntegerUnionEntry>,
    storage: &IntegerStorage,
    opts: &UnionOptions,
) -> crate::BuiltinResult<UnionEvaluation> {
    let mut order: Vec<_> = (0..entries.len()).collect();
    match opts.order {
        UnionOrder::Sorted => order.sort_by(|&a, &b| {
            integer_order::compare(&entries[a].value, &entries[b].value, false, false)
        }),
        UnionOrder::Stable => order.sort_by_key(|&index| entries[index].order_rank),
    }
    let values: Vec<_> = order
        .iter()
        .map(|&index| entries[index].value.clone())
        .collect();
    let mut ia = Vec::new();
    let mut ib = Vec::new();
    for &index in &order {
        let entry = &entries[index];
        if let Some(a_index) = entry.a_index {
            ia.push((a_index + 1) as f64);
        } else if let Some(b_index) = entry.b_index {
            ib.push((b_index + 1) as f64);
        }
    }
    let values = Tensor::new_integer(
        storage
            .from_exact_values_like(values)
            .map_err(|e| union_internal_error(format!("union: {e}")))?,
        vec![order.len(), 1],
    )
    .map_err(|e| union_internal_error(format!("union: {e}")))?;
    let ia_len = ia.len();
    let ib_len = ib.len();
    let ia = Tensor::new(ia, vec![ia_len, 1])
        .map_err(|e| union_internal_error(format!("union: {e}")))?;
    let ib = Tensor::new(ib, vec![ib_len, 1])
        .map_err(|e| union_internal_error(format!("union: {e}")))?;
    Ok(UnionEvaluation::new(Value::Tensor(values), ia, ib))
}

fn assemble_floating_row_union<T: SetFloat>(
    entries: Vec<FloatingRowUnionEntry<T>>,
    opts: &UnionOptions,
    cols: usize,
) -> crate::BuiltinResult<UnionEvaluation> {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    match opts.order {
        UnionOrder::Sorted => {
            order.sort_by(|&lhs, &rhs| {
                compare_floating_rows(&entries[lhs].row_data, &entries[rhs].row_data)
            });
        }
        UnionOrder::Stable => {
            order.sort_by_key(|&idx| entries[idx].order_rank);
        }
    }

    let unique_rows = order.len();
    let mut values = vec![T::default(); unique_rows * cols];
    let mut ia = Vec::new();
    let mut ib = Vec::new();

    for (row_pos, &entry_idx) in order.iter().enumerate() {
        let entry = &entries[entry_idx];
        for col in 0..cols {
            let dest = row_pos + col * unique_rows;
            values[dest] = entry.row_data[col];
        }
        if let Some(a_row) = entry.a_row {
            ia.push((a_row + 1) as f64);
        } else if let Some(b_row) = entry.b_row {
            ib.push((b_row + 1) as f64);
        }
    }

    let value_tensor =
        Tensor::from_numeric_storage(T::numeric_storage(values), vec![unique_rows, cols])
            .map_err(|e| union_internal_error(format!("union: {e}")))?;
    let ia_len = ia.len();
    let ib_len = ib.len();
    let ia_tensor = Tensor::new(ia, vec![ia_len, 1])
        .map_err(|e| union_internal_error(format!("union: {e}")))?;
    let ib_tensor = Tensor::new(ib, vec![ib_len, 1])
        .map_err(|e| union_internal_error(format!("union: {e}")))?;

    Ok(UnionEvaluation::new(
        tensor::tensor_into_value(value_tensor),
        ia_tensor,
        ib_tensor,
    ))
}

fn assemble_integer_row_union(
    entries: Vec<IntegerRowUnionEntry>,
    storage: &IntegerStorage,
    opts: &UnionOptions,
    cols: usize,
) -> crate::BuiltinResult<UnionEvaluation> {
    let mut order: Vec<_> = (0..entries.len()).collect();
    match opts.order {
        UnionOrder::Sorted => order.sort_by(|&a, &b| {
            for (left, right) in entries[a].row_data.iter().zip(&entries[b].row_data) {
                let ordering = integer_order::compare(left, right, false, false);
                if ordering != Ordering::Equal {
                    return ordering;
                }
            }
            Ordering::Equal
        }),
        UnionOrder::Stable => order.sort_by_key(|&index| entries[index].order_rank),
    }
    let rows = order.len();
    let mut values = Vec::with_capacity(rows * cols);
    for col in 0..cols {
        for &index in &order {
            values.push(entries[index].row_data[col].clone());
        }
    }
    let mut ia = Vec::new();
    let mut ib = Vec::new();
    for &index in &order {
        let entry = &entries[index];
        if let Some(a_row) = entry.a_row {
            ia.push((a_row + 1) as f64);
        } else if let Some(b_row) = entry.b_row {
            ib.push((b_row + 1) as f64);
        }
    }
    let values = Tensor::new_integer(
        storage
            .from_exact_values_like(values)
            .map_err(|e| union_internal_error(format!("union: {e}")))?,
        vec![rows, cols],
    )
    .map_err(|e| union_internal_error(format!("union: {e}")))?;
    let ia_len = ia.len();
    let ib_len = ib.len();
    let ia = Tensor::new(ia, vec![ia_len, 1])
        .map_err(|e| union_internal_error(format!("union: {e}")))?;
    let ib = Tensor::new(ib, vec![ib_len, 1])
        .map_err(|e| union_internal_error(format!("union: {e}")))?;
    Ok(UnionEvaluation::new(Value::Tensor(values), ia, ib))
}

fn assemble_complex_union<T: SetFloat>(
    entries: Vec<ComplexUnionEntry<T>>,
    opts: &UnionOptions,
) -> crate::BuiltinResult<UnionEvaluation> {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    match opts.order {
        UnionOrder::Sorted => {
            order.sort_by(|&lhs, &rhs| compare_complex(entries[lhs].value, entries[rhs].value));
        }
        UnionOrder::Stable => {
            order.sort_by_key(|&idx| entries[idx].order_rank);
        }
    }

    let mut values = Vec::with_capacity(order.len());
    let mut ia = Vec::new();
    let mut ib = Vec::new();
    for &idx in &order {
        let entry = &entries[idx];
        values.push(entry.value);
        if let Some(a_idx) = entry.a_index {
            ia.push((a_idx + 1) as f64);
        } else if let Some(b_idx) = entry.b_index {
            ib.push((b_idx + 1) as f64);
        }
    }

    let value_tensor =
        ComplexTensor::from_complex_storage(T::complex_storage(values), vec![order.len(), 1])
            .map_err(|e| union_internal_error(format!("union: {e}")))?;
    let ia_len = ia.len();
    let ib_len = ib.len();
    let ia_tensor = Tensor::new(ia, vec![ia_len, 1])
        .map_err(|e| union_internal_error(format!("union: {e}")))?;
    let ib_tensor = Tensor::new(ib, vec![ib_len, 1])
        .map_err(|e| union_internal_error(format!("union: {e}")))?;

    let value = if value_tensor.as_f32_slice().is_some() {
        Value::ComplexTensor(value_tensor)
    } else {
        complex_tensor_into_value(value_tensor)
    };
    Ok(UnionEvaluation::new(value, ia_tensor, ib_tensor))
}

fn assemble_complex_row_union<T: SetFloat>(
    entries: Vec<ComplexRowUnionEntry<T>>,
    opts: &UnionOptions,
    cols: usize,
) -> crate::BuiltinResult<UnionEvaluation> {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    match opts.order {
        UnionOrder::Sorted => {
            order.sort_by(|&lhs, &rhs| {
                compare_complex_rows(&entries[lhs].row_data, &entries[rhs].row_data)
            });
        }
        UnionOrder::Stable => {
            order.sort_by_key(|&idx| entries[idx].order_rank);
        }
    }

    let unique_rows = order.len();
    let mut values = vec![(T::default(), T::default()); unique_rows * cols];
    let mut ia = Vec::new();
    let mut ib = Vec::new();

    for (row_pos, &entry_idx) in order.iter().enumerate() {
        let entry = &entries[entry_idx];
        for col in 0..cols {
            let dest = row_pos + col * unique_rows;
            values[dest] = entry.row_data[col];
        }
        if let Some(a_row) = entry.a_row {
            ia.push((a_row + 1) as f64);
        } else if let Some(b_row) = entry.b_row {
            ib.push((b_row + 1) as f64);
        }
    }

    let value_tensor =
        ComplexTensor::from_complex_storage(T::complex_storage(values), vec![unique_rows, cols])
            .map_err(|e| union_internal_error(format!("union: {e}")))?;
    let ia_len = ia.len();
    let ib_len = ib.len();
    let ia_tensor = Tensor::new(ia, vec![ia_len, 1])
        .map_err(|e| union_internal_error(format!("union: {e}")))?;
    let ib_tensor = Tensor::new(ib, vec![ib_len, 1])
        .map_err(|e| union_internal_error(format!("union: {e}")))?;

    let value = if value_tensor.as_f32_slice().is_some() {
        Value::ComplexTensor(value_tensor)
    } else {
        complex_tensor_into_value(value_tensor)
    };
    Ok(UnionEvaluation::new(value, ia_tensor, ib_tensor))
}

fn assemble_char_union(
    entries: Vec<CharUnionEntry>,
    opts: &UnionOptions,
) -> crate::BuiltinResult<UnionEvaluation> {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    match opts.order {
        UnionOrder::Sorted => {
            order.sort_by(|&lhs, &rhs| entries[lhs].ch.cmp(&entries[rhs].ch));
        }
        UnionOrder::Stable => {
            order.sort_by_key(|&idx| entries[idx].order_rank);
        }
    }

    let mut values = Vec::with_capacity(order.len());
    let mut ia = Vec::new();
    let mut ib = Vec::new();
    for &idx in &order {
        let entry = &entries[idx];
        values.push(entry.ch);
        if let Some(a_idx) = entry.a_index {
            ia.push((a_idx + 1) as f64);
        } else if let Some(b_idx) = entry.b_index {
            ib.push((b_idx + 1) as f64);
        }
    }

    let value_array = CharArray::new(values, order.len(), 1)
        .map_err(|e| union_internal_error(format!("union: {e}")))?;
    let ia_len = ia.len();
    let ib_len = ib.len();
    let ia_tensor = Tensor::new(ia, vec![ia_len, 1])
        .map_err(|e| union_internal_error(format!("union: {e}")))?;
    let ib_tensor = Tensor::new(ib, vec![ib_len, 1])
        .map_err(|e| union_internal_error(format!("union: {e}")))?;

    Ok(UnionEvaluation::new(
        Value::CharArray(value_array),
        ia_tensor,
        ib_tensor,
    ))
}

fn assemble_char_row_union(
    entries: Vec<CharRowUnionEntry>,
    opts: &UnionOptions,
    cols: usize,
) -> crate::BuiltinResult<UnionEvaluation> {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    match opts.order {
        UnionOrder::Sorted => {
            order.sort_by(|&lhs, &rhs| {
                compare_char_rows(&entries[lhs].row_data, &entries[rhs].row_data)
            });
        }
        UnionOrder::Stable => {
            order.sort_by_key(|&idx| entries[idx].order_rank);
        }
    }

    let unique_rows = order.len();
    let mut values = vec!['\0'; unique_rows * cols];
    let mut ia = Vec::new();
    let mut ib = Vec::new();

    for (row_pos, &entry_idx) in order.iter().enumerate() {
        let entry = &entries[entry_idx];
        for col in 0..cols {
            let dest = row_pos * cols + col;
            values[dest] = entry.row_data[col];
        }
        if let Some(a_row) = entry.a_row {
            ia.push((a_row + 1) as f64);
        } else if let Some(b_row) = entry.b_row {
            ib.push((b_row + 1) as f64);
        }
    }

    let value_array = CharArray::new(values, unique_rows, cols)
        .map_err(|e| union_internal_error(format!("union: {e}")))?;
    let ia_len = ia.len();
    let ib_len = ib.len();
    let ia_tensor = Tensor::new(ia, vec![ia_len, 1])
        .map_err(|e| union_internal_error(format!("union: {e}")))?;
    let ib_tensor = Tensor::new(ib, vec![ib_len, 1])
        .map_err(|e| union_internal_error(format!("union: {e}")))?;

    Ok(UnionEvaluation::new(
        Value::CharArray(value_array),
        ia_tensor,
        ib_tensor,
    ))
}

fn assemble_string_union(
    entries: Vec<StringUnionEntry>,
    opts: &UnionOptions,
) -> crate::BuiltinResult<UnionEvaluation> {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    match opts.order {
        UnionOrder::Sorted => {
            order.sort_by(|&lhs, &rhs| entries[lhs].value.cmp(&entries[rhs].value));
        }
        UnionOrder::Stable => {
            order.sort_by_key(|&idx| entries[idx].order_rank);
        }
    }

    let mut values = Vec::with_capacity(order.len());
    let mut ia = Vec::new();
    let mut ib = Vec::new();
    for &idx in &order {
        let entry = &entries[idx];
        values.push(entry.value.clone());
        if let Some(a_idx) = entry.a_index {
            ia.push((a_idx + 1) as f64);
        } else if let Some(b_idx) = entry.b_index {
            ib.push((b_idx + 1) as f64);
        }
    }

    let value_array = StringArray::new(values, vec![order.len(), 1])
        .map_err(|e| union_internal_error(format!("union: {e}")))?;
    let ia_len = ia.len();
    let ib_len = ib.len();
    let ia_tensor = Tensor::new(ia, vec![ia_len, 1])
        .map_err(|e| union_internal_error(format!("union: {e}")))?;
    let ib_tensor = Tensor::new(ib, vec![ib_len, 1])
        .map_err(|e| union_internal_error(format!("union: {e}")))?;

    Ok(UnionEvaluation::new(
        Value::StringArray(value_array),
        ia_tensor,
        ib_tensor,
    ))
}

fn assemble_string_row_union(
    entries: Vec<StringRowUnionEntry>,
    opts: &UnionOptions,
    cols: usize,
) -> crate::BuiltinResult<UnionEvaluation> {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    match opts.order {
        UnionOrder::Sorted => {
            order.sort_by(|&lhs, &rhs| {
                compare_string_rows(&entries[lhs].row_data, &entries[rhs].row_data)
            });
        }
        UnionOrder::Stable => {
            order.sort_by_key(|&idx| entries[idx].order_rank);
        }
    }

    let unique_rows = order.len();
    let mut values = vec![String::new(); unique_rows * cols];
    let mut ia = Vec::new();
    let mut ib = Vec::new();

    for (row_pos, &entry_idx) in order.iter().enumerate() {
        let entry = &entries[entry_idx];
        for col in 0..cols {
            let dest = row_pos + col * unique_rows;
            values[dest] = entry.row_data[col].clone();
        }
        if let Some(a_row) = entry.a_row {
            ia.push((a_row + 1) as f64);
        } else if let Some(b_row) = entry.b_row {
            ib.push((b_row + 1) as f64);
        }
    }

    let value_array = StringArray::new(values, vec![unique_rows, cols])
        .map_err(|e| union_internal_error(format!("union: {e}")))?;
    let ia_len = ia.len();
    let ib_len = ib.len();
    let ia_tensor = Tensor::new(ia, vec![ia_len, 1])
        .map_err(|e| union_internal_error(format!("union: {e}")))?;
    let ib_tensor = Tensor::new(ib, vec![ib_len, 1])
        .map_err(|e| union_internal_error(format!("union: {e}")))?;

    Ok(UnionEvaluation::new(
        Value::StringArray(value_array),
        ia_tensor,
        ib_tensor,
    ))
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

fn complex_is_nan<T: SetFloat>(value: (T, T)) -> bool {
    value.0.is_nan() || value.1.is_nan()
}

fn compare_complex<T: SetFloat>(a: (T, T), b: (T, T)) -> Ordering {
    match (complex_is_nan(a), complex_is_nan(b)) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => {
            let mag_a = a.0.hypot(a.1);
            let mag_b = b.0.hypot(b.1);
            let mag_cmp = mag_a.compare(mag_b);
            if mag_cmp != Ordering::Equal {
                return mag_cmp;
            }
            let re_cmp = a.0.compare(b.0);
            if re_cmp != Ordering::Equal {
                return re_cmp;
            }
            a.1.compare(b.1)
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
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{IntValue, ResolveContext, Tensor, Type, Value};

    fn evaluate_sync(a: Value, b: Value, rest: &[Value]) -> crate::BuiltinResult<UnionEvaluation> {
        futures::executor::block_on(evaluate(a, b, rest))
    }

    fn builtin_sync(a: Value, b: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        futures::executor::block_on(union_builtin(a, b, rest))
    }

    #[test]
    fn registered_builtin_restores_resident_outputs_and_rejects_excess_arity() {
        test_support::with_test_provider(|provider| {
            let left = Tensor::new_integer(IntegerStorage::I32(vec![7, 2, 9]), vec![3, 1]).unwrap();
            let right = Tensor::new_integer(IntegerStorage::I32(vec![2, 7]), vec![2, 1]).unwrap();
            let left =
                Value::GpuTensor(gpu_helpers::upload_tensor(provider, &left).expect("upload left"));
            let right = Value::GpuTensor(
                gpu_helpers::upload_tensor(provider, &right).expect("upload right"),
            );

            {
                let _guard = crate::output_count::push_output_count(Some(3));
                let Value::OutputList(outputs) =
                    builtin_sync(left, right, Vec::new()).expect("resident union")
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
                    Some(&IntegerStorage::I32(vec![2, 7, 9]))
                );
            }

            let _guard = crate::output_count::push_output_count(Some(4));
            let err = builtin_sync(Value::Num(1.0), Value::Num(1.0), Vec::new())
                .expect_err("excess outputs must fail");
            assert_eq!(err.identifier(), UNION_ERROR_INVALID_ARGUMENT.identifier);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn union_numeric_sorted_default() {
        let a = Tensor::new(vec![5.0, 7.0, 1.0], vec![3, 1]).unwrap();
        let b = Tensor::new(vec![3.0, 1.0, 1.0], vec![3, 1]).unwrap();
        let eval = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[]).expect("union");
        match eval.values_value() {
            Value::Tensor(t) => {
                assert_eq!(t.materialize_f64(), vec![1.0, 3.0, 5.0, 7.0]);
                assert_eq!(t.shape, vec![4, 1]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        let ia = tensor::value_into_tensor_for("union", eval.ia_value()).expect("ia tensor");
        assert_eq!(ia.materialize_f64(), vec![3.0, 1.0, 2.0]);
        assert_eq!(ia.shape, vec![3, 1]);
        let ib = tensor::value_into_tensor_for("union", eval.ib_value()).expect("ib tensor");
        assert_eq!(ib.materialize_f64(), vec![1.0]);
        assert_eq!(ib.shape, vec![1, 1]);
    }

    #[test]
    fn union_preserves_native_single_elements_and_rows() {
        let a = Tensor::from_f32(vec![5.0, 7.0, 1.0], vec![3, 1]).unwrap();
        let b = Tensor::from_f32(vec![3.0, 1.0, 1.0], vec![3, 1]).unwrap();
        let values = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[])
            .expect("single union")
            .into_values_value();
        let Value::Tensor(values) = values else {
            panic!("expected native single values");
        };
        assert_eq!(
            values.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![1.0, 3.0, 5.0, 7.0])
        );

        let a = Tensor::from_f32(vec![1.0, 3.0, 1.0, 2.0, 4.0, 2.0], vec![3, 2]).unwrap();
        let b = Tensor::from_f32(vec![3.0, 5.0, 4.0, 6.0], vec![2, 2]).unwrap();
        let values = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[Value::from("rows")])
            .expect("single row union")
            .into_values_value();
        let Value::Tensor(values) = values else {
            panic!("expected native single rows");
        };
        assert_eq!(values.shape, vec![3, 2]);
        assert_eq!(
            values.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0])
        );
    }

    #[test]
    fn union_preserves_native_complex_single_elements_and_rows() {
        let a = ComplexTensor::from_f32(vec![(1.0, 1.0)], vec![1, 1]).unwrap();
        let b = ComplexTensor::from_f32(vec![(1.0, 1.0)], vec![1, 1]).unwrap();
        let values = evaluate_sync(Value::ComplexTensor(a), Value::ComplexTensor(b), &[])
            .expect("complex single union")
            .into_values_value();
        let Value::ComplexTensor(values) = values else {
            panic!("expected native complex single value");
        };
        assert_eq!(values.as_f32_slice(), Some(&[(1.0, 1.0)][..]));

        let a = ComplexTensor::from_f32(
            vec![
                (1.0, 0.0),
                (3.0, 0.0),
                (1.0, 0.0),
                (2.0, 1.0),
                (4.0, 1.0),
                (2.0, 1.0),
            ],
            vec![3, 2],
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
        .expect("complex single row union")
        .into_values_value();
        let Value::ComplexTensor(values) = values else {
            panic!("expected native complex single rows");
        };
        assert_eq!(values.shape, vec![3, 2]);
        assert_eq!(
            values.as_f32_slice(),
            Some(
                &[
                    (1.0, 0.0),
                    (3.0, 0.0),
                    (5.0, 0.0),
                    (2.0, 1.0),
                    (4.0, 1.0),
                    (6.0, 1.0),
                ][..]
            )
        );
    }

    #[test]
    fn union_preserves_exact_integer_elements_and_rows() {
        let a = Tensor::new_integer(
            runmat_builtins::IntegerStorage::U64(vec![u64::MAX, 0, 9_007_199_254_740_993]),
            vec![3, 1],
        )
        .expect("input");
        let b = Tensor::new_integer(runmat_builtins::IntegerStorage::U64(vec![0, 7]), vec![2, 1])
            .expect("input");
        let (values, ia, ib) = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[])
            .expect("union")
            .into_triple();
        let Value::Tensor(values) = values else {
            panic!("exact values");
        };
        assert_eq!(
            values.integer_storage(),
            Some(&runmat_builtins::IntegerStorage::U64(vec![
                0,
                7,
                9_007_199_254_740_993,
                u64::MAX
            ]))
        );
        let ia = tensor::value_into_tensor_for("union", ia).expect("indices");
        assert_eq!(ia.materialize_f64(), vec![2.0, 3.0, 1.0]);
        let ib = tensor::value_into_tensor_for("union", ib).expect("indices");
        assert_eq!(ib.materialize_f64(), vec![2.0]);

        let a = Tensor::new_integer(
            runmat_builtins::IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993, 0, 1]),
            vec![2, 2],
        )
        .expect("rows input");
        let b = Tensor::new_integer(
            runmat_builtins::IntegerStorage::U64(vec![9_007_199_254_740_993, 4, 1, 2]),
            vec![2, 2],
        )
        .expect("rows input");
        let (values, ia, ib) =
            evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[Value::from("rows")])
                .expect("union rows")
                .into_triple();
        let Value::Tensor(values) = values else {
            panic!("exact row values");
        };
        assert_eq!(
            values.integer_storage(),
            Some(&runmat_builtins::IntegerStorage::U64(vec![
                4,
                9_007_199_254_740_993,
                u64::MAX,
                2,
                1,
                0,
            ]))
        );
        let ia = tensor::value_into_tensor_for("union", ia).expect("row indices");
        assert_eq!(ia.materialize_f64(), vec![2.0, 1.0]);
        let ib = tensor::value_into_tensor_for("union", ib).expect("row indices");
        assert_eq!(ib.materialize_f64(), vec![2.0]);
    }

    #[test]
    fn union_rejects_mixed_nondouble_integer_classes() {
        let a = Tensor::new_integer(runmat_builtins::IntegerStorage::U16(vec![7, 2]), vec![2, 1])
            .expect("input");
        let b = Tensor::new_integer(runmat_builtins::IntegerStorage::I32(vec![2, 9]), vec![2, 1])
            .expect("input");
        let error = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[])
            .expect_err("mixed integer classes must reject");
        assert_eq!(
            error.identifier(),
            UNION_ERROR_NUMERIC_CLASS_MISMATCH.identifier
        );
    }

    #[test]
    fn union_preserves_every_exact_integer_class() {
        let cases = [
            runmat_builtins::IntegerStorage::I8(vec![i8::MAX, 0]),
            runmat_builtins::IntegerStorage::I16(vec![i16::MAX, 0]),
            runmat_builtins::IntegerStorage::I32(vec![i32::MAX, 0]),
            runmat_builtins::IntegerStorage::I64(vec![i64::MAX, 0]),
            runmat_builtins::IntegerStorage::U8(vec![u8::MAX, 0]),
            runmat_builtins::IntegerStorage::U16(vec![u16::MAX, 0]),
            runmat_builtins::IntegerStorage::U32(vec![u32::MAX, 0]),
            runmat_builtins::IntegerStorage::U64(vec![u64::MAX, 0]),
        ];
        for storage in cases {
            let expected = storage.clone();
            let a = Tensor::new_integer(storage, vec![2, 1]).expect("input");
            let b = Tensor::new_integer(expected.zeros_like(1), vec![1, 1]).expect("input");
            let values =
                evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[Value::from("stable")])
                    .expect("union")
                    .into_values_value();
            let Value::Tensor(values) = values else {
                panic!("exact values");
            };
            assert_eq!(values.integer_storage(), Some(&expected));
        }
    }

    #[test]
    fn union_type_resolver_numeric() {
        assert_eq!(
            set_values_output_type(
                &[Type::tensor(), Type::tensor()],
                &ResolveContext::new(Vec::new()),
            ),
            Type::tensor()
        );
    }

    #[test]
    fn union_type_resolver_string_array() {
        assert_eq!(
            set_values_output_type(
                &[Type::cell_of(Type::String), Type::String],
                &ResolveContext::new(Vec::new()),
            ),
            Type::cell_of(Type::String)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn union_numeric_stable_order() {
        let a = Tensor::new(vec![5.0, 7.0, 1.0], vec![3, 1]).unwrap();
        let b = Tensor::new(vec![3.0, 2.0, 4.0], vec![3, 1]).unwrap();
        let eval = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[Value::from("stable")])
            .expect("union");
        match eval.values_value() {
            Value::Tensor(t) => {
                assert_eq!(t.materialize_f64(), vec![5.0, 7.0, 1.0, 3.0, 2.0, 4.0]);
                assert_eq!(t.shape, vec![6, 1]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        let ia = tensor::value_into_tensor_for("union", eval.ia_value()).expect("ia tensor");
        assert_eq!(ia.materialize_f64(), vec![1.0, 2.0, 3.0]);
        let ib = tensor::value_into_tensor_for("union", eval.ib_value()).expect("ib tensor");
        assert_eq!(ib.materialize_f64(), vec![1.0, 2.0, 3.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn union_numeric_sorted_places_nan_last() {
        let a = Tensor::new(vec![f64::NAN, 1.0], vec![2, 1]).unwrap();
        let b = Tensor::new(vec![2.0, f64::NAN], vec![2, 1]).unwrap();
        let eval = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[]).expect("union");
        let values = tensor::value_into_tensor_for("union", eval.values_value()).expect("values");
        assert_eq!(values.shape, vec![3, 1]);
        assert_eq!(values.materialize_f64()[0], 1.0);
        assert_eq!(values.materialize_f64()[1], 2.0);
        assert!(values.materialize_f64()[2].is_nan());
        let ia = tensor::value_into_tensor_for("union", eval.ia_value()).expect("ia tensor");
        assert_eq!(ia.materialize_f64(), vec![2.0, 1.0]);
        let ib = tensor::value_into_tensor_for("union", eval.ib_value()).expect("ib tensor");
        assert_eq!(ib.materialize_f64(), vec![1.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn union_numeric_rows_sorted() {
        let a = Tensor::new(vec![1.0, 3.0, 1.0, 2.0, 4.0, 2.0], vec![3, 2]).unwrap();
        let b = Tensor::new(vec![3.0, 5.0, 4.0, 6.0], vec![2, 2]).unwrap();
        let eval = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[Value::from("rows")])
            .expect("union");
        match eval.values_value() {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 2]);
                assert_eq!(t.materialize_f64(), vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        let ia = tensor::value_into_tensor_for("union", eval.ia_value()).expect("ia tensor");
        assert_eq!(ia.materialize_f64(), vec![1.0, 2.0]);
        let ib = tensor::value_into_tensor_for("union", eval.ib_value()).expect("ib tensor");
        assert_eq!(ib.materialize_f64(), vec![2.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn union_numeric_rows_stable_preserves_first_occurrence() {
        let a = Tensor::new(vec![1.0, 3.0, 1.0, 2.0, 4.0, 2.0], vec![3, 2]).unwrap();
        let b = Tensor::new(vec![3.0, 5.0, 1.0, 4.0, 6.0, 2.0], vec![3, 2]).unwrap();
        let eval = evaluate_sync(
            Value::Tensor(a),
            Value::Tensor(b),
            &[Value::from("rows"), Value::from("stable")],
        )
        .expect("union");
        let (values, ia, ib) = eval.into_triple();
        match values {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 2]);
                assert_eq!(t.materialize_f64(), vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        let ia_tensor = tensor::value_into_tensor_for("union", ia).expect("ia tensor");
        assert_eq!(ia_tensor.materialize_f64(), vec![1.0, 2.0]);
        let ib_tensor = tensor::value_into_tensor_for("union", ib).expect("ib tensor");
        assert_eq!(ib_tensor.materialize_f64(), vec![2.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn union_char_elements() {
        let a = CharArray::new(vec!['m', 'z', 'm', 'a'], 2, 2).unwrap();
        let b = CharArray::new(vec!['a', 'x', 'm', 'a'], 2, 2).unwrap();
        let eval = evaluate_sync(Value::CharArray(a), Value::CharArray(b), &[]).expect("union");
        match eval.values_value() {
            Value::CharArray(arr) => {
                assert_eq!(arr.rows, 4);
                assert_eq!(arr.cols, 1);
                assert_eq!(arr.data, vec!['a', 'm', 'x', 'z']);
            }
            other => panic!("expected char array, got {other:?}"),
        }
        let ia = tensor::value_into_tensor_for("union", eval.ia_value()).expect("ia tensor");
        assert_eq!(ia.materialize_f64(), vec![4.0, 1.0, 3.0]);
        let ib = tensor::value_into_tensor_for("union", eval.ib_value()).expect("ib tensor");
        assert_eq!(ib.materialize_f64(), vec![3.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn union_string_rows_stable() {
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
        .expect("union");
        match eval.values_value() {
            Value::StringArray(arr) => {
                assert_eq!(arr.shape, vec![3, 2]);
                assert_eq!(
                    arr.data,
                    vec![
                        "alpha".to_string(),
                        "gamma".to_string(),
                        "delta".to_string(),
                        "beta".to_string(),
                        "beta".to_string(),
                        "beta".to_string()
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
        let ia = tensor::value_into_tensor_for("union", eval.ia_value()).expect("ia tensor");
        assert_eq!(ia.materialize_f64(), vec![1.0, 2.0]);
        let ib = tensor::value_into_tensor_for("union", eval.ib_value()).expect("ib tensor");
        assert_eq!(ib.materialize_f64(), vec![2.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn union_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let a = Tensor::new(vec![4.0, 1.0, 2.0], vec![3, 1]).unwrap();
            let b = Tensor::new(vec![2.0, 5.0], vec![2, 1]).unwrap();
            let view_a = HostTensorView {
                data: &a.materialize_f64(),
                shape: &a.shape,
            };
            let view_b = HostTensorView {
                data: &b.materialize_f64(),
                shape: &b.shape,
            };
            let handle_a = provider.upload(&view_a).expect("upload A");
            let handle_b = provider.upload(&view_b).expect("upload B");
            let eval = evaluate_sync(
                Value::GpuTensor(handle_a),
                Value::GpuTensor(handle_b),
                &[Value::from("stable")],
            )
            .expect("union");
            let values = tensor::value_into_tensor_for("union", eval.values_value()).unwrap();
            assert_eq!(values.materialize_f64(), vec![4.0, 1.0, 2.0, 5.0]);
            let ia = tensor::value_into_tensor_for("union", eval.ia_value()).unwrap();
            assert_eq!(ia.materialize_f64(), vec![1.0, 2.0, 3.0]);
            let ib = tensor::value_into_tensor_for("union", eval.ib_value()).unwrap();
            assert_eq!(ib.materialize_f64(), vec![2.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn union_rejects_legacy_option() {
        let tensor =
            Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).expect("tensor construction failed");
        let err = evaluate_sync(
            Value::Tensor(tensor.clone()),
            Value::Tensor(tensor),
            &[Value::from("legacy")],
        )
        .unwrap_err();
        assert_eq!(
            err.identifier(),
            UNION_ERROR_LEGACY_OPTION_UNSUPPORTED.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn union_rejects_conflicting_order_options() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).expect("tensor construction failed");
        let err = evaluate_sync(
            Value::Tensor(tensor.clone()),
            Value::Tensor(tensor),
            &[Value::from("stable"), Value::from("sorted")],
        )
        .unwrap_err();
        assert_eq!(
            err.identifier(),
            UNION_ERROR_CONFLICTING_ORDER_OPTIONS.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn union_rejects_unknown_option() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).expect("tensor construction failed");
        let err = evaluate_sync(
            Value::Tensor(tensor.clone()),
            Value::Tensor(tensor),
            &[Value::from("bogus")],
        )
        .unwrap_err();
        assert_eq!(err.identifier(), UNION_ERROR_UNKNOWN_OPTION.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn union_rows_dimension_mismatch() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err =
            evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[Value::from("rows")]).unwrap_err();
        assert_eq!(
            err.identifier(),
            UNION_ERROR_ROWS_COLUMN_MISMATCH.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn union_requires_matching_types() {
        let a = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let b = CharArray::new(vec!['a', 'b'], 1, 2).unwrap();
        let err = union_host(
            Value::Tensor(a),
            Value::CharArray(b),
            &UnionOptions {
                rows: false,
                order: UnionOrder::Sorted,
            },
        )
        .unwrap_err();
        assert_eq!(
            err.identifier(),
            UNION_ERROR_UNSUPPORTED_INPUT_TYPE.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn union_accepts_scalar_inputs() {
        let eval =
            evaluate_sync(Value::Int(IntValue::I32(1)), Value::Num(3.0), &[]).expect("union");
        match eval.values_value() {
            Value::Tensor(t) => {
                assert_eq!(t.materialize_f64(), vec![1.0, 3.0]);
                assert_eq!(t.shape, vec![2, 1]);
            }
            other => panic!("expected numeric tensor, got {other:?}"),
        }
        let ia = tensor::value_into_tensor_for("union", eval.ia_value()).unwrap();
        assert_eq!(ia.materialize_f64(), vec![1.0]);
        let ib = tensor::value_into_tensor_for("union", eval.ib_value()).unwrap();
        assert_eq!(ib.materialize_f64(), vec![1.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn union_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let a = Tensor::new(vec![4.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
        let b = Tensor::new(vec![2.0, 6.0, 3.0], vec![3, 1]).unwrap();

        let cpu_eval =
            evaluate_sync(Value::Tensor(a.clone()), Value::Tensor(b.clone()), &[]).expect("union");
        let cpu_values = tensor::value_into_tensor_for("union", cpu_eval.values_value()).unwrap();
        let cpu_ia = tensor::value_into_tensor_for("union", cpu_eval.ia_value()).unwrap();
        let cpu_ib = tensor::value_into_tensor_for("union", cpu_eval.ib_value()).unwrap();

        let provider = runmat_accelerate_api::provider().expect("provider");
        let view_a = HostTensorView {
            data: &a.materialize_f64(),
            shape: &a.shape,
        };
        let view_b = HostTensorView {
            data: &b.materialize_f64(),
            shape: &b.shape,
        };
        let handle_a = provider.upload(&view_a).expect("upload A");
        let handle_b = provider.upload(&view_b).expect("upload B");
        let gpu_eval = evaluate_sync(Value::GpuTensor(handle_a), Value::GpuTensor(handle_b), &[])
            .expect("union");
        let gpu_values = tensor::value_into_tensor_for("union", gpu_eval.values_value()).unwrap();
        let gpu_ia = tensor::value_into_tensor_for("union", gpu_eval.ia_value()).unwrap();
        let gpu_ib = tensor::value_into_tensor_for("union", gpu_eval.ib_value()).unwrap();

        assert_eq!(gpu_values.materialize_f64(), cpu_values.materialize_f64());
        assert_eq!(gpu_values.shape, cpu_values.shape);
        assert_eq!(gpu_ia.materialize_f64(), cpu_ia.materialize_f64());
        assert_eq!(gpu_ib.materialize_f64(), cpu_ib.materialize_f64());
    }
}
