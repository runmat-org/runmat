//! MATLAB-compatible `setdiff` builtin with GPU-aware semantics for RunMat.
//!
//! Provides element-wise and row-wise set difference with optional stable
//! ordering. GPU tensors use a provider hook or typed host fallback, then public
//! outputs are restored to the owning provider.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use runmat_accelerate_api::{GpuTensorHandle, SetdiffOptions, SetdiffOrder, SetdiffResult};
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
    NumericStorage, StringArray, Tensor, Value,
};

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

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::sorting_sets::setdiff")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "setdiff",
    op_kind: GpuOpKind::Custom("setdiff"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("setdiff")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: true,
    notes: "Providers may implement `setdiff`; exact typed fallback gathers when needed and restores difference values plus double indices to the input owner.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::array::sorting_sets::setdiff"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "setdiff",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "`setdiff` terminates fusion chains and materialises results on the host; upstream tensors are gathered when necessary.",
};

const BUILTIN_NAME: &str = "setdiff";

const SETDIFF_OUTPUT_C: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Values that appear in A but not in B.",
}];

const SETDIFF_OUTPUT_C_IA: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Values that appear in A but not in B.",
    },
    BuiltinParamDescriptor {
        name: "ia",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indices selecting retained values/rows from A.",
    },
];

const SETDIFF_INPUTS_A_B: [BuiltinParamDescriptor; 2] = [
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

const SETDIFF_INPUTS_A_B_OPTIONS: [BuiltinParamDescriptor; 3] = [
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

const SETDIFF_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "C = setdiff(A, B)",
        inputs: &SETDIFF_INPUTS_A_B,
        outputs: &SETDIFF_OUTPUT_C,
    },
    BuiltinSignatureDescriptor {
        label: "C = setdiff(A, B, option...)",
        inputs: &SETDIFF_INPUTS_A_B_OPTIONS,
        outputs: &SETDIFF_OUTPUT_C,
    },
    BuiltinSignatureDescriptor {
        label: "[C, ia] = setdiff(A, B)",
        inputs: &SETDIFF_INPUTS_A_B,
        outputs: &SETDIFF_OUTPUT_C_IA,
    },
    BuiltinSignatureDescriptor {
        label: "[C, ia] = setdiff(A, B, option...)",
        inputs: &SETDIFF_INPUTS_A_B_OPTIONS,
        outputs: &SETDIFF_OUTPUT_C_IA,
    },
];

const SETDIFF_ERROR_LEGACY_OPTION_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETDIFF.LEGACY_OPTION_UNSUPPORTED",
    identifier: Some("RunMat:setdiff:LegacyOptionUnsupported"),
    when: "Legacy compatibility options are requested.",
    message: "setdiff: the 'legacy' behaviour is not supported",
};

const SETDIFF_ERROR_CONFLICTING_ORDER_OPTIONS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETDIFF.CONFLICTING_ORDER_OPTIONS",
    identifier: Some("RunMat:setdiff:ConflictingOrderOptions"),
    when: "Both 'sorted' and 'stable' options are provided.",
    message: "setdiff: cannot combine 'sorted' with 'stable'",
};

const SETDIFF_ERROR_UNKNOWN_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETDIFF.UNKNOWN_OPTION",
    identifier: Some("RunMat:setdiff:UnknownOption"),
    when: "An unsupported option token is provided.",
    message: "setdiff: unrecognised option",
};

const SETDIFF_ERROR_ROWS_COLUMN_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETDIFF.ROWS_COLUMN_MISMATCH",
    identifier: Some("RunMat:setdiff:RowsColumnMismatch"),
    when: "'rows' mode is used and column counts differ.",
    message: "setdiff: inputs must have the same number of columns when using 'rows'",
};

const SETDIFF_ERROR_UNSUPPORTED_INPUT_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETDIFF.UNSUPPORTED_INPUT_TYPE",
    identifier: Some("RunMat:setdiff:UnsupportedInputType"),
    when: "Input values cannot be converted into supported setdiff domains.",
    message: "setdiff: unsupported input type",
};

const SETDIFF_ERROR_NUMERIC_CLASS_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETDIFF.NUMERIC_CLASS_MISMATCH",
    identifier: Some("RunMat:setdiff:NumericClassMismatch"),
    when: "Numeric inputs have incompatible nondouble classes.",
    message: "setdiff: numeric inputs must have the same class, except double may be combined with one nondouble class",
};

const SETDIFF_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETDIFF.INVALID_ARGUMENT",
    identifier: Some("RunMat:setdiff:InvalidArgument"),
    when: "Option arguments are not string-like where required.",
    message: "setdiff: expected string option arguments",
};

const SETDIFF_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETDIFF.INTERNAL",
    identifier: Some("RunMat:setdiff:Internal"),
    when: "Internal conversion/allocation/provider decode fails.",
    message: "setdiff: internal operation failed",
};

const SETDIFF_ERRORS: [BuiltinErrorDescriptor; 8] = [
    SETDIFF_ERROR_LEGACY_OPTION_UNSUPPORTED,
    SETDIFF_ERROR_CONFLICTING_ORDER_OPTIONS,
    SETDIFF_ERROR_UNKNOWN_OPTION,
    SETDIFF_ERROR_ROWS_COLUMN_MISMATCH,
    SETDIFF_ERROR_UNSUPPORTED_INPUT_TYPE,
    SETDIFF_ERROR_NUMERIC_CLASS_MISMATCH,
    SETDIFF_ERROR_INVALID_ARGUMENT,
    SETDIFF_ERROR_INTERNAL,
];

const SETDIFF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[C, ia] = setdiff(integer_A, integer_B, options)",
        inputs: &super::BINARY_SET_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "C preserves A's integer class, including when B is double; ia is one-based double. GPU supports integer classes through 32 bits and restores outputs after typed fallback.",
    }];

pub const SETDIFF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SETDIFF_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SETDIFF_ERRORS,
};

fn setdiff_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn setdiff_error(error: &'static BuiltinErrorDescriptor) -> crate::RuntimeError {
    setdiff_error_with(error, error.message)
}

fn setdiff_internal_error(message: impl Into<String>) -> crate::RuntimeError {
    setdiff_error_with(&SETDIFF_ERROR_INTERNAL, message)
}

#[runtime_builtin(
    name = "setdiff",
    category = "array/sorting_sets",
    summary = "Return values that appear in the first input but not the second.",
    keywords = "setdiff,difference,stable,rows,indices,gpu",
    accel = "array_construct",
    sink = true,
    type_resolver(set_values_output_type),
    descriptor(crate::builtins::array::sorting_sets::setdiff::SETDIFF_DESCRIPTOR),
    integer_capabilities(SETDIFF_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::array::sorting_sets::setdiff"
)]
async fn setdiff_builtin(a: Value, b: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 2) {
        return Err(setdiff_error_with(
            &SETDIFF_ERROR_INVALID_ARGUMENT,
            "setdiff: too many output arguments; maximum is 2",
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
                setdiff_internal_error,
            )?;
            return Ok(Value::OutputList(outputs));
        }
        let (values, ia) = eval.into_pair();
        let outputs = super::restore_set_outputs(
            provider,
            BUILTIN_NAME,
            vec![values, ia],
            setdiff_internal_error,
        )?;
        return Ok(Value::OutputList(outputs));
    }
    let mut outputs = super::restore_set_outputs(
        provider,
        BUILTIN_NAME,
        vec![eval.into_values_value()],
        setdiff_internal_error,
    )?;
    Ok(outputs.pop().expect("setdiff output"))
}

/// Evaluate the `setdiff` builtin once and expose all outputs.
pub async fn evaluate(
    a: Value,
    b: Value,
    rest: &[Value],
) -> crate::BuiltinResult<SetdiffEvaluation> {
    crate::builtins::common::validation::reject_typed_complex_integer(&a, "setdiff")?;
    crate::builtins::common::validation::reject_typed_complex_integer(&b, "setdiff")?;
    let opts = parse_options(rest)?;
    for value in [&a, &b] {
        if let Value::GpuTensor(handle) = value {
            if super::is_unsupported_set_gpu_integer(handle) {
                return Err(setdiff_error_with(
                    &SETDIFF_ERROR_UNSUPPORTED_INPUT_TYPE,
                    "setdiff: resident 64-bit integer inputs are not supported",
                ));
            }
        }
    }
    match (a, b) {
        (Value::GpuTensor(handle_a), Value::GpuTensor(handle_b)) => {
            setdiff_gpu_pair(handle_a, handle_b, &opts).await
        }
        (Value::GpuTensor(handle_a), other) => {
            setdiff_gpu_mixed(handle_a, other, &opts, true).await
        }
        (other, Value::GpuTensor(handle_b)) => {
            setdiff_gpu_mixed(handle_b, other, &opts, false).await
        }
        (left, right) => setdiff_host(left, right, &opts),
    }
}

fn parse_options(rest: &[Value]) -> crate::BuiltinResult<SetdiffOptions> {
    let mut opts = SetdiffOptions {
        rows: false,
        order: SetdiffOrder::Sorted,
    };
    let mut seen_order: Option<SetdiffOrder> = None;

    let tokens = tokens_from_values(rest);
    for (arg, token) in rest.iter().zip(tokens.iter()) {
        let text = match token {
            crate::builtins::common::arg_tokens::ArgToken::String(text) => text.as_str(),
            _ => {
                let text = tensor::value_to_string(arg)
                    .ok_or_else(|| setdiff_error(&SETDIFF_ERROR_INVALID_ARGUMENT))?;
                let lowered = text.trim().to_ascii_lowercase();
                parse_setdiff_option(&mut opts, &mut seen_order, &lowered)?;
                continue;
            }
        };
        parse_setdiff_option(&mut opts, &mut seen_order, text)?;
    }

    Ok(opts)
}

fn parse_setdiff_option(
    opts: &mut SetdiffOptions,
    seen_order: &mut Option<SetdiffOrder>,
    lowered: &str,
) -> crate::BuiltinResult<()> {
    match lowered {
        "rows" => opts.rows = true,
        "sorted" => {
            if let Some(prev) = seen_order {
                if *prev != SetdiffOrder::Sorted {
                    return Err(setdiff_error(&SETDIFF_ERROR_CONFLICTING_ORDER_OPTIONS));
                }
            }
            *seen_order = Some(SetdiffOrder::Sorted);
            opts.order = SetdiffOrder::Sorted;
        }
        "stable" => {
            if let Some(prev) = seen_order {
                if *prev != SetdiffOrder::Stable {
                    return Err(setdiff_error(&SETDIFF_ERROR_CONFLICTING_ORDER_OPTIONS));
                }
            }
            *seen_order = Some(SetdiffOrder::Stable);
            opts.order = SetdiffOrder::Stable;
        }
        "legacy" | "r2012a" => {
            return Err(setdiff_error(&SETDIFF_ERROR_LEGACY_OPTION_UNSUPPORTED));
        }
        other => {
            return Err(setdiff_error_with(
                &SETDIFF_ERROR_UNKNOWN_OPTION,
                format!("setdiff: unrecognised option '{other}'"),
            ))
        }
    }
    Ok(())
}

async fn setdiff_gpu_pair(
    handle_a: GpuTensorHandle,
    handle_b: GpuTensorHandle,
    opts: &SetdiffOptions,
) -> crate::BuiltinResult<SetdiffEvaluation> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle_a)
        .or_else(runmat_accelerate_api::provider)
    {
        match provider.setdiff(&handle_a, &handle_b, opts).await {
            Ok(result) => return SetdiffEvaluation::from_setdiff_result(result),
            Err(_) => {
                // Fall back to host gather when provider does not support setdiff.
            }
        }
    }
    let a_tensor = gpu_helpers::gather_tensor_async(&handle_a).await?;
    let b_tensor = gpu_helpers::gather_tensor_async(&handle_b).await?;
    setdiff_numeric(a_tensor, b_tensor, opts)
}

async fn setdiff_gpu_mixed(
    handle_gpu: GpuTensorHandle,
    other: Value,
    opts: &SetdiffOptions,
    gpu_is_a: bool,
) -> crate::BuiltinResult<SetdiffEvaluation> {
    let gpu_tensor = gpu_helpers::gather_tensor_async(&handle_gpu).await?;
    let other_tensor =
        tensor::value_into_tensor_for("setdiff", other).map_err(setdiff_internal_error)?;
    if gpu_is_a {
        setdiff_numeric(gpu_tensor, other_tensor, opts)
    } else {
        setdiff_numeric(other_tensor, gpu_tensor, opts)
    }
}

fn setdiff_host(
    a: Value,
    b: Value,
    opts: &SetdiffOptions,
) -> crate::BuiltinResult<SetdiffEvaluation> {
    match (a, b) {
        (Value::ComplexTensor(at), Value::ComplexTensor(bt)) => setdiff_complex(at, bt, opts),
        (Value::ComplexTensor(at), Value::Complex(re, im)) => {
            let bt = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| setdiff_internal_error(format!("setdiff: {e}")))?;
            setdiff_complex(at, bt, opts)
        }
        (Value::Complex(a_re, a_im), Value::ComplexTensor(bt)) => {
            let at = ComplexTensor::new(vec![(a_re, a_im)], vec![1, 1])
                .map_err(|e| setdiff_internal_error(format!("setdiff: {e}")))?;
            setdiff_complex(at, bt, opts)
        }
        (Value::Complex(a_re, a_im), Value::Complex(b_re, b_im)) => {
            let at = ComplexTensor::new(vec![(a_re, a_im)], vec![1, 1])
                .map_err(|e| setdiff_internal_error(format!("setdiff: {e}")))?;
            let bt = ComplexTensor::new(vec![(b_re, b_im)], vec![1, 1])
                .map_err(|e| setdiff_internal_error(format!("setdiff: {e}")))?;
            setdiff_complex(at, bt, opts)
        }

        (Value::CharArray(ac), Value::CharArray(bc)) => setdiff_char(ac, bc, opts),

        (Value::StringArray(astring), Value::StringArray(bstring)) => {
            setdiff_string(astring, bstring, opts)
        }
        (Value::StringArray(astring), Value::String(b)) => {
            let bstring = StringArray::new(vec![b], vec![1, 1])
                .map_err(|e| setdiff_internal_error(format!("setdiff: {e}")))?;
            setdiff_string(astring, bstring, opts)
        }
        (Value::String(a), Value::StringArray(bstring)) => {
            let astring = StringArray::new(vec![a], vec![1, 1])
                .map_err(|e| setdiff_internal_error(format!("setdiff: {e}")))?;
            setdiff_string(astring, bstring, opts)
        }
        (Value::String(a), Value::String(b)) => {
            let astring = StringArray::new(vec![a], vec![1, 1])
                .map_err(|e| setdiff_internal_error(format!("setdiff: {e}")))?;
            let bstring = StringArray::new(vec![b], vec![1, 1])
                .map_err(|e| setdiff_internal_error(format!("setdiff: {e}")))?;
            setdiff_string(astring, bstring, opts)
        }

        (left, right) => {
            let tensor_a = tensor::value_into_tensor_for("setdiff", left)
                .map_err(|e| setdiff_error_with(&SETDIFF_ERROR_UNSUPPORTED_INPUT_TYPE, e))?;
            let tensor_b = tensor::value_into_tensor_for("setdiff", right)
                .map_err(|e| setdiff_error_with(&SETDIFF_ERROR_UNSUPPORTED_INPUT_TYPE, e))?;
            setdiff_numeric(tensor_a, tensor_b, opts)
        }
    }
}

fn setdiff_numeric(
    a: Tensor,
    b: Tensor,
    opts: &SetdiffOptions,
) -> crate::BuiltinResult<SetdiffEvaluation> {
    let a_dtype = a.numeric_dtype();
    let b_dtype = b.numeric_dtype();
    if let (Some(a_storage), Some(b_storage)) = (a.integer_storage(), b.integer_storage()) {
        if a_storage.class_name() == b_storage.class_name() {
            return if opts.rows {
                setdiff_integer_rows(a_storage, a.shape.clone(), b_storage, b.shape.clone(), opts)
            } else {
                setdiff_integer_elements(a_storage, b_storage, opts)
            };
        }
        return Err(setdiff_error(&SETDIFF_ERROR_NUMERIC_CLASS_MISMATCH));
    }
    match (a.integer_storage(), b.integer_storage()) {
        (Some(storage), None) if b_dtype == NumericDType::F64 => {
            let target = IntegerTarget::from_storage(storage);
            let b = target.cast_tensor(b).map_err(setdiff_internal_error)?;
            return setdiff_numeric(a, b, opts);
        }
        _ => {}
    }
    if a_dtype != b_dtype && a_dtype != NumericDType::F64 && b_dtype != NumericDType::F64 {
        return Err(setdiff_error(&SETDIFF_ERROR_NUMERIC_CLASS_MISMATCH));
    }
    let a_shape = a.shape.clone();
    let b_shape = b.shape.clone();
    let a_storage = a
        .into_numeric_storage()
        .map_err(|e| setdiff_internal_error(format!("setdiff: {e}")))?;
    let b_storage = b
        .into_numeric_storage()
        .map_err(|e| setdiff_internal_error(format!("setdiff: {e}")))?;
    match (a_storage, b_storage) {
        (NumericStorage::F64(a), NumericStorage::F64(b)) => {
            setdiff_floating(a, a_shape, b, b_shape, opts)
        }
        (NumericStorage::F32(a), NumericStorage::F32(b)) => {
            setdiff_floating(a, a_shape, b, b_shape, opts)
        }
        (a, b) => setdiff_promoted_f64(a, a_shape, b, b_shape, opts),
    }
}

fn setdiff_promoted_f64(
    a: NumericStorage,
    a_shape: Vec<usize>,
    b: NumericStorage,
    b_shape: Vec<usize>,
    opts: &SetdiffOptions,
) -> crate::BuiltinResult<SetdiffEvaluation> {
    setdiff_floating(
        a.materialize_f64(),
        a_shape,
        b.materialize_f64(),
        b_shape,
        opts,
    )
}

fn setdiff_floating<T: SetFloat>(
    a: Vec<T>,
    a_shape: Vec<usize>,
    b: Vec<T>,
    b_shape: Vec<usize>,
    opts: &SetdiffOptions,
) -> crate::BuiltinResult<SetdiffEvaluation> {
    if opts.rows {
        setdiff_floating_rows(a, a_shape, b, b_shape, opts)
    } else {
        setdiff_floating_elements(a, b, opts)
    }
}

fn setdiff_integer_elements(
    a: &IntegerStorage,
    b: &IntegerStorage,
    opts: &SetdiffOptions,
) -> crate::BuiltinResult<SetdiffEvaluation> {
    let b_values: HashSet<_> = b.exact_values().into_iter().collect();
    let mut seen = HashSet::new();
    let mut entries = Vec::<IntegerDiffEntry>::new();
    for (index, value) in a.exact_values().into_iter().enumerate() {
        if b_values.contains(&value) || !seen.insert(value.clone()) {
            continue;
        }
        let order_rank = entries.len();
        entries.push(IntegerDiffEntry {
            value,
            index,
            order_rank,
        });
    }
    assemble_integer_setdiff(entries, a, opts)
}

fn setdiff_integer_rows(
    a_storage: &IntegerStorage,
    a_shape: Vec<usize>,
    b_storage: &IntegerStorage,
    b_shape: Vec<usize>,
    opts: &SetdiffOptions,
) -> crate::BuiltinResult<SetdiffEvaluation> {
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(setdiff_internal_error(
            "setdiff: 'rows' option requires 2-D numeric matrices",
        ));
    }
    if a_shape[1] != b_shape[1] {
        return Err(setdiff_error(&SETDIFF_ERROR_ROWS_COLUMN_MISMATCH));
    }
    let (rows_a, rows_b, cols) = (a_shape[0], b_shape[0], a_shape[1]);
    let a_values = a_storage.exact_values();
    let b_values = b_storage.exact_values();
    let b_rows: HashSet<Vec<IntValue>> = (0..rows_b)
        .map(|row| {
            (0..cols)
                .map(|col| b_values[row + col * rows_b].clone())
                .collect()
        })
        .collect();
    let mut seen = HashSet::new();
    let mut entries = Vec::<IntegerRowDiffEntry>::new();
    for row in 0..rows_a {
        let row_data: Vec<_> = (0..cols)
            .map(|col| a_values[row + col * rows_a].clone())
            .collect();
        if b_rows.contains(&row_data) || !seen.insert(row_data.clone()) {
            continue;
        }
        let order_rank = entries.len();
        entries.push(IntegerRowDiffEntry {
            row_data,
            row_index: row,
            order_rank,
        });
    }
    assemble_integer_row_setdiff(entries, a_storage, opts, cols)
}

/// Helper exposed for acceleration providers handling numeric tensors entirely on the host.
pub fn setdiff_numeric_from_tensors(
    a: Tensor,
    b: Tensor,
    opts: &SetdiffOptions,
) -> crate::BuiltinResult<SetdiffEvaluation> {
    setdiff_numeric(a, b, opts)
}

fn setdiff_floating_elements<T: SetFloat>(
    a_values: Vec<T>,
    b_values: Vec<T>,
    opts: &SetdiffOptions,
) -> crate::BuiltinResult<SetdiffEvaluation> {
    let mut b_keys: HashSet<u64> = HashSet::new();
    for &value in &b_values {
        b_keys.insert(value.canonical_key());
    }

    let mut seen: HashMap<u64, usize> = HashMap::new();
    let mut entries = Vec::<FloatingDiffEntry<T>>::new();
    let mut order_counter = 0usize;

    for (idx, &value) in a_values.iter().enumerate() {
        let key = value.canonical_key();
        if b_keys.contains(&key) {
            continue;
        }
        if seen.contains_key(&key) {
            continue;
        }
        let entry_idx = entries.len();
        entries.push(FloatingDiffEntry {
            value,
            index: idx,
            order_rank: order_counter,
        });
        seen.insert(key, entry_idx);
        order_counter += 1;
    }

    assemble_floating_setdiff(entries, opts)
}

fn setdiff_floating_rows<T: SetFloat>(
    a_values: Vec<T>,
    a_shape: Vec<usize>,
    b_values: Vec<T>,
    b_shape: Vec<usize>,
    opts: &SetdiffOptions,
) -> crate::BuiltinResult<SetdiffEvaluation> {
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(setdiff_internal_error(
            "setdiff: 'rows' option requires 2-D numeric matrices",
        ));
    }
    if a_shape[1] != b_shape[1] {
        return Err(setdiff_error(&SETDIFF_ERROR_ROWS_COLUMN_MISMATCH));
    }

    let rows_a = a_shape[0];
    let rows_b = b_shape[0];
    let cols = a_shape[1];

    let mut b_keys: HashSet<FloatingRowKey> = HashSet::new();
    for r in 0..rows_b {
        let mut row_values = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r + c * rows_b;
            row_values.push(b_values[idx]);
        }
        b_keys.insert(FloatingRowKey::from_slice(&row_values));
    }

    let mut seen: HashSet<FloatingRowKey> = HashSet::new();
    let mut entries = Vec::<FloatingRowDiffEntry<T>>::new();
    let mut order_counter = 0usize;

    for r in 0..rows_a {
        let mut row_values = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r + c * rows_a;
            row_values.push(a_values[idx]);
        }
        let key = FloatingRowKey::from_slice(&row_values);
        if b_keys.contains(&key) {
            continue;
        }
        if !seen.insert(key) {
            continue;
        }
        entries.push(FloatingRowDiffEntry {
            row_data: row_values,
            row_index: r,
            order_rank: order_counter,
        });
        order_counter += 1;
    }

    assemble_floating_row_setdiff(entries, opts, cols)
}

fn setdiff_complex(
    a: ComplexTensor,
    b: ComplexTensor,
    opts: &SetdiffOptions,
) -> crate::BuiltinResult<SetdiffEvaluation> {
    let a_shape = a.shape.clone();
    let b_shape = b.shape.clone();
    match (a.into_complex_storage(), b.into_complex_storage()) {
        (ComplexStorage::F64(a), ComplexStorage::F64(b)) => {
            setdiff_floating_complex(a, a_shape, b, b_shape, opts)
        }
        (ComplexStorage::F32(a), ComplexStorage::F32(b)) => {
            setdiff_floating_complex(a, a_shape, b, b_shape, opts)
        }
        (a, b) => setdiff_promoted_complex_f64(a, a_shape, b, b_shape, opts),
    }
}

fn setdiff_promoted_complex_f64(
    a: ComplexStorage,
    a_shape: Vec<usize>,
    b: ComplexStorage,
    b_shape: Vec<usize>,
    opts: &SetdiffOptions,
) -> crate::BuiltinResult<SetdiffEvaluation> {
    setdiff_floating_complex(
        a.materialize_f64(),
        a_shape,
        b.materialize_f64(),
        b_shape,
        opts,
    )
}

fn setdiff_floating_complex<T: SetFloat>(
    a: Vec<(T, T)>,
    a_shape: Vec<usize>,
    b: Vec<(T, T)>,
    b_shape: Vec<usize>,
    opts: &SetdiffOptions,
) -> crate::BuiltinResult<SetdiffEvaluation> {
    if opts.rows {
        setdiff_complex_rows(a, a_shape, b, b_shape, opts)
    } else {
        setdiff_complex_elements(a, b, opts)
    }
}

fn setdiff_complex_elements<T: SetFloat>(
    a: Vec<(T, T)>,
    b: Vec<(T, T)>,
    opts: &SetdiffOptions,
) -> crate::BuiltinResult<SetdiffEvaluation> {
    let mut b_keys: HashSet<ComplexKey> = HashSet::new();
    for &value in &b {
        b_keys.insert(ComplexKey::new(value));
    }

    let mut seen: HashSet<ComplexKey> = HashSet::new();
    let mut entries = Vec::<ComplexDiffEntry<T>>::new();
    let mut order_counter = 0usize;

    for (idx, &value) in a.iter().enumerate() {
        let key = ComplexKey::new(value);
        if b_keys.contains(&key) {
            continue;
        }
        if !seen.insert(key) {
            continue;
        }
        entries.push(ComplexDiffEntry {
            value,
            index: idx,
            order_rank: order_counter,
        });
        order_counter += 1;
    }

    assemble_complex_setdiff(entries, opts)
}

fn setdiff_complex_rows<T: SetFloat>(
    a: Vec<(T, T)>,
    a_shape: Vec<usize>,
    b: Vec<(T, T)>,
    b_shape: Vec<usize>,
    opts: &SetdiffOptions,
) -> crate::BuiltinResult<SetdiffEvaluation> {
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(setdiff_internal_error(
            "setdiff: 'rows' option requires 2-D complex matrices",
        ));
    }
    if a_shape[1] != b_shape[1] {
        return Err(setdiff_error(&SETDIFF_ERROR_ROWS_COLUMN_MISMATCH));
    }

    let rows_a = a_shape[0];
    let rows_b = b_shape[0];
    let cols = a_shape[1];

    let mut b_keys: HashSet<Vec<ComplexKey>> = HashSet::new();
    for r in 0..rows_b {
        let mut key_row = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r + c * rows_b;
            key_row.push(ComplexKey::new(b[idx]));
        }
        b_keys.insert(key_row);
    }

    let mut seen: HashSet<Vec<ComplexKey>> = HashSet::new();
    let mut entries = Vec::<ComplexRowDiffEntry<T>>::new();
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
        if b_keys.contains(&key_row) {
            continue;
        }
        if !seen.insert(key_row) {
            continue;
        }
        entries.push(ComplexRowDiffEntry {
            row_data: row_values,
            row_index: r,
            order_rank: order_counter,
        });
        order_counter += 1;
    }

    assemble_complex_row_setdiff(entries, opts, cols)
}

fn setdiff_char(
    a: CharArray,
    b: CharArray,
    opts: &SetdiffOptions,
) -> crate::BuiltinResult<SetdiffEvaluation> {
    if opts.rows {
        setdiff_char_rows(a, b, opts)
    } else {
        setdiff_char_elements(a, b, opts)
    }
}

fn setdiff_char_elements(
    a: CharArray,
    b: CharArray,
    opts: &SetdiffOptions,
) -> crate::BuiltinResult<SetdiffEvaluation> {
    let mut b_keys: HashSet<u32> = HashSet::new();
    for ch in &b.data {
        b_keys.insert(*ch as u32);
    }

    let mut seen: HashSet<u32> = HashSet::new();
    let mut entries = Vec::<CharDiffEntry>::new();
    let mut order_counter = 0usize;

    for col in 0..a.cols {
        for row in 0..a.rows {
            let linear_idx = row + col * a.rows;
            let data_idx = row * a.cols + col;
            let ch = a.data[data_idx];
            let key = ch as u32;
            if b_keys.contains(&key) {
                continue;
            }
            if !seen.insert(key) {
                continue;
            }
            entries.push(CharDiffEntry {
                ch,
                index: linear_idx,
                order_rank: order_counter,
            });
            order_counter += 1;
        }
    }

    assemble_char_setdiff(entries, opts)
}

fn setdiff_char_rows(
    a: CharArray,
    b: CharArray,
    opts: &SetdiffOptions,
) -> crate::BuiltinResult<SetdiffEvaluation> {
    if a.cols != b.cols {
        return Err(setdiff_error(&SETDIFF_ERROR_ROWS_COLUMN_MISMATCH));
    }

    let rows_a = a.rows;
    let rows_b = b.rows;
    let cols = a.cols;

    let mut b_keys: HashSet<RowCharKey> = HashSet::new();
    for r in 0..rows_b {
        let mut row_values = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r * cols + c;
            row_values.push(b.data[idx]);
        }
        b_keys.insert(RowCharKey::from_slice(&row_values));
    }

    let mut seen: HashSet<RowCharKey> = HashSet::new();
    let mut entries = Vec::<CharRowDiffEntry>::new();
    let mut order_counter = 0usize;

    for r in 0..rows_a {
        let mut row_values = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r * cols + c;
            row_values.push(a.data[idx]);
        }
        let key = RowCharKey::from_slice(&row_values);
        if b_keys.contains(&key) {
            continue;
        }
        if !seen.insert(key) {
            continue;
        }
        entries.push(CharRowDiffEntry {
            row_data: row_values,
            row_index: r,
            order_rank: order_counter,
        });
        order_counter += 1;
    }

    assemble_char_row_setdiff(entries, opts, cols)
}

fn setdiff_string(
    a: StringArray,
    b: StringArray,
    opts: &SetdiffOptions,
) -> crate::BuiltinResult<SetdiffEvaluation> {
    if opts.rows {
        setdiff_string_rows(a, b, opts)
    } else {
        setdiff_string_elements(a, b, opts)
    }
}

fn setdiff_string_elements(
    a: StringArray,
    b: StringArray,
    opts: &SetdiffOptions,
) -> crate::BuiltinResult<SetdiffEvaluation> {
    let mut b_keys: HashSet<String> = HashSet::new();
    for value in &b.data {
        b_keys.insert(value.clone());
    }

    let mut seen: HashSet<String> = HashSet::new();
    let mut entries = Vec::<StringDiffEntry>::new();
    let mut order_counter = 0usize;

    for (idx, value) in a.data.iter().enumerate() {
        if b_keys.contains(value) {
            continue;
        }
        if !seen.insert(value.clone()) {
            continue;
        }
        entries.push(StringDiffEntry {
            value: value.clone(),
            index: idx,
            order_rank: order_counter,
        });
        order_counter += 1;
    }

    assemble_string_setdiff(entries, opts)
}

fn setdiff_string_rows(
    a: StringArray,
    b: StringArray,
    opts: &SetdiffOptions,
) -> crate::BuiltinResult<SetdiffEvaluation> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(setdiff_internal_error(
            "setdiff: 'rows' option requires 2-D string arrays",
        ));
    }
    if a.shape[1] != b.shape[1] {
        return Err(setdiff_error(&SETDIFF_ERROR_ROWS_COLUMN_MISMATCH));
    }

    let rows_a = a.shape[0];
    let rows_b = b.shape[0];
    let cols = a.shape[1];

    let mut b_keys: HashSet<RowStringKey> = HashSet::new();
    for r in 0..rows_b {
        let mut row_values = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r + c * rows_b;
            row_values.push(b.data[idx].clone());
        }
        b_keys.insert(RowStringKey(row_values.clone()));
    }

    let mut seen: HashSet<RowStringKey> = HashSet::new();
    let mut entries = Vec::<StringRowDiffEntry>::new();
    let mut order_counter = 0usize;

    for r in 0..rows_a {
        let mut row_values = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r + c * rows_a;
            row_values.push(a.data[idx].clone());
        }
        let key = RowStringKey(row_values.clone());
        if b_keys.contains(&key) {
            continue;
        }
        if !seen.insert(key) {
            continue;
        }
        entries.push(StringRowDiffEntry {
            row_data: row_values,
            row_index: r,
            order_rank: order_counter,
        });
        order_counter += 1;
    }

    assemble_string_row_setdiff(entries, opts, cols)
}

fn assemble_floating_setdiff<T: SetFloat>(
    entries: Vec<FloatingDiffEntry<T>>,
    opts: &SetdiffOptions,
) -> crate::BuiltinResult<SetdiffEvaluation> {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    match opts.order {
        SetdiffOrder::Sorted => {
            order.sort_by(|&lhs, &rhs| entries[lhs].value.compare(entries[rhs].value));
        }
        SetdiffOrder::Stable => {
            order.sort_by_key(|&idx| entries[idx].order_rank);
        }
    }

    let mut values = Vec::with_capacity(order.len());
    let mut ia = Vec::with_capacity(order.len());
    for &idx in &order {
        let entry = &entries[idx];
        values.push(entry.value);
        ia.push((entry.index + 1) as f64);
    }

    let value_tensor =
        Tensor::from_numeric_storage(T::numeric_storage(values), vec![order.len(), 1])
            .map_err(|e| setdiff_internal_error(format!("setdiff: {e}")))?;
    let ia_tensor = Tensor::new(ia, vec![order.len(), 1])
        .map_err(|e| setdiff_internal_error(format!("setdiff: {e}")))?;

    Ok(SetdiffEvaluation::new(
        Value::Tensor(value_tensor),
        ia_tensor,
    ))
}

fn assemble_integer_setdiff(
    entries: Vec<IntegerDiffEntry>,
    storage: &IntegerStorage,
    opts: &SetdiffOptions,
) -> crate::BuiltinResult<SetdiffEvaluation> {
    let mut order: Vec<_> = (0..entries.len()).collect();
    match opts.order {
        SetdiffOrder::Sorted => order.sort_by(|&a, &b| {
            integer_order::compare(&entries[a].value, &entries[b].value, false, false)
        }),
        SetdiffOrder::Stable => order.sort_by_key(|&index| entries[index].order_rank),
    }
    let values: Vec<_> = order
        .iter()
        .map(|&index| entries[index].value.clone())
        .collect();
    let ia: Vec<_> = order
        .iter()
        .map(|&index| (entries[index].index + 1) as f64)
        .collect();
    let values = Tensor::new_integer(
        storage
            .from_exact_values_like(values)
            .map_err(|e| setdiff_internal_error(format!("setdiff: {e}")))?,
        vec![order.len(), 1],
    )
    .map_err(|e| setdiff_internal_error(format!("setdiff: {e}")))?;
    let ia = Tensor::new(ia, vec![order.len(), 1])
        .map_err(|e| setdiff_internal_error(format!("setdiff: {e}")))?;
    Ok(SetdiffEvaluation::new(Value::Tensor(values), ia))
}

fn assemble_floating_row_setdiff<T: SetFloat>(
    entries: Vec<FloatingRowDiffEntry<T>>,
    opts: &SetdiffOptions,
    cols: usize,
) -> crate::BuiltinResult<SetdiffEvaluation> {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    match opts.order {
        SetdiffOrder::Sorted => {
            order.sort_by(|&lhs, &rhs| {
                compare_floating_rows(&entries[lhs].row_data, &entries[rhs].row_data)
            });
        }
        SetdiffOrder::Stable => {
            order.sort_by_key(|&idx| entries[idx].order_rank);
        }
    }

    let unique_rows = order.len();
    let mut values = vec![T::default(); unique_rows * cols];
    let mut ia = Vec::with_capacity(unique_rows);

    for (row_pos, &entry_idx) in order.iter().enumerate() {
        let entry = &entries[entry_idx];
        for col in 0..cols {
            let dest = row_pos + col * unique_rows;
            values[dest] = entry.row_data[col];
        }
        ia.push((entry.row_index + 1) as f64);
    }

    let value_tensor =
        Tensor::from_numeric_storage(T::numeric_storage(values), vec![unique_rows, cols])
            .map_err(|e| setdiff_internal_error(format!("setdiff: {e}")))?;
    let ia_tensor = Tensor::new(ia, vec![unique_rows, 1])
        .map_err(|e| setdiff_internal_error(format!("setdiff: {e}")))?;

    Ok(SetdiffEvaluation::new(
        Value::Tensor(value_tensor),
        ia_tensor,
    ))
}

fn assemble_integer_row_setdiff(
    entries: Vec<IntegerRowDiffEntry>,
    storage: &IntegerStorage,
    opts: &SetdiffOptions,
    cols: usize,
) -> crate::BuiltinResult<SetdiffEvaluation> {
    let mut order: Vec<_> = (0..entries.len()).collect();
    match opts.order {
        SetdiffOrder::Sorted => order.sort_by(|&a, &b| {
            for (left, right) in entries[a].row_data.iter().zip(&entries[b].row_data) {
                let ordering = integer_order::compare(left, right, false, false);
                if ordering != Ordering::Equal {
                    return ordering;
                }
            }
            Ordering::Equal
        }),
        SetdiffOrder::Stable => order.sort_by_key(|&index| entries[index].order_rank),
    }
    let rows = order.len();
    let mut values = Vec::with_capacity(rows * cols);
    for col in 0..cols {
        for &index in &order {
            values.push(entries[index].row_data[col].clone());
        }
    }
    let ia: Vec<_> = order
        .iter()
        .map(|&index| (entries[index].row_index + 1) as f64)
        .collect();
    let values = Tensor::new_integer(
        storage
            .from_exact_values_like(values)
            .map_err(|e| setdiff_internal_error(format!("setdiff: {e}")))?,
        vec![rows, cols],
    )
    .map_err(|e| setdiff_internal_error(format!("setdiff: {e}")))?;
    let ia = Tensor::new(ia, vec![rows, 1])
        .map_err(|e| setdiff_internal_error(format!("setdiff: {e}")))?;
    Ok(SetdiffEvaluation::new(Value::Tensor(values), ia))
}

fn assemble_complex_setdiff<T: SetFloat>(
    entries: Vec<ComplexDiffEntry<T>>,
    opts: &SetdiffOptions,
) -> crate::BuiltinResult<SetdiffEvaluation> {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    match opts.order {
        SetdiffOrder::Sorted => {
            order.sort_by(|&lhs, &rhs| compare_complex(entries[lhs].value, entries[rhs].value));
        }
        SetdiffOrder::Stable => {
            order.sort_by_key(|&idx| entries[idx].order_rank);
        }
    }

    let mut values = Vec::with_capacity(order.len());
    let mut ia = Vec::with_capacity(order.len());
    for &idx in &order {
        let entry = &entries[idx];
        values.push(entry.value);
        ia.push((entry.index + 1) as f64);
    }

    let value_tensor =
        ComplexTensor::from_complex_storage(T::complex_storage(values), vec![order.len(), 1])
            .map_err(|e| setdiff_internal_error(format!("setdiff: {e}")))?;
    let ia_tensor = Tensor::new(ia, vec![order.len(), 1])
        .map_err(|e| setdiff_internal_error(format!("setdiff: {e}")))?;

    let value = if value_tensor.as_f32_slice().is_some() {
        Value::ComplexTensor(value_tensor)
    } else {
        complex_tensor_into_value(value_tensor)
    };
    Ok(SetdiffEvaluation::new(value, ia_tensor))
}

fn assemble_complex_row_setdiff<T: SetFloat>(
    entries: Vec<ComplexRowDiffEntry<T>>,
    opts: &SetdiffOptions,
    cols: usize,
) -> crate::BuiltinResult<SetdiffEvaluation> {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    match opts.order {
        SetdiffOrder::Sorted => {
            order.sort_by(|&lhs, &rhs| {
                compare_complex_rows(&entries[lhs].row_data, &entries[rhs].row_data)
            });
        }
        SetdiffOrder::Stable => {
            order.sort_by_key(|&idx| entries[idx].order_rank);
        }
    }

    let unique_rows = order.len();
    let mut values = vec![(T::default(), T::default()); unique_rows * cols];
    let mut ia = Vec::with_capacity(unique_rows);

    for (row_pos, &entry_idx) in order.iter().enumerate() {
        let entry = &entries[entry_idx];
        for col in 0..cols {
            let dest = row_pos + col * unique_rows;
            values[dest] = entry.row_data[col];
        }
        ia.push((entry.row_index + 1) as f64);
    }

    let value_tensor =
        ComplexTensor::from_complex_storage(T::complex_storage(values), vec![unique_rows, cols])
            .map_err(|e| setdiff_internal_error(format!("setdiff: {e}")))?;
    let ia_tensor = Tensor::new(ia, vec![unique_rows, 1])
        .map_err(|e| setdiff_internal_error(format!("setdiff: {e}")))?;

    let value = if value_tensor.as_f32_slice().is_some() {
        Value::ComplexTensor(value_tensor)
    } else {
        complex_tensor_into_value(value_tensor)
    };
    Ok(SetdiffEvaluation::new(value, ia_tensor))
}

fn assemble_char_setdiff(
    entries: Vec<CharDiffEntry>,
    opts: &SetdiffOptions,
) -> crate::BuiltinResult<SetdiffEvaluation> {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    match opts.order {
        SetdiffOrder::Sorted => {
            order.sort_by(|&lhs, &rhs| entries[lhs].ch.cmp(&entries[rhs].ch));
        }
        SetdiffOrder::Stable => {
            order.sort_by_key(|&idx| entries[idx].order_rank);
        }
    }

    let mut values = Vec::with_capacity(order.len());
    let mut ia = Vec::with_capacity(order.len());
    for &idx in &order {
        let entry = &entries[idx];
        values.push(entry.ch);
        ia.push((entry.index + 1) as f64);
    }

    let value_array = CharArray::new(values, order.len(), 1)
        .map_err(|e| setdiff_internal_error(format!("setdiff: {e}")))?;
    let ia_tensor = Tensor::new(ia, vec![order.len(), 1])
        .map_err(|e| setdiff_internal_error(format!("setdiff: {e}")))?;

    Ok(SetdiffEvaluation::new(
        Value::CharArray(value_array),
        ia_tensor,
    ))
}

fn assemble_char_row_setdiff(
    entries: Vec<CharRowDiffEntry>,
    opts: &SetdiffOptions,
    cols: usize,
) -> crate::BuiltinResult<SetdiffEvaluation> {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    match opts.order {
        SetdiffOrder::Sorted => {
            order.sort_by(|&lhs, &rhs| {
                compare_char_rows(&entries[lhs].row_data, &entries[rhs].row_data)
            });
        }
        SetdiffOrder::Stable => {
            order.sort_by_key(|&idx| entries[idx].order_rank);
        }
    }

    let unique_rows = order.len();
    let mut values = vec!['\0'; unique_rows * cols];
    let mut ia = Vec::with_capacity(unique_rows);

    for (row_pos, &entry_idx) in order.iter().enumerate() {
        let entry = &entries[entry_idx];
        for col in 0..cols {
            let dest = row_pos * cols + col;
            values[dest] = entry.row_data[col];
        }
        ia.push((entry.row_index + 1) as f64);
    }

    let value_array = CharArray::new(values, unique_rows, cols)
        .map_err(|e| setdiff_internal_error(format!("setdiff: {e}")))?;
    let ia_tensor = Tensor::new(ia, vec![unique_rows, 1])
        .map_err(|e| setdiff_internal_error(format!("setdiff: {e}")))?;

    Ok(SetdiffEvaluation::new(
        Value::CharArray(value_array),
        ia_tensor,
    ))
}

fn assemble_string_setdiff(
    entries: Vec<StringDiffEntry>,
    opts: &SetdiffOptions,
) -> crate::BuiltinResult<SetdiffEvaluation> {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    match opts.order {
        SetdiffOrder::Sorted => {
            order.sort_by(|&lhs, &rhs| entries[lhs].value.cmp(&entries[rhs].value));
        }
        SetdiffOrder::Stable => {
            order.sort_by_key(|&idx| entries[idx].order_rank);
        }
    }

    let mut values = Vec::with_capacity(order.len());
    let mut ia = Vec::with_capacity(order.len());
    for &idx in &order {
        let entry = &entries[idx];
        values.push(entry.value.clone());
        ia.push((entry.index + 1) as f64);
    }

    let value_array = StringArray::new(values, vec![order.len(), 1])
        .map_err(|e| setdiff_internal_error(format!("setdiff: {e}")))?;
    let ia_tensor = Tensor::new(ia, vec![order.len(), 1])
        .map_err(|e| setdiff_internal_error(format!("setdiff: {e}")))?;

    Ok(SetdiffEvaluation::new(
        Value::StringArray(value_array),
        ia_tensor,
    ))
}

fn assemble_string_row_setdiff(
    entries: Vec<StringRowDiffEntry>,
    opts: &SetdiffOptions,
    cols: usize,
) -> crate::BuiltinResult<SetdiffEvaluation> {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    match opts.order {
        SetdiffOrder::Sorted => {
            order.sort_by(|&lhs, &rhs| {
                compare_string_rows(&entries[lhs].row_data, &entries[rhs].row_data)
            });
        }
        SetdiffOrder::Stable => {
            order.sort_by_key(|&idx| entries[idx].order_rank);
        }
    }

    let unique_rows = order.len();
    let mut values = vec![String::new(); unique_rows * cols];
    let mut ia = Vec::with_capacity(unique_rows);

    for (row_pos, &entry_idx) in order.iter().enumerate() {
        let entry = &entries[entry_idx];
        for col in 0..cols {
            let dest = row_pos + col * unique_rows;
            values[dest] = entry.row_data[col].clone();
        }
        ia.push((entry.row_index + 1) as f64);
    }

    let value_array = StringArray::new(values, vec![unique_rows, cols])
        .map_err(|e| setdiff_internal_error(format!("setdiff: {e}")))?;
    let ia_tensor = Tensor::new(ia, vec![unique_rows, 1])
        .map_err(|e| setdiff_internal_error(format!("setdiff: {e}")))?;

    Ok(SetdiffEvaluation::new(
        Value::StringArray(value_array),
        ia_tensor,
    ))
}

#[derive(Clone, Copy, Debug)]
struct FloatingDiffEntry<T> {
    value: T,
    index: usize,
    order_rank: usize,
}

#[derive(Clone, Debug)]
struct IntegerDiffEntry {
    value: IntValue,
    index: usize,
    order_rank: usize,
}

#[derive(Clone, Debug)]
struct FloatingRowDiffEntry<T> {
    row_data: Vec<T>,
    row_index: usize,
    order_rank: usize,
}

#[derive(Clone, Debug)]
struct IntegerRowDiffEntry {
    row_data: Vec<IntValue>,
    row_index: usize,
    order_rank: usize,
}

#[derive(Clone, Copy, Debug)]
struct ComplexDiffEntry<T> {
    value: (T, T),
    index: usize,
    order_rank: usize,
}

#[derive(Clone, Debug)]
struct ComplexRowDiffEntry<T> {
    row_data: Vec<(T, T)>,
    row_index: usize,
    order_rank: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CharDiffEntry {
    ch: char,
    index: usize,
    order_rank: usize,
}

#[derive(Clone, Debug)]
struct CharRowDiffEntry {
    row_data: Vec<char>,
    row_index: usize,
    order_rank: usize,
}

#[derive(Clone, Debug)]
struct StringDiffEntry {
    value: String,
    index: usize,
    order_rank: usize,
}

#[derive(Clone, Debug)]
struct StringRowDiffEntry {
    row_data: Vec<String>,
    row_index: usize,
    order_rank: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FloatingRowKey(Vec<u64>);

impl FloatingRowKey {
    fn from_slice<T: SetFloat>(values: &[T]) -> Self {
        FloatingRowKey(values.iter().map(|&value| value.canonical_key()).collect())
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

#[derive(Debug)]
pub struct SetdiffEvaluation {
    values: Value,
    ia: Tensor,
}

impl SetdiffEvaluation {
    fn new(values: Value, ia: Tensor) -> Self {
        Self { values, ia }
    }

    pub fn from_setdiff_result(result: SetdiffResult) -> crate::BuiltinResult<Self> {
        let SetdiffResult { values, ia } = result;
        let values_tensor = Tensor::new(values.data, values.shape)
            .map_err(|e| setdiff_internal_error(format!("setdiff: {e}")))?;
        let ia_tensor = Tensor::new(ia.data, ia.shape)
            .map_err(|e| setdiff_internal_error(format!("setdiff: {e}")))?;
        Ok(SetdiffEvaluation::new(
            Value::Tensor(values_tensor),
            ia_tensor,
        ))
    }

    pub fn into_numeric_setdiff_result(self) -> crate::BuiltinResult<SetdiffResult> {
        let SetdiffEvaluation { values, ia } = self;
        let values_tensor = tensor::value_into_tensor_for("setdiff", values)
            .map_err(|e| setdiff_internal_error(e))?;
        Ok(SetdiffResult {
            values: tensor::tensor_into_host_f64_owned(values_tensor),
            ia: tensor::tensor_into_host_f64_owned(ia),
        })
    }

    pub fn into_values_value(self) -> Value {
        self.values
    }

    pub fn into_pair(self) -> (Value, Value) {
        let ia = tensor::tensor_into_value(self.ia);
        (self.values, ia)
    }

    pub fn values_value(&self) -> Value {
        self.values.clone()
    }

    pub fn ia_value(&self) -> Value {
        tensor::tensor_into_value(self.ia.clone())
    }
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
    use runmat_builtins::{ResolveContext, Type};
    use runmat_value::{CharArray, StringArray, Tensor, Value};

    fn evaluate_sync(
        a: Value,
        b: Value,
        rest: &[Value],
    ) -> crate::BuiltinResult<SetdiffEvaluation> {
        futures::executor::block_on(evaluate(a, b, rest))
    }

    fn builtin_sync(a: Value, b: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        futures::executor::block_on(setdiff_builtin(a, b, rest))
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
                let _guard = crate::output_count::push_output_count(Some(2));
                let Value::OutputList(outputs) =
                    builtin_sync(left, right, Vec::new()).expect("resident setdiff")
                else {
                    panic!("expected output list");
                };
                assert_eq!(outputs.len(), 2);
                assert!(outputs
                    .iter()
                    .all(|output| matches!(output, Value::GpuTensor(_))));
                assert_eq!(
                    test_support::gather(outputs[0].clone())
                        .expect("gather values")
                        .integer_storage(),
                    Some(&IntegerStorage::I32(vec![9]))
                );
            }

            let _guard = crate::output_count::push_output_count(Some(3));
            let err = builtin_sync(Value::Num(1.0), Value::Num(1.0), Vec::new())
                .expect_err("excess outputs must fail");
            assert_eq!(err.identifier(), SETDIFF_ERROR_INVALID_ARGUMENT.identifier);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn setdiff_numeric_sorted_default() {
        let a = Tensor::new(vec![5.0, 7.0, 5.0, 1.0], vec![4, 1]).unwrap();
        let b = Tensor::new(vec![7.0, 1.0, 3.0], vec![3, 1]).unwrap();
        let eval = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[]).expect("setdiff");
        match eval.values_value() {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 1]);
                assert_eq!(t.materialize_f64(), vec![5.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        let ia = tensor::value_into_tensor_for("setdiff", eval.ia_value()).expect("ia tensor");
        assert_eq!(ia.materialize_f64(), vec![1.0]);
    }

    #[test]
    fn setdiff_preserves_native_single_elements_and_rows() {
        let a = Tensor::from_f32(vec![5.0, 7.0, 1.0], vec![3, 1]).unwrap();
        let b = Tensor::from_f32(vec![7.0, 1.0], vec![2, 1]).unwrap();
        let values = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[])
            .expect("single setdiff")
            .into_values_value();
        let Value::Tensor(values) = values else {
            panic!("expected native single values");
        };
        assert_eq!(
            values.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![5.0])
        );

        let a = Tensor::from_f32(vec![1.0, 3.0, 1.0, 2.0, 4.0, 2.0], vec![3, 2]).unwrap();
        let b = Tensor::from_f32(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let values = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[Value::from("rows")])
            .expect("single row setdiff")
            .into_values_value();
        let Value::Tensor(values) = values else {
            panic!("expected native single rows");
        };
        assert_eq!(values.shape, vec![1, 2]);
        assert_eq!(
            values.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![1.0, 2.0])
        );
    }

    #[test]
    fn setdiff_preserves_native_complex_single_elements_and_rows() {
        let a = ComplexTensor::from_f32(vec![(1.0, 1.0), (2.0, 0.0)], vec![2, 1]).unwrap();
        let b = ComplexTensor::from_f32(vec![(2.0, 0.0)], vec![1, 1]).unwrap();
        let values = evaluate_sync(Value::ComplexTensor(a), Value::ComplexTensor(b), &[])
            .expect("complex single setdiff")
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
        let b = ComplexTensor::from_f32(vec![(3.0, 0.0), (4.0, 1.0)], vec![1, 2]).unwrap();
        let values = evaluate_sync(
            Value::ComplexTensor(a),
            Value::ComplexTensor(b),
            &[Value::from("rows")],
        )
        .expect("complex single row setdiff")
        .into_values_value();
        let Value::ComplexTensor(values) = values else {
            panic!("expected native complex single rows");
        };
        assert_eq!(values.shape, vec![1, 2]);
        assert_eq!(values.as_f32_slice(), Some(&[(1.0, 0.0), (2.0, 1.0)][..]));
    }

    #[test]
    fn setdiff_preserves_exact_integer_elements_and_rows() {
        let a = Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![u64::MAX, 0, 9_007_199_254_740_993]),
            vec![3, 1],
        )
        .expect("input");
        let b = Tensor::new_integer(runmat_value::IntegerStorage::U64(vec![0]), vec![1, 1])
            .expect("input");
        let (values, ia) = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[])
            .expect("setdiff")
            .into_pair();
        let Value::Tensor(values) = values else {
            panic!("exact values");
        };
        assert_eq!(
            values.integer_storage(),
            Some(&runmat_value::IntegerStorage::U64(vec![
                9_007_199_254_740_993,
                u64::MAX
            ]))
        );
        let ia = tensor::value_into_tensor_for("setdiff", ia).expect("indices");
        assert_eq!(ia.materialize_f64(), vec![3.0, 1.0]);

        let a = Tensor::new_integer(
            runmat_value::IntegerStorage::I64(vec![i64::MAX, 4, 0, 2]),
            vec![2, 2],
        )
        .expect("rows input");
        let b = Tensor::new_integer(
            runmat_value::IntegerStorage::I64(vec![i64::MAX, 0]),
            vec![1, 2],
        )
        .expect("rows input");
        let (values, ia) =
            evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[Value::from("rows")])
                .expect("setdiff rows")
                .into_pair();
        let Value::Tensor(values) = values else {
            panic!("exact row values");
        };
        assert_eq!(
            values.integer_storage(),
            Some(&runmat_value::IntegerStorage::I64(vec![4, 2]))
        );
        let ia = tensor::value_into_tensor_for("setdiff", ia).expect("row indices");
        assert_eq!(ia.materialize_f64(), vec![2.0]);
    }

    #[test]
    fn setdiff_rejects_mixed_nondouble_integer_classes() {
        let a = Tensor::new_integer(runmat_value::IntegerStorage::U16(vec![7, 2, 9]), vec![3, 1])
            .expect("input");
        let b = Tensor::new_integer(runmat_value::IntegerStorage::I32(vec![2]), vec![1, 1])
            .expect("input");
        let error = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[])
            .expect_err("mixed integer classes must reject");
        assert_eq!(
            error.identifier(),
            SETDIFF_ERROR_NUMERIC_CLASS_MISMATCH.identifier
        );
    }

    #[test]
    fn setdiff_type_resolver_numeric() {
        assert_eq!(
            set_values_output_type(
                &[Type::tensor(), Type::tensor()],
                &ResolveContext::new(Vec::new()),
            ),
            Type::tensor()
        );
    }

    #[test]
    fn setdiff_type_resolver_string_array() {
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
    fn setdiff_numeric_stable() {
        let a = Tensor::new(vec![4.0, 2.0, 4.0, 1.0, 3.0], vec![5, 1]).unwrap();
        let b = Tensor::new(vec![3.0, 4.0, 5.0, 1.0], vec![4, 1]).unwrap();
        let eval = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[Value::from("stable")])
            .expect("setdiff");
        match eval.values_value() {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 1]);
                assert_eq!(t.materialize_f64(), vec![2.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        let ia = tensor::value_into_tensor_for("setdiff", eval.ia_value()).expect("ia tensor");
        assert_eq!(ia.materialize_f64(), vec![2.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn setdiff_numeric_rows_sorted() {
        let a = Tensor::new(vec![1.0, 3.0, 1.0, 2.0, 4.0, 2.0], vec![3, 2]).unwrap();
        let b = Tensor::new(vec![3.0, 5.0, 4.0, 6.0], vec![2, 2]).unwrap();
        let eval = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[Value::from("rows")])
            .expect("setdiff");
        match eval.values_value() {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.materialize_f64(), vec![1.0, 2.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        let ia = tensor::value_into_tensor_for("setdiff", eval.ia_value()).expect("ia tensor");
        assert_eq!(ia.materialize_f64(), vec![1.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn setdiff_numeric_removes_nan() {
        let a = Tensor::new(vec![f64::NAN, 2.0, 3.0], vec![3, 1]).unwrap();
        let b = Tensor::new(vec![f64::NAN], vec![1, 1]).unwrap();
        let eval = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[]).expect("setdiff");
        let values = tensor::value_into_tensor_for("setdiff", eval.values_value()).expect("values");
        assert_eq!(values.materialize_f64(), vec![2.0, 3.0]);
        let ia = tensor::value_into_tensor_for("setdiff", eval.ia_value()).expect("ia tensor");
        assert_eq!(ia.materialize_f64(), vec![2.0, 3.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn setdiff_char_elements() {
        let a = CharArray::new(vec!['m', 'z', 'm', 'a'], 2, 2).unwrap();
        let b = CharArray::new(vec!['a', 'x', 'm', 'a'], 2, 2).unwrap();
        let eval = evaluate_sync(Value::CharArray(a), Value::CharArray(b), &[]).expect("setdiff");
        match eval.values_value() {
            Value::CharArray(arr) => {
                assert_eq!(arr.rows, 1);
                assert_eq!(arr.cols, 1);
                assert_eq!(arr.data, vec!['z']);
            }
            other => panic!("expected char array, got {other:?}"),
        }
        let ia = tensor::value_into_tensor_for("setdiff", eval.ia_value()).expect("ia tensor");
        assert_eq!(ia.materialize_f64(), vec![3.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn setdiff_string_rows_stable() {
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
        .expect("setdiff");
        match eval.values_value() {
            Value::StringArray(arr) => {
                assert_eq!(arr.shape, vec![1, 2]);
                assert_eq!(arr.data, vec!["alpha".to_string(), "beta".to_string()]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
        let ia = tensor::value_into_tensor_for("setdiff", eval.ia_value()).expect("ia tensor");
        assert_eq!(ia.materialize_f64(), vec![1.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn setdiff_type_mismatch_errors() {
        let err = evaluate_sync(Value::from(1.0), Value::String("a".into()), &[]).unwrap_err();
        assert_eq!(
            err.identifier(),
            SETDIFF_ERROR_UNSUPPORTED_INPUT_TYPE.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn setdiff_rows_dimension_mismatch_reports_identifier() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("tensor a");
        let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).expect("tensor b");
        let err =
            evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[Value::from("rows")]).unwrap_err();
        assert_eq!(
            err.identifier(),
            SETDIFF_ERROR_ROWS_COLUMN_MISMATCH.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn setdiff_rejects_legacy_option() {
        let err = evaluate_sync(Value::from(1.0), Value::from(2.0), &[Value::from("legacy")])
            .unwrap_err();
        assert_eq!(
            err.identifier(),
            SETDIFF_ERROR_LEGACY_OPTION_UNSUPPORTED.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn setdiff_rejects_conflicting_order_options() {
        let err = evaluate_sync(
            Value::from(1.0),
            Value::from(2.0),
            &[Value::from("stable"), Value::from("sorted")],
        )
        .unwrap_err();
        assert_eq!(
            err.identifier(),
            SETDIFF_ERROR_CONFLICTING_ORDER_OPTIONS.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn setdiff_rejects_unknown_option() {
        let err =
            evaluate_sync(Value::from(1.0), Value::from(2.0), &[Value::from("bogus")]).unwrap_err();
        assert_eq!(err.identifier(), SETDIFF_ERROR_UNKNOWN_OPTION.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn setdiff_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor_a = Tensor::new(vec![10.0, 4.0, 6.0, 4.0], vec![4, 1]).unwrap();
            let tensor_b = Tensor::new(vec![6.0, 4.0, 2.0], vec![3, 1]).unwrap();
            let view_a = HostTensorView {
                data: &tensor_a.materialize_f64(),
                shape: &tensor_a.shape,
            };
            let view_b = HostTensorView {
                data: &tensor_b.materialize_f64(),
                shape: &tensor_b.shape,
            };
            let handle_a = provider.upload(&view_a).expect("upload a");
            let handle_b = provider.upload(&view_b).expect("upload b");
            let eval = evaluate_sync(Value::GpuTensor(handle_a), Value::GpuTensor(handle_b), &[])
                .expect("setdiff");
            match eval.values_value() {
                Value::Tensor(t) => {
                    assert_eq!(t.materialize_f64(), vec![10.0]);
                }
                other => panic!("expected tensor result, got {other:?}"),
            }
            let ia = tensor::value_into_tensor_for("setdiff", eval.ia_value()).expect("ia tensor");
            assert_eq!(ia.materialize_f64(), vec![1.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn setdiff_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let a = Tensor::new(vec![8.0, 4.0, 2.0, 4.0], vec![4, 1]).unwrap();
        let b = Tensor::new(vec![2.0, 5.0], vec![2, 1]).unwrap();

        let cpu_eval = evaluate_sync(Value::Tensor(a.clone()), Value::Tensor(b.clone()), &[])
            .expect("setdiff");
        let cpu_values = tensor::value_into_tensor_for("setdiff", cpu_eval.values_value()).unwrap();
        let cpu_ia = tensor::value_into_tensor_for("setdiff", cpu_eval.ia_value()).unwrap();

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
            .expect("setdiff");
        let gpu_values = tensor::value_into_tensor_for("setdiff", gpu_eval.values_value()).unwrap();
        let gpu_ia = tensor::value_into_tensor_for("setdiff", gpu_eval.ia_value()).unwrap();

        assert_eq!(gpu_values.materialize_f64(), cpu_values.materialize_f64());
        assert_eq!(gpu_values.shape, cpu_values.shape);
        assert_eq!(gpu_ia.materialize_f64(), cpu_ia.materialize_f64());
        assert_eq!(gpu_ia.shape, cpu_ia.shape);
    }
}
