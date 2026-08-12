//! MATLAB-compatible `intersect` builtin with GPU-aware semantics for RunMat.
//!
//! Supports element-wise and row-wise intersections with optional stable ordering,
//! and index outputs that mirror MathWorks MATLAB semantics. GPU tensors use
//! typed host fallback and their public outputs are restored to the owning
//! provider.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use runmat_accelerate_api::GpuTensorHandle;
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
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::math::elementwise::integer_cast::IntegerTarget;

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::array::sorting_sets::intersect"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "intersect",
    op_kind: GpuOpKind::Custom("intersect"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: true,
    notes: "Exact typed fallback gathers when needed and restores intersection values plus double indices to the input owner.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::array::sorting_sets::intersect"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "intersect",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "`intersect` materialises its inputs and terminates fusion chains; upstream GPU tensors are gathered when necessary.",
};

const BUILTIN_NAME: &str = "intersect";

const INTERSECT_OUTPUT_C: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Intersection values or rows.",
}];

const INTERSECT_OUTPUT_C_IA: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Intersection values or rows.",
    },
    BuiltinParamDescriptor {
        name: "ia",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indices selecting matching elements/rows in A.",
    },
];

const INTERSECT_OUTPUT_C_IA_IB: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Intersection values or rows.",
    },
    BuiltinParamDescriptor {
        name: "ia",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indices selecting matching elements/rows in A.",
    },
    BuiltinParamDescriptor {
        name: "ib",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indices selecting matching elements/rows in B.",
    },
];

const INTERSECT_INPUTS_A_B: [BuiltinParamDescriptor; 2] = [
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

const INTERSECT_INPUTS_A_B_OPTIONS: [BuiltinParamDescriptor; 3] = [
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

const INTERSECT_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "C = intersect(A, B)",
        inputs: &INTERSECT_INPUTS_A_B,
        outputs: &INTERSECT_OUTPUT_C,
    },
    BuiltinSignatureDescriptor {
        label: "C = intersect(A, B, option...)",
        inputs: &INTERSECT_INPUTS_A_B_OPTIONS,
        outputs: &INTERSECT_OUTPUT_C,
    },
    BuiltinSignatureDescriptor {
        label: "[C, ia] = intersect(A, B)",
        inputs: &INTERSECT_INPUTS_A_B,
        outputs: &INTERSECT_OUTPUT_C_IA,
    },
    BuiltinSignatureDescriptor {
        label: "[C, ia] = intersect(A, B, option...)",
        inputs: &INTERSECT_INPUTS_A_B_OPTIONS,
        outputs: &INTERSECT_OUTPUT_C_IA,
    },
    BuiltinSignatureDescriptor {
        label: "[C, ia, ib] = intersect(A, B)",
        inputs: &INTERSECT_INPUTS_A_B,
        outputs: &INTERSECT_OUTPUT_C_IA_IB,
    },
    BuiltinSignatureDescriptor {
        label: "[C, ia, ib] = intersect(A, B, option...)",
        inputs: &INTERSECT_INPUTS_A_B_OPTIONS,
        outputs: &INTERSECT_OUTPUT_C_IA_IB,
    },
];

const INTERSECT_ERROR_LEGACY_OPTION_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTERSECT.LEGACY_OPTION_UNSUPPORTED",
    identifier: Some("RunMat:intersect:LegacyOptionUnsupported"),
    when: "Legacy compatibility options are requested.",
    message: "intersect: the 'legacy' behaviour is not supported",
};

const INTERSECT_ERROR_CONFLICTING_ORDER_OPTIONS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTERSECT.CONFLICTING_ORDER_OPTIONS",
    identifier: Some("RunMat:intersect:ConflictingOrderOptions"),
    when: "Both 'sorted' and 'stable' options are provided.",
    message: "intersect: cannot combine 'sorted' with 'stable'",
};

const INTERSECT_ERROR_UNKNOWN_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTERSECT.UNKNOWN_OPTION",
    identifier: Some("RunMat:intersect:UnknownOption"),
    when: "An unsupported option token is provided.",
    message: "intersect: unrecognised option",
};

const INTERSECT_ERROR_ROWS_COLUMN_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTERSECT.ROWS_COLUMN_MISMATCH",
    identifier: Some("RunMat:intersect:RowsColumnMismatch"),
    when: "'rows' mode is used and column counts differ.",
    message: "intersect: inputs must have the same number of columns when using 'rows'",
};

const INTERSECT_ERROR_UNSUPPORTED_INPUT_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTERSECT.UNSUPPORTED_INPUT_TYPE",
    identifier: Some("RunMat:intersect:UnsupportedInputType"),
    when: "Input values cannot be converted into supported intersect domains.",
    message: "intersect: unsupported input type",
};

const INTERSECT_ERROR_NUMERIC_CLASS_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTERSECT.NUMERIC_CLASS_MISMATCH",
    identifier: Some("RunMat:intersect:NumericClassMismatch"),
    when: "Numeric inputs have incompatible nondouble classes.",
    message: "intersect: numeric inputs must have the same class, except double may be combined with one nondouble class",
};

const INTERSECT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTERSECT.INVALID_ARGUMENT",
    identifier: Some("RunMat:intersect:InvalidArgument"),
    when: "Option arguments are not string-like where required.",
    message: "intersect: expected string option arguments",
};

const INTERSECT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTERSECT.INTERNAL",
    identifier: Some("RunMat:intersect:Internal"),
    when: "Internal conversion/allocation/provider decode fails.",
    message: "intersect: internal operation failed",
};

const INTERSECT_ERRORS: [BuiltinErrorDescriptor; 8] = [
    INTERSECT_ERROR_LEGACY_OPTION_UNSUPPORTED,
    INTERSECT_ERROR_CONFLICTING_ORDER_OPTIONS,
    INTERSECT_ERROR_UNKNOWN_OPTION,
    INTERSECT_ERROR_ROWS_COLUMN_MISMATCH,
    INTERSECT_ERROR_UNSUPPORTED_INPUT_TYPE,
    INTERSECT_ERROR_NUMERIC_CLASS_MISMATCH,
    INTERSECT_ERROR_INVALID_ARGUMENT,
    INTERSECT_ERROR_INTERNAL,
];

const INTERSECT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[C, ia, ib] = intersect(integer_A, integer_B, options)",
        inputs: &super::BINARY_SET_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "C preserves the common nondouble integer class, including when paired with double; ia and ib are one-based double. GPU supports integer classes through 32 bits and restores outputs after typed fallback.",
    }];

pub const INTERSECT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &INTERSECT_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &INTERSECT_ERRORS,
};

fn intersect_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn intersect_error(error: &'static BuiltinErrorDescriptor) -> crate::RuntimeError {
    intersect_error_with(error, error.message)
}

fn intersect_internal_error(message: impl Into<String>) -> crate::RuntimeError {
    intersect_error_with(&INTERSECT_ERROR_INTERNAL, message)
}

#[runtime_builtin(
    name = "intersect",
    category = "array/sorting_sets",
    summary = "Return common elements or rows across arrays with index outputs.",
    keywords = "intersect,set,stable,rows,indices,gpu",
    accel = "array_construct",
    sink = true,
    type_resolver(set_values_output_type),
    descriptor(crate::builtins::array::sorting_sets::intersect::INTERSECT_DESCRIPTOR),
    integer_capabilities(INTERSECT_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::array::sorting_sets::intersect"
)]
async fn intersect_builtin(a: Value, b: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 3) {
        return Err(intersect_error_with(
            &INTERSECT_ERROR_INVALID_ARGUMENT,
            "intersect: too many output arguments; maximum is 3",
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
                intersect_internal_error,
            )?;
            return Ok(Value::OutputList(outputs));
        }
        if out_count == 2 {
            let (values, ia) = eval.into_pair();
            let outputs = super::restore_set_outputs(
                provider,
                BUILTIN_NAME,
                vec![values, ia],
                intersect_internal_error,
            )?;
            return Ok(Value::OutputList(outputs));
        }
        let (values, ia, ib) = eval.into_triple();
        let outputs = super::restore_set_outputs(
            provider,
            BUILTIN_NAME,
            vec![values, ia, ib],
            intersect_internal_error,
        )?;
        return Ok(Value::OutputList(outputs));
    }
    let mut outputs = super::restore_set_outputs(
        provider,
        BUILTIN_NAME,
        vec![eval.into_values_value()],
        intersect_internal_error,
    )?;
    Ok(outputs.pop().expect("intersect output"))
}

/// Evaluate the `intersect` builtin once and expose all outputs.
pub async fn evaluate(
    a: Value,
    b: Value,
    rest: &[Value],
) -> crate::BuiltinResult<IntersectEvaluation> {
    crate::builtins::common::validation::reject_typed_complex_integer(&a, "intersect")?;
    crate::builtins::common::validation::reject_typed_complex_integer(&b, "intersect")?;
    let opts = parse_options(rest)?;
    for value in [&a, &b] {
        if let Value::GpuTensor(handle) = value {
            if super::is_unsupported_set_gpu_integer(handle) {
                return Err(intersect_error_with(
                    &INTERSECT_ERROR_UNSUPPORTED_INPUT_TYPE,
                    "intersect: resident 64-bit integer inputs are not supported",
                ));
            }
        }
    }
    match (a, b) {
        (Value::GpuTensor(handle_a), Value::GpuTensor(handle_b)) => {
            intersect_gpu_pair(handle_a, handle_b, &opts).await
        }
        (Value::GpuTensor(handle_a), other) => {
            intersect_gpu_mixed(handle_a, other, &opts, true).await
        }
        (other, Value::GpuTensor(handle_b)) => {
            intersect_gpu_mixed(handle_b, other, &opts, false).await
        }
        (left, right) => intersect_host(left, right, &opts),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IntersectOrder {
    Sorted,
    Stable,
}

#[derive(Debug, Clone)]
struct IntersectOptions {
    rows: bool,
    order: IntersectOrder,
}

fn parse_options(rest: &[Value]) -> crate::BuiltinResult<IntersectOptions> {
    let mut opts = IntersectOptions {
        rows: false,
        order: IntersectOrder::Sorted,
    };
    let mut seen_order: Option<IntersectOrder> = None;

    let tokens = tokens_from_values(rest);
    for (arg, token) in rest.iter().zip(tokens.iter()) {
        let text = match token {
            crate::builtins::common::arg_tokens::ArgToken::String(text) => text.as_str(),
            _ => {
                let text = tensor::value_to_string(arg)
                    .ok_or_else(|| intersect_error(&INTERSECT_ERROR_INVALID_ARGUMENT))?;
                let lowered = text.trim().to_ascii_lowercase();
                parse_intersect_option(&mut opts, &mut seen_order, &lowered)?;
                continue;
            }
        };
        parse_intersect_option(&mut opts, &mut seen_order, text)?;
    }

    Ok(opts)
}

fn parse_intersect_option(
    opts: &mut IntersectOptions,
    seen_order: &mut Option<IntersectOrder>,
    lowered: &str,
) -> crate::BuiltinResult<()> {
    match lowered {
        "rows" => opts.rows = true,
        "sorted" => {
            if let Some(prev) = seen_order {
                if *prev != IntersectOrder::Sorted {
                    return Err(intersect_error(&INTERSECT_ERROR_CONFLICTING_ORDER_OPTIONS));
                }
            }
            *seen_order = Some(IntersectOrder::Sorted);
            opts.order = IntersectOrder::Sorted;
        }
        "stable" => {
            if let Some(prev) = seen_order {
                if *prev != IntersectOrder::Stable {
                    return Err(intersect_error(&INTERSECT_ERROR_CONFLICTING_ORDER_OPTIONS));
                }
            }
            *seen_order = Some(IntersectOrder::Stable);
            opts.order = IntersectOrder::Stable;
        }
        "legacy" | "r2012a" => {
            return Err(intersect_error(&INTERSECT_ERROR_LEGACY_OPTION_UNSUPPORTED));
        }
        other => {
            return Err(intersect_error_with(
                &INTERSECT_ERROR_UNKNOWN_OPTION,
                format!("intersect: unrecognised option '{other}'"),
            ))
        }
    }
    Ok(())
}

async fn intersect_gpu_pair(
    handle_a: GpuTensorHandle,
    handle_b: GpuTensorHandle,
    opts: &IntersectOptions,
) -> crate::BuiltinResult<IntersectEvaluation> {
    let tensor_a = gpu_helpers::gather_tensor_async(&handle_a).await?;
    let tensor_b = gpu_helpers::gather_tensor_async(&handle_b).await?;
    intersect_numeric(tensor_a, tensor_b, opts)
}

async fn intersect_gpu_mixed(
    handle_gpu: GpuTensorHandle,
    other: Value,
    opts: &IntersectOptions,
    gpu_is_a: bool,
) -> crate::BuiltinResult<IntersectEvaluation> {
    let tensor_gpu = gpu_helpers::gather_tensor_async(&handle_gpu).await?;
    let tensor_other = tensor::value_into_tensor_for("intersect", other)
        .map_err(|e| intersect_internal_error(e))?;
    if gpu_is_a {
        intersect_numeric(tensor_gpu, tensor_other, opts)
    } else {
        intersect_numeric(tensor_other, tensor_gpu, opts)
    }
}

fn intersect_host(
    a: Value,
    b: Value,
    opts: &IntersectOptions,
) -> crate::BuiltinResult<IntersectEvaluation> {
    match (a, b) {
        (Value::ComplexTensor(at), Value::ComplexTensor(bt)) => intersect_complex(at, bt, opts),
        (Value::ComplexTensor(at), Value::Complex(re, im)) => {
            let bt = scalar_complex_tensor(re, im)?;
            intersect_complex(at, bt, opts)
        }
        (Value::Complex(re, im), Value::ComplexTensor(bt)) => {
            let at = scalar_complex_tensor(re, im)?;
            intersect_complex(at, bt, opts)
        }
        (Value::Complex(a_re, a_im), Value::Complex(b_re, b_im)) => {
            let at = scalar_complex_tensor(a_re, a_im)?;
            let bt = scalar_complex_tensor(b_re, b_im)?;
            intersect_complex(at, bt, opts)
        }
        (Value::ComplexTensor(at), other) => {
            let bt = value_into_complex_tensor(other)?;
            intersect_complex(at, bt, opts)
        }
        (other, Value::ComplexTensor(bt)) => {
            let at = value_into_complex_tensor(other)?;
            intersect_complex(at, bt, opts)
        }
        (Value::Complex(re, im), other) => {
            let at = scalar_complex_tensor(re, im)?;
            let bt = value_into_complex_tensor(other)?;
            intersect_complex(at, bt, opts)
        }
        (other, Value::Complex(re, im)) => {
            let at = value_into_complex_tensor(other)?;
            let bt = scalar_complex_tensor(re, im)?;
            intersect_complex(at, bt, opts)
        }

        (Value::CharArray(ac), Value::CharArray(bc)) => intersect_char(ac, bc, opts),

        (Value::StringArray(astring), Value::StringArray(bstring)) => {
            intersect_string(astring, bstring, opts)
        }
        (Value::StringArray(astring), Value::String(b)) => {
            let bstring = StringArray::new(vec![b], vec![1, 1])
                .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;
            intersect_string(astring, bstring, opts)
        }
        (Value::String(a), Value::StringArray(bstring)) => {
            let astring = StringArray::new(vec![a], vec![1, 1])
                .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;
            intersect_string(astring, bstring, opts)
        }
        (Value::String(a), Value::String(b)) => {
            let astring = StringArray::new(vec![a], vec![1, 1])
                .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;
            let bstring = StringArray::new(vec![b], vec![1, 1])
                .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;
            intersect_string(astring, bstring, opts)
        }

        (left, right) => {
            let tensor_a = tensor::value_into_tensor_for("intersect", left)
                .map_err(|e| intersect_error_with(&INTERSECT_ERROR_UNSUPPORTED_INPUT_TYPE, e))?;
            let tensor_b = tensor::value_into_tensor_for("intersect", right)
                .map_err(|e| intersect_error_with(&INTERSECT_ERROR_UNSUPPORTED_INPUT_TYPE, e))?;
            intersect_numeric(tensor_a, tensor_b, opts)
        }
    }
}

fn intersect_numeric(
    a: Tensor,
    b: Tensor,
    opts: &IntersectOptions,
) -> crate::BuiltinResult<IntersectEvaluation> {
    let a_dtype = a.numeric_dtype();
    let b_dtype = b.numeric_dtype();
    if let (Some(a_storage), Some(b_storage)) = (a.integer_storage(), b.integer_storage()) {
        if a_storage.class_name() == b_storage.class_name() {
            return if opts.rows {
                intersect_integer_rows(a_storage, a.shape.clone(), b_storage, b.shape.clone(), opts)
            } else {
                intersect_integer_elements(a_storage, b_storage, opts)
            };
        }
        return Err(intersect_error(&INTERSECT_ERROR_NUMERIC_CLASS_MISMATCH));
    }
    match (a.integer_storage(), b.integer_storage()) {
        (Some(storage), None) if b_dtype == NumericDType::F64 => {
            let target = IntegerTarget::from_storage(storage);
            let b = target.cast_tensor(b).map_err(intersect_internal_error)?;
            return intersect_numeric(a, b, opts);
        }
        (None, Some(storage)) if a_dtype == NumericDType::F64 => {
            let target = IntegerTarget::from_storage(storage);
            let a = target.cast_tensor(a).map_err(intersect_internal_error)?;
            return intersect_numeric(a, b, opts);
        }
        _ => {}
    }
    if a_dtype != b_dtype && a_dtype != NumericDType::F64 && b_dtype != NumericDType::F64 {
        return Err(intersect_error(&INTERSECT_ERROR_NUMERIC_CLASS_MISMATCH));
    }
    let a_shape = a.shape.clone();
    let b_shape = b.shape.clone();
    let a_storage = a
        .into_numeric_storage()
        .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;
    let b_storage = b
        .into_numeric_storage()
        .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;
    match (a_storage, b_storage) {
        (NumericStorage::F64(a), NumericStorage::F64(b)) => {
            intersect_floating(a, a_shape, b, b_shape, opts)
        }
        (NumericStorage::F32(a), NumericStorage::F32(b)) => {
            intersect_floating(a, a_shape, b, b_shape, opts)
        }
        (a, b) => intersect_promoted_f64(a, a_shape, b, b_shape, opts),
    }
}

fn intersect_promoted_f64(
    a: NumericStorage,
    a_shape: Vec<usize>,
    b: NumericStorage,
    b_shape: Vec<usize>,
    opts: &IntersectOptions,
) -> crate::BuiltinResult<IntersectEvaluation> {
    intersect_floating(
        a.materialize_f64(),
        a_shape,
        b.materialize_f64(),
        b_shape,
        opts,
    )
}

fn intersect_floating<T: SetFloat>(
    a: Vec<T>,
    a_shape: Vec<usize>,
    b: Vec<T>,
    b_shape: Vec<usize>,
    opts: &IntersectOptions,
) -> crate::BuiltinResult<IntersectEvaluation> {
    if opts.rows {
        intersect_floating_rows(a, a_shape, b, b_shape, opts)
    } else {
        intersect_floating_elements(a, b, opts)
    }
}

fn intersect_integer_elements(
    a: &IntegerStorage,
    b: &IntegerStorage,
    opts: &IntersectOptions,
) -> crate::BuiltinResult<IntersectEvaluation> {
    let mut b_map = HashMap::<IntValue, usize>::new();
    for (index, value) in b.exact_values().into_iter().enumerate() {
        b_map.entry(value).or_insert(index);
    }
    let mut seen = HashSet::<IntValue>::new();
    let mut entries = Vec::<IntegerIntersectEntry>::new();
    for (a_index, value) in a.exact_values().into_iter().enumerate() {
        if seen.contains(&value) {
            continue;
        }
        if let Some(&b_index) = b_map.get(&value) {
            let order_rank = entries.len();
            entries.push(IntegerIntersectEntry {
                value: value.clone(),
                a_index,
                b_index,
                order_rank,
            });
            seen.insert(value);
        }
    }
    assemble_integer_intersect(entries, a, opts)
}

fn intersect_integer_rows(
    a_storage: &IntegerStorage,
    a_shape: Vec<usize>,
    b_storage: &IntegerStorage,
    b_shape: Vec<usize>,
    opts: &IntersectOptions,
) -> crate::BuiltinResult<IntersectEvaluation> {
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(intersect_internal_error(
            "intersect: 'rows' option requires 2-D numeric matrices",
        ));
    }
    if a_shape[1] != b_shape[1] {
        return Err(intersect_error(&INTERSECT_ERROR_ROWS_COLUMN_MISMATCH));
    }
    let (rows_a, rows_b, cols) = (a_shape[0], b_shape[0], a_shape[1]);
    let a_values = a_storage.exact_values();
    let b_values = b_storage.exact_values();
    let mut b_map = HashMap::<Vec<IntValue>, usize>::new();
    for row in 0..rows_b {
        let key: Vec<_> = (0..cols)
            .map(|col| b_values[row + col * rows_b].clone())
            .collect();
        b_map.entry(key).or_insert(row);
    }
    let mut seen = HashSet::<Vec<IntValue>>::new();
    let mut entries = Vec::<IntegerRowIntersectEntry>::new();
    for row in 0..rows_a {
        let values: Vec<_> = (0..cols)
            .map(|col| a_values[row + col * rows_a].clone())
            .collect();
        if seen.contains(&values) {
            continue;
        }
        if let Some(&b_row) = b_map.get(&values) {
            let order_rank = entries.len();
            entries.push(IntegerRowIntersectEntry {
                row_data: values.clone(),
                a_row: row,
                b_row,
                order_rank,
            });
            seen.insert(values);
        }
    }
    assemble_integer_row_intersect(entries, a_storage, opts, cols)
}

fn intersect_floating_elements<T: SetFloat>(
    a_values: Vec<T>,
    b_values: Vec<T>,
    opts: &IntersectOptions,
) -> crate::BuiltinResult<IntersectEvaluation> {
    let mut b_map: HashMap<u64, usize> = HashMap::new();
    for (idx, &value) in b_values.iter().enumerate() {
        let key = value.canonical_key();
        b_map.entry(key).or_insert(idx);
    }

    let mut seen: HashSet<u64> = HashSet::new();
    let mut entries = Vec::<FloatingIntersectEntry<T>>::new();
    let mut order_counter = 0usize;

    for (idx, &value) in a_values.iter().enumerate() {
        let key = value.canonical_key();
        if seen.contains(&key) {
            continue;
        }
        if let Some(&b_idx) = b_map.get(&key) {
            entries.push(FloatingIntersectEntry {
                value,
                a_index: idx,
                b_index: b_idx,
                order_rank: order_counter,
            });
            seen.insert(key);
            order_counter += 1;
        }
    }

    assemble_floating_intersect(entries, opts)
}

fn intersect_floating_rows<T: SetFloat>(
    a_values: Vec<T>,
    a_shape: Vec<usize>,
    b_values: Vec<T>,
    b_shape: Vec<usize>,
    opts: &IntersectOptions,
) -> crate::BuiltinResult<IntersectEvaluation> {
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(intersect_internal_error(
            "intersect: 'rows' option requires 2-D numeric matrices",
        ));
    }
    if a_shape[1] != b_shape[1] {
        return Err(intersect_error(&INTERSECT_ERROR_ROWS_COLUMN_MISMATCH));
    }
    let rows_a = a_shape[0];
    let cols = a_shape[1];
    let rows_b = b_shape[0];

    let mut b_map: HashMap<FloatingRowKey, usize> = HashMap::new();
    for r in 0..rows_b {
        let mut row_values = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r + c * rows_b;
            row_values.push(b_values[idx]);
        }
        let key = FloatingRowKey::from_slice(&row_values);
        b_map.entry(key).or_insert(r);
    }

    let mut seen: HashSet<FloatingRowKey> = HashSet::new();
    let mut entries = Vec::<FloatingRowIntersectEntry<T>>::new();
    let mut order_counter = 0usize;

    for r in 0..rows_a {
        let mut row_values = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r + c * rows_a;
            row_values.push(a_values[idx]);
        }
        let key = FloatingRowKey::from_slice(&row_values);
        if seen.contains(&key) {
            continue;
        }
        if let Some(&b_row) = b_map.get(&key) {
            entries.push(FloatingRowIntersectEntry {
                row_data: row_values,
                a_row: r,
                b_row,
                order_rank: order_counter,
            });
            seen.insert(key);
            order_counter += 1;
        }
    }

    assemble_floating_row_intersect(entries, opts, cols)
}

#[cfg(test)]
fn intersect_numeric_elements(
    a: Tensor,
    b: Tensor,
    opts: &IntersectOptions,
) -> crate::BuiltinResult<IntersectEvaluation> {
    intersect_numeric(a, b, opts)
}

#[cfg(test)]
fn intersect_numeric_rows(
    a: Tensor,
    b: Tensor,
    opts: &IntersectOptions,
) -> crate::BuiltinResult<IntersectEvaluation> {
    intersect_numeric(a, b, opts)
}

fn intersect_complex(
    a: ComplexTensor,
    b: ComplexTensor,
    opts: &IntersectOptions,
) -> crate::BuiltinResult<IntersectEvaluation> {
    let a_shape = a.shape.clone();
    let b_shape = b.shape.clone();
    match (a.into_complex_storage(), b.into_complex_storage()) {
        (ComplexStorage::F64(a), ComplexStorage::F64(b)) => {
            intersect_floating_complex(a, a_shape, b, b_shape, opts)
        }
        (ComplexStorage::F32(a), ComplexStorage::F32(b)) => {
            intersect_floating_complex(a, a_shape, b, b_shape, opts)
        }
        (a, b) => intersect_promoted_complex_f64(a, a_shape, b, b_shape, opts),
    }
}

fn intersect_promoted_complex_f64(
    a: ComplexStorage,
    a_shape: Vec<usize>,
    b: ComplexStorage,
    b_shape: Vec<usize>,
    opts: &IntersectOptions,
) -> crate::BuiltinResult<IntersectEvaluation> {
    intersect_floating_complex(
        a.materialize_f64(),
        a_shape,
        b.materialize_f64(),
        b_shape,
        opts,
    )
}

fn intersect_floating_complex<T: SetFloat>(
    a: Vec<(T, T)>,
    a_shape: Vec<usize>,
    b: Vec<(T, T)>,
    b_shape: Vec<usize>,
    opts: &IntersectOptions,
) -> crate::BuiltinResult<IntersectEvaluation> {
    if opts.rows {
        intersect_complex_rows(a, a_shape, b, b_shape, opts)
    } else {
        intersect_complex_elements(a, b, opts)
    }
}

fn intersect_complex_elements<T: SetFloat>(
    a: Vec<(T, T)>,
    b: Vec<(T, T)>,
    opts: &IntersectOptions,
) -> crate::BuiltinResult<IntersectEvaluation> {
    let mut b_map: HashMap<ComplexKey, usize> = HashMap::new();
    for (idx, &value) in b.iter().enumerate() {
        let key = ComplexKey::new(value);
        b_map.entry(key).or_insert(idx);
    }

    let mut seen: HashSet<ComplexKey> = HashSet::new();
    let mut entries = Vec::<ComplexIntersectEntry<T>>::new();
    let mut order_counter = 0usize;

    for (idx, &value) in a.iter().enumerate() {
        let key = ComplexKey::new(value);
        if seen.contains(&key) {
            continue;
        }
        if let Some(&b_idx) = b_map.get(&key) {
            entries.push(ComplexIntersectEntry {
                value,
                a_index: idx,
                b_index: b_idx,
                order_rank: order_counter,
            });
            seen.insert(key);
            order_counter += 1;
        }
    }

    assemble_complex_intersect(entries, opts)
}

fn intersect_complex_rows<T: SetFloat>(
    a: Vec<(T, T)>,
    a_shape: Vec<usize>,
    b: Vec<(T, T)>,
    b_shape: Vec<usize>,
    opts: &IntersectOptions,
) -> crate::BuiltinResult<IntersectEvaluation> {
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(intersect_internal_error(
            "intersect: 'rows' option requires 2-D complex matrices",
        ));
    }
    if a_shape[1] != b_shape[1] {
        return Err(intersect_error(&INTERSECT_ERROR_ROWS_COLUMN_MISMATCH));
    }
    let rows_a = a_shape[0];
    let cols = a_shape[1];
    let rows_b = b_shape[0];

    let mut b_map: HashMap<Vec<ComplexKey>, usize> = HashMap::new();
    for r in 0..rows_b {
        let mut row_keys = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r + c * rows_b;
            row_keys.push(ComplexKey::new(b[idx]));
        }
        b_map.entry(row_keys).or_insert(r);
    }

    let mut seen: HashSet<Vec<ComplexKey>> = HashSet::new();
    let mut entries = Vec::<ComplexRowIntersectEntry<T>>::new();
    let mut order_counter = 0usize;

    for r in 0..rows_a {
        let mut row_values = Vec::with_capacity(cols);
        let mut row_keys = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r + c * rows_a;
            let value = a[idx];
            row_values.push(value);
            row_keys.push(ComplexKey::new(value));
        }
        if seen.contains(&row_keys) {
            continue;
        }
        if let Some(&b_row) = b_map.get(&row_keys) {
            entries.push(ComplexRowIntersectEntry {
                row_data: row_values,
                a_row: r,
                b_row,
                order_rank: order_counter,
            });
            seen.insert(row_keys);
            order_counter += 1;
        }
    }

    assemble_complex_row_intersect(entries, opts, cols)
}

fn intersect_char(
    a: CharArray,
    b: CharArray,
    opts: &IntersectOptions,
) -> crate::BuiltinResult<IntersectEvaluation> {
    if opts.rows {
        intersect_char_rows(a, b, opts)
    } else {
        intersect_char_elements(a, b, opts)
    }
}

fn intersect_char_elements(
    a: CharArray,
    b: CharArray,
    opts: &IntersectOptions,
) -> crate::BuiltinResult<IntersectEvaluation> {
    let mut seen: HashSet<u32> = HashSet::new();
    let mut entries = Vec::<CharIntersectEntry>::new();
    let mut order_counter = 0usize;

    for col in 0..a.cols {
        for row in 0..a.rows {
            let linear_idx = row + col * a.rows;
            let data_idx = row * a.cols + col;
            let ch = a.data[data_idx];
            let key = ch as u32;
            if seen.contains(&key) {
                continue;
            }
            if let Some(b_idx) = find_char_index(&b, ch) {
                entries.push(CharIntersectEntry {
                    ch,
                    a_index: linear_idx,
                    b_index: b_idx,
                    order_rank: order_counter,
                });
                seen.insert(key);
                order_counter += 1;
            }
        }
    }

    assemble_char_intersect(entries, opts, &b)
}

fn intersect_char_rows(
    a: CharArray,
    b: CharArray,
    opts: &IntersectOptions,
) -> crate::BuiltinResult<IntersectEvaluation> {
    if a.cols != b.cols {
        return Err(intersect_error(&INTERSECT_ERROR_ROWS_COLUMN_MISMATCH));
    }
    let rows_a = a.rows;
    let rows_b = b.rows;
    let cols = a.cols;

    let mut b_map: HashMap<RowCharKey, usize> = HashMap::new();
    for r in 0..rows_b {
        let mut row_values = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r * cols + c;
            row_values.push(b.data[idx]);
        }
        let key = RowCharKey::from_slice(&row_values);
        b_map.entry(key).or_insert(r);
    }

    let mut seen: HashSet<RowCharKey> = HashSet::new();
    let mut entries = Vec::<CharRowIntersectEntry>::new();
    let mut order_counter = 0usize;

    for r in 0..rows_a {
        let mut row_values = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r * cols + c;
            row_values.push(a.data[idx]);
        }
        let key = RowCharKey::from_slice(&row_values);
        if seen.contains(&key) {
            continue;
        }
        if let Some(&b_row) = b_map.get(&key) {
            entries.push(CharRowIntersectEntry {
                row_data: row_values,
                a_row: r,
                b_row,
                order_rank: order_counter,
            });
            seen.insert(key);
            order_counter += 1;
        }
    }

    assemble_char_row_intersect(entries, opts, cols)
}

fn find_char_index(array: &CharArray, target: char) -> Option<usize> {
    for col in 0..array.cols {
        for row in 0..array.rows {
            let data_idx = row * array.cols + col;
            if array.data[data_idx] == target {
                return Some(row + col * array.rows);
            }
        }
    }
    None
}

fn intersect_string(
    a: StringArray,
    b: StringArray,
    opts: &IntersectOptions,
) -> crate::BuiltinResult<IntersectEvaluation> {
    if opts.rows {
        intersect_string_rows(a, b, opts)
    } else {
        intersect_string_elements(a, b, opts)
    }
}

fn intersect_string_elements(
    a: StringArray,
    b: StringArray,
    opts: &IntersectOptions,
) -> crate::BuiltinResult<IntersectEvaluation> {
    let mut b_map: HashMap<String, usize> = HashMap::new();
    for (idx, value) in b.data.iter().enumerate() {
        b_map.entry(value.clone()).or_insert(idx);
    }

    let mut seen: HashSet<String> = HashSet::new();
    let mut entries = Vec::<StringIntersectEntry>::new();
    let mut order_counter = 0usize;

    for (idx, value) in a.data.iter().enumerate() {
        if seen.contains(value) {
            continue;
        }
        if let Some(&b_idx) = b_map.get(value) {
            entries.push(StringIntersectEntry {
                value: value.clone(),
                a_index: idx,
                b_index: b_idx,
                order_rank: order_counter,
            });
            seen.insert(value.clone());
            order_counter += 1;
        }
    }

    assemble_string_intersect(entries, opts)
}

fn intersect_string_rows(
    a: StringArray,
    b: StringArray,
    opts: &IntersectOptions,
) -> crate::BuiltinResult<IntersectEvaluation> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(intersect_internal_error(
            "intersect: 'rows' option requires 2-D string arrays",
        ));
    }
    if a.shape[1] != b.shape[1] {
        return Err(intersect_error(&INTERSECT_ERROR_ROWS_COLUMN_MISMATCH));
    }
    let rows_a = a.shape[0];
    let cols = a.shape[1];
    let rows_b = b.shape[0];

    let mut b_map: HashMap<RowStringKey, usize> = HashMap::new();
    for r in 0..rows_b {
        let mut row_values = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r + c * rows_b;
            row_values.push(b.data[idx].clone());
        }
        let key = RowStringKey::from_slice(&row_values);
        b_map.entry(key).or_insert(r);
    }

    let mut seen: HashSet<RowStringKey> = HashSet::new();
    let mut entries = Vec::<StringRowIntersectEntry>::new();
    let mut order_counter = 0usize;

    for r in 0..rows_a {
        let mut row_values = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r + c * rows_a;
            row_values.push(a.data[idx].clone());
        }
        let key = RowStringKey::from_slice(&row_values);
        if seen.contains(&key) {
            continue;
        }
        if let Some(&b_row) = b_map.get(&key) {
            entries.push(StringRowIntersectEntry {
                row_data: row_values,
                a_row: r,
                b_row,
                order_rank: order_counter,
            });
            seen.insert(key);
            order_counter += 1;
        }
    }

    assemble_string_row_intersect(entries, opts, cols)
}

#[derive(Debug, Clone)]
pub struct IntersectEvaluation {
    values: Value,
    ia: Tensor,
    ib: Tensor,
}

impl IntersectEvaluation {
    fn new(values: Value, ia: Tensor, ib: Tensor) -> Self {
        Self { values, ia, ib }
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
struct FloatingIntersectEntry<T> {
    value: T,
    a_index: usize,
    b_index: usize,
    order_rank: usize,
}

#[derive(Debug)]
struct IntegerIntersectEntry {
    value: IntValue,
    a_index: usize,
    b_index: usize,
    order_rank: usize,
}

#[derive(Debug)]
struct FloatingRowIntersectEntry<T> {
    row_data: Vec<T>,
    a_row: usize,
    b_row: usize,
    order_rank: usize,
}

#[derive(Debug)]
struct IntegerRowIntersectEntry {
    row_data: Vec<IntValue>,
    a_row: usize,
    b_row: usize,
    order_rank: usize,
}

#[derive(Debug)]
struct ComplexIntersectEntry<T> {
    value: (T, T),
    a_index: usize,
    b_index: usize,
    order_rank: usize,
}

#[derive(Debug)]
struct ComplexRowIntersectEntry<T> {
    row_data: Vec<(T, T)>,
    a_row: usize,
    b_row: usize,
    order_rank: usize,
}

#[derive(Debug)]
struct CharIntersectEntry {
    ch: char,
    a_index: usize,
    b_index: usize,
    order_rank: usize,
}

#[derive(Debug)]
struct CharRowIntersectEntry {
    row_data: Vec<char>,
    a_row: usize,
    b_row: usize,
    order_rank: usize,
}

#[derive(Debug)]
struct StringIntersectEntry {
    value: String,
    a_index: usize,
    b_index: usize,
    order_rank: usize,
}

#[derive(Debug)]
struct StringRowIntersectEntry {
    row_data: Vec<String>,
    a_row: usize,
    b_row: usize,
    order_rank: usize,
}

fn assemble_floating_intersect<T: SetFloat>(
    entries: Vec<FloatingIntersectEntry<T>>,
    opts: &IntersectOptions,
) -> crate::BuiltinResult<IntersectEvaluation> {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    match opts.order {
        IntersectOrder::Sorted => {
            order.sort_by(|&lhs, &rhs| entries[lhs].value.compare(entries[rhs].value));
        }
        IntersectOrder::Stable => {
            order.sort_by_key(|&idx| entries[idx].order_rank);
        }
    }

    let mut values = Vec::with_capacity(order.len());
    let mut ia = Vec::with_capacity(order.len());
    let mut ib = Vec::with_capacity(order.len());
    for &idx in &order {
        let entry = &entries[idx];
        values.push(entry.value);
        ia.push((entry.a_index + 1) as f64);
        ib.push((entry.b_index + 1) as f64);
    }

    let value_tensor =
        Tensor::from_numeric_storage(T::numeric_storage(values), vec![order.len(), 1])
            .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;
    let ia_tensor = Tensor::new(ia, vec![order.len(), 1])
        .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;
    let ib_tensor = Tensor::new(ib, vec![order.len(), 1])
        .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;

    Ok(IntersectEvaluation::new(
        tensor::tensor_into_value(value_tensor),
        ia_tensor,
        ib_tensor,
    ))
}

fn assemble_integer_intersect(
    entries: Vec<IntegerIntersectEntry>,
    storage: &IntegerStorage,
    opts: &IntersectOptions,
) -> crate::BuiltinResult<IntersectEvaluation> {
    let mut order: Vec<_> = (0..entries.len()).collect();
    match opts.order {
        IntersectOrder::Sorted => order.sort_by(|&a, &b| {
            integer_order::compare(&entries[a].value, &entries[b].value, false, false)
        }),
        IntersectOrder::Stable => order.sort_by_key(|&index| entries[index].order_rank),
    }
    let values: Vec<_> = order
        .iter()
        .map(|&index| entries[index].value.clone())
        .collect();
    let ia: Vec<_> = order
        .iter()
        .map(|&index| (entries[index].a_index + 1) as f64)
        .collect();
    let ib: Vec<_> = order
        .iter()
        .map(|&index| (entries[index].b_index + 1) as f64)
        .collect();
    let values = Tensor::new_integer(
        storage
            .from_exact_values_like(values)
            .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?,
        vec![order.len(), 1],
    )
    .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;
    let ia = Tensor::new(ia, vec![order.len(), 1])
        .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;
    let ib = Tensor::new(ib, vec![order.len(), 1])
        .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;
    Ok(IntersectEvaluation::new(Value::Tensor(values), ia, ib))
}

fn assemble_floating_row_intersect<T: SetFloat>(
    entries: Vec<FloatingRowIntersectEntry<T>>,
    opts: &IntersectOptions,
    cols: usize,
) -> crate::BuiltinResult<IntersectEvaluation> {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    match opts.order {
        IntersectOrder::Sorted => {
            order.sort_by(|&lhs, &rhs| {
                compare_floating_rows(&entries[lhs].row_data, &entries[rhs].row_data)
            });
        }
        IntersectOrder::Stable => {
            order.sort_by_key(|&idx| entries[idx].order_rank);
        }
    }

    let rows_out = order.len();
    let mut values = vec![T::default(); rows_out * cols];
    let mut ia = Vec::with_capacity(rows_out);
    let mut ib = Vec::with_capacity(rows_out);

    for (row_pos, &entry_idx) in order.iter().enumerate() {
        let entry = &entries[entry_idx];
        for col in 0..cols {
            let dest = row_pos + col * rows_out;
            values[dest] = entry.row_data[col];
        }
        ia.push((entry.a_row + 1) as f64);
        ib.push((entry.b_row + 1) as f64);
    }

    let value_tensor =
        Tensor::from_numeric_storage(T::numeric_storage(values), vec![rows_out, cols])
            .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;
    let ia_tensor = Tensor::new(ia, vec![rows_out, 1])
        .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;
    let ib_tensor = Tensor::new(ib, vec![rows_out, 1])
        .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;

    Ok(IntersectEvaluation::new(
        tensor::tensor_into_value(value_tensor),
        ia_tensor,
        ib_tensor,
    ))
}

fn assemble_integer_row_intersect(
    entries: Vec<IntegerRowIntersectEntry>,
    storage: &IntegerStorage,
    opts: &IntersectOptions,
    cols: usize,
) -> crate::BuiltinResult<IntersectEvaluation> {
    let mut order: Vec<_> = (0..entries.len()).collect();
    match opts.order {
        IntersectOrder::Sorted => order.sort_by(|&a, &b| {
            for (left, right) in entries[a].row_data.iter().zip(&entries[b].row_data) {
                let ordering = integer_order::compare(left, right, false, false);
                if ordering != Ordering::Equal {
                    return ordering;
                }
            }
            Ordering::Equal
        }),
        IntersectOrder::Stable => order.sort_by_key(|&index| entries[index].order_rank),
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
        .map(|&index| (entries[index].a_row + 1) as f64)
        .collect();
    let ib: Vec<_> = order
        .iter()
        .map(|&index| (entries[index].b_row + 1) as f64)
        .collect();
    let values = Tensor::new_integer(
        storage
            .from_exact_values_like(values)
            .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?,
        vec![rows, cols],
    )
    .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;
    let ia = Tensor::new(ia, vec![rows, 1])
        .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;
    let ib = Tensor::new(ib, vec![rows, 1])
        .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;
    Ok(IntersectEvaluation::new(Value::Tensor(values), ia, ib))
}

fn assemble_complex_intersect<T: SetFloat>(
    entries: Vec<ComplexIntersectEntry<T>>,
    opts: &IntersectOptions,
) -> crate::BuiltinResult<IntersectEvaluation> {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    match opts.order {
        IntersectOrder::Sorted => {
            order.sort_by(|&lhs, &rhs| compare_complex(entries[lhs].value, entries[rhs].value));
        }
        IntersectOrder::Stable => {
            order.sort_by_key(|&idx| entries[idx].order_rank);
        }
    }

    let mut values = Vec::with_capacity(order.len());
    let mut ia = Vec::with_capacity(order.len());
    let mut ib = Vec::with_capacity(order.len());
    for &idx in &order {
        let entry = &entries[idx];
        values.push(entry.value);
        ia.push((entry.a_index + 1) as f64);
        ib.push((entry.b_index + 1) as f64);
    }

    let value_tensor =
        ComplexTensor::from_complex_storage(T::complex_storage(values), vec![order.len(), 1])
            .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;
    let ia_tensor = Tensor::new(ia, vec![order.len(), 1])
        .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;
    let ib_tensor = Tensor::new(ib, vec![order.len(), 1])
        .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;

    let value = if value_tensor.as_f32_slice().is_some() {
        Value::ComplexTensor(value_tensor)
    } else {
        complex_tensor_into_value(value_tensor)
    };
    Ok(IntersectEvaluation::new(value, ia_tensor, ib_tensor))
}

fn assemble_complex_row_intersect<T: SetFloat>(
    entries: Vec<ComplexRowIntersectEntry<T>>,
    opts: &IntersectOptions,
    cols: usize,
) -> crate::BuiltinResult<IntersectEvaluation> {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    match opts.order {
        IntersectOrder::Sorted => {
            order.sort_by(|&lhs, &rhs| {
                compare_complex_rows(&entries[lhs].row_data, &entries[rhs].row_data)
            });
        }
        IntersectOrder::Stable => {
            order.sort_by_key(|&idx| entries[idx].order_rank);
        }
    }

    let rows_out = order.len();
    let mut values = vec![(T::default(), T::default()); rows_out * cols];
    let mut ia = Vec::with_capacity(rows_out);
    let mut ib = Vec::with_capacity(rows_out);

    for (row_pos, &entry_idx) in order.iter().enumerate() {
        let entry = &entries[entry_idx];
        for col in 0..cols {
            let dest = row_pos + col * rows_out;
            values[dest] = entry.row_data[col];
        }
        ia.push((entry.a_row + 1) as f64);
        ib.push((entry.b_row + 1) as f64);
    }

    let value_tensor =
        ComplexTensor::from_complex_storage(T::complex_storage(values), vec![rows_out, cols])
            .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;
    let ia_tensor = Tensor::new(ia, vec![rows_out, 1])
        .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;
    let ib_tensor = Tensor::new(ib, vec![rows_out, 1])
        .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;

    let value = if value_tensor.as_f32_slice().is_some() {
        Value::ComplexTensor(value_tensor)
    } else {
        complex_tensor_into_value(value_tensor)
    };
    Ok(IntersectEvaluation::new(value, ia_tensor, ib_tensor))
}

fn assemble_char_intersect(
    entries: Vec<CharIntersectEntry>,
    opts: &IntersectOptions,
    b: &CharArray,
) -> crate::BuiltinResult<IntersectEvaluation> {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    match opts.order {
        IntersectOrder::Sorted => {
            order.sort_by(|&lhs, &rhs| entries[lhs].ch.cmp(&entries[rhs].ch));
        }
        IntersectOrder::Stable => {
            order.sort_by_key(|&idx| entries[idx].order_rank);
        }
    }

    let mut values = Vec::with_capacity(order.len());
    let mut ia = Vec::with_capacity(order.len());
    let mut ib = Vec::with_capacity(order.len());
    for &idx in &order {
        let entry = &entries[idx];
        values.push(entry.ch);
        ia.push((entry.a_index + 1) as f64);
        let b_idx = find_char_index(b, entry.ch).unwrap_or(entry.b_index);
        ib.push((b_idx + 1) as f64);
    }

    let value_array = CharArray::new(values, order.len(), 1)
        .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;
    let ia_tensor = Tensor::new(ia, vec![order.len(), 1])
        .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;
    let ib_tensor = Tensor::new(ib, vec![order.len(), 1])
        .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;

    Ok(IntersectEvaluation::new(
        Value::CharArray(value_array),
        ia_tensor,
        ib_tensor,
    ))
}

fn assemble_char_row_intersect(
    entries: Vec<CharRowIntersectEntry>,
    opts: &IntersectOptions,
    cols: usize,
) -> crate::BuiltinResult<IntersectEvaluation> {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    match opts.order {
        IntersectOrder::Sorted => {
            order.sort_by(|&lhs, &rhs| {
                compare_char_rows(&entries[lhs].row_data, &entries[rhs].row_data)
            });
        }
        IntersectOrder::Stable => {
            order.sort_by_key(|&idx| entries[idx].order_rank);
        }
    }

    let rows_out = order.len();
    let mut values = vec!['\0'; rows_out * cols];
    let mut ia = Vec::with_capacity(rows_out);
    let mut ib = Vec::with_capacity(rows_out);

    for (row_pos, &entry_idx) in order.iter().enumerate() {
        let entry = &entries[entry_idx];
        for col in 0..cols {
            let dest = row_pos * cols + col;
            values[dest] = entry.row_data[col];
        }
        ia.push((entry.a_row + 1) as f64);
        ib.push((entry.b_row + 1) as f64);
    }

    let value_array = CharArray::new(values, rows_out, cols)
        .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;
    let ia_tensor = Tensor::new(ia, vec![rows_out, 1])
        .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;
    let ib_tensor = Tensor::new(ib, vec![rows_out, 1])
        .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;

    Ok(IntersectEvaluation::new(
        Value::CharArray(value_array),
        ia_tensor,
        ib_tensor,
    ))
}

fn assemble_string_intersect(
    entries: Vec<StringIntersectEntry>,
    opts: &IntersectOptions,
) -> crate::BuiltinResult<IntersectEvaluation> {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    match opts.order {
        IntersectOrder::Sorted => {
            order.sort_by(|&lhs, &rhs| entries[lhs].value.cmp(&entries[rhs].value));
        }
        IntersectOrder::Stable => {
            order.sort_by_key(|&idx| entries[idx].order_rank);
        }
    }

    let mut values = Vec::with_capacity(order.len());
    let mut ia = Vec::with_capacity(order.len());
    let mut ib = Vec::with_capacity(order.len());
    for &idx in &order {
        let entry = &entries[idx];
        values.push(entry.value.clone());
        ia.push((entry.a_index + 1) as f64);
        ib.push((entry.b_index + 1) as f64);
    }

    let value_array = StringArray::new(values, vec![order.len(), 1])
        .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;
    let ia_tensor = Tensor::new(ia, vec![order.len(), 1])
        .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;
    let ib_tensor = Tensor::new(ib, vec![order.len(), 1])
        .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;

    Ok(IntersectEvaluation::new(
        Value::StringArray(value_array),
        ia_tensor,
        ib_tensor,
    ))
}

fn assemble_string_row_intersect(
    entries: Vec<StringRowIntersectEntry>,
    opts: &IntersectOptions,
    cols: usize,
) -> crate::BuiltinResult<IntersectEvaluation> {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    match opts.order {
        IntersectOrder::Sorted => {
            order.sort_by(|&lhs, &rhs| {
                compare_string_rows(&entries[lhs].row_data, &entries[rhs].row_data)
            });
        }
        IntersectOrder::Stable => {
            order.sort_by_key(|&idx| entries[idx].order_rank);
        }
    }

    let rows_out = order.len();
    let mut values = vec![String::new(); rows_out * cols];
    let mut ia = Vec::with_capacity(rows_out);
    let mut ib = Vec::with_capacity(rows_out);

    for (row_pos, &entry_idx) in order.iter().enumerate() {
        let entry = &entries[entry_idx];
        for col in 0..cols {
            let dest = row_pos + col * rows_out;
            values[dest] = entry.row_data[col].clone();
        }
        ia.push((entry.a_row + 1) as f64);
        ib.push((entry.b_row + 1) as f64);
    }

    let value_array = StringArray::new(values, vec![rows_out, cols])
        .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;
    let ia_tensor = Tensor::new(ia, vec![rows_out, 1])
        .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;
    let ib_tensor = Tensor::new(ib, vec![rows_out, 1])
        .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;

    Ok(IntersectEvaluation::new(
        Value::StringArray(value_array),
        ia_tensor,
        ib_tensor,
    ))
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

impl RowStringKey {
    fn from_slice(values: &[String]) -> Self {
        RowStringKey(values.to_vec())
    }
}

fn scalar_complex_tensor(re: f64, im: f64) -> crate::BuiltinResult<ComplexTensor> {
    ComplexTensor::new(vec![(re, im)], vec![1, 1])
        .map_err(|e| intersect_internal_error(format!("intersect: {e}")))
}

fn tensor_to_complex_owned(name: &str, tensor: Tensor) -> crate::BuiltinResult<ComplexTensor> {
    let shape = tensor.shape.clone();
    let complex = tensor
        .into_numeric_storage()
        .map_err(|e| intersect_internal_error(format!("{name}: {e}")))?
        .materialize_f64()
        .into_iter()
        .map(|real| (real, 0.0))
        .collect();
    ComplexTensor::new(complex, shape).map_err(|e| intersect_internal_error(format!("{name}: {e}")))
}

fn value_into_complex_tensor(value: Value) -> crate::BuiltinResult<ComplexTensor> {
    match value {
        Value::ComplexTensor(tensor) => Ok(tensor),
        Value::Complex(re, im) => scalar_complex_tensor(re, im),
        other => {
            let tensor = tensor::value_into_tensor_for("intersect", other)
                .map_err(|e| intersect_internal_error(e))?;
            tensor_to_complex_owned("intersect", tensor)
        }
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

    fn evaluate_sync(
        a: Value,
        b: Value,
        rest: &[Value],
    ) -> crate::BuiltinResult<IntersectEvaluation> {
        futures::executor::block_on(evaluate(a, b, rest))
    }

    fn builtin_sync(a: Value, b: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        futures::executor::block_on(intersect_builtin(a, b, rest))
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
                    builtin_sync(left, right, Vec::new()).expect("resident intersect")
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
                    Some(&IntegerStorage::I32(vec![2, 7]))
                );
            }

            let _guard = crate::output_count::push_output_count(Some(4));
            let err = builtin_sync(Value::Num(1.0), Value::Num(1.0), Vec::new())
                .expect_err("excess outputs must fail");
            assert_eq!(
                err.identifier(),
                INTERSECT_ERROR_INVALID_ARGUMENT.identifier
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn intersect_numeric_sorted() {
        let a = Tensor::new(vec![5.0, 7.0, 5.0, 1.0], vec![4, 1]).unwrap();
        let b = Tensor::new(vec![7.0, 1.0, 3.0], vec![3, 1]).unwrap();
        let eval = intersect_numeric_elements(
            a,
            b,
            &IntersectOptions {
                rows: false,
                order: IntersectOrder::Sorted,
            },
        )
        .expect("intersect");
        let values = tensor::value_into_tensor_for("intersect", eval.values_value()).unwrap();
        assert_eq!(values.materialize_f64(), vec![1.0, 7.0]);
        let ia = tensor::value_into_tensor_for("intersect", eval.ia_value()).unwrap();
        let ib = tensor::value_into_tensor_for("intersect", eval.ib_value()).unwrap();
        assert_eq!(ia.materialize_f64(), vec![4.0, 2.0]);
        assert_eq!(ib.materialize_f64(), vec![2.0, 1.0]);
    }

    #[test]
    fn intersect_preserves_native_single_elements_and_rows() {
        let a = Tensor::from_f32(vec![5.0, 7.0, 5.0, 1.0], vec![4, 1]).unwrap();
        let b = Tensor::from_f32(vec![7.0, 1.0, 3.0], vec![3, 1]).unwrap();
        let values = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[])
            .expect("single intersect")
            .values_value();
        let Value::Tensor(values) = values else {
            panic!("expected native single values");
        };
        assert_eq!(
            values.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![1.0, 7.0])
        );

        let a = Tensor::from_f32(vec![1.0, 3.0, 1.0, 2.0, 4.0, 2.0], vec![3, 2]).unwrap();
        let b = Tensor::from_f32(vec![1.0, 5.0, 2.0, 6.0], vec![2, 2]).unwrap();
        let values = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[Value::from("rows")])
            .expect("single row intersect")
            .values_value();
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
    fn intersect_preserves_native_complex_single_elements_and_rows() {
        let a =
            ComplexTensor::from_f32(vec![(1.0, 1.0), (0.0, 2.0), (1.0, -1.0)], vec![3, 1]).unwrap();
        let b = ComplexTensor::from_f32(vec![(0.0, 2.0), (4.0, 0.0)], vec![2, 1]).unwrap();
        let values = evaluate_sync(Value::ComplexTensor(a), Value::ComplexTensor(b), &[])
            .expect("complex single intersect")
            .values_value();
        let Value::ComplexTensor(values) = values else {
            panic!("expected native complex single value");
        };
        assert_eq!(values.as_f32_slice(), Some(&[(0.0, 2.0)][..]));

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
            vec![(1.0, 0.0), (5.0, 0.0), (2.0, 1.0), (6.0, 1.0)],
            vec![2, 2],
        )
        .unwrap();
        let values = evaluate_sync(
            Value::ComplexTensor(a),
            Value::ComplexTensor(b),
            &[Value::from("rows")],
        )
        .expect("complex single row intersect")
        .values_value();
        let Value::ComplexTensor(values) = values else {
            panic!("expected native complex single rows");
        };
        assert_eq!(values.shape, vec![1, 2]);
        assert_eq!(values.as_f32_slice(), Some(&[(1.0, 0.0), (2.0, 1.0)][..]));
    }

    #[test]
    fn intersect_preserves_exact_integer_elements_and_rows() {
        let a = Tensor::new_integer(
            runmat_builtins::IntegerStorage::U64(vec![u64::MAX, 0, 9_007_199_254_740_993]),
            vec![3, 1],
        )
        .expect("input");
        let b = Tensor::new_integer(
            runmat_builtins::IntegerStorage::U64(vec![0, u64::MAX]),
            vec![2, 1],
        )
        .expect("input");
        let (values, ia, ib) = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[])
            .expect("intersect")
            .into_triple();
        let Value::Tensor(values) = values else {
            panic!("exact values");
        };
        assert_eq!(
            values.integer_storage(),
            Some(&runmat_builtins::IntegerStorage::U64(vec![0, u64::MAX]))
        );
        let Value::Tensor(ia) = ia else {
            panic!("indices");
        };
        assert_eq!(ia.materialize_f64(), vec![2.0, 1.0]);
        let Value::Tensor(ib) = ib else {
            panic!("indices");
        };
        assert_eq!(ib.materialize_f64(), vec![1.0, 2.0]);
    }

    #[test]
    fn intersect_rejects_mixed_nondouble_integer_classes() {
        let a = Tensor::new_integer(IntegerStorage::U16(vec![7, 2, 9, 7]), vec![4, 1]).unwrap();
        let b = Tensor::new_integer(IntegerStorage::I32(vec![2, 7]), vec![2, 1]).unwrap();

        let error = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[])
            .expect_err("mixed integer classes must reject");
        assert_eq!(
            error.identifier(),
            INTERSECT_ERROR_NUMERIC_CLASS_MISMATCH.identifier
        );
    }

    #[test]
    fn intersect_type_resolver_numeric() {
        assert_eq!(
            set_values_output_type(&[Type::tensor()], &ResolveContext::new(Vec::new())),
            Type::tensor()
        );
    }

    #[test]
    fn intersect_type_resolver_string_array() {
        assert_eq!(
            set_values_output_type(
                &[Type::cell_of(Type::String)],
                &ResolveContext::new(Vec::new()),
            ),
            Type::cell_of(Type::String)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn intersect_numeric_stable() {
        let a = Tensor::new(vec![4.0, 2.0, 4.0, 1.0, 3.0], vec![5, 1]).unwrap();
        let b = Tensor::new(vec![3.0, 4.0, 5.0, 1.0], vec![4, 1]).unwrap();
        let eval = intersect_numeric_elements(
            a,
            b,
            &IntersectOptions {
                rows: false,
                order: IntersectOrder::Stable,
            },
        )
        .expect("intersect");
        let values = tensor::value_into_tensor_for("intersect", eval.values_value()).unwrap();
        assert_eq!(values.materialize_f64(), vec![4.0, 1.0, 3.0]);
        let ia = tensor::value_into_tensor_for("intersect", eval.ia_value()).unwrap();
        let ib = tensor::value_into_tensor_for("intersect", eval.ib_value()).unwrap();
        assert_eq!(ia.materialize_f64(), vec![1.0, 4.0, 5.0]);
        assert_eq!(ib.materialize_f64(), vec![2.0, 4.0, 1.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn intersect_numeric_handles_nan() {
        let a = Tensor::new(vec![f64::NAN, 1.0, f64::NAN], vec![3, 1]).unwrap();
        let b = Tensor::new(vec![2.0, f64::NAN], vec![2, 1]).unwrap();
        let eval = intersect_numeric_elements(
            a,
            b,
            &IntersectOptions {
                rows: false,
                order: IntersectOrder::Sorted,
            },
        )
        .expect("intersect");
        let values = tensor::value_into_tensor_for("intersect", eval.values_value()).unwrap();
        assert_eq!(values.materialize_f64().len(), 1);
        assert!(values.materialize_f64()[0].is_nan());
        let ia = tensor::value_into_tensor_for("intersect", eval.ia_value()).unwrap();
        let ib = tensor::value_into_tensor_for("intersect", eval.ib_value()).unwrap();
        assert_eq!(ia.materialize_f64(), vec![1.0]);
        assert_eq!(ib.materialize_f64(), vec![2.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn intersect_complex_with_real_inputs() {
        let complex =
            ComplexTensor::new(vec![(1.0, 0.0), (2.0, 0.0), (3.0, 1.0)], vec![3, 1]).unwrap();
        let real = Tensor::new(vec![2.0, 4.0, 1.0], vec![3, 1]).unwrap();
        let real_complex = tensor_to_complex_owned("intersect", real).unwrap();
        let eval = intersect_complex(
            complex,
            real_complex,
            &IntersectOptions {
                rows: false,
                order: IntersectOrder::Sorted,
            },
        )
        .expect("intersect complex");
        match eval.values_value() {
            Value::ComplexTensor(t) => {
                assert_eq!(t.materialize_f64(), vec![(1.0, 0.0), (2.0, 0.0)]);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
        let ia = tensor::value_into_tensor_for("intersect", eval.ia_value()).unwrap();
        let ib = tensor::value_into_tensor_for("intersect", eval.ib_value()).unwrap();
        assert_eq!(ia.materialize_f64(), vec![1.0, 2.0]);
        assert_eq!(ib.materialize_f64(), vec![3.0, 1.0]);
    }

    #[test]
    fn intersect_complex_real_alignment_reads_typed_integer_storage_exactly() {
        let real =
            Tensor::new_integer(IntegerStorage::I64(vec![i64::MIN, -7, 3]), vec![3, 1]).unwrap();
        let complex = ComplexTensor::new(vec![(-7.0, 0.0), (4.0, 0.0)], vec![2, 1]).unwrap();

        let eval = evaluate_sync(Value::Tensor(real), Value::ComplexTensor(complex), &[])
            .expect("intersect");
        let Value::Complex(re, im) = eval.values_value() else {
            panic!("expected complex scalar");
        };
        assert_eq!(re, -7.0);
        assert_eq!(im, 0.0);
        let ia = tensor::value_into_tensor_for("intersect", eval.ia_value()).unwrap();
        let ib = tensor::value_into_tensor_for("intersect", eval.ib_value()).unwrap();
        assert_eq!(ia.materialize_f64(), vec![2.0]);
        assert_eq!(ib.materialize_f64(), vec![1.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn intersect_numeric_rows_default() {
        let a = Tensor::new(vec![1.0, 3.0, 1.0, 2.0, 4.0, 2.0], vec![3, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 5.0, 2.0, 6.0], vec![2, 2]).unwrap();
        let eval = intersect_numeric_rows(
            a,
            b,
            &IntersectOptions {
                rows: true,
                order: IntersectOrder::Sorted,
            },
        )
        .expect("intersect rows");
        let values = tensor::value_into_tensor_for("intersect", eval.values_value()).unwrap();
        assert_eq!(values.shape, vec![1, 2]);
        assert_eq!(values.materialize_f64(), vec![1.0, 2.0]);
        let ia = tensor::value_into_tensor_for("intersect", eval.ia_value()).unwrap();
        let ib = tensor::value_into_tensor_for("intersect", eval.ib_value()).unwrap();
        assert_eq!(ia.materialize_f64(), vec![1.0]);
        assert_eq!(ib.materialize_f64(), vec![1.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn intersect_char_elements_basic() {
        let a = CharArray::new("cab".chars().collect(), 1, 3).unwrap();
        let b = CharArray::new("bcd".chars().collect(), 1, 3).unwrap();
        assert_eq!(find_char_index(&b, 'b'), Some(0));
        assert_eq!(find_char_index(&b, 'c'), Some(1));
        let b_for_eval = CharArray::new("bcd".chars().collect(), 1, 3).unwrap();
        let eval = intersect_char_elements(
            a,
            b_for_eval,
            &IntersectOptions {
                rows: false,
                order: IntersectOrder::Sorted,
            },
        )
        .expect("intersect char");
        match eval.values_value() {
            Value::CharArray(arr) => {
                assert_eq!(arr.rows, 2);
                assert_eq!(arr.cols, 1);
                assert_eq!(arr.data, vec!['b', 'c']);
            }
            other => panic!("expected char array, got {other:?}"),
        }
        let ia = tensor::value_into_tensor_for("intersect", eval.ia_value()).unwrap();
        let ib = tensor::value_into_tensor_for("intersect", eval.ib_value()).unwrap();
        assert_eq!(ia.materialize_f64(), vec![3.0, 1.0]);
        assert_eq!(ib.materialize_f64(), vec![1.0, 2.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn intersect_string_elements_stable() {
        let a = StringArray::new(
            vec!["apple".into(), "orange".into(), "pear".into()],
            vec![3, 1],
        )
        .unwrap();
        let b = StringArray::new(
            vec!["pear".into(), "grape".into(), "orange".into()],
            vec![3, 1],
        )
        .unwrap();
        let eval = intersect_string_elements(
            a,
            b,
            &IntersectOptions {
                rows: false,
                order: IntersectOrder::Stable,
            },
        )
        .expect("intersect string");
        match eval.values_value() {
            Value::StringArray(arr) => {
                assert_eq!(arr.shape, vec![2, 1]);
                assert_eq!(arr.data, vec!["orange".to_string(), "pear".to_string()]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
        let ia = tensor::value_into_tensor_for("intersect", eval.ia_value()).unwrap();
        let ib = tensor::value_into_tensor_for("intersect", eval.ib_value()).unwrap();
        assert_eq!(ia.materialize_f64(), vec![2.0, 3.0]);
        assert_eq!(ib.materialize_f64(), vec![3.0, 1.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn intersect_rejects_legacy_option() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = evaluate_sync(
            Value::Tensor(tensor.clone()),
            Value::Tensor(tensor),
            &[Value::from("legacy")],
        )
        .unwrap_err();
        assert_eq!(
            err.identifier(),
            INTERSECT_ERROR_LEGACY_OPTION_UNSUPPORTED.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn intersect_rejects_conflicting_order_options() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = evaluate_sync(
            Value::Tensor(tensor.clone()),
            Value::Tensor(tensor),
            &[Value::from("stable"), Value::from("sorted")],
        )
        .unwrap_err();
        assert_eq!(
            err.identifier(),
            INTERSECT_ERROR_CONFLICTING_ORDER_OPTIONS.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn intersect_rejects_unknown_option() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = evaluate_sync(
            Value::Tensor(tensor.clone()),
            Value::Tensor(tensor),
            &[Value::from("bogus")],
        )
        .unwrap_err();
        assert_eq!(err.identifier(), INTERSECT_ERROR_UNKNOWN_OPTION.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn intersect_rows_dimension_mismatch() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = intersect_numeric_rows(
            a,
            b,
            &IntersectOptions {
                rows: true,
                order: IntersectOrder::Sorted,
            },
        )
        .unwrap_err();
        assert_eq!(
            err.identifier(),
            INTERSECT_ERROR_ROWS_COLUMN_MISMATCH.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn intersect_mixed_types_error() {
        let a = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let b = CharArray::new(vec!['a', 'b'], 1, 2).unwrap();
        let err = intersect_host(
            Value::Tensor(a),
            Value::CharArray(b),
            &IntersectOptions {
                rows: false,
                order: IntersectOrder::Sorted,
            },
        )
        .unwrap_err();
        assert_eq!(
            err.identifier(),
            INTERSECT_ERROR_UNSUPPORTED_INPUT_TYPE.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn intersect_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let a = Tensor::new(vec![4.0, 1.0, 2.0, 1.0], vec![4, 1]).unwrap();
            let b = Tensor::new(vec![2.0, 5.0, 1.0], vec![3, 1]).unwrap();
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
            let eval = evaluate_sync(Value::GpuTensor(handle_a), Value::GpuTensor(handle_b), &[])
                .expect("intersect");
            let values = tensor::value_into_tensor_for("intersect", eval.values_value()).unwrap();
            assert_eq!(values.materialize_f64(), vec![1.0, 2.0]);
            let ia = tensor::value_into_tensor_for("intersect", eval.ia_value()).unwrap();
            let ib = tensor::value_into_tensor_for("intersect", eval.ib_value()).unwrap();
            assert_eq!(ia.materialize_f64(), vec![2.0, 3.0]);
            assert_eq!(ib.materialize_f64(), vec![3.0, 1.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn intersect_two_outputs_from_evaluate() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let b = Tensor::new(vec![3.0, 1.0], vec![2, 1]).unwrap();
        let eval = intersect_numeric_elements(
            a,
            b,
            &IntersectOptions {
                rows: false,
                order: IntersectOrder::Sorted,
            },
        )
        .unwrap();
        let (_c, ia) = eval.clone().into_pair();
        let ia_tensor = tensor::value_into_tensor_for("intersect", ia).unwrap();
        assert_eq!(ia_tensor.materialize_f64(), vec![1.0, 3.0]);
        let (_c, ia2, ib2) = eval.into_triple();
        let ia_tensor2 = tensor::value_into_tensor_for("intersect", ia2).unwrap();
        let ib_tensor2 = tensor::value_into_tensor_for("intersect", ib2).unwrap();
        assert_eq!(ia_tensor2.materialize_f64(), vec![1.0, 3.0]);
        assert_eq!(ib_tensor2.materialize_f64(), vec![2.0, 1.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn intersect_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let a = Tensor::new(vec![4.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
        let b = Tensor::new(vec![2.0, 6.0, 3.0], vec![3, 1]).unwrap();

        let cpu_eval = intersect_numeric_elements(
            a.clone(),
            b.clone(),
            &IntersectOptions {
                rows: false,
                order: IntersectOrder::Sorted,
            },
        )
        .unwrap();
        let cpu_values =
            tensor::value_into_tensor_for("intersect", cpu_eval.values_value()).unwrap();
        let cpu_ia = tensor::value_into_tensor_for("intersect", cpu_eval.ia_value()).unwrap();
        let cpu_ib = tensor::value_into_tensor_for("intersect", cpu_eval.ib_value()).unwrap();

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
            .expect("intersect");
        let gpu_values =
            tensor::value_into_tensor_for("intersect", gpu_eval.values_value()).unwrap();
        let gpu_ia = tensor::value_into_tensor_for("intersect", gpu_eval.ia_value()).unwrap();
        let gpu_ib = tensor::value_into_tensor_for("intersect", gpu_eval.ib_value()).unwrap();

        assert_eq!(gpu_values.materialize_f64(), cpu_values.materialize_f64());
        assert_eq!(gpu_ia.materialize_f64(), cpu_ia.materialize_f64());
        assert_eq!(gpu_ib.materialize_f64(), cpu_ib.materialize_f64());
    }
}
