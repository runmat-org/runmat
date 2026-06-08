//! MATLAB-compatible `intersect` builtin with GPU-aware semantics for RunMat.
//!
//! Supports element-wise and row-wise intersections with optional stable ordering,
//! and index outputs that mirror MathWorks MATLAB semantics. GPU tensors are
//! gathered to host memory unless a provider supplies a dedicated `intersect`
//! kernel hook.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, ComplexTensor, StringArray, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use super::type_resolvers::set_values_output_type;
use crate::build_runtime_error;
use crate::builtins::common::arg_tokens::tokens_from_values;
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::array::sorting_sets::intersect"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "intersect",
    op_kind: GpuOpKind::Custom("intersect"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("intersect")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: true,
    notes:
        "Providers may expose a dedicated intersect hook; otherwise tensors are gathered and processed on the host.",
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

const INTERSECT_ERRORS: [BuiltinErrorDescriptor; 7] = [
    INTERSECT_ERROR_LEGACY_OPTION_UNSUPPORTED,
    INTERSECT_ERROR_CONFLICTING_ORDER_OPTIONS,
    INTERSECT_ERROR_UNKNOWN_OPTION,
    INTERSECT_ERROR_ROWS_COLUMN_MISMATCH,
    INTERSECT_ERROR_UNSUPPORTED_INPUT_TYPE,
    INTERSECT_ERROR_INVALID_ARGUMENT,
    INTERSECT_ERROR_INTERNAL,
];

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
    builtin_path = "crate::builtins::array::sorting_sets::intersect"
)]
async fn intersect_builtin(a: Value, b: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let eval = evaluate(a, b, &rest).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::OutputList(vec![eval.into_values_value()]));
        }
        if out_count == 2 {
            let (values, ia) = eval.into_pair();
            return Ok(Value::OutputList(vec![values, ia]));
        }
        let (values, ia, ib) = eval.into_triple();
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![values, ia, ib],
        ));
    }
    Ok(eval.into_values_value())
}

/// Evaluate the `intersect` builtin once and expose all outputs.
pub async fn evaluate(
    a: Value,
    b: Value,
    rest: &[Value],
) -> crate::BuiltinResult<IntersectEvaluation> {
    let opts = parse_options(rest)?;
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
    if opts.rows {
        intersect_numeric_rows(a, b, opts)
    } else {
        intersect_numeric_elements(a, b, opts)
    }
}

fn intersect_numeric_elements(
    a: Tensor,
    b: Tensor,
    opts: &IntersectOptions,
) -> crate::BuiltinResult<IntersectEvaluation> {
    let mut b_map: HashMap<u64, usize> = HashMap::new();
    for (idx, &value) in b.data.iter().enumerate() {
        let key = canonicalize_f64(value);
        b_map.entry(key).or_insert(idx);
    }

    let mut seen: HashSet<u64> = HashSet::new();
    let mut entries = Vec::<NumericIntersectEntry>::new();
    let mut order_counter = 0usize;

    for (idx, &value) in a.data.iter().enumerate() {
        let key = canonicalize_f64(value);
        if seen.contains(&key) {
            continue;
        }
        if let Some(&b_idx) = b_map.get(&key) {
            entries.push(NumericIntersectEntry {
                value,
                a_index: idx,
                b_index: b_idx,
                order_rank: order_counter,
            });
            seen.insert(key);
            order_counter += 1;
        }
    }

    assemble_numeric_intersect(entries, opts)
}

fn intersect_numeric_rows(
    a: Tensor,
    b: Tensor,
    opts: &IntersectOptions,
) -> crate::BuiltinResult<IntersectEvaluation> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(intersect_internal_error(
            "intersect: 'rows' option requires 2-D numeric matrices",
        ));
    }
    if a.shape[1] != b.shape[1] {
        return Err(intersect_error(&INTERSECT_ERROR_ROWS_COLUMN_MISMATCH));
    }
    let rows_a = a.shape[0];
    let cols = a.shape[1];
    let rows_b = b.shape[0];

    let mut b_map: HashMap<NumericRowKey, usize> = HashMap::new();
    for r in 0..rows_b {
        let mut row_values = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r + c * rows_b;
            row_values.push(b.data[idx]);
        }
        let key = NumericRowKey::from_slice(&row_values);
        b_map.entry(key).or_insert(r);
    }

    let mut seen: HashSet<NumericRowKey> = HashSet::new();
    let mut entries = Vec::<NumericRowIntersectEntry>::new();
    let mut order_counter = 0usize;

    for r in 0..rows_a {
        let mut row_values = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r + c * rows_a;
            row_values.push(a.data[idx]);
        }
        let key = NumericRowKey::from_slice(&row_values);
        if seen.contains(&key) {
            continue;
        }
        if let Some(&b_row) = b_map.get(&key) {
            entries.push(NumericRowIntersectEntry {
                row_data: row_values,
                a_row: r,
                b_row,
                order_rank: order_counter,
            });
            seen.insert(key);
            order_counter += 1;
        }
    }

    assemble_numeric_row_intersect(entries, opts, cols)
}

fn intersect_complex(
    a: ComplexTensor,
    b: ComplexTensor,
    opts: &IntersectOptions,
) -> crate::BuiltinResult<IntersectEvaluation> {
    if opts.rows {
        intersect_complex_rows(a, b, opts)
    } else {
        intersect_complex_elements(a, b, opts)
    }
}

fn intersect_complex_elements(
    a: ComplexTensor,
    b: ComplexTensor,
    opts: &IntersectOptions,
) -> crate::BuiltinResult<IntersectEvaluation> {
    let mut b_map: HashMap<ComplexKey, usize> = HashMap::new();
    for (idx, &value) in b.data.iter().enumerate() {
        let key = ComplexKey::new(value);
        b_map.entry(key).or_insert(idx);
    }

    let mut seen: HashSet<ComplexKey> = HashSet::new();
    let mut entries = Vec::<ComplexIntersectEntry>::new();
    let mut order_counter = 0usize;

    for (idx, &value) in a.data.iter().enumerate() {
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

fn intersect_complex_rows(
    a: ComplexTensor,
    b: ComplexTensor,
    opts: &IntersectOptions,
) -> crate::BuiltinResult<IntersectEvaluation> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(intersect_internal_error(
            "intersect: 'rows' option requires 2-D complex matrices",
        ));
    }
    if a.shape[1] != b.shape[1] {
        return Err(intersect_error(&INTERSECT_ERROR_ROWS_COLUMN_MISMATCH));
    }
    let rows_a = a.shape[0];
    let cols = a.shape[1];
    let rows_b = b.shape[0];

    let mut b_map: HashMap<Vec<ComplexKey>, usize> = HashMap::new();
    for r in 0..rows_b {
        let mut row_keys = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r + c * rows_b;
            row_keys.push(ComplexKey::new(b.data[idx]));
        }
        b_map.entry(row_keys).or_insert(r);
    }

    let mut seen: HashSet<Vec<ComplexKey>> = HashSet::new();
    let mut entries = Vec::<ComplexRowIntersectEntry>::new();
    let mut order_counter = 0usize;

    for r in 0..rows_a {
        let mut row_values = Vec::with_capacity(cols);
        let mut row_keys = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r + c * rows_a;
            let value = a.data[idx];
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
struct NumericIntersectEntry {
    value: f64,
    a_index: usize,
    b_index: usize,
    order_rank: usize,
}

#[derive(Debug)]
struct NumericRowIntersectEntry {
    row_data: Vec<f64>,
    a_row: usize,
    b_row: usize,
    order_rank: usize,
}

#[derive(Debug)]
struct ComplexIntersectEntry {
    value: (f64, f64),
    a_index: usize,
    b_index: usize,
    order_rank: usize,
}

#[derive(Debug)]
struct ComplexRowIntersectEntry {
    row_data: Vec<(f64, f64)>,
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

fn assemble_numeric_intersect(
    entries: Vec<NumericIntersectEntry>,
    opts: &IntersectOptions,
) -> crate::BuiltinResult<IntersectEvaluation> {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    match opts.order {
        IntersectOrder::Sorted => {
            order.sort_by(|&lhs, &rhs| compare_f64(entries[lhs].value, entries[rhs].value));
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

    let value_tensor = Tensor::new(values, vec![order.len(), 1])
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

fn assemble_numeric_row_intersect(
    entries: Vec<NumericRowIntersectEntry>,
    opts: &IntersectOptions,
    cols: usize,
) -> crate::BuiltinResult<IntersectEvaluation> {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    match opts.order {
        IntersectOrder::Sorted => {
            order.sort_by(|&lhs, &rhs| {
                compare_numeric_rows(&entries[lhs].row_data, &entries[rhs].row_data)
            });
        }
        IntersectOrder::Stable => {
            order.sort_by_key(|&idx| entries[idx].order_rank);
        }
    }

    let rows_out = order.len();
    let mut values = vec![0.0f64; rows_out * cols];
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

    let value_tensor = Tensor::new(values, vec![rows_out, cols])
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

fn assemble_complex_intersect(
    entries: Vec<ComplexIntersectEntry>,
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

    let value_tensor = ComplexTensor::new(values, vec![order.len(), 1])
        .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;
    let ia_tensor = Tensor::new(ia, vec![order.len(), 1])
        .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;
    let ib_tensor = Tensor::new(ib, vec![order.len(), 1])
        .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;

    Ok(IntersectEvaluation::new(
        complex_tensor_into_value(value_tensor),
        ia_tensor,
        ib_tensor,
    ))
}

fn assemble_complex_row_intersect(
    entries: Vec<ComplexRowIntersectEntry>,
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
    let mut values = vec![(0.0f64, 0.0f64); rows_out * cols];
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

    let value_tensor = ComplexTensor::new(values, vec![rows_out, cols])
        .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;
    let ia_tensor = Tensor::new(ia, vec![rows_out, 1])
        .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;
    let ib_tensor = Tensor::new(ib, vec![rows_out, 1])
        .map_err(|e| intersect_internal_error(format!("intersect: {e}")))?;

    Ok(IntersectEvaluation::new(
        complex_tensor_into_value(value_tensor),
        ia_tensor,
        ib_tensor,
    ))
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
struct NumericRowKey(Vec<u64>);

impl NumericRowKey {
    fn from_slice(values: &[f64]) -> Self {
        NumericRowKey(values.iter().map(|&v| canonicalize_f64(v)).collect())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ComplexKey {
    re: u64,
    im: u64,
}

impl ComplexKey {
    fn new(value: (f64, f64)) -> Self {
        Self {
            re: canonicalize_f64(value.0),
            im: canonicalize_f64(value.1),
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
    let Tensor { data, shape, .. } = tensor;
    let complex: Vec<(f64, f64)> = data.into_iter().map(|re| (re, 0.0)).collect();
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

fn canonicalize_f64(value: f64) -> u64 {
    if value.is_nan() {
        0x7ff8_0000_0000_0000u64
    } else if value == 0.0 {
        0u64
    } else {
        value.to_bits()
    }
}

fn compare_f64(a: f64, b: f64) -> Ordering {
    if a.is_nan() {
        if b.is_nan() {
            Ordering::Equal
        } else {
            Ordering::Greater
        }
    } else if b.is_nan() {
        Ordering::Less
    } else {
        a.partial_cmp(&b).unwrap_or(Ordering::Equal)
    }
}

fn compare_numeric_rows(a: &[f64], b: &[f64]) -> Ordering {
    for (lhs, rhs) in a.iter().zip(b.iter()) {
        let ord = compare_f64(*lhs, *rhs);
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

fn complex_is_nan(value: (f64, f64)) -> bool {
    value.0.is_nan() || value.1.is_nan()
}

fn compare_complex(a: (f64, f64), b: (f64, f64)) -> Ordering {
    match (complex_is_nan(a), complex_is_nan(b)) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => {
            let mag_a = a.0.hypot(a.1);
            let mag_b = b.0.hypot(b.1);
            let mag_cmp = compare_f64(mag_a, mag_b);
            if mag_cmp != Ordering::Equal {
                return mag_cmp;
            }
            let re_cmp = compare_f64(a.0, b.0);
            if re_cmp != Ordering::Equal {
                return re_cmp;
            }
            compare_f64(a.1, b.1)
        }
    }
}

fn compare_complex_rows(a: &[(f64, f64)], b: &[(f64, f64)]) -> Ordering {
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
        assert_eq!(values.data, vec![1.0, 7.0]);
        let ia = tensor::value_into_tensor_for("intersect", eval.ia_value()).unwrap();
        let ib = tensor::value_into_tensor_for("intersect", eval.ib_value()).unwrap();
        assert_eq!(ia.data, vec![4.0, 2.0]);
        assert_eq!(ib.data, vec![2.0, 1.0]);
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
        assert_eq!(values.data, vec![4.0, 1.0, 3.0]);
        let ia = tensor::value_into_tensor_for("intersect", eval.ia_value()).unwrap();
        let ib = tensor::value_into_tensor_for("intersect", eval.ib_value()).unwrap();
        assert_eq!(ia.data, vec![1.0, 4.0, 5.0]);
        assert_eq!(ib.data, vec![2.0, 4.0, 1.0]);
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
        assert_eq!(values.data.len(), 1);
        assert!(values.data[0].is_nan());
        let ia = tensor::value_into_tensor_for("intersect", eval.ia_value()).unwrap();
        let ib = tensor::value_into_tensor_for("intersect", eval.ib_value()).unwrap();
        assert_eq!(ia.data, vec![1.0]);
        assert_eq!(ib.data, vec![2.0]);
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
                assert_eq!(t.data, vec![(1.0, 0.0), (2.0, 0.0)]);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
        let ia = tensor::value_into_tensor_for("intersect", eval.ia_value()).unwrap();
        let ib = tensor::value_into_tensor_for("intersect", eval.ib_value()).unwrap();
        assert_eq!(ia.data, vec![1.0, 2.0]);
        assert_eq!(ib.data, vec![3.0, 1.0]);
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
        assert_eq!(values.data, vec![1.0, 2.0]);
        let ia = tensor::value_into_tensor_for("intersect", eval.ia_value()).unwrap();
        let ib = tensor::value_into_tensor_for("intersect", eval.ib_value()).unwrap();
        assert_eq!(ia.data, vec![1.0]);
        assert_eq!(ib.data, vec![1.0]);
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
        assert_eq!(ia.data, vec![3.0, 1.0]);
        assert_eq!(ib.data, vec![1.0, 2.0]);
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
        assert_eq!(ia.data, vec![2.0, 3.0]);
        assert_eq!(ib.data, vec![3.0, 1.0]);
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
                data: &a.data,
                shape: &a.shape,
            };
            let view_b = HostTensorView {
                data: &b.data,
                shape: &b.shape,
            };
            let handle_a = provider.upload(&view_a).expect("upload A");
            let handle_b = provider.upload(&view_b).expect("upload B");
            let eval = evaluate_sync(Value::GpuTensor(handle_a), Value::GpuTensor(handle_b), &[])
                .expect("intersect");
            let values = tensor::value_into_tensor_for("intersect", eval.values_value()).unwrap();
            assert_eq!(values.data, vec![1.0, 2.0]);
            let ia = tensor::value_into_tensor_for("intersect", eval.ia_value()).unwrap();
            let ib = tensor::value_into_tensor_for("intersect", eval.ib_value()).unwrap();
            assert_eq!(ia.data, vec![2.0, 3.0]);
            assert_eq!(ib.data, vec![3.0, 1.0]);
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
        assert_eq!(ia_tensor.data, vec![1.0, 3.0]);
        let (_c, ia2, ib2) = eval.into_triple();
        let ia_tensor2 = tensor::value_into_tensor_for("intersect", ia2).unwrap();
        let ib_tensor2 = tensor::value_into_tensor_for("intersect", ib2).unwrap();
        assert_eq!(ia_tensor2.data, vec![1.0, 3.0]);
        assert_eq!(ib_tensor2.data, vec![2.0, 1.0]);
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
            data: &a.data,
            shape: &a.shape,
        };
        let view_b = HostTensorView {
            data: &b.data,
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

        assert_eq!(gpu_values.data, cpu_values.data);
        assert_eq!(gpu_ia.data, cpu_ia.data);
        assert_eq!(gpu_ib.data, cpu_ib.data);
    }
}
