//! MATLAB-compatible `union` builtin with GPU-aware semantics for RunMat.
//!
//! Handles element-wise and row-wise unions with optional stable ordering and
//! index outputs that mirror MathWorks MATLAB semantics. GPU tensors are
//! gathered to host memory unless a provider supplies a dedicated `union`
//! kernel hook.

use std::cmp::Ordering;
use std::collections::{hash_map::Entry, HashMap};

use runmat_accelerate_api::{
    GpuTensorHandle, HostTensorOwned, UnionOptions, UnionOrder, UnionResult,
};
use runmat_builtins::{CharArray, ComplexTensor, StringArray, Tensor, Value};
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

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::sorting_sets::union")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "union",
    op_kind: GpuOpKind::Custom("union"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("union")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: true,
    notes: "Providers may expose a dedicated union hook; otherwise tensors are gathered and processed on the host.",
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

fn union_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message).with_builtin("union").build()
}

#[runtime_builtin(
    name = "union",
    category = "array/sorting_sets",
    summary = "Combine two arrays, returning their union with MATLAB-compatible ordering and index outputs.",
    keywords = "union,set,stable,rows,indices,gpu",
    accel = "array_construct",
    sink = true,
    type_resolver(set_values_output_type),
    builtin_path = "crate::builtins::array::sorting_sets::union"
)]
async fn union_builtin(a: Value, b: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
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

/// Evaluate the `union` builtin once and expose all outputs.
pub async fn evaluate(a: Value, b: Value, rest: &[Value]) -> crate::BuiltinResult<UnionEvaluation> {
    let opts = parse_options(rest)?;
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
                    .ok_or_else(|| union_error("union: expected string option arguments"))?;
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
                    return Err(union_error("union: cannot combine 'sorted' with 'stable'"));
                }
            }
            *seen_order = Some(UnionOrder::Sorted);
            opts.order = UnionOrder::Sorted;
        }
        "stable" => {
            if let Some(prev) = seen_order {
                if *prev != UnionOrder::Stable {
                    return Err(union_error("union: cannot combine 'sorted' with 'stable'"));
                }
            }
            *seen_order = Some(UnionOrder::Stable);
            opts.order = UnionOrder::Stable;
        }
        "legacy" | "r2012a" => {
            return Err(union_error(
                "union: the 'legacy' behaviour is not supported",
            ));
        }
        other => return Err(union_error(format!("union: unrecognised option '{other}'"))),
    }
    Ok(())
}

async fn union_gpu_pair(
    handle_a: GpuTensorHandle,
    handle_b: GpuTensorHandle,
    opts: &UnionOptions,
) -> crate::BuiltinResult<UnionEvaluation> {
    if let Some(provider) = runmat_accelerate_api::provider() {
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
    let tensor_other = tensor::value_into_tensor_for("union", other).map_err(|e| union_error(e))?;
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
                .map_err(|e| union_error(format!("union: {e}")))?;
            union_complex(at, bt, opts)
        }
        (Value::Complex(re, im), Value::ComplexTensor(bt)) => {
            let at = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| union_error(format!("union: {e}")))?;
            union_complex(at, bt, opts)
        }
        (Value::Complex(a_re, a_im), Value::Complex(b_re, b_im)) => {
            let at = ComplexTensor::new(vec![(a_re, a_im)], vec![1, 1])
                .map_err(|e| union_error(format!("union: {e}")))?;
            let bt = ComplexTensor::new(vec![(b_re, b_im)], vec![1, 1])
                .map_err(|e| union_error(format!("union: {e}")))?;
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
                .map_err(|e| union_error(format!("union: {e}")))?;
            union_string(astring, bstring, opts)
        }
        (Value::String(a), Value::StringArray(bstring)) => {
            let astring = StringArray::new(vec![a], vec![1, 1])
                .map_err(|e| union_error(format!("union: {e}")))?;
            union_string(astring, bstring, opts)
        }
        (Value::String(a), Value::String(b)) => {
            let astring = StringArray::new(vec![a], vec![1, 1])
                .map_err(|e| union_error(format!("union: {e}")))?;
            let bstring = StringArray::new(vec![b], vec![1, 1])
                .map_err(|e| union_error(format!("union: {e}")))?;
            union_string(astring, bstring, opts)
        }

        // Fallback to numeric (includes tensors, logical arrays, ints, bools, doubles)
        (left, right) => {
            let tensor_a =
                tensor::value_into_tensor_for("union", left).map_err(|e| union_error(e))?;
            let tensor_b =
                tensor::value_into_tensor_for("union", right).map_err(|e| union_error(e))?;
            union_numeric(tensor_a, tensor_b, opts)
        }
    }
}

fn union_numeric(
    a: Tensor,
    b: Tensor,
    opts: &UnionOptions,
) -> crate::BuiltinResult<UnionEvaluation> {
    if opts.rows {
        union_numeric_rows(a, b, opts)
    } else {
        union_numeric_elements(a, b, opts)
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

fn union_numeric_elements(
    a: Tensor,
    b: Tensor,
    opts: &UnionOptions,
) -> crate::BuiltinResult<UnionEvaluation> {
    let mut entries = Vec::<NumericUnionEntry>::new();
    let mut map: HashMap<u64, usize> = HashMap::new();
    let mut order_counter = 0usize;

    for (idx, &value) in a.data.iter().enumerate() {
        let key = canonicalize_f64(value);
        match map.entry(key) {
            Entry::Occupied(_) => {
                // Already recorded from A; keep first occurrence only.
            }
            Entry::Vacant(v) => {
                let entry_idx = entries.len();
                entries.push(NumericUnionEntry {
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

    for (idx, &value) in b.data.iter().enumerate() {
        let key = canonicalize_f64(value);
        match map.entry(key) {
            Entry::Occupied(occ) => {
                let entry = &mut entries[*occ.get()];
                if entry.a_index.is_none() && entry.b_index.is_none() {
                    entry.b_index = Some(idx);
                }
            }
            Entry::Vacant(v) => {
                let entry_idx = entries.len();
                entries.push(NumericUnionEntry {
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

    assemble_numeric_union(entries, opts)
}

fn union_numeric_rows(
    a: Tensor,
    b: Tensor,
    opts: &UnionOptions,
) -> crate::BuiltinResult<UnionEvaluation> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(union_error(
            "union: 'rows' option requires 2-D numeric matrices",
        ));
    }
    if a.shape[1] != b.shape[1] {
        return Err(union_error(
            "union: inputs must have the same number of columns when using 'rows'",
        ));
    }
    let rows_a = a.shape[0];
    let cols = a.shape[1];
    let rows_b = b.shape[0];

    let mut entries = Vec::<NumericRowUnionEntry>::new();
    let mut map: HashMap<NumericRowKey, usize> = HashMap::new();
    let mut order_counter = 0usize;

    for r in 0..rows_a {
        let mut row_values = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r + c * rows_a;
            row_values.push(a.data[idx]);
        }
        let key = NumericRowKey::from_slice(&row_values);
        match map.entry(key) {
            Entry::Occupied(_) => {}
            Entry::Vacant(v) => {
                let entry_idx = entries.len();
                entries.push(NumericRowUnionEntry {
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
            row_values.push(b.data[idx]);
        }
        let key = NumericRowKey::from_slice(&row_values);
        match map.entry(key) {
            Entry::Occupied(occ) => {
                let entry = &mut entries[*occ.get()];
                if entry.a_row.is_none() && entry.b_row.is_none() {
                    entry.b_row = Some(r);
                }
            }
            Entry::Vacant(v) => {
                let entry_idx = entries.len();
                entries.push(NumericRowUnionEntry {
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

    assemble_numeric_row_union(entries, opts, cols)
}

fn union_complex(
    a: ComplexTensor,
    b: ComplexTensor,
    opts: &UnionOptions,
) -> crate::BuiltinResult<UnionEvaluation> {
    if opts.rows {
        union_complex_rows(a, b, opts)
    } else {
        union_complex_elements(a, b, opts)
    }
}

fn union_complex_elements(
    a: ComplexTensor,
    b: ComplexTensor,
    opts: &UnionOptions,
) -> crate::BuiltinResult<UnionEvaluation> {
    let mut entries = Vec::<ComplexUnionEntry>::new();
    let mut map: HashMap<ComplexKey, usize> = HashMap::new();
    let mut order_counter = 0usize;

    for (idx, &value) in a.data.iter().enumerate() {
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

    for (idx, &value) in b.data.iter().enumerate() {
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

fn union_complex_rows(
    a: ComplexTensor,
    b: ComplexTensor,
    opts: &UnionOptions,
) -> crate::BuiltinResult<UnionEvaluation> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(union_error(
            "union: 'rows' option requires 2-D complex matrices",
        ));
    }
    if a.shape[1] != b.shape[1] {
        return Err(union_error(
            "union: inputs must have the same number of columns when using 'rows'",
        ));
    }
    let rows_a = a.shape[0];
    let cols = a.shape[1];
    let rows_b = b.shape[0];

    let mut entries = Vec::<ComplexRowUnionEntry>::new();
    let mut map: HashMap<Vec<ComplexKey>, usize> = HashMap::new();
    let mut order_counter = 0usize;

    for r in 0..rows_a {
        let mut row_values = Vec::with_capacity(cols);
        let mut key_row = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r + c * rows_a;
            let value = a.data[idx];
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
            let value = b.data[idx];
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
        return Err(union_error(
            "union: inputs must have the same number of columns when using 'rows'",
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
        return Err(union_error(
            "union: 'rows' option requires 2-D string arrays",
        ));
    }
    if a.shape[1] != b.shape[1] {
        return Err(union_error(
            "union: inputs must have the same number of columns when using 'rows'",
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
            .map_err(|e| union_error(format!("union: {e}")))?;
        let ia_tensor =
            Tensor::new(ia.data, ia.shape).map_err(|e| union_error(format!("union: {e}")))?;
        let ib_tensor =
            Tensor::new(ib.data, ib.shape).map_err(|e| union_error(format!("union: {e}")))?;
        Ok(UnionEvaluation::new(
            tensor::tensor_into_value(values_tensor),
            ia_tensor,
            ib_tensor,
        ))
    }

    pub fn into_numeric_union_result(self) -> crate::BuiltinResult<UnionResult> {
        let UnionEvaluation { values, ia, ib } = self;
        let values_tensor =
            tensor::value_into_tensor_for("union", values).map_err(|e| union_error(e))?;
        Ok(UnionResult {
            values: HostTensorOwned {
                data: values_tensor.data,
                shape: values_tensor.shape,
            },
            ia: HostTensorOwned {
                data: ia.data,
                shape: ia.shape,
            },
            ib: HostTensorOwned {
                data: ib.data,
                shape: ib.shape,
            },
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
struct NumericUnionEntry {
    value: f64,
    a_index: Option<usize>,
    b_index: Option<usize>,
    order_rank: usize,
}

#[derive(Debug)]
struct NumericRowUnionEntry {
    row_data: Vec<f64>,
    a_row: Option<usize>,
    b_row: Option<usize>,
    order_rank: usize,
}

#[derive(Debug)]
struct ComplexUnionEntry {
    value: (f64, f64),
    a_index: Option<usize>,
    b_index: Option<usize>,
    order_rank: usize,
}

#[derive(Debug)]
struct ComplexRowUnionEntry {
    row_data: Vec<(f64, f64)>,
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

fn assemble_numeric_union(
    entries: Vec<NumericUnionEntry>,
    opts: &UnionOptions,
) -> crate::BuiltinResult<UnionEvaluation> {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    match opts.order {
        UnionOrder::Sorted => {
            order.sort_by(|&lhs, &rhs| compare_f64(entries[lhs].value, entries[rhs].value));
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

    let value_tensor = Tensor::new(values, vec![order.len(), 1])
        .map_err(|e| union_error(format!("union: {e}")))?;
    let ia_len = ia.len();
    let ib_len = ib.len();
    let ia_tensor =
        Tensor::new(ia, vec![ia_len, 1]).map_err(|e| union_error(format!("union: {e}")))?;
    let ib_tensor =
        Tensor::new(ib, vec![ib_len, 1]).map_err(|e| union_error(format!("union: {e}")))?;

    Ok(UnionEvaluation::new(
        tensor::tensor_into_value(value_tensor),
        ia_tensor,
        ib_tensor,
    ))
}

fn assemble_numeric_row_union(
    entries: Vec<NumericRowUnionEntry>,
    opts: &UnionOptions,
    cols: usize,
) -> crate::BuiltinResult<UnionEvaluation> {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    match opts.order {
        UnionOrder::Sorted => {
            order.sort_by(|&lhs, &rhs| {
                compare_numeric_rows(&entries[lhs].row_data, &entries[rhs].row_data)
            });
        }
        UnionOrder::Stable => {
            order.sort_by_key(|&idx| entries[idx].order_rank);
        }
    }

    let unique_rows = order.len();
    let mut values = vec![0.0f64; unique_rows * cols];
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

    let value_tensor = Tensor::new(values, vec![unique_rows, cols])
        .map_err(|e| union_error(format!("union: {e}")))?;
    let ia_len = ia.len();
    let ib_len = ib.len();
    let ia_tensor =
        Tensor::new(ia, vec![ia_len, 1]).map_err(|e| union_error(format!("union: {e}")))?;
    let ib_tensor =
        Tensor::new(ib, vec![ib_len, 1]).map_err(|e| union_error(format!("union: {e}")))?;

    Ok(UnionEvaluation::new(
        tensor::tensor_into_value(value_tensor),
        ia_tensor,
        ib_tensor,
    ))
}

fn assemble_complex_union(
    entries: Vec<ComplexUnionEntry>,
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

    let value_tensor = ComplexTensor::new(values, vec![order.len(), 1])
        .map_err(|e| union_error(format!("union: {e}")))?;
    let ia_len = ia.len();
    let ib_len = ib.len();
    let ia_tensor =
        Tensor::new(ia, vec![ia_len, 1]).map_err(|e| union_error(format!("union: {e}")))?;
    let ib_tensor =
        Tensor::new(ib, vec![ib_len, 1]).map_err(|e| union_error(format!("union: {e}")))?;

    Ok(UnionEvaluation::new(
        complex_tensor_into_value(value_tensor),
        ia_tensor,
        ib_tensor,
    ))
}

fn assemble_complex_row_union(
    entries: Vec<ComplexRowUnionEntry>,
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
    let mut values = vec![(0.0, 0.0); unique_rows * cols];
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

    let value_tensor = ComplexTensor::new(values, vec![unique_rows, cols])
        .map_err(|e| union_error(format!("union: {e}")))?;
    let ia_len = ia.len();
    let ib_len = ib.len();
    let ia_tensor =
        Tensor::new(ia, vec![ia_len, 1]).map_err(|e| union_error(format!("union: {e}")))?;
    let ib_tensor =
        Tensor::new(ib, vec![ib_len, 1]).map_err(|e| union_error(format!("union: {e}")))?;

    Ok(UnionEvaluation::new(
        complex_tensor_into_value(value_tensor),
        ia_tensor,
        ib_tensor,
    ))
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

    let value_array =
        CharArray::new(values, order.len(), 1).map_err(|e| union_error(format!("union: {e}")))?;
    let ia_len = ia.len();
    let ib_len = ib.len();
    let ia_tensor =
        Tensor::new(ia, vec![ia_len, 1]).map_err(|e| union_error(format!("union: {e}")))?;
    let ib_tensor =
        Tensor::new(ib, vec![ib_len, 1]).map_err(|e| union_error(format!("union: {e}")))?;

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
        .map_err(|e| union_error(format!("union: {e}")))?;
    let ia_len = ia.len();
    let ib_len = ib.len();
    let ia_tensor =
        Tensor::new(ia, vec![ia_len, 1]).map_err(|e| union_error(format!("union: {e}")))?;
    let ib_tensor =
        Tensor::new(ib, vec![ib_len, 1]).map_err(|e| union_error(format!("union: {e}")))?;

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
        .map_err(|e| union_error(format!("union: {e}")))?;
    let ia_len = ia.len();
    let ib_len = ib.len();
    let ia_tensor =
        Tensor::new(ia, vec![ia_len, 1]).map_err(|e| union_error(format!("union: {e}")))?;
    let ib_tensor =
        Tensor::new(ib, vec![ib_len, 1]).map_err(|e| union_error(format!("union: {e}")))?;

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
        .map_err(|e| union_error(format!("union: {e}")))?;
    let ia_len = ia.len();
    let ib_len = ib.len();
    let ia_tensor =
        Tensor::new(ia, vec![ia_len, 1]).map_err(|e| union_error(format!("union: {e}")))?;
    let ib_tensor =
        Tensor::new(ib, vec![ib_len, 1]).map_err(|e| union_error(format!("union: {e}")))?;

    Ok(UnionEvaluation::new(
        Value::StringArray(value_array),
        ia_tensor,
        ib_tensor,
    ))
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
    use runmat_builtins::{IntValue, ResolveContext, Tensor, Type, Value};

    fn error_message(err: crate::RuntimeError) -> String {
        err.message().to_string()
    }

    fn evaluate_sync(a: Value, b: Value, rest: &[Value]) -> crate::BuiltinResult<UnionEvaluation> {
        futures::executor::block_on(evaluate(a, b, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn union_numeric_sorted_default() {
        let a = Tensor::new(vec![5.0, 7.0, 1.0], vec![3, 1]).unwrap();
        let b = Tensor::new(vec![3.0, 1.0, 1.0], vec![3, 1]).unwrap();
        let eval = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[]).expect("union");
        match eval.values_value() {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![1.0, 3.0, 5.0, 7.0]);
                assert_eq!(t.shape, vec![4, 1]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        let ia = tensor::value_into_tensor_for("union", eval.ia_value()).expect("ia tensor");
        assert_eq!(ia.data, vec![3.0, 1.0, 2.0]);
        assert_eq!(ia.shape, vec![3, 1]);
        let ib = tensor::value_into_tensor_for("union", eval.ib_value()).expect("ib tensor");
        assert_eq!(ib.data, vec![1.0]);
        assert_eq!(ib.shape, vec![1, 1]);
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
                assert_eq!(t.data, vec![5.0, 7.0, 1.0, 3.0, 2.0, 4.0]);
                assert_eq!(t.shape, vec![6, 1]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        let ia = tensor::value_into_tensor_for("union", eval.ia_value()).expect("ia tensor");
        assert_eq!(ia.data, vec![1.0, 2.0, 3.0]);
        let ib = tensor::value_into_tensor_for("union", eval.ib_value()).expect("ib tensor");
        assert_eq!(ib.data, vec![1.0, 2.0, 3.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn union_numeric_sorted_places_nan_last() {
        let a = Tensor::new(vec![f64::NAN, 1.0], vec![2, 1]).unwrap();
        let b = Tensor::new(vec![2.0, f64::NAN], vec![2, 1]).unwrap();
        let eval = evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[]).expect("union");
        let values = tensor::value_into_tensor_for("union", eval.values_value()).expect("values");
        assert_eq!(values.shape, vec![3, 1]);
        assert_eq!(values.data[0], 1.0);
        assert_eq!(values.data[1], 2.0);
        assert!(values.data[2].is_nan());
        let ia = tensor::value_into_tensor_for("union", eval.ia_value()).expect("ia tensor");
        assert_eq!(ia.data, vec![2.0, 1.0]);
        let ib = tensor::value_into_tensor_for("union", eval.ib_value()).expect("ib tensor");
        assert_eq!(ib.data, vec![1.0]);
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
                assert_eq!(t.data, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        let ia = tensor::value_into_tensor_for("union", eval.ia_value()).expect("ia tensor");
        assert_eq!(ia.data, vec![1.0, 2.0]);
        let ib = tensor::value_into_tensor_for("union", eval.ib_value()).expect("ib tensor");
        assert_eq!(ib.data, vec![2.0]);
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
                assert_eq!(t.data, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        let ia_tensor = tensor::value_into_tensor_for("union", ia).expect("ia tensor");
        assert_eq!(ia_tensor.data, vec![1.0, 2.0]);
        let ib_tensor = tensor::value_into_tensor_for("union", ib).expect("ib tensor");
        assert_eq!(ib_tensor.data, vec![2.0]);
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
        assert_eq!(ia.data, vec![4.0, 1.0, 3.0]);
        let ib = tensor::value_into_tensor_for("union", eval.ib_value()).expect("ib tensor");
        assert_eq!(ib.data, vec![3.0]);
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
        assert_eq!(ia.data, vec![1.0, 2.0]);
        let ib = tensor::value_into_tensor_for("union", eval.ib_value()).expect("ib tensor");
        assert_eq!(ib.data, vec![2.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn union_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let a = Tensor::new(vec![4.0, 1.0, 2.0], vec![3, 1]).unwrap();
            let b = Tensor::new(vec![2.0, 5.0], vec![2, 1]).unwrap();
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
            let eval = evaluate_sync(
                Value::GpuTensor(handle_a),
                Value::GpuTensor(handle_b),
                &[Value::from("stable")],
            )
            .expect("union");
            let values = tensor::value_into_tensor_for("union", eval.values_value()).unwrap();
            assert_eq!(values.data, vec![4.0, 1.0, 2.0, 5.0]);
            let ia = tensor::value_into_tensor_for("union", eval.ia_value()).unwrap();
            assert_eq!(ia.data, vec![1.0, 2.0, 3.0]);
            let ib = tensor::value_into_tensor_for("union", eval.ib_value()).unwrap();
            assert_eq!(ib.data, vec![2.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn union_rejects_legacy_option() {
        let tensor =
            Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).expect("tensor construction failed");
        let err = error_message(
            evaluate_sync(
                Value::Tensor(tensor.clone()),
                Value::Tensor(tensor),
                &[Value::from("legacy")],
            )
            .unwrap_err(),
        );
        assert!(err.contains("legacy"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn union_rows_dimension_mismatch() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = error_message(
            evaluate_sync(Value::Tensor(a), Value::Tensor(b), &[Value::from("rows")]).unwrap_err(),
        );
        assert!(err.contains("same number of columns"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn union_requires_matching_types() {
        let a = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let b = CharArray::new(vec!['a', 'b'], 1, 2).unwrap();
        let err = error_message(
            union_host(
                Value::Tensor(a),
                Value::CharArray(b),
                &UnionOptions {
                    rows: false,
                    order: UnionOrder::Sorted,
                },
            )
            .unwrap_err(),
        );
        assert!(err.contains("unsupported input type"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn union_accepts_scalar_inputs() {
        let eval =
            evaluate_sync(Value::Int(IntValue::I32(1)), Value::Num(3.0), &[]).expect("union");
        match eval.values_value() {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![1.0, 3.0]);
                assert_eq!(t.shape, vec![2, 1]);
            }
            other => panic!("expected numeric tensor, got {other:?}"),
        }
        let ia = tensor::value_into_tensor_for("union", eval.ia_value()).unwrap();
        assert_eq!(ia.data, vec![1.0]);
        let ib = tensor::value_into_tensor_for("union", eval.ib_value()).unwrap();
        assert_eq!(ib.data, vec![1.0]);
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
            .expect("union");
        let gpu_values = tensor::value_into_tensor_for("union", gpu_eval.values_value()).unwrap();
        let gpu_ia = tensor::value_into_tensor_for("union", gpu_eval.ia_value()).unwrap();
        let gpu_ib = tensor::value_into_tensor_for("union", gpu_eval.ib_value()).unwrap();

        assert_eq!(gpu_values.data, cpu_values.data);
        assert_eq!(gpu_values.shape, cpu_values.shape);
        assert_eq!(gpu_ia.data, cpu_ia.data);
        assert_eq!(gpu_ib.data, cpu_ib.data);
    }
}
