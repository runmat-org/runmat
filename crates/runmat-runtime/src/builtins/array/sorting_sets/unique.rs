//! MATLAB-compatible `unique` builtin with GPU-aware semantics for RunMat.
//!
//! The implementation mirrors MathWorks MATLAB behavioural details for sorted
//! and stable orderings, row-wise uniqueness, and index outputs. GPU tensors
//! are gathered to host memory today, but the builtin is registered as a sink
//! so future providers can add device-side kernels without impacting callers.

use std::cmp::Ordering;
use std::collections::HashMap;

use runmat_accelerate_api::{
    GpuTensorHandle, HostTensorOwned, UniqueOccurrence, UniqueOptions, UniqueOrder, UniqueResult,
};
use runmat_builtins::{CharArray, ComplexTensor, StringArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::build_runtime_error;
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "unique",
        builtin_path = "crate::builtins::array::sorting_sets::unique"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "unique"
category: "array/sorting_sets"
keywords: ["unique", "set", "distinct", "stable", "rows", "indices", "gpu", "string", "char"]
summary: "Return the unique elements or rows of arrays with optional index outputs."
references:
  - https://www.mathworks.com/help/matlab/ref/unique.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Uses the provider `unique` hook when available; default providers download to host memory and reuse the CPU implementation."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::array::sorting_sets::unique::tests"
  integration: "builtins::array::sorting_sets::unique::tests::unique_gpu_roundtrip"
---

# What does the `unique` function do in MATLAB / RunMat?
`unique` removes duplicates from its input while providing optional index outputs
that map between the original data and the returned distinct values (or rows).
By default results are sorted, but you can request stable order, operate on rows,
and choose whether the first or last occurrence is retained.

## How does the `unique` function behave in MATLAB / RunMat?
- `unique(A)` flattens numeric, logical, character, string, or complex arrays column-major into a vector of unique values sorted ascending.
- `[C, IA] = unique(A)` also returns the indices of the selected occurrences (`IA`) so that `C = A(IA)`.
- `[C, IA, IC] = unique(A)` provides `IC`, the mapping from each element of `A` to the corresponding index in `C`.
- `unique(A, 'stable')` preserves the first appearance order rather than sorting.
- `unique(A, 'rows')` treats each row as an observation and returns a matrix (or char/string array) whose rows are unique.
- `unique(A, 'last')` or `'first'` controls which occurrence contributes to `IA` (defaults to `'first'`).
- Combinations such as `unique(A, 'rows', 'stable', 'last')` follow MATLAB's precedence rules; mutually exclusive flags (e.g. `'sorted'` with `'stable'`) are rejected.
- Empty inputs return empty outputs with consistent dimensions.
- Legacy switches such as `'legacy'` or `'R2012a'` are not supported; RunMat always follows the modern MATLAB semantics.

## `unique` Function GPU Execution Behaviour
`unique` is registered as a residency sink. When the provider exposes the custom
`unique` hook, the runtime can execute the operation entirely on the device and keep
results resident. If the active provider does not implement that hook, RunMat gathers
the data to host memory, performs the CPU implementation, and returns host-resident
outputs so subsequent MATLAB code observes the same values and ordering.

## Examples of using the `unique` function in MATLAB / RunMat

### Getting Sorted Unique Values
```matlab
A = [3 1 3 2];
C = unique(A);
```
Expected output:
```matlab
C =
     1
     2
     3
```

### Preserving Input Order with `'stable'`
```matlab
A = [4 2 4 1 2];
C = unique(A, 'stable');
```
Expected output:
```matlab
C =
     4
     2
     1
```

### Returning Indices for Reconstruction
```matlab
A = [7 5 7 3];
[C, IA, IC] = unique(A);
reconstructed = C(IC);
```
Expected output:
```matlab
C =
     3
     5
     7
IA =
     4
     2
     1
IC =
     3
     2
     3
     1
reconstructed =
     7
     5
     7
     3
```

### Finding Unique Rows in a Matrix
```matlab
A = [1 3; 1 3; 2 4; 1 2];
[C, IA, IC] = unique(A, 'rows');
```
Expected output:
```matlab
C =
     1     2
     1     3
     2     4
IA =
     4
     1
     3
IC =
     2
     2
     3
     1
```

### Selecting Last Occurrences
```matlab
A = [9 8 9 7 8];
[C, IA] = unique(A, 'last');
```
Expected output:
```matlab
C =
     7
     8
     9
IA =
     4
     5
     3
```

### Working with Empty Arrays
```matlab
A = zeros(0, 3);
[C, IA, IC] = unique(A, 'rows');
```
Expected output:
```matlab
C =
IA =
IC =
```
all outputs are empty with compatible dimensions.).map_err(Into::into)
### Using `unique` on GPU Arrays
```matlab
G = gpuArray([5 3 5 1]);
[C, IA, IC] = unique(G, 'stable');
```
RunMat gathers `G` to the host (until providers implement a device kernel) and returns:
```matlab
C =
     5
     3
     1
IA =
     1
     2
     4
IC =
     1
     2
     1
     3
```

### Unique Characters in a Char Array
```matlab
chars = ['m','z'; 'm','a'];
[C, IA] = unique(chars);
```
Expected output (`C` is a column vector of characters):
```matlab
C =
    a
    m
    z
IA =
    4
    1
    3
```

### Unique Strings with Row Deduplication
```matlab
S = ["alpha" "beta"; "alpha" "beta"; "gamma" "beta"];
[C, IA, IC] = unique(S, 'rows', 'stable');
```
Expected output:
```matlab
C =
  2x2 string array
    "alpha"    "beta"
    "gamma"    "beta"
IA =
     1
     3
IC =
     1
     1
     2
```

## FAQ

### Which ordering does `unique` use by default?
Results are sorted in ascending order unless you pass `'stable'`, which preserves the first occurrence order.

### How are the index outputs defined?
`IA` indexes into the original data (or rows). `IC` is a column vector mapping each element (or row) of the input to the position of the corresponding unique value in `C`.

### What do `'first'` and `'last'` control?
They determine whether `IA` references the first or last occurrence of each distinct value/row. They do not affect `C` or `IC`.

### Can I combine `'rows'` with `'stable'` or `'last'`?
Yes. All permutations of `'rows'`, `'stable'`/`'sorted'`, and `'first'`/`'last'` are accepted. The runtime enforces MATLAB's validation rules.

### Does `unique` support complex numbers or characters?
Yes. Complex values use magnitude ordering for the sorted output, and character or string arrays produce results in their native container types (char arrays and string arrays respectively).

### How does `unique` treat NaN values?
All NaN values are considered equal. Sorted outputs place NaNs at the end; stable outputs keep their original relative order.

### Are GPU arrays supported?
Yes. When a provider lacks a native kernel, RunMat gathers GPU arrays to host memory and executes the host implementation, guaranteeing MATLAB-compatible output.

### Does `unique` preserve array shape?
Scalar outputs remain scalars. Otherwise, values are returned as column vectors (for element mode) or matrices with the same number of columns as the input (for `'rows'`).

### What happens with empty inputs?
Empty inputs (including empty matrices) return empty outputs with matching dimensions, and index outputs are empty column vectors.

### Is `unique` stable?
Sorting is stable where applicable; ties preserve their relative order. You can also request `'stable'` explicitly.

## See Also
[sort](./sort), [sortrows](./sortrows), [argsort](./argsort)

## Source & Feedback
- Source code: [`crates/runmat-runtime/src/builtins/array/sorting_sets/unique.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/sorting_sets/unique.rs)
- Found a bug? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::sorting_sets::unique")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "unique",
    op_kind: GpuOpKind::Custom("unique"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("unique")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: true,
    notes: "Providers may implement the `unique` hook; default providers download tensors and reuse the CPU implementation.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::array::sorting_sets::unique"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "unique",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "`unique` terminates fusion chains and materialises results on the host; upstream tensors are gathered when necessary.",
};

fn unique_error(message: impl Into<String>) -> crate::RuntimeControlFlow {
    build_runtime_error(message)
        .with_builtin("unique")
        .build()
        .into()
}

#[runtime_builtin(
    name = "unique",
    category = "array/sorting_sets",
    summary = "Return the unique elements or rows of arrays with optional index outputs.",
    keywords = "unique,set,distinct,stable,rows,indices,gpu",
    accel = "array_construct",
    sink = true,
    builtin_path = "crate::builtins::array::sorting_sets::unique"
)]
fn unique_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    Ok(evaluate(value, &rest)?.into_values_value())
}

/// Evaluate `unique` once and expose all outputs to the caller.
pub fn evaluate(value: Value, rest: &[Value]) -> crate::BuiltinResult<UniqueEvaluation> {
    let opts = parse_options(rest)?;
    match value {
        Value::GpuTensor(handle) => unique_gpu(handle, &opts),
        other => unique_host(other, &opts),
    }
}

fn parse_options(rest: &[Value]) -> crate::BuiltinResult<UniqueOptions> {
    let mut opts = UniqueOptions {
        rows: false,
        order: UniqueOrder::Sorted,
        occurrence: UniqueOccurrence::First,
    };
    let mut seen_order: Option<UniqueOrder> = None;
    let mut seen_occurrence: Option<UniqueOccurrence> = None;

    for arg in rest {
        let text = tensor::value_to_string(arg)
            .ok_or_else(|| unique_error("unique: expected string option arguments"))?;
        let lowered = text.trim().to_ascii_lowercase();
        match lowered.as_str() {
            "sorted" => {
                if let Some(prev) = seen_order {
                    if prev != UniqueOrder::Sorted {
                        return Err(unique_error("unique: cannot combine 'sorted' with 'stable'"));
                    }
                }
                seen_order = Some(UniqueOrder::Sorted);
                opts.order = UniqueOrder::Sorted;
            }
            "stable" => {
                if let Some(prev) = seen_order {
                    if prev != UniqueOrder::Stable {
                        return Err(unique_error("unique: cannot combine 'sorted' with 'stable'"));
                    }
                }
                seen_order = Some(UniqueOrder::Stable);
                opts.order = UniqueOrder::Stable;
            }
            "rows" => {
                opts.rows = true;
            }
            "first" => {
                if let Some(prev) = seen_occurrence {
                    if prev != UniqueOccurrence::First {
                        return Err(unique_error("unique: cannot combine 'first' with 'last'"));
                    }
                }
                seen_occurrence = Some(UniqueOccurrence::First);
                opts.occurrence = UniqueOccurrence::First;
            }
            "last" => {
                if let Some(prev) = seen_occurrence {
                    if prev != UniqueOccurrence::Last {
                        return Err(unique_error("unique: cannot combine 'first' with 'last'"));
                    }
                }
                seen_occurrence = Some(UniqueOccurrence::Last);
                opts.occurrence = UniqueOccurrence::Last;
            }
            "legacy" | "r2012a" => {
                return Err(unique_error("unique: the 'legacy' behaviour is not supported"));
            }
            other => {
                return Err(unique_error(format!("unique: unrecognised option '{other}'")));
            }
        }
    }

    Ok(opts)
}

fn unique_gpu(handle: GpuTensorHandle, opts: &UniqueOptions) -> crate::BuiltinResult<UniqueEvaluation> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Ok(result) = provider.unique(&handle, opts) {
            return UniqueEvaluation::from_unique_result(result);
        }
    }
    let tensor = gpu_helpers::gather_tensor(&handle)?;
    unique_numeric_from_tensor(tensor, opts)
}

fn unique_host(value: Value, opts: &UniqueOptions) -> crate::BuiltinResult<UniqueEvaluation> {
    match value {
        Value::Tensor(tensor) => unique_numeric_from_tensor(tensor, opts),
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1]).map_err(|e| unique_error(format!("unique: {e}")))?;
            unique_numeric_from_tensor(tensor, opts)
        }
        Value::Int(i) => {
            let tensor = Tensor::new(vec![i.to_f64()], vec![1, 1])
                .map_err(|e| unique_error(format!("unique: {e}")))?;
            unique_numeric_from_tensor(tensor, opts)
        }
        Value::Bool(b) => {
            let tensor = Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|e| unique_error(format!("unique: {e}")))?;
            unique_numeric_from_tensor(tensor, opts)
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| unique_error(e))?;
            unique_numeric_from_tensor(tensor, opts)
        }
        Value::ComplexTensor(tensor) => unique_complex_from_tensor(tensor, opts),
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| unique_error(format!("unique: {e}")))?;
            unique_complex_from_tensor(tensor, opts)
        }
        Value::CharArray(array) => unique_char_array(array, opts),
        Value::StringArray(array) => unique_string_array(array, opts),
        Value::String(s) => {
            let array = StringArray::new(vec![s], vec![1, 1]).map_err(|e| unique_error(format!("unique: {e}")))?;
            unique_string_array(array, opts)
        }
        other => Err(unique_error(format!(
            "unique: unsupported input type {:?}; expected numeric, logical, char, string, or complex values",
            other
        ))
        .into()),
    }
}

pub fn unique_numeric_from_tensor(
    tensor: Tensor,
    opts: &UniqueOptions,
) -> crate::BuiltinResult<UniqueEvaluation> {
    if opts.rows {
        unique_numeric_rows(tensor, opts)
    } else {
        unique_numeric_elements(tensor, opts)
    }
}

fn unique_numeric_elements(
    tensor: Tensor,
    opts: &UniqueOptions,
) -> crate::BuiltinResult<UniqueEvaluation> {
    let len = tensor.data.len();
    if len == 0 {
        let values = Tensor::new(Vec::new(), vec![0, 1]).map_err(|e| unique_error(format!("unique: {e}")))?;
        let ia = Tensor::new(Vec::new(), vec![0, 1]).map_err(|e| unique_error(format!("unique: {e}")))?;
        let ic = Tensor::new(Vec::new(), vec![0, 1]).map_err(|e| unique_error(format!("unique: {e}")))?;
        return Ok(UniqueEvaluation::new(
            tensor::tensor_into_value(values),
            ia,
            ic,
        ));
    }

    let mut entries = Vec::<NumericElementEntry>::new();
    let mut map: HashMap<u64, usize> = HashMap::new();
    let mut element_entry_index = Vec::with_capacity(len);

    for (idx, &value) in tensor.data.iter().enumerate() {
        let key = canonicalize_f64(value);
        match map.get(&key) {
            Some(&entry_idx) => {
                entries[entry_idx].last = idx;
                element_entry_index.push(entry_idx);
            }
            None => {
                let entry_idx = entries.len();
                entries.push(NumericElementEntry {
                    value,
                    first: idx,
                    last: idx,
                });
                map.insert(key, entry_idx);
                element_entry_index.push(entry_idx);
            }
        }
    }

    let mut order: Vec<usize> = (0..entries.len()).collect();
    if opts.order == UniqueOrder::Sorted {
        order.sort_by(|&a, &b| compare_f64(entries[a].value, entries[b].value));
    }

    let mut entry_to_position = vec![0usize; entries.len()];
    for (pos, &entry_idx) in order.iter().enumerate() {
        entry_to_position[entry_idx] = pos;
    }

    let mut values = Vec::with_capacity(order.len());
    let mut ia = Vec::with_capacity(order.len());
    for &entry_idx in &order {
        let entry = &entries[entry_idx];
        values.push(entry.value);
        let occurrence = match opts.occurrence {
            UniqueOccurrence::First => entry.first,
            UniqueOccurrence::Last => entry.last,
        };
        ia.push((occurrence + 1) as f64);
    }

    let mut ic = Vec::with_capacity(len);
    for entry_idx in element_entry_index {
        let pos = entry_to_position[entry_idx];
        ic.push((pos + 1) as f64);
    }

    let value_tensor =
        Tensor::new(values, vec![order.len(), 1]).map_err(|e| unique_error(format!("unique: {e}")))?;
    let ia_tensor = Tensor::new(ia, vec![order.len(), 1]).map_err(|e| unique_error(format!("unique: {e}")))?;
    let ic_tensor = Tensor::new(ic, vec![len, 1]).map_err(|e| unique_error(format!("unique: {e}")))?;

    Ok(UniqueEvaluation::new(
        tensor::tensor_into_value(value_tensor),
        ia_tensor,
        ic_tensor,
    ))
}

fn unique_numeric_rows(tensor: Tensor, opts: &UniqueOptions) -> crate::BuiltinResult<UniqueEvaluation> {
    if tensor.shape.len() != 2 {
        return Err(unique_error("unique: 'rows' option requires a 2-D matrix input"));
    }
    let rows = tensor.shape[0];
    let cols = tensor.shape[1];

    if rows == 0 || cols == 0 {
        let values = Tensor::new(Vec::new(), vec![0, cols]).map_err(|e| unique_error(format!("unique: {e}")))?;
        let ia = Tensor::new(Vec::new(), vec![0, 1]).map_err(|e| unique_error(format!("unique: {e}")))?;
        let ic = Tensor::new(Vec::new(), vec![rows, 1]).map_err(|e| unique_error(format!("unique: {e}")))?;
        return Ok(UniqueEvaluation::new(
            tensor::tensor_into_value(values),
            ia,
            ic,
        ));
    }

    let mut entries = Vec::<NumericRowEntry>::new();
    let mut map: HashMap<NumericRowKey, usize> = HashMap::new();
    let mut row_entry_index = Vec::with_capacity(rows);

    for r in 0..rows {
        let mut row_values = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r + c * rows;
            row_values.push(tensor.data[idx]);
        }
        let key = NumericRowKey::from_slice(&row_values);
        match map.get(&key) {
            Some(&entry_idx) => {
                entries[entry_idx].last = r;
                row_entry_index.push(entry_idx);
            }
            None => {
                let entry_idx = entries.len();
                entries.push(NumericRowEntry {
                    row_data: row_values.clone(),
                    first: r,
                    last: r,
                });
                map.insert(key, entry_idx);
                row_entry_index.push(entry_idx);
            }
        }
    }

    let mut order: Vec<usize> = (0..entries.len()).collect();
    if opts.order == UniqueOrder::Sorted {
        order.sort_by(|&a, &b| compare_numeric_rows(&entries[a].row_data, &entries[b].row_data));
    }

    let mut entry_to_position = vec![0usize; entries.len()];
    for (pos, &entry_idx) in order.iter().enumerate() {
        entry_to_position[entry_idx] = pos;
    }

    let unique_rows_count = order.len();
    let mut values = vec![0.0f64; unique_rows_count * cols];
    for (row_pos, &entry_idx) in order.iter().enumerate() {
        let row = &entries[entry_idx].row_data;
        for (col, value) in row.iter().enumerate().take(cols) {
            let dest = row_pos + col * unique_rows_count;
            values[dest] = *value;
        }
    }

    let mut ia = Vec::with_capacity(unique_rows_count);
    for &entry_idx in &order {
        let entry = &entries[entry_idx];
        let occurrence = match opts.occurrence {
            UniqueOccurrence::First => entry.first,
            UniqueOccurrence::Last => entry.last,
        };
        ia.push((occurrence + 1) as f64);
    }

    let mut ic = Vec::with_capacity(rows);
    for entry_idx in row_entry_index {
        let pos = entry_to_position[entry_idx];
        ic.push((pos + 1) as f64);
    }

    let value_tensor =
        Tensor::new(values, vec![unique_rows_count, cols]).map_err(|e| unique_error(format!("unique: {e}")))?;
    let ia_tensor =
        Tensor::new(ia, vec![unique_rows_count, 1]).map_err(|e| unique_error(format!("unique: {e}")))?;
    let ic_tensor = Tensor::new(ic, vec![rows, 1]).map_err(|e| unique_error(format!("unique: {e}")))?;

    Ok(UniqueEvaluation::new(
        tensor::tensor_into_value(value_tensor),
        ia_tensor,
        ic_tensor,
    ))
}

fn unique_complex_from_tensor(
    tensor: ComplexTensor,
    opts: &UniqueOptions,
) -> crate::BuiltinResult<UniqueEvaluation> {
    if opts.rows {
        unique_complex_rows(tensor, opts)
    } else {
        unique_complex_elements(tensor, opts)
    }
}

fn unique_complex_elements(
    tensor: ComplexTensor,
    opts: &UniqueOptions,
) -> crate::BuiltinResult<UniqueEvaluation> {
    let len = tensor.data.len();
    if len == 0 {
        let values =
            ComplexTensor::new(Vec::new(), vec![0, 1]).map_err(|e| unique_error(format!("unique: {e}")))?;
        let ia = Tensor::new(Vec::new(), vec![0, 1]).map_err(|e| unique_error(format!("unique: {e}")))?;
        let ic = Tensor::new(Vec::new(), vec![0, 1]).map_err(|e| unique_error(format!("unique: {e}")))?;
        return Ok(UniqueEvaluation::new(
            complex_tensor_into_value(values),
            ia,
            ic,
        ));
    }

    let mut entries = Vec::<ComplexElementEntry>::new();
    let mut map: HashMap<ComplexKey, usize> = HashMap::new();
    let mut element_entry_index = Vec::with_capacity(len);

    for (idx, &value) in tensor.data.iter().enumerate() {
        let key = ComplexKey::new(value);
        match map.get(&key) {
            Some(&entry_idx) => {
                entries[entry_idx].last = idx;
                element_entry_index.push(entry_idx);
            }
            None => {
                let entry_idx = entries.len();
                entries.push(ComplexElementEntry {
                    value,
                    first: idx,
                    last: idx,
                });
                map.insert(key, entry_idx);
                element_entry_index.push(entry_idx);
            }
        }
    }

    let mut order: Vec<usize> = (0..entries.len()).collect();
    if opts.order == UniqueOrder::Sorted {
        order.sort_by(|&a, &b| compare_complex(entries[a].value, entries[b].value));
    }

    let mut entry_to_position = vec![0usize; entries.len()];
    for (pos, &entry_idx) in order.iter().enumerate() {
        entry_to_position[entry_idx] = pos;
    }

    let mut values = Vec::with_capacity(order.len());
    let mut ia = Vec::with_capacity(order.len());
    for &entry_idx in &order {
        let entry = &entries[entry_idx];
        values.push(entry.value);
        let occurrence = match opts.occurrence {
            UniqueOccurrence::First => entry.first,
            UniqueOccurrence::Last => entry.last,
        };
        ia.push((occurrence + 1) as f64);
    }

    let mut ic = Vec::with_capacity(len);
    for entry_idx in element_entry_index {
        let pos = entry_to_position[entry_idx];
        ic.push((pos + 1) as f64);
    }

    let value_tensor =
        ComplexTensor::new(values, vec![order.len(), 1]).map_err(|e| unique_error(format!("unique: {e}")))?;
    let ia_tensor = Tensor::new(ia, vec![order.len(), 1]).map_err(|e| unique_error(format!("unique: {e}")))?;
    let ic_tensor = Tensor::new(ic, vec![len, 1]).map_err(|e| unique_error(format!("unique: {e}")))?;

    Ok(UniqueEvaluation::new(
        complex_tensor_into_value(value_tensor),
        ia_tensor,
        ic_tensor,
    ))
}

fn unique_complex_rows(
    tensor: ComplexTensor,
    opts: &UniqueOptions,
) -> crate::BuiltinResult<UniqueEvaluation> {
    if tensor.shape.len() != 2 {
        return Err(unique_error("unique: 'rows' option requires a 2-D matrix input"));
    }
    let rows = tensor.shape[0];
    let cols = tensor.shape[1];

    if rows == 0 || cols == 0 {
        let values =
            ComplexTensor::new(Vec::new(), vec![rows, cols]).map_err(|e| unique_error(format!("unique: {e}")))?;
        let ia = Tensor::new(Vec::new(), vec![0, 1]).map_err(|e| unique_error(format!("unique: {e}")))?;
        let ic = Tensor::new(Vec::new(), vec![rows, 1]).map_err(|e| unique_error(format!("unique: {e}")))?;
        return Ok(UniqueEvaluation::new(
            complex_tensor_into_value(values),
            ia,
            ic,
        ));
    }

    let mut entries = Vec::<ComplexRowEntry>::new();
    let mut map: HashMap<Vec<ComplexKey>, usize> = HashMap::new();
    let mut row_entry_index = Vec::with_capacity(rows);

    for r in 0..rows {
        let mut row_values = Vec::with_capacity(cols);
        let mut key_row = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r + c * rows;
            let value = tensor.data[idx];
            row_values.push(value);
            key_row.push(ComplexKey::new(value));
        }
        match map.get(&key_row) {
            Some(&entry_idx) => {
                entries[entry_idx].last = r;
                row_entry_index.push(entry_idx);
            }
            None => {
                let entry_idx = entries.len();
                entries.push(ComplexRowEntry {
                    row_data: row_values.clone(),
                    first: r,
                    last: r,
                });
                map.insert(key_row, entry_idx);
                row_entry_index.push(entry_idx);
            }
        }
    }

    let mut order: Vec<usize> = (0..entries.len()).collect();
    if opts.order == UniqueOrder::Sorted {
        order.sort_by(|&a, &b| compare_complex_rows(&entries[a].row_data, &entries[b].row_data));
    }

    let mut entry_to_position = vec![0usize; entries.len()];
    for (pos, &entry_idx) in order.iter().enumerate() {
        entry_to_position[entry_idx] = pos;
    }

    let unique_rows_count = order.len();
    let mut values = vec![(0.0, 0.0); unique_rows_count * cols];
    for (row_pos, &entry_idx) in order.iter().enumerate() {
        let row = &entries[entry_idx].row_data;
        for (col, value) in row.iter().enumerate().take(cols) {
            let dest = row_pos + col * unique_rows_count;
            values[dest] = *value;
        }
    }

    let mut ia = Vec::with_capacity(unique_rows_count);
    for &entry_idx in &order {
        let entry = &entries[entry_idx];
        let occurrence = match opts.occurrence {
            UniqueOccurrence::First => entry.first,
            UniqueOccurrence::Last => entry.last,
        };
        ia.push((occurrence + 1) as f64);
    }

    let mut ic = Vec::with_capacity(rows);
    for entry_idx in row_entry_index {
        let pos = entry_to_position[entry_idx];
        ic.push((pos + 1) as f64);
    }

    let value_tensor = ComplexTensor::new(values, vec![unique_rows_count, cols])
        .map_err(|e| unique_error(format!("unique: {e}")))?;
    let ia_tensor =
        Tensor::new(ia, vec![unique_rows_count, 1]).map_err(|e| unique_error(format!("unique: {e}")))?;
    let ic_tensor = Tensor::new(ic, vec![rows, 1]).map_err(|e| unique_error(format!("unique: {e}")))?;

    Ok(UniqueEvaluation::new(
        complex_tensor_into_value(value_tensor),
        ia_tensor,
        ic_tensor,
    ))
}

fn unique_char_array(array: CharArray, opts: &UniqueOptions) -> crate::BuiltinResult<UniqueEvaluation> {
    if opts.rows {
        unique_char_rows(array, opts)
    } else {
        unique_char_elements(array, opts)
    }
}

fn unique_char_elements(
    array: CharArray,
    opts: &UniqueOptions,
) -> crate::BuiltinResult<UniqueEvaluation> {
    let rows = array.rows;
    let cols = array.cols;
    let total = rows * cols;
    if total == 0 {
        let values = CharArray::new(Vec::new(), 0, 0).map_err(|e| unique_error(format!("unique: {e}")))?;
        let ia = Tensor::new(Vec::new(), vec![0, 1]).map_err(|e| unique_error(format!("unique: {e}")))?;
        let ic = Tensor::new(Vec::new(), vec![0, 1]).map_err(|e| unique_error(format!("unique: {e}")))?;
        return Ok(UniqueEvaluation::new(Value::CharArray(values), ia, ic));
    }

    let mut entries = Vec::<CharElementEntry>::new();
    let mut map: HashMap<u32, usize> = HashMap::new();
    let mut element_entry_index = Vec::with_capacity(total);

    for col in 0..cols {
        for row in 0..rows {
            let linear_idx = row + col * rows;
            let data_idx = row * cols + col;
            let ch = array.data[data_idx];
            let key = ch as u32;
            match map.get(&key) {
                Some(&entry_idx) => {
                    entries[entry_idx].last = linear_idx;
                    element_entry_index.push(entry_idx);
                }
                None => {
                    let entry_idx = entries.len();
                    entries.push(CharElementEntry {
                        ch,
                        first: linear_idx,
                        last: linear_idx,
                    });
                    map.insert(key, entry_idx);
                    element_entry_index.push(entry_idx);
                }
            }
        }
    }

    let mut order: Vec<usize> = (0..entries.len()).collect();
    if opts.order == UniqueOrder::Sorted {
        order.sort_by(|&a, &b| entries[a].ch.cmp(&entries[b].ch));
    }

    let mut entry_to_position = vec![0usize; entries.len()];
    for (pos, &entry_idx) in order.iter().enumerate() {
        entry_to_position[entry_idx] = pos;
    }

    let mut values = Vec::with_capacity(order.len());
    let mut ia = Vec::with_capacity(order.len());
    for &entry_idx in &order {
        let entry = &entries[entry_idx];
        values.push(entry.ch);
        let occurrence = match opts.occurrence {
            UniqueOccurrence::First => entry.first,
            UniqueOccurrence::Last => entry.last,
        };
        ia.push((occurrence + 1) as f64);
    }

    let mut ic = Vec::with_capacity(total);
    for entry_idx in element_entry_index {
        let pos = entry_to_position[entry_idx];
        ic.push((pos + 1) as f64);
    }

    let value_array = CharArray::new(values, order.len(), 1).map_err(|e| unique_error(format!("unique: {e}")))?;
    let ia_tensor = Tensor::new(ia, vec![order.len(), 1]).map_err(|e| unique_error(format!("unique: {e}")))?;
    let ic_tensor = Tensor::new(ic, vec![total, 1]).map_err(|e| unique_error(format!("unique: {e}")))?;

    Ok(UniqueEvaluation::new(
        Value::CharArray(value_array),
        ia_tensor,
        ic_tensor,
    ))
}

fn unique_char_rows(array: CharArray, opts: &UniqueOptions) -> crate::BuiltinResult<UniqueEvaluation> {
    let rows = array.rows;
    let cols = array.cols;
    if rows == 0 {
        let values = CharArray::new(Vec::new(), 0, cols).map_err(|e| unique_error(format!("unique: {e}")))?;
        let ia = Tensor::new(Vec::new(), vec![0, 1]).map_err(|e| unique_error(format!("unique: {e}")))?;
        let ic = Tensor::new(Vec::new(), vec![0, 1]).map_err(|e| unique_error(format!("unique: {e}")))?;
        return Ok(UniqueEvaluation::new(Value::CharArray(values), ia, ic));
    }

    let mut entries = Vec::<CharRowEntry>::new();
    let mut map: HashMap<RowCharKey, usize> = HashMap::new();
    let mut row_entry_index = Vec::with_capacity(rows);

    for r in 0..rows {
        let start = r * cols;
        let end = start + cols;
        let slice = &array.data[start..end];
        let key = RowCharKey::from_slice(slice);
        match map.get(&key) {
            Some(&entry_idx) => {
                entries[entry_idx].last = r;
                row_entry_index.push(entry_idx);
            }
            None => {
                let entry_idx = entries.len();
                entries.push(CharRowEntry {
                    row_data: slice.to_vec(),
                    first: r,
                    last: r,
                });
                map.insert(key, entry_idx);
                row_entry_index.push(entry_idx);
            }
        }
    }

    let mut order: Vec<usize> = (0..entries.len()).collect();
    if opts.order == UniqueOrder::Sorted {
        order.sort_by(|&a, &b| compare_char_rows(&entries[a].row_data, &entries[b].row_data));
    }

    let mut entry_to_position = vec![0usize; entries.len()];
    for (pos, &entry_idx) in order.iter().enumerate() {
        entry_to_position[entry_idx] = pos;
    }

    let unique_rows_count = order.len();
    let mut values = vec!['\0'; unique_rows_count * cols];
    for (row_pos, &entry_idx) in order.iter().enumerate() {
        let row = &entries[entry_idx].row_data;
        for col in 0..cols {
            let dest = row_pos * cols + col;
            if col < row.len() {
                values[dest] = row[col];
            }
        }
    }

    let mut ia = Vec::with_capacity(unique_rows_count);
    for &entry_idx in &order {
        let entry = &entries[entry_idx];
        let occurrence = match opts.occurrence {
            UniqueOccurrence::First => entry.first,
            UniqueOccurrence::Last => entry.last,
        };
        ia.push((occurrence + 1) as f64);
    }

    let mut ic = Vec::with_capacity(rows);
    for entry_idx in row_entry_index {
        let pos = entry_to_position[entry_idx];
        ic.push((pos + 1) as f64);
    }

    let value_array =
        CharArray::new(values, unique_rows_count, cols).map_err(|e| unique_error(format!("unique: {e}")))?;
    let ia_tensor =
        Tensor::new(ia, vec![unique_rows_count, 1]).map_err(|e| unique_error(format!("unique: {e}")))?;
    let ic_tensor = Tensor::new(ic, vec![rows, 1]).map_err(|e| unique_error(format!("unique: {e}")))?;

    Ok(UniqueEvaluation::new(
        Value::CharArray(value_array),
        ia_tensor,
        ic_tensor,
    ))
}

fn unique_string_array(
    array: StringArray,
    opts: &UniqueOptions,
) -> crate::BuiltinResult<UniqueEvaluation> {
    if opts.rows {
        unique_string_rows(array, opts)
    } else {
        unique_string_elements(array, opts)
    }
}

fn unique_string_elements(
    array: StringArray,
    opts: &UniqueOptions,
) -> crate::BuiltinResult<UniqueEvaluation> {
    let len = array.data.len();
    if len == 0 {
        let values =
            StringArray::new(Vec::new(), vec![0, 1]).map_err(|e| unique_error(format!("unique: {e}")))?;
        let ia = Tensor::new(Vec::new(), vec![0, 1]).map_err(|e| unique_error(format!("unique: {e}")))?;
        let ic = Tensor::new(Vec::new(), vec![0, 1]).map_err(|e| unique_error(format!("unique: {e}")))?;
        return Ok(UniqueEvaluation::new(Value::StringArray(values), ia, ic));
    }

    let mut entries = Vec::<StringElementEntry>::new();
    let mut map: HashMap<String, usize> = HashMap::new();
    let mut element_entry_index = Vec::with_capacity(len);

    for (idx, value) in array.data.iter().enumerate() {
        match map.get(value) {
            Some(&entry_idx) => {
                entries[entry_idx].last = idx;
                element_entry_index.push(entry_idx);
            }
            None => {
                let entry_idx = entries.len();
                entries.push(StringElementEntry {
                    value: value.clone(),
                    first: idx,
                    last: idx,
                });
                map.insert(value.clone(), entry_idx);
                element_entry_index.push(entry_idx);
            }
        }
    }

    let mut order: Vec<usize> = (0..entries.len()).collect();
    if opts.order == UniqueOrder::Sorted {
        order.sort_by(|&a, &b| entries[a].value.cmp(&entries[b].value));
    }

    let mut entry_to_position = vec![0usize; entries.len()];
    for (pos, &entry_idx) in order.iter().enumerate() {
        entry_to_position[entry_idx] = pos;
    }

    let mut values = Vec::with_capacity(order.len());
    let mut ia = Vec::with_capacity(order.len());
    for &entry_idx in &order {
        let entry = &entries[entry_idx];
        values.push(entry.value.clone());
        let occurrence = match opts.occurrence {
            UniqueOccurrence::First => entry.first,
            UniqueOccurrence::Last => entry.last,
        };
        ia.push((occurrence + 1) as f64);
    }

    let mut ic = Vec::with_capacity(len);
    for entry_idx in element_entry_index {
        let pos = entry_to_position[entry_idx];
        ic.push((pos + 1) as f64);
    }

    let value_array =
        StringArray::new(values, vec![order.len(), 1]).map_err(|e| unique_error(format!("unique: {e}")))?;
    let ia_tensor = Tensor::new(ia, vec![order.len(), 1]).map_err(|e| unique_error(format!("unique: {e}")))?;
    let ic_tensor = Tensor::new(ic, vec![len, 1]).map_err(|e| unique_error(format!("unique: {e}")))?;

    Ok(UniqueEvaluation::new(
        Value::StringArray(value_array),
        ia_tensor,
        ic_tensor,
    ))
}

fn unique_string_rows(
    array: StringArray,
    opts: &UniqueOptions,
) -> crate::BuiltinResult<UniqueEvaluation> {
    if array.shape.len() != 2 {
        return Err(unique_error("unique: 'rows' option requires a 2-D matrix input"));
    }
    let rows = array.shape[0];
    let cols = array.shape[1];

    if rows == 0 {
        let values =
            StringArray::new(Vec::new(), vec![0, cols]).map_err(|e| unique_error(format!("unique: {e}")))?;
        let ia = Tensor::new(Vec::new(), vec![0, 1]).map_err(|e| unique_error(format!("unique: {e}")))?;
        let ic = Tensor::new(Vec::new(), vec![0, 1]).map_err(|e| unique_error(format!("unique: {e}")))?;
        return Ok(UniqueEvaluation::new(Value::StringArray(values), ia, ic));
    }

    let mut entries = Vec::<StringRowEntry>::new();
    let mut map: HashMap<RowStringKey, usize> = HashMap::new();
    let mut row_entry_index = Vec::with_capacity(rows);

    for r in 0..rows {
        let mut row_values = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r + c * rows;
            row_values.push(array.data[idx].clone());
        }
        let key = RowStringKey(row_values.clone());
        match map.get(&key) {
            Some(&entry_idx) => {
                entries[entry_idx].last = r;
                row_entry_index.push(entry_idx);
            }
            None => {
                let entry_idx = entries.len();
                entries.push(StringRowEntry {
                    row_data: row_values.clone(),
                    first: r,
                    last: r,
                });
                map.insert(key, entry_idx);
                row_entry_index.push(entry_idx);
            }
        }
    }

    let mut order: Vec<usize> = (0..entries.len()).collect();
    if opts.order == UniqueOrder::Sorted {
        order.sort_by(|&a, &b| compare_string_rows(&entries[a].row_data, &entries[b].row_data));
    }

    let mut entry_to_position = vec![0usize; entries.len()];
    for (pos, &entry_idx) in order.iter().enumerate() {
        entry_to_position[entry_idx] = pos;
    }

    let unique_rows_count = order.len();
    let mut values = vec![String::new(); unique_rows_count * cols];
    for (row_pos, &entry_idx) in order.iter().enumerate() {
        let row = &entries[entry_idx].row_data;
        for (col, value) in row.iter().enumerate().take(cols) {
            let dest = row_pos + col * unique_rows_count;
            values[dest] = value.clone();
        }
    }

    let mut ia = Vec::with_capacity(unique_rows_count);
    for &entry_idx in &order {
        let entry = &entries[entry_idx];
        let occurrence = match opts.occurrence {
            UniqueOccurrence::First => entry.first,
            UniqueOccurrence::Last => entry.last,
        };
        ia.push((occurrence + 1) as f64);
    }

    let mut ic = Vec::with_capacity(rows);
    for entry_idx in row_entry_index {
        let pos = entry_to_position[entry_idx];
        ic.push((pos + 1) as f64);
    }

    let value_array = StringArray::new(values, vec![unique_rows_count, cols])
        .map_err(|e| unique_error(format!("unique: {e}")))?;
    let ia_tensor =
        Tensor::new(ia, vec![unique_rows_count, 1]).map_err(|e| unique_error(format!("unique: {e}")))?;
    let ic_tensor = Tensor::new(ic, vec![rows, 1]).map_err(|e| unique_error(format!("unique: {e}")))?;

    Ok(UniqueEvaluation::new(
        Value::StringArray(value_array),
        ia_tensor,
        ic_tensor,
    ))
}

#[derive(Debug)]
struct NumericElementEntry {
    value: f64,
    first: usize,
    last: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct NumericRowKey(Vec<u64>);

impl NumericRowKey {
    fn from_slice(values: &[f64]) -> Self {
        NumericRowKey(values.iter().map(|&v| canonicalize_f64(v)).collect())
    }
}

#[derive(Debug, Clone)]
struct NumericRowEntry {
    row_data: Vec<f64>,
    first: usize,
    last: usize,
}

#[derive(Debug)]
struct ComplexElementEntry {
    value: (f64, f64),
    first: usize,
    last: usize,
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

#[derive(Debug, Clone)]
struct ComplexRowEntry {
    row_data: Vec<(f64, f64)>,
    first: usize,
    last: usize,
}

#[derive(Debug)]
struct CharElementEntry {
    ch: char,
    first: usize,
    last: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RowCharKey(Vec<u32>);

impl RowCharKey {
    fn from_slice(values: &[char]) -> Self {
        RowCharKey(values.iter().map(|&ch| ch as u32).collect())
    }
}

#[derive(Debug, Clone)]
struct CharRowEntry {
    row_data: Vec<char>,
    first: usize,
    last: usize,
}

#[derive(Debug, Clone)]
struct StringElementEntry {
    value: String,
    first: usize,
    last: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RowStringKey(Vec<String>);

#[derive(Debug, Clone)]
struct StringRowEntry {
    row_data: Vec<String>,
    first: usize,
    last: usize,
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

#[derive(Debug)]
pub struct UniqueEvaluation {
    values: Value,
    ia: Tensor,
    ic: Tensor,
}

impl UniqueEvaluation {
    fn new(values: Value, ia: Tensor, ic: Tensor) -> Self {
        Self { values, ia, ic }
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
        let ic = tensor::tensor_into_value(self.ic);
        (self.values, ia, ic)
    }

    pub fn from_unique_result(result: UniqueResult) -> crate::BuiltinResult<Self> {
        let UniqueResult { values, ia, ic } = result;
        let values_tensor =
            Tensor::new(values.data, values.shape).map_err(|e| unique_error(format!("unique: {e}")))?;
        let ia_tensor = Tensor::new(ia.data, ia.shape).map_err(|e| unique_error(format!("unique: {e}")))?;
        let ic_tensor = Tensor::new(ic.data, ic.shape).map_err(|e| unique_error(format!("unique: {e}")))?;
        Ok(UniqueEvaluation::new(
            tensor::tensor_into_value(values_tensor),
            ia_tensor,
            ic_tensor,
        ))
    }

    pub fn into_numeric_unique_result(self) -> crate::BuiltinResult<UniqueResult> {
        let UniqueEvaluation { values, ia, ic } = self;
        let values_tensor = tensor::value_into_tensor_for("unique", values)
            .map_err(|e| unique_error(e))?;
        Ok(UniqueResult {
            values: HostTensorOwned {
                data: values_tensor.data,
                shape: values_tensor.shape,
            },
            ia: HostTensorOwned {
                data: ia.data,
                shape: ia.shape,
            },
            ic: HostTensorOwned {
                data: ic.data,
                shape: ic.shape,
            },
        })
    }

    pub fn ia_value(&self) -> Value {
        tensor::tensor_into_value(self.ia.clone())
    }

    pub fn ic_value(&self) -> Value {
        tensor::tensor_into_value(self.ic.clone())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{CharArray, IntValue, LogicalArray, StringArray, Tensor, Value};

    fn error_message(flow: crate::RuntimeControlFlow) -> String {
        match flow {
            crate::RuntimeControlFlow::Error(err) => err.message().to_string(),
            crate::RuntimeControlFlow::Suspend(_) => panic!("unexpected suspend"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_sorted_default() {
        let tensor = Tensor::new(vec![3.0, 1.0, 3.0, 2.0], vec![4, 1]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[]).expect("unique");
        let (values, ia, ic) = eval.into_triple();
        match values {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![1.0, 2.0, 3.0]);
                assert_eq!(t.shape, vec![3, 1]);
            }
            Value::Num(_) => panic!("expected tensor result"),
            other => panic!("unexpected result {other:?}"),
        }
        match ia {
            Value::Tensor(t) => assert_eq!(t.data, vec![2.0, 4.0, 1.0]),
            other => panic!("unexpected IA {other:?}"),
        }
        match ic {
            Value::Tensor(t) => assert_eq!(t.data, vec![3.0, 1.0, 3.0, 2.0]),
            other => panic!("unexpected IC {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_sorted_handles_nan() {
        let tensor = Tensor::new(vec![f64::NAN, 2.0, f64::NAN, 1.0], vec![4, 1]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[]).expect("unique");
        let (values, ..) = eval.into_triple();
        match values {
            Value::Tensor(t) => {
                assert_eq!(t.data.len(), 3);
                assert_eq!(t.data[0], 1.0);
                assert_eq!(t.data[1], 2.0);
                assert!(t.data[2].is_nan());
            }
            other => panic!("unexpected values {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_stable_with_nan() {
        let tensor = Tensor::new(vec![f64::NAN, 2.0, f64::NAN, 1.0], vec![4, 1]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[Value::from("stable")]).expect("unique");
        let (values, ..) = eval.into_triple();
        match values {
            Value::Tensor(t) => {
                assert!(t.data[0].is_nan());
                assert_eq!(t.data[1], 2.0);
                assert_eq!(t.data[2], 1.0);
            }
            other => panic!("unexpected values {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_stable_preserves_order() {
        let tensor = Tensor::new(vec![4.0, 2.0, 4.0, 1.0, 2.0], vec![5, 1]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[Value::from("stable")]).expect("unique");
        let (values, ia) = eval.into_pair();
        match values {
            Value::Tensor(t) => assert_eq!(t.data, vec![4.0, 2.0, 1.0]),
            other => panic!("unexpected values {other:?}"),
        }
        match ia {
            Value::Tensor(t) => assert_eq!(t.data, vec![1.0, 2.0, 4.0]),
            other => panic!("unexpected IA {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_last_occurrence() {
        let tensor = Tensor::new(vec![9.0, 8.0, 9.0, 7.0, 8.0], vec![5, 1]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[Value::from("last")]).expect("unique");
        let (values, ia, ic) = eval.into_triple();
        match values {
            Value::Tensor(t) => assert_eq!(t.data, vec![7.0, 8.0, 9.0]),
            other => panic!("unexpected values {other:?}"),
        }
        match ia {
            Value::Tensor(t) => assert_eq!(t.data, vec![4.0, 5.0, 3.0]),
            other => panic!("unexpected IA {other:?}"),
        }
        match ic {
            Value::Tensor(t) => assert_eq!(t.data, vec![3.0, 2.0, 3.0, 1.0, 2.0]),
            other => panic!("unexpected IC {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_rows_sorted_default() {
        let tensor = Tensor::new(vec![1.0, 1.0, 2.0, 1.0, 3.0, 3.0, 4.0, 2.0], vec![4, 2]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[Value::from("rows")]).expect("unique");
        let (values, ia, ic) = eval.into_triple();
        match values {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 2]);
                assert_eq!(t.data, vec![1.0, 1.0, 2.0, 2.0, 3.0, 4.0]);
            }
            other => panic!("unexpected values {other:?}"),
        }
        match ia {
            Value::Tensor(t) => assert_eq!(t.data, vec![4.0, 1.0, 3.0]),
            other => panic!("unexpected IA {other:?}"),
        }
        match ic {
            Value::Tensor(t) => assert_eq!(t.data, vec![2.0, 2.0, 3.0, 1.0]),
            other => panic!("unexpected IC {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_rows_stable_last() {
        let tensor = Tensor::new(vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0], vec![3, 2]).unwrap();
        let eval = evaluate(
            Value::Tensor(tensor),
            &[
                Value::from("rows"),
                Value::from("stable"),
                Value::from("last"),
            ],
        )
        .expect("unique");
        let (values, ia, ic) = eval.into_triple();
        match values {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![1.0, 2.0, 1.0, 2.0]);
            }
            other => panic!("unexpected values {other:?}"),
        }
        match ia {
            Value::Tensor(t) => assert_eq!(t.data, vec![2.0, 3.0]),
            other => panic!("unexpected IA {other:?}"),
        }
        match ic {
            Value::Tensor(t) => assert_eq!(t.data, vec![1.0, 1.0, 2.0]),
            other => panic!("unexpected IC {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_char_elements_sorted() {
        let chars = CharArray::new(vec!['m', 'z', 'm', 'a'], 2, 2).unwrap();
        let eval = evaluate(Value::CharArray(chars), &[]).expect("unique");
        let (values, ia, ic) = eval.into_triple();
        match values {
            Value::CharArray(arr) => {
                assert_eq!(arr.rows, 3);
                assert_eq!(arr.cols, 1);
                assert_eq!(arr.data, vec!['a', 'm', 'z']);
            }
            other => panic!("unexpected values {other:?}"),
        }
        match ia {
            Value::Tensor(t) => assert_eq!(t.data, vec![4.0, 1.0, 3.0]),
            other => panic!("unexpected IA {other:?}"),
        }
        match ic {
            Value::Tensor(t) => assert_eq!(t.data, vec![2.0, 2.0, 3.0, 1.0]),
            other => panic!("unexpected IC {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_char_rows_last() {
        let chars = CharArray::new(vec!['a', 'b', 'a', 'b', 'a', 'c'], 3, 2).unwrap();
        let eval = evaluate(
            Value::CharArray(chars),
            &[Value::from("rows"), Value::from("last")],
        )
        .expect("unique");
        let (values, ia, ic) = eval.into_triple();
        match values {
            Value::CharArray(arr) => {
                assert_eq!(arr.rows, 2);
                assert_eq!(arr.cols, 2);
                assert_eq!(arr.data, vec!['a', 'b', 'a', 'c']);
            }
            other => panic!("unexpected values {other:?}"),
        }
        match ia {
            Value::Tensor(t) => assert_eq!(t.data, vec![2.0, 3.0]),
            other => panic!("unexpected IA {other:?}"),
        }
        match ic {
            Value::Tensor(t) => assert_eq!(t.data, vec![1.0, 1.0, 2.0]),
            other => panic!("unexpected IC {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_string_elements_stable() {
        let array = StringArray::new(
            vec!["beta".into(), "alpha".into(), "beta".into()],
            vec![3, 1],
        )
        .unwrap();
        let eval = evaluate(Value::StringArray(array), &[Value::from("stable")]).expect("unique");
        let (values, ia, ic) = eval.into_triple();
        match values {
            Value::StringArray(sa) => {
                assert_eq!(sa.data, vec!["beta", "alpha"]);
                assert_eq!(sa.shape, vec![2, 1]);
            }
            other => panic!("unexpected values {other:?}"),
        }
        match ia {
            Value::Tensor(t) => assert_eq!(t.data, vec![1.0, 2.0]),
            other => panic!("unexpected IA {other:?}"),
        }
        match ic {
            Value::Tensor(t) => assert_eq!(t.data, vec![1.0, 2.0, 1.0]),
            other => panic!("unexpected IC {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_string_rows() {
        let array = StringArray::new(
            vec![
                "alpha".into(),
                "alpha".into(),
                "gamma".into(),
                "beta".into(),
                "beta".into(),
                "beta".into(),
            ],
            vec![3, 2],
        )
        .unwrap();
        let eval = evaluate(
            Value::StringArray(array),
            &[Value::from("rows"), Value::from("stable")],
        )
        .expect("unique");
        let (values, ia, ic) = eval.into_triple();
        match values {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![2, 2]);
                assert_eq!(sa.data, vec!["alpha", "gamma", "beta", "beta"]);
            }
            other => panic!("unexpected values {other:?}"),
        }
        match ia {
            Value::Tensor(t) => assert_eq!(t.data, vec![1.0, 3.0]),
            other => panic!("unexpected IA {other:?}"),
        }
        match ic {
            Value::Tensor(t) => assert_eq!(t.data, vec![1.0, 1.0, 2.0]),
            other => panic!("unexpected IC {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_complex_sorted() {
        let tensor = ComplexTensor::new(
            vec![(1.0, 1.0), (0.0, 2.0), (1.0, -1.0), (0.0, 2.0)],
            vec![4, 1],
        )
        .unwrap();
        let eval = evaluate(Value::ComplexTensor(tensor), &[]).expect("unique");
        let (values, ..) = eval.into_triple();
        match values {
            Value::ComplexTensor(t) => {
                assert_eq!(t.data.len(), 3);
                assert_eq!(t.data[0], (1.0, -1.0));
                assert_eq!(t.data[1], (1.0, 1.0));
                assert_eq!(t.data[2], (0.0, 2.0));
            }
            other => panic!("unexpected values {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_handles_logical_arrays() {
        let logical = LogicalArray::new(vec![1, 0, 1, 1], vec![4, 1]).unwrap();
        let eval = evaluate(Value::LogicalArray(logical), &[]).expect("unique");
        let values = eval.into_values_value();
        match values {
            Value::Tensor(t) => assert_eq!(t.data, vec![0.0, 1.0]),
            other => panic!("unexpected values {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![5.0, 3.0, 5.0, 1.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let eval =
                evaluate(Value::GpuTensor(handle), &[Value::from("stable")]).expect("unique");
            let values = eval.into_values_value();
            match values {
                Value::Tensor(t) => assert_eq!(t.data, vec![5.0, 3.0, 1.0]),
                other => panic!("unexpected values {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn unique_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![5.0, 3.0, 5.0, 1.0, 2.0], vec![5, 1]).unwrap();
        let host_eval = evaluate(Value::Tensor(tensor.clone()), &[]).expect("host unique");
        let (host_values, host_ia, host_ic) = host_eval.into_triple();

        let provider = runmat_accelerate_api::provider().expect("provider registered");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_eval = evaluate(Value::GpuTensor(handle.clone()), &[]).expect("gpu unique");
        let (gpu_values, gpu_ia, gpu_ic) = gpu_eval.into_triple();
        let _ = provider.free(&handle);

        let host_values = test_support::gather(host_values).expect("gather host values");
        let host_ia = test_support::gather(host_ia).expect("gather host ia");
        let host_ic = test_support::gather(host_ic).expect("gather host ic");
        let gpu_values = test_support::gather(gpu_values).expect("gather gpu values");
        let gpu_ia = test_support::gather(gpu_ia).expect("gather gpu ia");
        let gpu_ic = test_support::gather(gpu_ic).expect("gather gpu ic");

        assert_eq!(gpu_values.shape, host_values.shape);
        assert_eq!(gpu_values.data, host_values.data);
        assert_eq!(gpu_ia.data, host_ia.data);
        assert_eq!(gpu_ic.data, host_ic.data);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_rejects_legacy_option() {
        let tensor = Tensor::new(vec![1.0, 1.0], vec![2, 1]).unwrap();
        let err = error_message(evaluate(Value::Tensor(tensor), &[Value::from("legacy")]).unwrap_err());
        assert!(err.contains("legacy"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_conflicting_order_flags() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = error_message(
            evaluate(
                Value::Tensor(tensor),
                &[Value::from("stable"), Value::from("sorted")],
            )
            .unwrap_err(),
        );
        assert!(err.contains("stable"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_conflicting_occurrence_flags() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = error_message(
            evaluate(
                Value::Tensor(tensor),
                &[Value::from("first"), Value::from("last")],
            )
            .unwrap_err(),
        );
        assert!(err.contains("first"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_rows_requires_two_dimensional_input() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1, 1]).unwrap();
        let err = error_message(evaluate(Value::Tensor(tensor), &[Value::from("rows")]).unwrap_err());
        assert!(err.contains("2-D matrix"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_handles_empty_rows() {
        let tensor = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[Value::from("rows")]).expect("unique");
        let (values, ia, ic) = eval.into_triple();
        match values {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 3]);
                assert!(t.data.is_empty());
            }
            other => panic!("unexpected values {other:?}"),
        }
        match ia {
            Value::Tensor(t) => assert!(t.data.is_empty()),
            other => panic!("unexpected IA {other:?}"),
        }
        match ic {
            Value::Tensor(t) => assert!(t.data.is_empty()),
            other => panic!("unexpected IC {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_accepts_integer_scalars() {
        let eval = evaluate(Value::Int(IntValue::I32(42)), &[]).expect("unique");
        let values = eval.into_values_value();
        match values {
            Value::Num(n) => assert_eq!(n, 42.0),
            other => panic!("unexpected values {other:?}"),
        }
    }
}
