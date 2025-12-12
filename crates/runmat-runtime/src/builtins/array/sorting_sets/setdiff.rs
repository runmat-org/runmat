//! MATLAB-compatible `setdiff` builtin with GPU-aware semantics for RunMat.
//!
//! Provides element-wise and row-wise set difference with optional stable
//! ordering. GPU tensors are gathered to host memory today, but the builtin is
//! registered as a residency sink so future providers can implement device-side
//! kernels without impacting behaviour.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use runmat_accelerate_api::{
    GpuTensorHandle, HostTensorOwned, SetdiffOptions, SetdiffOrder, SetdiffResult,
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
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "setdiff",
        builtin_path = "crate::builtins::array::sorting_sets::setdiff"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "setdiff"
category: "array/sorting_sets"
keywords: ["setdiff", "difference", "stable", "rows", "indices", "gpu"]
summary: "Return values that appear in the first input but not the second, matching MATLAB ordering rules."
references:
  - https://www.mathworks.com/help/matlab/ref/setdiff.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "When providers lack a dedicated `setdiff` hook, RunMat gathers GPU tensors to host memory and reuses the CPU path."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::array::sorting_sets::setdiff::tests"
  integration: "builtins::array::sorting_sets::setdiff::tests::setdiff_gpu_roundtrip"
---

# What does the `setdiff` function do in MATLAB / RunMat?
`setdiff(A, B)` returns the set of values (or rows) that appear in `A` but not in `B`. Results are
unique, and the function can operate in sorted or stable order as well as row mode.

## How does the `setdiff` function behave in MATLAB / RunMat?
- `setdiff(A, B)` flattens inputs column-major, removes duplicates, subtracts the values of `B` from `A`,
  and returns the remaining elements **sorted** ascending by default.
- `[C, IA] = setdiff(A, B)` also returns indices so that `C = A(IA)`.
- `setdiff(A, B, 'stable')` preserves the first appearance order from `A` instead of sorting.
- `setdiff(A, B, 'rows')` treats each row as an element. Inputs must share the same number of columns.
- Character arrays, string arrays, logical arrays, numeric types, and complex values are all supported.
- Legacy flags (`'legacy'`, `'R2012a'`) are not supported; RunMat always follows modern MATLAB semantics.

## `setdiff` Function GPU Execution Behaviour
`setdiff` is registered as a residency sink. When tensors reside on the GPU and the active provider
does not yet implement a `setdiff` hook, RunMat gathers them to host memory, performs the CPU
implementation, and materialises host-resident results. Future providers can wire a custom hook to
perform the set difference directly on-device without affecting existing callers.

## Examples of using the `setdiff` function in MATLAB / RunMat

### Finding values exclusive to the first numeric vector
```matlab
A = [5 7 5 1];
B = [7 1 3];
[C, IA] = setdiff(A, B);
```
Expected output:
```matlab
C =
     5
IA =
     1
```

### Preserving input order with `'stable'`
```matlab
A = [4 2 4 1 3];
B = [3 4 5 1];
[C, IA] = setdiff(A, B, 'stable');
```
Expected output:
```matlab
C =
     2
IA =
     2
```

### Working with rows of numeric matrices
```matlab
A = [1 2; 3 4; 1 2];
B = [3 4; 5 6];
[C, IA] = setdiff(A, B, 'rows');
```
Expected output:
```matlab
C =
     1     2
IA =
     1
```

### Computing set difference for character data
```matlab
A = ['m','z'; 'm','a'];
B = ['a','x'; 'm','a'];
[C, IA] = setdiff(A, B);
```
Expected output:
```matlab
C =
    m
IA =
     1
```

### Subtracting string arrays by row
```matlab
A = ["alpha" "beta"; "gamma" "beta"];
B = ["gamma" "beta"; "delta" "beta"];
[C, IA] = setdiff(A, B, 'rows', 'stable');
```
Expected output:
```matlab
C =
  1x2 string array
    "alpha"    "beta"
IA =
     1
```

### Using `setdiff` with GPU arrays
```matlab
G = gpuArray([10 4 6 4]);
H = gpuArray([6 4 2]);
C = setdiff(G, H);
```
RunMat gathers `G` and `H` to the host (until providers implement a GPU hook) and returns:
```matlab
C =
    10
```

## FAQ

### What ordering does `setdiff` use by default?
Results are sorted ascending. Specify `'stable'` to preserve the first appearance order from the first input.

### How are the index outputs defined?
`IA` points to the positions in `A` that correspond to each element (or row) returned in `C`, using MATLAB's one-based indexing.

### Can I combine `'rows'` with `'stable'`?
Yes. `'rows'` can be paired with either `'sorted'` (default) or `'stable'`. Other option combinations that conflict (e.g. `'sorted'` with `'stable'`) are rejected.

### Does `setdiff` remove `NaN` values from `A` when they exist in `B`?
Yes. `NaN` values are considered equal. If `B` contains `NaN`, all `NaN` entries from `A` are removed.

### Are complex numbers supported?
Absolutely. Complex values use MATLAB's ordering rules (magnitude, then real part, then imaginary part) for the sorted output.

### Does GPU execution change the results?
No. Until providers supply a device implementation, RunMat gathers GPU inputs and executes the CPU path to guarantee MATLAB-compatible behaviour.

### What happens if the inputs have different classes?
RunMat follows MATLAB's rules: both inputs must share the same class (numeric/logical, complex, char, or string). Mixed-class inputs raise descriptive errors.

### Can I request `'legacy'` behaviour?
No. RunMat implements the modern semantics only. Passing `'legacy'` or `'R2012a'` results in an error.

## See Also
[unique](./unique), [union](./union), [intersect](./intersect), [ismember](./ismember), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- Implementation: `crates/runmat-runtime/src/builtins/array/sorting_sets/setdiff.rs`
- Issues / feedback: https://github.com/runmat-org/runmat/issues/new/choose
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::sorting_sets::setdiff")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "setdiff",
    op_kind: GpuOpKind::Custom("setdiff"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("setdiff")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: true,
    notes: "Providers may implement `setdiff`; until then tensors are gathered and processed on the host.",
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
    notes: "`setdiff` materialises its inputs and terminates fusion chains; upstream GPU tensors are gathered if needed.",
};

#[runtime_builtin(
    name = "setdiff",
    category = "array/sorting_sets",
    summary = "Return the values that appear in the first input but not the second.",
    keywords = "setdiff,difference,stable,rows,indices,gpu",
    accel = "array_construct",
    sink = true,
    builtin_path = "crate::builtins::array::sorting_sets::setdiff"
)]
fn setdiff_builtin(a: Value, b: Value, rest: Vec<Value>) -> Result<Value, String> {
    evaluate(a, b, &rest).map(|eval| eval.into_values_value())
}

/// Evaluate `setdiff` once and expose all outputs to the caller.
pub fn evaluate(a: Value, b: Value, rest: &[Value]) -> Result<SetdiffEvaluation, String> {
    let opts = parse_options(rest)?;
    match (a, b) {
        (Value::GpuTensor(handle_a), Value::GpuTensor(handle_b)) => {
            setdiff_gpu_pair(handle_a, handle_b, &opts)
        }
        (Value::GpuTensor(handle_a), other) => setdiff_gpu_mixed(handle_a, other, &opts, true),
        (other, Value::GpuTensor(handle_b)) => setdiff_gpu_mixed(handle_b, other, &opts, false),
        (left, right) => setdiff_host(left, right, &opts),
    }
}

fn parse_options(rest: &[Value]) -> Result<SetdiffOptions, String> {
    let mut opts = SetdiffOptions {
        rows: false,
        order: SetdiffOrder::Sorted,
    };
    let mut seen_order: Option<SetdiffOrder> = None;

    for arg in rest {
        let text = tensor::value_to_string(arg)
            .ok_or_else(|| "setdiff: expected string option arguments".to_string())?;
        let lowered = text.trim().to_ascii_lowercase();
        match lowered.as_str() {
            "rows" => opts.rows = true,
            "sorted" => {
                if let Some(prev) = seen_order {
                    if prev != SetdiffOrder::Sorted {
                        return Err("setdiff: cannot combine 'sorted' with 'stable'".to_string());
                    }
                }
                seen_order = Some(SetdiffOrder::Sorted);
                opts.order = SetdiffOrder::Sorted;
            }
            "stable" => {
                if let Some(prev) = seen_order {
                    if prev != SetdiffOrder::Stable {
                        return Err("setdiff: cannot combine 'sorted' with 'stable'".to_string());
                    }
                }
                seen_order = Some(SetdiffOrder::Stable);
                opts.order = SetdiffOrder::Stable;
            }
            "legacy" | "r2012a" => {
                return Err("setdiff: the 'legacy' behaviour is not supported".to_string());
            }
            other => return Err(format!("setdiff: unrecognised option '{other}'")),
        }
    }

    Ok(opts)
}

fn setdiff_gpu_pair(
    handle_a: GpuTensorHandle,
    handle_b: GpuTensorHandle,
    opts: &SetdiffOptions,
) -> Result<SetdiffEvaluation, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        match provider.setdiff(&handle_a, &handle_b, opts) {
            Ok(result) => return SetdiffEvaluation::from_setdiff_result(result),
            Err(_) => {
                // Fall back to host gather when provider does not support setdiff.
            }
        }
    }
    let a_tensor = gpu_helpers::gather_tensor(&handle_a)?;
    let b_tensor = gpu_helpers::gather_tensor(&handle_b)?;
    setdiff_numeric(a_tensor, b_tensor, opts)
}

fn setdiff_gpu_mixed(
    handle_gpu: GpuTensorHandle,
    other: Value,
    opts: &SetdiffOptions,
    gpu_is_a: bool,
) -> Result<SetdiffEvaluation, String> {
    let gpu_tensor = gpu_helpers::gather_tensor(&handle_gpu)?;
    let other_tensor = tensor::value_into_tensor_for("setdiff", other)?;
    if gpu_is_a {
        setdiff_numeric(gpu_tensor, other_tensor, opts)
    } else {
        setdiff_numeric(other_tensor, gpu_tensor, opts)
    }
}

fn setdiff_host(a: Value, b: Value, opts: &SetdiffOptions) -> Result<SetdiffEvaluation, String> {
    match (a, b) {
        (Value::ComplexTensor(at), Value::ComplexTensor(bt)) => setdiff_complex(at, bt, opts),
        (Value::ComplexTensor(at), Value::Complex(re, im)) => {
            let bt = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| format!("setdiff: {e}"))?;
            setdiff_complex(at, bt, opts)
        }
        (Value::Complex(a_re, a_im), Value::ComplexTensor(bt)) => {
            let at = ComplexTensor::new(vec![(a_re, a_im)], vec![1, 1])
                .map_err(|e| format!("setdiff: {e}"))?;
            setdiff_complex(at, bt, opts)
        }
        (Value::Complex(a_re, a_im), Value::Complex(b_re, b_im)) => {
            let at = ComplexTensor::new(vec![(a_re, a_im)], vec![1, 1])
                .map_err(|e| format!("setdiff: {e}"))?;
            let bt = ComplexTensor::new(vec![(b_re, b_im)], vec![1, 1])
                .map_err(|e| format!("setdiff: {e}"))?;
            setdiff_complex(at, bt, opts)
        }

        (Value::CharArray(ac), Value::CharArray(bc)) => setdiff_char(ac, bc, opts),

        (Value::StringArray(astring), Value::StringArray(bstring)) => {
            setdiff_string(astring, bstring, opts)
        }
        (Value::StringArray(astring), Value::String(b)) => {
            let bstring =
                StringArray::new(vec![b], vec![1, 1]).map_err(|e| format!("setdiff: {e}"))?;
            setdiff_string(astring, bstring, opts)
        }
        (Value::String(a), Value::StringArray(bstring)) => {
            let astring =
                StringArray::new(vec![a], vec![1, 1]).map_err(|e| format!("setdiff: {e}"))?;
            setdiff_string(astring, bstring, opts)
        }
        (Value::String(a), Value::String(b)) => {
            let astring =
                StringArray::new(vec![a], vec![1, 1]).map_err(|e| format!("setdiff: {e}"))?;
            let bstring =
                StringArray::new(vec![b], vec![1, 1]).map_err(|e| format!("setdiff: {e}"))?;
            setdiff_string(astring, bstring, opts)
        }

        (left, right) => {
            let tensor_a = tensor::value_into_tensor_for("setdiff", left)?;
            let tensor_b = tensor::value_into_tensor_for("setdiff", right)?;
            setdiff_numeric(tensor_a, tensor_b, opts)
        }
    }
}

fn setdiff_numeric(
    a: Tensor,
    b: Tensor,
    opts: &SetdiffOptions,
) -> Result<SetdiffEvaluation, String> {
    if opts.rows {
        setdiff_numeric_rows(a, b, opts)
    } else {
        setdiff_numeric_elements(a, b, opts)
    }
}

/// Helper exposed for acceleration providers handling numeric tensors entirely on the host.
pub fn setdiff_numeric_from_tensors(
    a: Tensor,
    b: Tensor,
    opts: &SetdiffOptions,
) -> Result<SetdiffEvaluation, String> {
    setdiff_numeric(a, b, opts)
}

fn setdiff_numeric_elements(
    a: Tensor,
    b: Tensor,
    opts: &SetdiffOptions,
) -> Result<SetdiffEvaluation, String> {
    let mut b_keys: HashSet<u64> = HashSet::new();
    for &value in &b.data {
        b_keys.insert(canonicalize_f64(value));
    }

    let mut seen: HashMap<u64, usize> = HashMap::new();
    let mut entries = Vec::<NumericDiffEntry>::new();
    let mut order_counter = 0usize;

    for (idx, &value) in a.data.iter().enumerate() {
        let key = canonicalize_f64(value);
        if b_keys.contains(&key) {
            continue;
        }
        if seen.contains_key(&key) {
            continue;
        }
        let entry_idx = entries.len();
        entries.push(NumericDiffEntry {
            value,
            index: idx,
            order_rank: order_counter,
        });
        seen.insert(key, entry_idx);
        order_counter += 1;
    }

    assemble_numeric_setdiff(entries, opts)
}

fn setdiff_numeric_rows(
    a: Tensor,
    b: Tensor,
    opts: &SetdiffOptions,
) -> Result<SetdiffEvaluation, String> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err("setdiff: 'rows' option requires 2-D numeric matrices".to_string());
    }
    if a.shape[1] != b.shape[1] {
        return Err(
            "setdiff: inputs must have the same number of columns when using 'rows'".to_string(),
        );
    }

    let rows_a = a.shape[0];
    let rows_b = b.shape[0];
    let cols = a.shape[1];

    let mut b_keys: HashSet<NumericRowKey> = HashSet::new();
    for r in 0..rows_b {
        let mut row_values = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r + c * rows_b;
            row_values.push(b.data[idx]);
        }
        b_keys.insert(NumericRowKey::from_slice(&row_values));
    }

    let mut seen: HashSet<NumericRowKey> = HashSet::new();
    let mut entries = Vec::<NumericRowDiffEntry>::new();
    let mut order_counter = 0usize;

    for r in 0..rows_a {
        let mut row_values = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r + c * rows_a;
            row_values.push(a.data[idx]);
        }
        let key = NumericRowKey::from_slice(&row_values);
        if b_keys.contains(&key) {
            continue;
        }
        if !seen.insert(key) {
            continue;
        }
        entries.push(NumericRowDiffEntry {
            row_data: row_values,
            row_index: r,
            order_rank: order_counter,
        });
        order_counter += 1;
    }

    assemble_numeric_row_setdiff(entries, opts, cols)
}

fn setdiff_complex(
    a: ComplexTensor,
    b: ComplexTensor,
    opts: &SetdiffOptions,
) -> Result<SetdiffEvaluation, String> {
    if opts.rows {
        setdiff_complex_rows(a, b, opts)
    } else {
        setdiff_complex_elements(a, b, opts)
    }
}

fn setdiff_complex_elements(
    a: ComplexTensor,
    b: ComplexTensor,
    opts: &SetdiffOptions,
) -> Result<SetdiffEvaluation, String> {
    let mut b_keys: HashSet<ComplexKey> = HashSet::new();
    for &value in &b.data {
        b_keys.insert(ComplexKey::new(value));
    }

    let mut seen: HashSet<ComplexKey> = HashSet::new();
    let mut entries = Vec::<ComplexDiffEntry>::new();
    let mut order_counter = 0usize;

    for (idx, &value) in a.data.iter().enumerate() {
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

fn setdiff_complex_rows(
    a: ComplexTensor,
    b: ComplexTensor,
    opts: &SetdiffOptions,
) -> Result<SetdiffEvaluation, String> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err("setdiff: 'rows' option requires 2-D complex matrices".to_string());
    }
    if a.shape[1] != b.shape[1] {
        return Err(
            "setdiff: inputs must have the same number of columns when using 'rows'".to_string(),
        );
    }

    let rows_a = a.shape[0];
    let rows_b = b.shape[0];
    let cols = a.shape[1];

    let mut b_keys: HashSet<Vec<ComplexKey>> = HashSet::new();
    for r in 0..rows_b {
        let mut key_row = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = r + c * rows_b;
            key_row.push(ComplexKey::new(b.data[idx]));
        }
        b_keys.insert(key_row);
    }

    let mut seen: HashSet<Vec<ComplexKey>> = HashSet::new();
    let mut entries = Vec::<ComplexRowDiffEntry>::new();
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
) -> Result<SetdiffEvaluation, String> {
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
) -> Result<SetdiffEvaluation, String> {
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
) -> Result<SetdiffEvaluation, String> {
    if a.cols != b.cols {
        return Err(
            "setdiff: inputs must have the same number of columns when using 'rows'".to_string(),
        );
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
) -> Result<SetdiffEvaluation, String> {
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
) -> Result<SetdiffEvaluation, String> {
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
) -> Result<SetdiffEvaluation, String> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err("setdiff: 'rows' option requires 2-D string arrays".to_string());
    }
    if a.shape[1] != b.shape[1] {
        return Err(
            "setdiff: inputs must have the same number of columns when using 'rows'".to_string(),
        );
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

fn assemble_numeric_setdiff(
    entries: Vec<NumericDiffEntry>,
    opts: &SetdiffOptions,
) -> Result<SetdiffEvaluation, String> {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    match opts.order {
        SetdiffOrder::Sorted => {
            order.sort_by(|&lhs, &rhs| compare_f64(entries[lhs].value, entries[rhs].value));
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
        Tensor::new(values, vec![order.len(), 1]).map_err(|e| format!("setdiff: {e}"))?;
    let ia_tensor = Tensor::new(ia, vec![order.len(), 1]).map_err(|e| format!("setdiff: {e}"))?;

    Ok(SetdiffEvaluation::new(
        Value::Tensor(value_tensor),
        ia_tensor,
    ))
}

fn assemble_numeric_row_setdiff(
    entries: Vec<NumericRowDiffEntry>,
    opts: &SetdiffOptions,
    cols: usize,
) -> Result<SetdiffEvaluation, String> {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    match opts.order {
        SetdiffOrder::Sorted => {
            order.sort_by(|&lhs, &rhs| {
                compare_numeric_rows(&entries[lhs].row_data, &entries[rhs].row_data)
            });
        }
        SetdiffOrder::Stable => {
            order.sort_by_key(|&idx| entries[idx].order_rank);
        }
    }

    let unique_rows = order.len();
    let mut values = vec![0.0f64; unique_rows * cols];
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
        Tensor::new(values, vec![unique_rows, cols]).map_err(|e| format!("setdiff: {e}"))?;
    let ia_tensor = Tensor::new(ia, vec![unique_rows, 1]).map_err(|e| format!("setdiff: {e}"))?;

    Ok(SetdiffEvaluation::new(
        Value::Tensor(value_tensor),
        ia_tensor,
    ))
}

fn assemble_complex_setdiff(
    entries: Vec<ComplexDiffEntry>,
    opts: &SetdiffOptions,
) -> Result<SetdiffEvaluation, String> {
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
        ComplexTensor::new(values, vec![order.len(), 1]).map_err(|e| format!("setdiff: {e}"))?;
    let ia_tensor = Tensor::new(ia, vec![order.len(), 1]).map_err(|e| format!("setdiff: {e}"))?;

    Ok(SetdiffEvaluation::new(
        complex_tensor_into_value(value_tensor),
        ia_tensor,
    ))
}

fn assemble_complex_row_setdiff(
    entries: Vec<ComplexRowDiffEntry>,
    opts: &SetdiffOptions,
    cols: usize,
) -> Result<SetdiffEvaluation, String> {
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
    let mut values = vec![(0.0f64, 0.0f64); unique_rows * cols];
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
        ComplexTensor::new(values, vec![unique_rows, cols]).map_err(|e| format!("setdiff: {e}"))?;
    let ia_tensor = Tensor::new(ia, vec![unique_rows, 1]).map_err(|e| format!("setdiff: {e}"))?;

    Ok(SetdiffEvaluation::new(
        complex_tensor_into_value(value_tensor),
        ia_tensor,
    ))
}

fn assemble_char_setdiff(
    entries: Vec<CharDiffEntry>,
    opts: &SetdiffOptions,
) -> Result<SetdiffEvaluation, String> {
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

    let value_array =
        CharArray::new(values, order.len(), 1).map_err(|e| format!("setdiff: {e}"))?;
    let ia_tensor = Tensor::new(ia, vec![order.len(), 1]).map_err(|e| format!("setdiff: {e}"))?;

    Ok(SetdiffEvaluation::new(
        Value::CharArray(value_array),
        ia_tensor,
    ))
}

fn assemble_char_row_setdiff(
    entries: Vec<CharRowDiffEntry>,
    opts: &SetdiffOptions,
    cols: usize,
) -> Result<SetdiffEvaluation, String> {
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

    let value_array =
        CharArray::new(values, unique_rows, cols).map_err(|e| format!("setdiff: {e}"))?;
    let ia_tensor = Tensor::new(ia, vec![unique_rows, 1]).map_err(|e| format!("setdiff: {e}"))?;

    Ok(SetdiffEvaluation::new(
        Value::CharArray(value_array),
        ia_tensor,
    ))
}

fn assemble_string_setdiff(
    entries: Vec<StringDiffEntry>,
    opts: &SetdiffOptions,
) -> Result<SetdiffEvaluation, String> {
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

    let value_array =
        StringArray::new(values, vec![order.len(), 1]).map_err(|e| format!("setdiff: {e}"))?;
    let ia_tensor = Tensor::new(ia, vec![order.len(), 1]).map_err(|e| format!("setdiff: {e}"))?;

    Ok(SetdiffEvaluation::new(
        Value::StringArray(value_array),
        ia_tensor,
    ))
}

fn assemble_string_row_setdiff(
    entries: Vec<StringRowDiffEntry>,
    opts: &SetdiffOptions,
    cols: usize,
) -> Result<SetdiffEvaluation, String> {
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

    let value_array =
        StringArray::new(values, vec![unique_rows, cols]).map_err(|e| format!("setdiff: {e}"))?;
    let ia_tensor = Tensor::new(ia, vec![unique_rows, 1]).map_err(|e| format!("setdiff: {e}"))?;

    Ok(SetdiffEvaluation::new(
        Value::StringArray(value_array),
        ia_tensor,
    ))
}

#[derive(Clone, Copy, Debug)]
struct NumericDiffEntry {
    value: f64,
    index: usize,
    order_rank: usize,
}

#[derive(Clone, Debug)]
struct NumericRowDiffEntry {
    row_data: Vec<f64>,
    row_index: usize,
    order_rank: usize,
}

#[derive(Clone, Copy, Debug)]
struct ComplexDiffEntry {
    value: (f64, f64),
    index: usize,
    order_rank: usize,
}

#[derive(Clone, Debug)]
struct ComplexRowDiffEntry {
    row_data: Vec<(f64, f64)>,
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

pub struct SetdiffEvaluation {
    values: Value,
    ia: Tensor,
}

impl SetdiffEvaluation {
    fn new(values: Value, ia: Tensor) -> Self {
        Self { values, ia }
    }

    pub fn from_setdiff_result(result: SetdiffResult) -> Result<Self, String> {
        let SetdiffResult { values, ia } = result;
        let values_tensor =
            Tensor::new(values.data, values.shape).map_err(|e| format!("setdiff: {e}"))?;
        let ia_tensor = Tensor::new(ia.data, ia.shape).map_err(|e| format!("setdiff: {e}"))?;
        Ok(SetdiffEvaluation::new(
            Value::Tensor(values_tensor),
            ia_tensor,
        ))
    }

    pub fn into_numeric_setdiff_result(self) -> Result<SetdiffResult, String> {
        let SetdiffEvaluation { values, ia } = self;
        let values_tensor = tensor::value_into_tensor_for("setdiff", values)?;
        Ok(SetdiffResult {
            values: HostTensorOwned {
                data: values_tensor.data,
                shape: values_tensor.shape,
            },
            ia: HostTensorOwned {
                data: ia.data,
                shape: ia.shape,
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

    pub fn values_value(&self) -> Value {
        self.values.clone()
    }

    pub fn ia_value(&self) -> Value {
        tensor::tensor_into_value(self.ia.clone())
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
    use runmat_builtins::{CharArray, StringArray, Tensor, Value};

    #[test]
    fn setdiff_numeric_sorted_default() {
        let a = Tensor::new(vec![5.0, 7.0, 5.0, 1.0], vec![4, 1]).unwrap();
        let b = Tensor::new(vec![7.0, 1.0, 3.0], vec![3, 1]).unwrap();
        let eval = evaluate(Value::Tensor(a), Value::Tensor(b), &[]).expect("setdiff");
        match eval.values_value() {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 1]);
                assert_eq!(t.data, vec![5.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        let ia = tensor::value_into_tensor_for("setdiff", eval.ia_value()).expect("ia tensor");
        assert_eq!(ia.data, vec![1.0]);
    }

    #[test]
    fn setdiff_numeric_stable() {
        let a = Tensor::new(vec![4.0, 2.0, 4.0, 1.0, 3.0], vec![5, 1]).unwrap();
        let b = Tensor::new(vec![3.0, 4.0, 5.0, 1.0], vec![4, 1]).unwrap();
        let eval = evaluate(Value::Tensor(a), Value::Tensor(b), &[Value::from("stable")])
            .expect("setdiff");
        match eval.values_value() {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 1]);
                assert_eq!(t.data, vec![2.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        let ia = tensor::value_into_tensor_for("setdiff", eval.ia_value()).expect("ia tensor");
        assert_eq!(ia.data, vec![2.0]);
    }

    #[test]
    fn setdiff_numeric_rows_sorted() {
        let a = Tensor::new(vec![1.0, 3.0, 1.0, 2.0, 4.0, 2.0], vec![3, 2]).unwrap();
        let b = Tensor::new(vec![3.0, 5.0, 4.0, 6.0], vec![2, 2]).unwrap();
        let eval =
            evaluate(Value::Tensor(a), Value::Tensor(b), &[Value::from("rows")]).expect("setdiff");
        match eval.values_value() {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.data, vec![1.0, 2.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        let ia = tensor::value_into_tensor_for("setdiff", eval.ia_value()).expect("ia tensor");
        assert_eq!(ia.data, vec![1.0]);
    }

    #[test]
    fn setdiff_numeric_removes_nan() {
        let a = Tensor::new(vec![f64::NAN, 2.0, 3.0], vec![3, 1]).unwrap();
        let b = Tensor::new(vec![f64::NAN], vec![1, 1]).unwrap();
        let eval = evaluate(Value::Tensor(a), Value::Tensor(b), &[]).expect("setdiff");
        let values = tensor::value_into_tensor_for("setdiff", eval.values_value()).expect("values");
        assert_eq!(values.data, vec![2.0, 3.0]);
        let ia = tensor::value_into_tensor_for("setdiff", eval.ia_value()).expect("ia tensor");
        assert_eq!(ia.data, vec![2.0, 3.0]);
    }

    #[test]
    fn setdiff_char_elements() {
        let a = CharArray::new(vec!['m', 'z', 'm', 'a'], 2, 2).unwrap();
        let b = CharArray::new(vec!['a', 'x', 'm', 'a'], 2, 2).unwrap();
        let eval = evaluate(Value::CharArray(a), Value::CharArray(b), &[]).expect("setdiff");
        match eval.values_value() {
            Value::CharArray(arr) => {
                assert_eq!(arr.rows, 1);
                assert_eq!(arr.cols, 1);
                assert_eq!(arr.data, vec!['z']);
            }
            other => panic!("expected char array, got {other:?}"),
        }
        let ia = tensor::value_into_tensor_for("setdiff", eval.ia_value()).expect("ia tensor");
        assert_eq!(ia.data, vec![3.0]);
    }

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
        let eval = evaluate(
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
        assert_eq!(ia.data, vec![1.0]);
    }

    #[test]
    fn setdiff_type_mismatch_errors() {
        let result = evaluate(Value::from(1.0), Value::String("a".into()), &[]);
        assert!(result.is_err());
    }

    #[test]
    fn setdiff_rejects_legacy_option() {
        let result = evaluate(Value::from(1.0), Value::from(2.0), &[Value::from("legacy")]);
        assert!(result
            .err()
            .unwrap()
            .contains("setdiff: the 'legacy' behaviour is not supported"));
    }

    #[test]
    fn setdiff_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor_a = Tensor::new(vec![10.0, 4.0, 6.0, 4.0], vec![4, 1]).unwrap();
            let tensor_b = Tensor::new(vec![6.0, 4.0, 2.0], vec![3, 1]).unwrap();
            let view_a = HostTensorView {
                data: &tensor_a.data,
                shape: &tensor_a.shape,
            };
            let view_b = HostTensorView {
                data: &tensor_b.data,
                shape: &tensor_b.shape,
            };
            let handle_a = provider.upload(&view_a).expect("upload a");
            let handle_b = provider.upload(&view_b).expect("upload b");
            let eval = evaluate(Value::GpuTensor(handle_a), Value::GpuTensor(handle_b), &[])
                .expect("setdiff");
            match eval.values_value() {
                Value::Tensor(t) => {
                    assert_eq!(t.data, vec![10.0]);
                }
                other => panic!("expected tensor result, got {other:?}"),
            }
            let ia = tensor::value_into_tensor_for("setdiff", eval.ia_value()).expect("ia tensor");
            assert_eq!(ia.data, vec![1.0]);
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn setdiff_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let a = Tensor::new(vec![8.0, 4.0, 2.0, 4.0], vec![4, 1]).unwrap();
        let b = Tensor::new(vec![2.0, 5.0], vec![2, 1]).unwrap();

        let cpu_eval =
            evaluate(Value::Tensor(a.clone()), Value::Tensor(b.clone()), &[]).expect("setdiff");
        let cpu_values = tensor::value_into_tensor_for("setdiff", cpu_eval.values_value()).unwrap();
        let cpu_ia = tensor::value_into_tensor_for("setdiff", cpu_eval.ia_value()).unwrap();

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
        let gpu_eval =
            evaluate(Value::GpuTensor(handle_a), Value::GpuTensor(handle_b), &[]).expect("setdiff");
        let gpu_values = tensor::value_into_tensor_for("setdiff", gpu_eval.values_value()).unwrap();
        let gpu_ia = tensor::value_into_tensor_for("setdiff", gpu_eval.ia_value()).unwrap();

        assert_eq!(gpu_values.data, cpu_values.data);
        assert_eq!(gpu_values.shape, cpu_values.shape);
        assert_eq!(gpu_ia.data, cpu_ia.data);
        assert_eq!(gpu_ia.shape, cpu_ia.shape);
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
