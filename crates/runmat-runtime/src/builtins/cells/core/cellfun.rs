//! MATLAB-compatible `cellfun` builtin with host execution semantics for RunMat.

use runmat_builtins::{
    CellArray, Closure, ComplexTensor, LogicalArray, StructValue, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::shape::{dims_to_row_tensor, value_numel};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{
    build_runtime_error, gather_if_needed, make_cell_with_shape, BuiltinResult, RuntimeError,
};

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "cellfun",
        builtin_path = "crate::builtins::cells::core::cellfun"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "cellfun"
category: "cells/core"
keywords: ["cellfun", "cell arrays", "functional programming", "uniformoutput", "errorhandler"]
summary: "Apply a function to every element of one or more cell arrays, mirroring MATLAB's cellfun behaviour."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "cellfun executes on the host. GPU-resident elements are gathered automatically before the callback is invoked."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::cells::core::cellfun::tests"
  integration: "builtins::cells::core::cellfun::tests::cellfun_error_handler_recovers"
---

# What does the `cellfun` function do in MATLAB / RunMat?
`cellfun` maps a function over one or more cell arrays. Each cell's contents become
inputs to the supplied function handle (or builtin name), and the results are collected
either into a numeric/logical/complex array (`'UniformOutput', true`, the default) or
into a new cell array (`'UniformOutput', false`). RunMat follows MATLAB's semantics for
result collection, broadcasting extra arguments, the `'ErrorHandler'` hook, and the
`'UniformOutput'` toggle.

## How does the `cellfun` function behave in MATLAB / RunMat?
- All cell array inputs must share the same size. Extra arguments that are **not** cell
  arrays are treated as constants and forwarded to every invocation.
- The callback may be supplied as a function handle (`@sin`), an anonymous function, or
  one of MATLAB's short string identifiers (for example `'isempty'` or `'isclass'`).
- With `'UniformOutput', true` (default) the callback must return scalars (numeric,
  logical, or complex). The results are packed into an array that mirrors the cell array
  shape. Mixed real/complex outputs automatically promote to complex.
- With `'UniformOutput', false` the callback may return arbitrary values; RunMat stores
  them in a new cell array that matches the input dimensions.
- `'ErrorHandler', handler` intercepts runtime errors. When the callback throws, RunMat
  calls `handler(errStruct, cellValue1, cellValue2, ..., extraArg1, extraArg2, ...)`.
  The structure contains the fields `identifier`, `message`, `index` (1-based linear
  index), and `indices` (1×N vector of subscripts). The handler's return value is
  inserted into the result.
- GPU values inside the cells are gathered to host memory before the callback runs so
  that uniform outputs can be materialised in CPU tensors today.

## `cellfun` GPU Execution Behaviour
`cellfun` itself is a host operation. RunMat gathers GPU-resident inputs before calling
the callback and stores the outputs on the host. This guarantees deterministic behaviour
even when the callback returns GPU arrays. Future acceleration hooks can elide the gather
step when providers support nested callbacks, but no provider work is required for
correctness today.

## Examples of using the `cellfun` function in MATLAB / RunMat

### Computing string lengths across a cell array

```matlab
C = {'apple', 'banana', 'cherry'};
lengths = cellfun(@length, C);
```

Expected output:
```matlab
lengths =
     5     6     6
```

### Mapping across two cell arrays in lockstep

```matlab
A = {1, 2, 3};
B = {4, 5, 6};
totals = cellfun(@plus, A, B);
```

Expected output:
```matlab
totals =
     5     7     9
```

### Keeping non-scalar outputs as cells

```matlab
names = {'Ada', 'Linus', 'Katherine'};
upper = cellfun(@upper, names, 'UniformOutput', false);
```

Expected output:
```matlab
upper =
  1×3 cell array
    {'ADA'}    {'LINUS'}    {'KATHERINE'}
```

### Supplying additional arguments alongside each cell

```matlab
C = {magic(3), eye(3)};
diag3 = cellfun('size', C, 2);
```

Expected output:
```matlab
diag3 =
     3     3
```

### Handling callback failures with `'ErrorHandler'`

```matlab
C = {'42', 'NaN', '3.14'};
handler = @(err, value) NaN;
numbers = cellfun(@str2double, C, 'ErrorHandler', handler);
```

Expected output:
```matlab
numbers =
   42     NaN    3.1400
```

### Using builtin short names instead of function handles

```matlab
cells = {[], 1:5, zeros(0, 3)};
empty_flags = cellfun('isempty', cells);
```

Expected output:
```matlab
empty_flags =
     1     0     1
```

### Returning rich metadata in the error handler

```matlab
cells = {1, 'two', 3};
eh = @(err, value) sprintf('%s @ %d', err.identifier, err.index);
results = cellfun(@(x) x + 1, cells, 'UniformOutput', false, 'ErrorHandler', eh);
```

Expected output:
```matlab
results =
  1×3 cell array
    {[2]}    {'MATLAB:UndefinedFunction @ 2'}    {[4]}
```

## FAQ

### Does `cellfun` modify the original cell array?
No. The builtin reads the cell contents, calls the supplied function, and returns a new
array. The original cell array is never mutated.

### What happens when the callback returns mixed real and complex values?
RunMat promotes the entire output to a complex array when `'UniformOutput', true`. Real
results are converted to complex numbers with zero imaginary parts to match MATLAB.

### Can the callback return strings or structs?
Yes, but you must specify `'UniformOutput', false`. That tells RunMat to collect results
into a cell array rather than forcing scalar concatenation.

### Are GPU arrays supported?
Yes. Inputs that live on the GPU are automatically gathered before the callback executes
so the function can operate on host values. The final result is host-resident today.

### How are errors reported to `'ErrorHandler'`?
The handler receives a structure with `identifier`, `message`, `index`, and `indices`
fields followed by the values that triggered the error. Returning a value from the
handler lets execution continue; rethrowing propagates the failure just like MATLAB.

### Do extra non-cell arguments need to be the same size as the cells?
No. Extra arguments are treated as constants and forwarded unchanged for every element.
Only the cell array inputs must agree on size.

### Can `cellfun` be used with anonymous functions that capture variables?
Absolutely. Captures are forwarded to the generated closure before the cell elements,
so anonymous functions behave the same way they do in MATLAB.

### What are the default output types for empty inputs?
When the cell arrays are empty and `'UniformOutput', true`, RunMat returns an empty
double array (size matches the input). For `'UniformOutput', false`, the result is an
empty cell array.

### Does `cellfun` support MATLAB's `'isclass'` short name?
Yes. RunMat maps `'isclass'` to the `class` builtin internally so you can write
`cellfun('isclass', C, 'double')` just like in MATLAB.

## See Also
[cell](./cell), [cell2mat](./cell2mat), [mat2cell](./mat2cell), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- Source code: [`crates/runmat-runtime/src/builtins/cells/core/cellfun.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/cells/core/cellfun.rs)
- Issue tracker: [RunMat GitHub Issues](https://github.com/runmat-org/runmat/issues/new/choose)
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::cells::core::cellfun")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cellfun",
    op_kind: GpuOpKind::Custom("host-cell-map"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Executes on the host and gathers GPU-resident inputs before evaluating callbacks.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::cells::core::cellfun")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cellfun",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Callback execution happens on the host; fusion planners should treat cellfun as a fusion barrier.",
};

const IDENT_INVALID_INPUT: &str = "MATLAB:cellfun:InvalidInput";
const IDENT_UNIFORM_OUTPUT: &str = "MATLAB:cellfun:UniformOutput";
const IDENT_FUNCTION_ERROR: &str = "MATLAB:cellfun:FunctionError";

fn cellfun_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("cellfun").build()
}

fn cellfun_error_with_identifier(message: impl Into<String>, identifier: &str) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin("cellfun")
        .with_identifier(identifier)
        .build()
}

#[runtime_builtin(
    name = "cellfun",
    category = "cells/core",
    summary = "Apply a function to the contents of each cell array element.",
    keywords = "cellfun,cell,array,functional",
    accel = "host",
    builtin_path = "crate::builtins::cells::core::cellfun"
)]
fn cellfun_builtin(func: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let callable = Callable::from_function(func)?;
    let mut args = rest;

    let mut uniform_output = true;
    let mut error_handler: Option<Callable> = None;

    while args.len() >= 2 {
        let name_candidate = args[args.len() - 2].clone();
        let Some(name) = extract_string(&name_candidate) else {
            break;
        };
        let value = args.pop().expect("value present");
        args.pop();
        match name.to_ascii_lowercase().as_str() {
            "uniformoutput" => {
                uniform_output = parse_uniform_output(value)?;
            }
            "errorhandler" => {
                error_handler = Some(Callable::from_function(value)?);
            }
            unknown => {
                return Err(cellfun_error_with_identifier(
                    format!("cellfun: unknown name-value argument '{unknown}'"),
                    IDENT_INVALID_INPUT,
                ));
            }
        }
    }

    if args.is_empty() {
        return Err(cellfun_error_with_identifier(
            "cellfun: expected at least one cell array input",
            IDENT_INVALID_INPUT,
        ));
    }

    let mut cell_inputs: Vec<CellArray> = Vec::new();
    let mut extra_args: Vec<Value> = Vec::new();
    let mut seen_non_cell = false;

    for value in args.into_iter() {
        match value {
            Value::Cell(ca) if !seen_non_cell => cell_inputs.push(ca),
            Value::Cell(_) => {
                return Err(cellfun_error_with_identifier(
                    "cellfun: cell array inputs must precede extra arguments",
                    IDENT_INVALID_INPUT,
                ));
            }
            other => {
                seen_non_cell = true;
                extra_args.push(other);
            }
        }
    }

    if cell_inputs.is_empty() {
        return Err(cellfun_error_with_identifier(
            "cellfun: expected at least one cell array input",
            IDENT_INVALID_INPUT,
        ));
    }

    let reference_shape = cell_inputs[0].shape.clone();
    for (idx, ca) in cell_inputs.iter().enumerate().skip(1) {
        if ca.shape != reference_shape {
            return Err(cellfun_error_with_identifier(
                format!(
                    "cellfun: cell array input {} does not match the size of the first input",
                    idx + 1
                ),
                IDENT_INVALID_INPUT,
            ));
        }
    }

    if uniform_output {
        execute_uniform(
            &callable,
            &cell_inputs,
            &extra_args,
            error_handler,
            &reference_shape,
        )
    } else {
        execute_cell(
            &callable,
            &cell_inputs,
            &extra_args,
            error_handler,
            &reference_shape,
        )
    }
}

fn execute_uniform(
    callable: &Callable,
    cell_inputs: &[CellArray],
    extra_args: &[Value],
    error_handler: Option<Callable>,
    shape: &[usize],
) -> BuiltinResult<Value> {
    let element_count = total_len(shape).ok_or_else(|| {
        cellfun_error_with_identifier(
            "cellfun: cell array size exceeds platform limits",
            IDENT_INVALID_INPUT,
        )
    })?;

    let host_extra_args = prepare_extra_args(extra_args)?;
    let mut collector = UniformCollector::Pending;
    let mut cell_values: Vec<Value> = Vec::with_capacity(cell_inputs.len());
    let mut call_args: Vec<Value> = Vec::with_capacity(cell_inputs.len() + host_extra_args.len());

    for linear_idx in 0..element_count {
        cell_values.clear();
        for cell in cell_inputs {
            let raw = deref_cell_value(cell, linear_idx);
            let host_value = gather_if_needed(&raw)?;
            cell_values.push(host_value);
        }
        call_args.clear();
        call_args.extend(cell_values.iter().cloned());
        call_args.extend(host_extra_args.iter().cloned());

        let result = match callable.call(&call_args) {
            Ok(value) => value,
            Err(err) => {
                let Some(handler) = error_handler.as_ref() else {
                    return Err(err);
                };
                let err_value = make_error_struct(&err, linear_idx, shape)?;
                let mut handler_args =
                    Vec::with_capacity(1 + cell_values.len() + host_extra_args.len());
                handler_args.push(err_value);
                handler_args.extend(cell_values.clone());
                handler_args.extend(host_extra_args.iter().cloned());
                handler.call(&handler_args)?
            }
        };

        let host_value = gather_if_needed(&result)?;
        collector.push(&host_value)?;
    }

    collector.finish(shape)
}

fn execute_cell(
    callable: &Callable,
    cell_inputs: &[CellArray],
    extra_args: &[Value],
    error_handler: Option<Callable>,
    shape: &[usize],
) -> BuiltinResult<Value> {
    let element_count = total_len(shape).ok_or_else(|| {
        cellfun_error_with_identifier(
            "cellfun: cell array size exceeds platform limits",
            IDENT_INVALID_INPUT,
        )
    })?;
    let host_extra_args = prepare_extra_args(extra_args)?;
    let mut outputs: Vec<Value> = Vec::with_capacity(element_count);
    let mut cell_values: Vec<Value> = Vec::with_capacity(cell_inputs.len());
    let mut call_args: Vec<Value> = Vec::with_capacity(cell_inputs.len() + host_extra_args.len());

    for linear_idx in 0..element_count {
        cell_values.clear();
        for cell in cell_inputs {
            let raw = deref_cell_value(cell, linear_idx);
            let host_value = gather_if_needed(&raw)?;
            cell_values.push(host_value);
        }
        call_args.clear();
        call_args.extend(cell_values.iter().cloned());
        call_args.extend(host_extra_args.iter().cloned());

        let result = match callable.call(&call_args) {
            Ok(value) => value,
            Err(err) => {
                let Some(handler) = error_handler.as_ref() else {
                    return Err(err);
                };
                let err_value = make_error_struct(&err, linear_idx, shape)?;
                let mut handler_args =
                    Vec::with_capacity(1 + cell_values.len() + host_extra_args.len());
                handler_args.push(err_value);
                handler_args.extend(cell_values.clone());
                handler_args.extend(host_extra_args.iter().cloned());
                handler.call(&handler_args)?
            }
        };

        let host_value = gather_if_needed(&result)?;
        outputs.push(host_value);
    }

    make_cell_with_shape(outputs, shape.to_vec())
        .map_err(|e| cellfun_error(format!("cellfun: {e}")))
}

fn deref_cell_value(cell: &CellArray, index: usize) -> Value {
    cell.data
        .get(index)
        .map(|ptr| (**ptr).clone())
        .unwrap_or(Value::Num(f64::NAN))
}

fn total_len(shape: &[usize]) -> Option<usize> {
    if shape.is_empty() {
        Some(0)
    } else {
        shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
    }
}

fn extract_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        _ => None,
    }
}

fn prepare_extra_args(extra_args: &[Value]) -> BuiltinResult<Vec<Value>> {
    let mut host_args = Vec::with_capacity(extra_args.len());
    for arg in extra_args {
        host_args.push(gather_if_needed(arg)?);
    }
    Ok(host_args)
}

fn parse_uniform_output(value: Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(b) => Ok(b),
        Value::Num(n) => Ok(n != 0.0),
        Value::Int(iv) => Ok(iv.to_f64() != 0.0),
        Value::String(s) => parse_bool_string(&s).ok_or_else(|| {
            cellfun_error_with_identifier(
                "cellfun: UniformOutput must be logical true or false",
                IDENT_UNIFORM_OUTPUT,
            )
        }),
        Value::CharArray(ca) if ca.rows == 1 => {
            let s: String = ca.data.iter().collect();
            parse_bool_string(&s).ok_or_else(|| {
                cellfun_error_with_identifier(
                    "cellfun: UniformOutput must be logical true or false",
                    IDENT_UNIFORM_OUTPUT,
                )
            })
        }
        other => Err(cellfun_error_with_identifier(
            format!("cellfun: UniformOutput must be logical true or false, got {other:?}"),
            IDENT_UNIFORM_OUTPUT,
        )),
    }
}

fn parse_bool_string(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "on" => Some(true),
        "false" | "off" => Some(false),
        _ => None,
    }
}

fn make_error_struct(
    raw_error: &RuntimeError,
    linear_index: usize,
    shape: &[usize],
) -> BuiltinResult<Value> {
    let (identifier, message) = error_identifier_and_message(raw_error);
    let mut st = StructValue::new();
    st.fields
        .insert("identifier".to_string(), Value::String(identifier));
    st.fields
        .insert("message".to_string(), Value::String(message));
    st.fields
        .insert("index".to_string(), Value::Num((linear_index + 1) as f64));
    let subs = linear_to_indices(linear_index, shape);
    let subs_tensor =
        dims_to_row_tensor(&subs).map_err(|e| cellfun_error(format!("cellfun: {e}")))?;
    st.fields
        .insert("indices".to_string(), Value::Tensor(subs_tensor));
    Ok(Value::Struct(st))
}

fn error_identifier_and_message(error: &RuntimeError) -> (String, String) {
    if let Some(identifier) = error.identifier() {
        return (identifier.to_string(), error.message().to_string());
    }
    split_error_message(error.message())
}

fn split_error_message(raw: &str) -> (String, String) {
    let trimmed = raw.trim();
    let mut indices = trimmed.match_indices(':');
    if let Some((_, _)) = indices.next() {
        if let Some((second_idx, _)) = indices.next() {
            let identifier = trimmed[..second_idx].trim().to_string();
            let message = trimmed[second_idx + 1..].trim().to_string();
            if !identifier.is_empty() && identifier.contains(':') {
                return (
                    identifier,
                    if message.is_empty() {
                        trimmed.to_string()
                    } else {
                        message
                    },
                );
            }
        } else if trimmed.len() >= 7
            && (trimmed[..7].eq_ignore_ascii_case("matlab:")
                || trimmed[..7].eq_ignore_ascii_case("runmat:"))
        {
            return (trimmed.to_string(), String::new());
        }
    }
    (IDENT_FUNCTION_ERROR.to_string(), trimmed.to_string())
}

fn linear_to_indices(mut index: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![1];
    }
    let mut subs = Vec::with_capacity(shape.len());
    for &dim in shape {
        if dim == 0 {
            subs.push(1);
            continue;
        }
        let coord = (index % dim) + 1;
        subs.push(coord);
        index /= dim;
    }
    subs
}

#[derive(Clone)]
enum Callable {
    Builtin { name: String },
    Closure(Closure),
    Special(SpecialCallable),
}

impl Callable {
    fn from_function(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::String(s) => Self::from_text(&s, true),
            Value::CharArray(ca) => {
                if ca.rows != 1 {
                    Err(cellfun_error_with_identifier(
                        "cellfun: function name must be a character vector or string scalar",
                        IDENT_INVALID_INPUT,
                    ))
                } else {
                    let text: String = ca.data.iter().collect();
                    Self::from_text(&text, true)
                }
            }
            Value::StringArray(sa) => {
                if sa.data.len() == 1 {
                    Self::from_text(&sa.data[0], true)
                } else {
                    Err(cellfun_error_with_identifier(
                        "cellfun: function name must be a character vector or string scalar",
                        IDENT_INVALID_INPUT,
                    ))
                }
            }
            Value::FunctionHandle(name) => Self::from_text(&name, false),
            Value::Closure(c) => Ok(Callable::Closure(c)),
            other => Err(cellfun_error_with_identifier(
                format!("cellfun: expected function handle or builtin name, got {other:?}"),
                IDENT_INVALID_INPUT,
            )),
        }
    }

    fn from_text(text: &str, fold_case: bool) -> BuiltinResult<Self> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Err(cellfun_error_with_identifier(
                "cellfun: expected function handle or builtin name, got empty string",
                IDENT_INVALID_INPUT,
            ));
        }
        if let Some(rest) = trimmed.strip_prefix('@') {
            let name = rest.trim();
            if name.is_empty() {
                Err(cellfun_error_with_identifier(
                    "cellfun: empty function handle",
                    IDENT_INVALID_INPUT,
                ))
            } else {
                Ok(Callable::Builtin {
                    name: name.to_string(),
                })
            }
        } else {
            let lowered = trimmed.to_ascii_lowercase();
            if fold_case && lowered == "isclass" {
                Ok(Callable::Special(SpecialCallable::IsClass))
            } else if fold_case && lowered == "prodofsize" {
                Ok(Callable::Special(SpecialCallable::ProdOfSize))
            } else {
                let name = if fold_case {
                    lowered
                } else {
                    trimmed.to_string()
                };
                Ok(Callable::Builtin { name })
            }
        }
    }

    fn call(&self, args: &[Value]) -> BuiltinResult<Value> {
        match self {
            Callable::Builtin { name } => crate::call_builtin(name, args),
            Callable::Closure(c) => {
                let mut captures = c.captures.clone();
                captures.extend_from_slice(args);
                crate::call_builtin(&c.function_name, &captures)
            }
            Callable::Special(special) => special.call(args),
        }
    }
}

#[derive(Clone)]
enum SpecialCallable {
    ProdOfSize,
    IsClass,
}

impl SpecialCallable {
    fn call(&self, args: &[Value]) -> BuiltinResult<Value> {
        match self {
            SpecialCallable::ProdOfSize => {
                let value = args.first().ok_or_else(|| {
                    cellfun_error_with_identifier(
                        "cellfun: prodofsize requires one input",
                        IDENT_INVALID_INPUT,
                    )
                })?;
                Ok(Value::Num(value_numel(value) as f64))
            }
            SpecialCallable::IsClass => {
                if args.len() < 2 {
                    return Err(cellfun_error_with_identifier(
                        "cellfun: 'isclass' requires a class name argument",
                        IDENT_INVALID_INPUT,
                    ));
                }
                let left = args[0].clone();
                let class_name = extract_string(&args[1]).ok_or_else(|| {
                    cellfun_error_with_identifier(
                        "cellfun: class name must be a string scalar",
                        IDENT_INVALID_INPUT,
                    )
                })?;
                let class_value = crate::call_builtin("class", &[left])?;
                let class_str = extract_string(&class_value).ok_or_else(|| {
                    cellfun_error_with_identifier(
                        "cellfun: failed to evaluate class name",
                        IDENT_FUNCTION_ERROR,
                    )
                })?;
                Ok(Value::Bool(
                    class_str.eq_ignore_ascii_case(class_name.trim()),
                ))
            }
        }
    }
}

enum UniformCollector {
    Pending,
    Double(Vec<f64>),
    Logical(Vec<u8>),
    Complex(Vec<(f64, f64)>),
}

impl UniformCollector {
    fn push(&mut self, value: &Value) -> BuiltinResult<()> {
        match self {
            UniformCollector::Pending => match classify_value(value)? {
                ClassifiedValue::Logical(b) => {
                    *self = UniformCollector::Logical(vec![b as u8]);
                    Ok(())
                }
                ClassifiedValue::Double(d) => {
                    *self = UniformCollector::Double(vec![d]);
                    Ok(())
                }
                ClassifiedValue::Complex(c) => {
                    *self = UniformCollector::Complex(vec![c]);
                    Ok(())
                }
            },
            UniformCollector::Logical(bits) => match classify_value(value)? {
                ClassifiedValue::Logical(b) => {
                    bits.push(b as u8);
                    Ok(())
                }
                ClassifiedValue::Double(d) => {
                    let mut data: Vec<f64> = bits
                        .iter()
                        .map(|&bit| if bit != 0 { 1.0 } else { 0.0 })
                        .collect();
                    data.push(d);
                    *self = UniformCollector::Double(data);
                    Ok(())
                }
                ClassifiedValue::Complex(c) => {
                    let mut data: Vec<(f64, f64)> = bits
                        .iter()
                        .map(|&bit| if bit != 0 { (1.0, 0.0) } else { (0.0, 0.0) })
                        .collect();
                    data.push(c);
                    *self = UniformCollector::Complex(data);
                    Ok(())
                }
            },
            UniformCollector::Double(data) => match classify_value(value)? {
                ClassifiedValue::Logical(b) => {
                    data.push(if b { 1.0 } else { 0.0 });
                    Ok(())
                }
                ClassifiedValue::Double(d) => {
                    data.push(d);
                    Ok(())
                }
                ClassifiedValue::Complex(c) => {
                    let promoted: Vec<(f64, f64)> = data.iter().map(|&v| (v, 0.0)).collect();
                    let mut complex = promoted;
                    complex.push(c);
                    *self = UniformCollector::Complex(complex);
                    Ok(())
                }
            },
            UniformCollector::Complex(data) => match classify_value(value)? {
                ClassifiedValue::Logical(b) => {
                    data.push((if b { 1.0 } else { 0.0 }, 0.0));
                    Ok(())
                }
                ClassifiedValue::Double(d) => {
                    data.push((d, 0.0));
                    Ok(())
                }
                ClassifiedValue::Complex(c) => {
                    data.push(c);
                    Ok(())
                }
            },
        }
    }

    fn finish(self, shape: &[usize]) -> BuiltinResult<Value> {
        match self {
            UniformCollector::Pending => {
                let total = total_len(shape).unwrap_or(0);
                let data = vec![0.0; total];
                let tensor = Tensor::new(data, shape.to_vec())
                    .map_err(|e| cellfun_error(format!("cellfun: {e}")))?;
                Ok(Value::Tensor(tensor))
            }
            UniformCollector::Double(data) => {
                let tensor = Tensor::new(data, shape.to_vec())
                    .map_err(|e| cellfun_error(format!("cellfun: {e}")))?;
                Ok(Value::Tensor(tensor))
            }
            UniformCollector::Logical(bits) => {
                let logical = LogicalArray::new(bits, shape.to_vec())
                    .map_err(|e| cellfun_error(format!("cellfun: {e}")))?;
                Ok(Value::LogicalArray(logical))
            }
            UniformCollector::Complex(data) => {
                let complex = ComplexTensor::new(data, shape.to_vec())
                    .map_err(|e| cellfun_error(format!("cellfun: {e}")))?;
                Ok(Value::ComplexTensor(complex))
            }
        }
    }
}

enum ClassifiedValue {
    Logical(bool),
    Double(f64),
    Complex((f64, f64)),
}

fn classify_value(value: &Value) -> BuiltinResult<ClassifiedValue> {
    match value {
        Value::Bool(b) => Ok(ClassifiedValue::Logical(*b)),
        Value::Num(n) => Ok(ClassifiedValue::Double(*n)),
        Value::Int(iv) => Ok(ClassifiedValue::Double(iv.to_f64())),
        Value::Complex(re, im) => Ok(ClassifiedValue::Complex((*re, *im))),
        Value::Tensor(t) if t.data.len() == 1 => Ok(ClassifiedValue::Double(t.data[0])),
        Value::LogicalArray(la) if la.data.len() == 1 => {
            Ok(ClassifiedValue::Logical(la.data[0] != 0))
        }
        Value::ComplexTensor(ct) if ct.data.len() == 1 => Ok(ClassifiedValue::Complex(ct.data[0])),
        _ => Err(cellfun_error_with_identifier(
            "cellfun: callback must return scalar values when 'UniformOutput' is true",
            IDENT_UNIFORM_OUTPUT,
        )),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{IntValue, StringArray};
    use std::convert::TryInto;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cellfun_length_uniform_default() {
        let cell = crate::make_cell(
            vec![
                Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap()),
                Value::Tensor(Tensor::new(vec![4.0, 5.0, 6.0, 7.0], vec![1, 4]).unwrap()),
                Value::Tensor(Tensor::new(vec![8.0, 9.0], vec![1, 2]).unwrap()),
            ],
            1,
            3,
        )
        .expect("cell");
        let result =
            cellfun_builtin(Value::String("@length".into()), vec![cell]).expect("cellfun length");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                assert_eq!(t.data, vec![3.0, 4.0, 2.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cellfun_multiple_cells_plus() {
        let left = crate::make_cell(
            vec![Value::Num(1.0), Value::Num(2.0), Value::Num(3.0)],
            1,
            3,
        )
        .expect("cell");
        let right = crate::make_cell(
            vec![Value::Num(4.0), Value::Num(5.0), Value::Num(6.0)],
            1,
            3,
        )
        .expect("cell");
        let result = cellfun_builtin(Value::String("@__cellfun_add".into()), vec![left, right])
            .expect("cellfun add");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![5.0, 7.0, 9.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cellfun_uniform_false_returns_cells() {
        let cell = crate::make_cell(
            vec![
                Value::String("Ada".into()),
                Value::String("Linus".into()),
                Value::String("Katherine".into()),
            ],
            1,
            3,
        )
        .expect("cell");
        let result = cellfun_builtin(
            Value::String("@upper".into()),
            vec![
                cell,
                Value::String("UniformOutput".into()),
                Value::Bool(false),
            ],
        )
        .expect("cellfun upper");
        match result {
            Value::Cell(ca) => {
                assert_eq!(ca.shape, vec![1, 3]);
                let upper_a = (*ca.data[0]).clone();
                let upper_b = (*ca.data[1]).clone();
                let upper_c = (*ca.data[2]).clone();
                assert_eq!(extract_string(&upper_a).unwrap(), "ADA");
                assert_eq!(extract_string(&upper_b).unwrap(), "LINUS");
                assert_eq!(extract_string(&upper_c).unwrap(), "KATHERINE");
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cellfun_error_handler_recovers() {
        let cells = crate::make_cell(
            vec![Value::Num(1.0), Value::Num(2.0), Value::Num(3.0)],
            1,
            3,
        )
        .expect("cell");
        let handler = Value::Closure(Closure {
            function_name: "__cellfun_test_handler".into(),
            captures: vec![Value::Num(0.0)],
        });
        let result = cellfun_builtin(
            Value::String("@nonexistent_builtin".into()),
            vec![cells, Value::String("ErrorHandler".into()), handler],
        )
        .expect("cellfun error handler");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![0.0, 0.0, 0.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cellfun_string_identifier() {
        let cells = crate::make_cell(
            vec![
                Value::CharArray(runmat_builtins::CharArray::new_row("")),
                Value::CharArray(runmat_builtins::CharArray::new_row("abc")),
                Value::CharArray(runmat_builtins::CharArray::new_row("")),
            ],
            1,
            3,
        )
        .expect("cell");
        let result = cellfun_builtin(
            Value::CharArray(runmat_builtins::CharArray::new_row("isempty")),
            vec![cells],
        )
        .expect("isempty");
        match result {
            Value::LogicalArray(la) => {
                assert_eq!(la.shape, vec![1, 3]);
                assert_eq!(la.data, vec![1, 0, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cellfun_string_array_identifier() {
        let cells = crate::make_cell(
            vec![Value::CharArray(runmat_builtins::CharArray::new_row(""))],
            1,
            1,
        )
        .expect("cell");
        let sa = StringArray::new(vec!["isempty".into()], vec![1, 1]).unwrap();
        let result =
            cellfun_builtin(Value::StringArray(sa), vec![cells]).expect("cellfun string array");
        match result {
            Value::LogicalArray(la) => {
                assert_eq!(la.shape, vec![1, 1]);
                assert_eq!(la.data, vec![1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cellfun_uniform_true_non_scalar_errors() {
        let cells = crate::make_cell(
            vec![Value::Tensor(
                Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap(),
            )],
            1,
            1,
        )
        .expect("cell");
        let err = cellfun_builtin(Value::String("@eye".into()), vec![cells])
            .unwrap_err()
            .to_string();
        assert!(
            err.to_ascii_lowercase().contains("uniformoutput"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cellfun_uniform_promotes_logical_to_double() {
        let cells = crate::make_cell(vec![Value::Bool(true), Value::Num(2.5)], 1, 2).unwrap();
        let result = cellfun_builtin(Value::String("@__cellfun_identity".into()), vec![cells])
            .expect("cellfun identity");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.data, vec![1.0, 2.5]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cellfun_uniform_promotes_double_to_complex() {
        let cells =
            crate::make_cell(vec![Value::Num(2.0), Value::Complex(0.0, 1.0)], 1, 2).unwrap();
        let result = cellfun_builtin(Value::String("@__cellfun_identity".into()), vec![cells])
            .expect("cellfun identity");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![1, 2]);
                assert_eq!(ct.data, vec![(2.0, 0.0), (0.0, 1.0)]);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cellfun_errors_on_mismatched_cell_sizes() {
        let first = crate::make_cell(vec![Value::Num(1.0), Value::Num(2.0)], 1, 2).unwrap();
        let second = crate::make_cell(vec![Value::Num(3.0)], 1, 1).unwrap();
        let err = cellfun_builtin(
            Value::String("@__cellfun_identity".into()),
            vec![first, second],
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.to_ascii_lowercase().contains("size"),
            "expected size mismatch error, got: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cellfun_uniformoutput_accepts_char_flags() {
        let strings =
            crate::make_cell(vec![Value::String("Ada".into())], 1, 1).expect("cell creation");
        let result = cellfun_builtin(
            Value::String("@upper".into()),
            vec![
                strings,
                Value::CharArray(runmat_builtins::CharArray::new_row("UniformOutput")),
                Value::CharArray(runmat_builtins::CharArray::new_row("off")),
            ],
        )
        .expect("cellfun upper char flag");
        assert!(
            matches!(result, Value::Cell(_)),
            "expected cell array result when UniformOutput is 'off'"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cellfun_isclass_special_case() {
        let ints = crate::make_cell(
            vec![
                Value::Int(IntValue::I32(5)),
                Value::Num(std::f64::consts::PI),
                Value::Int(IntValue::I16(2)),
            ],
            1,
            3,
        )
        .expect("cell");
        let result = cellfun_builtin(
            Value::String("isclass".into()),
            vec![ints, Value::String("int32".into())],
        )
        .expect("cellfun isclass");
        match result {
            Value::LogicalArray(la) => {
                assert_eq!(la.data, vec![1, 0, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cellfun_passes_additional_arguments() {
        let matrices = crate::make_cell(
            vec![
                Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap()),
                Value::Tensor(Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap()),
            ],
            1,
            2,
        )
        .expect("cell");
        let dimension = Value::Num(2.0);
        let result = cellfun_builtin(Value::String("size".into()), vec![matrices, dimension])
            .expect("cellfun size");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![2.0, 2.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cellfun_handles_string_array_uniform_false() {
        let sa = StringArray::new(vec!["foo".into(), "bar".into()], vec![1, 2]).unwrap();
        let cell = crate::make_cell(vec![Value::StringArray(sa)], 1, 1).unwrap();
        let result = cellfun_builtin(
            Value::String("@strlength".into()),
            vec![
                cell,
                Value::String("UniformOutput".into()),
                Value::Bool(false),
            ],
        )
        .unwrap();
        match result {
            Value::Cell(ca) => {
                assert_eq!(ca.shape, vec![1, 1]);
                let inner = (*ca.data[0]).clone();
                match inner {
                    Value::Tensor(t) => assert_eq!(t.data, vec![3.0, 3.0]),
                    _ => panic!("expected tensor inside cell"),
                }
            }
            other => panic!("expected cell, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cellfun_gathers_gpu_inputs() {
        test_support::with_test_provider(|provider| {
            let angle = std::f64::consts::PI / 6.0;
            let tensor = Tensor::new(vec![angle], vec![1, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let cell = crate::make_cell(vec![Value::GpuTensor(handle)], 1, 1).expect("cell");
            let result =
                cellfun_builtin(Value::String("@sin".into()), vec![cell]).expect("cellfun sin");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![1, 1]);
            let expected = angle.sin();
            assert!((gathered.data[0] - expected).abs() < 1e-12);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cellfun_with_wgpu_provider_handles_gpu_cells() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");

        let value = Tensor::new(vec![0.25], vec![1, 1]).unwrap();
        let view = HostTensorView {
            data: &value.data,
            shape: &value.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let cell = crate::make_cell(vec![Value::GpuTensor(handle)], 1, 1).expect("cell");

        let result =
            cellfun_builtin(Value::String("@sin".into()), vec![cell]).expect("cellfun sin");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![1, 1]);
        let expected = value.data[0].sin();
        assert!((gathered.data[0] - expected).abs() < 1e-12);
    }

    #[runmat_macros::runtime_builtin(
        name = "__cellfun_test_handler",
        builtin_path = "crate::builtins::cells::core::cellfun::tests"
    )]
    fn cellfun_test_handler(
        seed: Value,
        _err: Value,
        rest: Vec<Value>,
    ) -> crate::BuiltinResult<Value> {
        // Return the captured seed regardless of the inputs; ensure rest is present for coverage.
        let _ = rest;
        Ok(seed)
    }

    #[runmat_macros::runtime_builtin(
        name = "__cellfun_add",
        builtin_path = "crate::builtins::cells::core::cellfun::tests"
    )]
    fn cellfun_add(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
        let a: f64 = (&lhs).try_into()?;
        let b: f64 = (&rhs).try_into()?;
        Ok(Value::Num(a + b))
    }

    #[runmat_macros::runtime_builtin(
        name = "__cellfun_identity",
        builtin_path = "crate::builtins::cells::core::cellfun::tests"
    )]
    fn cellfun_identity(value: Value) -> crate::BuiltinResult<Value> {
        Ok(value)
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = crate::builtins::common::test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
