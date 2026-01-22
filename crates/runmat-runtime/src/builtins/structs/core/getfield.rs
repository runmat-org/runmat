//! MATLAB-compatible `getfield` builtin with struct array and object support.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::indexing::perform_indexing;
use crate::make_cell_with_shape;
use crate::{
    build_runtime_error, call_builtin_async, gather_if_needed_async, BuiltinResult, RuntimeError,
};
use runmat_builtins::{
    Access, CellArray, CharArray, ComplexTensor, HandleRef, Listener, LogicalArray, MException,
    ObjectInstance, StructValue, Tensor, Value,
};
use runmat_macros::runtime_builtin;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "getfield",
        builtin_path = "crate::builtins::structs::core::getfield"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "getfield"
category: "structs/core"
keywords: ["getfield", "struct", "struct array", "object property", "metadata"]
summary: "Access a field or property from structs, struct arrays, or MATLAB-style objects."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Runs on the host. Values that already reside on the GPU stay resident; no kernels are dispatched."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::structs::core::getfield::tests"
  integration: "runmat_ignition::tests::functions::member_get_set_and_method_call_skeleton"
---

# What does the `getfield` function do in MATLAB / RunMat?
`value = getfield(S, field)` returns the contents of `S.field`. RunMat matches MATLAB by
supporting nested field access, struct arrays (via index cells), and MATLAB-style objects
created with `new_object`.

## How does the `getfield` function behave in MATLAB / RunMat?
- Field names must be character vectors or string scalars. Several field names in a row
  navigate nested structs: `getfield(S, "outer", "inner")` is equivalent to
  `S.outer.inner`.
- When `S` is a struct array, `getfield(S, "field")` examines the first element by default.
  Provide indices in a cell array to target another element:
  `getfield(S, {k}, "field")` yields `S(k).field`.
- After a field name you may supply an index cell to subscript the field value, e.g.
  `getfield(S, "values", {row, col})`. Each position accepts positive integers or the
  keyword `end` to reference the last element in that dimension.
- MATLAB-style objects honour property attributes: static or private properties raise errors,
  while dependent properties invoke `get.<name>` when available.
- Handle objects dereference to their underlying instance automatically. Deleted handles raise
  the standard MATLAB-style error.
- `MException` values expose the `message`, `identifier`, and `stack` fields for compatibility
  with MATLAB error handling.

## `getfield` Function GPU Execution Behaviour
`getfield` is metadata-only. When structs or objects contain GPU tensors, the tensors
remain on the device. The builtin manipulates only the host-side metadata and does not
dispatch GPU kernels or gather buffers back to the CPU. Results inherit residency from the
values being returned.

## Examples of using the `getfield` function in MATLAB / RunMat

### Reading a scalar struct field
```matlab
stats = struct("mean", 42, "stdev", 3.5);
mu = getfield(stats, "mean");
```

Expected output:
```matlab
mu = 42
```

### Navigating nested structs with multiple field names
```matlab
cfg = struct("solver", struct("name", "cg", "tolerance", 1e-6));
tol = getfield(cfg, "solver", "tolerance");
```

Expected output:
```matlab
tol = 1.0000e-06
```

### Accessing an element of a struct array
```matlab
people = struct("name", {"Ada", "Grace"}, "id", {101, 102});
lastId = getfield(people, {2}, "id");
```

Expected output:
```matlab
lastId = 102
```

### Reading the first element of a struct array automatically
```matlab
people = struct("name", {"Ada", "Grace"}, "id", {101, 102});
firstName = getfield(people, "name");
```

Expected output:
```matlab
firstName = "Ada"
```

### Using `end` to reference the last item in a field
```matlab
series = struct("values", {1:5});
lastValue = getfield(series, "values", {"end"});
```

Expected output:
```matlab
lastValue = 5
```

### Indexing into a numeric field value
```matlab
measurements = struct("values", {[1 2 3; 4 5 6]});
entry = getfield(measurements, "values", {2, 3});
```

Expected output:
```matlab
entry = 6
```

### Gathering the exception message from a try/catch block
```matlab
try
    error("MATLAB:domainError", "Bad input");
catch e
    msg = getfield(e, "message");
end
```

Expected output:
```matlab
msg = 'Bad input'
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You do not have to move data between CPU and GPU explicitly when calling `getfield`. The
builtin works entirely with metadata and returns handles or tensors in whatever memory space
they already inhabit.

## FAQ

### Does `getfield` work on non-struct values?
No. The first argument must be a scalar struct, a struct array (possibly empty), an object
instance created with `new_object`, a handle object, or an `MException`. Passing other
types raises an error.

### How do I access nested struct fields?
Provide every level explicitly: `getfield(S, "parent", "child", "leaf")` traverses the same
path as `S.parent.child.leaf`.

### How do I read from a struct array?
Supply a cell array of indices before the first field name. For example,
`getfield(S, {3}, "value")` mirrors `S(3).value`. Indices are one-based like MATLAB.

### Can I index the value stored in a field?
Yes. You may supply scalars or `end` inside the index cell to reference elements of the
field value.

### Do dependent properties run their getter methods?
Yes. If a property is marked `Dependent` and a `get.propertyName` builtin exists, `getfield`
invokes it. Otherwise the backing field `<property>_backing` is inspected.

### What happens when I query a deleted handle object?
RunMat mirrors MATLAB by raising `Invalid or deleted handle object 'ClassName'.`

## See Also
[fieldnames](./fieldnames), [isfield](./isfield), [setfield](./setfield), [struct](./struct), [class](./class)
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::structs::core::getfield")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "getfield",
    op_kind: GpuOpKind::Custom("getfield"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Pure metadata operation; acceleration providers do not participate.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::structs::core::getfield")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "getfield",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Acts as a fusion barrier because it inspects metadata on the host.",
};

const BUILTIN_NAME: &str = "getfield";

fn getfield_flow(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn remap_getfield_flow(err: RuntimeError, prefix: Option<&str>) -> RuntimeError {
    let mut message = err.message().to_string();
    if let Some(prefix) = prefix {
        if !message.starts_with(prefix) {
            message = format!("{prefix}{message}");
        }
    }
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = err.identifier() {
        builder = builder.with_identifier(identifier);
    }
    builder.with_source(err).build()
}

fn is_undefined_function(err: &RuntimeError) -> bool {
    err.identifier() == Some("MATLAB:UndefinedFunction")
        || err.message().contains("MATLAB:UndefinedFunction")
}

#[runtime_builtin(
    name = "getfield",
    category = "structs/core",
    summary = "Access a field or property from structs, struct arrays, or MATLAB-style objects.",
    keywords = "getfield,struct,object,field access",
    builtin_path = "crate::builtins::structs::core::getfield"
)]
async fn getfield_builtin(base: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let parsed = parse_arguments(rest)?;

    let mut current = base;
    if let Some(index) = parsed.leading_index {
        current = apply_indices(current, &index).await?;
    }

    for step in parsed.fields {
        current = get_field_value(current, &step.name).await?;
        if let Some(index) = step.index {
            current = apply_indices(current, &index).await?;
        }
    }

    Ok(current)
}

#[derive(Default)]
struct ParsedArguments {
    leading_index: Option<IndexSelector>,
    fields: Vec<FieldStep>,
}

struct FieldStep {
    name: String,
    index: Option<IndexSelector>,
}

#[derive(Clone)]
struct IndexSelector {
    components: Vec<IndexComponent>,
}

#[derive(Clone)]
enum IndexComponent {
    Scalar(usize),
    End,
}

fn parse_arguments(mut rest: Vec<Value>) -> BuiltinResult<ParsedArguments> {
    if rest.is_empty() {
        return Err(getfield_flow("getfield: expected at least one field name"));
    }

    let mut parsed = ParsedArguments::default();
    if let Some(first) = rest.first() {
        if is_index_selector(first) {
            let value = rest.remove(0);
            parsed.leading_index = Some(parse_index_selector(value)?);
        }
    }

    if rest.is_empty() {
        return Err(getfield_flow("getfield: expected field name after indices"));
    }

    let mut iter = rest.into_iter().peekable();
    while let Some(arg) = iter.next() {
        let field_name = parse_field_name(arg)?;
        let mut step = FieldStep {
            name: field_name,
            index: None,
        };
        if let Some(next) = iter.peek() {
            if is_index_selector(next) {
                let selector = iter.next().unwrap();
                step.index = Some(parse_index_selector(selector)?);
            }
        }
        parsed.fields.push(step);
    }

    if parsed.fields.is_empty() {
        return Err(getfield_flow("getfield: expected field name arguments"));
    }

    Ok(parsed)
}

fn is_index_selector(value: &Value) -> bool {
    matches!(value, Value::Cell(_))
}

fn parse_index_selector(value: Value) -> BuiltinResult<IndexSelector> {
    let Value::Cell(cell) = value else {
        return Err(getfield_flow(
            "getfield: indices must be provided in a cell array",
        ));
    };

    let mut components = Vec::with_capacity(cell.data.len());
    for handle in &cell.data {
        let entry = unsafe { &*handle.as_raw() };
        components.push(parse_index_component(entry)?);
    }

    Ok(IndexSelector { components })
}

fn parse_index_component(value: &Value) -> BuiltinResult<IndexComponent> {
    match value {
        Value::CharArray(ca) => {
            let text: String = ca.data.iter().collect();
            parse_index_text(text.trim())
        }
        Value::String(s) => parse_index_text(s.trim()),
        Value::StringArray(sa) if sa.data.len() == 1 => parse_index_text(sa.data[0].trim()),
        _ => {
            let idx = parse_positive_scalar(value).map_err(|err| {
                getfield_flow(format!(
                    "getfield: invalid index element ({})",
                    err.message()
                ))
            })?;
            Ok(IndexComponent::Scalar(idx))
        }
    }
}

fn parse_index_text(text: &str) -> BuiltinResult<IndexComponent> {
    if text.eq_ignore_ascii_case("end") {
        return Ok(IndexComponent::End);
    }
    if text == ":" {
        return Err(getfield_flow(
            "getfield: ':' indexing is not currently supported",
        ));
    }
    if text.is_empty() {
        return Err(getfield_flow("getfield: index elements must not be empty"));
    }
    if let Ok(value) = text.parse::<usize>() {
        if value == 0 {
            return Err(getfield_flow("getfield: index must be >= 1"));
        }
        return Ok(IndexComponent::Scalar(value));
    }
    Err(getfield_flow(format!(
        "getfield: invalid index element '{}'",
        text
    )))
}

fn parse_positive_scalar(value: &Value) -> BuiltinResult<usize> {
    let number = match value {
        Value::Int(i) => i.to_i64() as f64,
        Value::Num(n) => *n,
        Value::Tensor(t) if t.data.len() == 1 => t.data[0],
        _ => {
            let repr = format!("{value:?}");
            return Err(getfield_flow(format!(
                "expected positive integer index, got {repr}"
            )));
        }
    };

    if !number.is_finite() {
        return Err(getfield_flow("index must be a finite number"));
    }
    if number.fract() != 0.0 {
        return Err(getfield_flow("index must be an integer"));
    }
    if number <= 0.0 {
        return Err(getfield_flow("index must be >= 1"));
    }
    if number > usize::MAX as f64 {
        return Err(getfield_flow("index exceeds platform limits"));
    }
    Ok(number as usize)
}

fn parse_field_name(value: Value) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s),
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                Ok(sa.data[0].clone())
            } else {
                Err(getfield_flow(
                    "getfield: field names must be scalar string arrays or character vectors",
                ))
            }
        }
        Value::CharArray(ca) => {
            if ca.rows == 1 {
                Ok(ca.data.iter().collect())
            } else {
                Err(getfield_flow(
                    "getfield: field names must be 1-by-N character vectors",
                ))
            }
        }
        other => Err(getfield_flow(format!(
            "getfield: expected field name, got {other:?}"
        ))),
    }
}

async fn apply_indices(value: Value, selector: &IndexSelector) -> BuiltinResult<Value> {
    if selector.components.is_empty() {
        return Err(getfield_flow(
            "getfield: index cell must contain at least one element",
        ));
    }

    let value = match value {
        Value::GpuTensor(handle) => gather_if_needed_async(&Value::GpuTensor(handle))
            .await
            .map_err(|flow| remap_getfield_flow(flow, Some("getfield: ")))?,
        other => other,
    };

    let resolved = resolve_indices(&value, selector)?;
    let resolved_f64: Vec<f64> = resolved.iter().map(|&idx| idx as f64).collect();

    match &value {
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(logical)
                .map_err(|e| getfield_flow(format!("getfield: {e}")))?;
            let scratch = Value::Tensor(tensor);
            let indexed = perform_indexing(&scratch, &resolved_f64)
                .await
                .map_err(|err| remap_getfield_flow(err, Some("getfield: ")))?;
            match indexed {
                Value::Num(n) => Ok(Value::Bool(n != 0.0)),
                Value::Tensor(t) => {
                    let bits: Vec<u8> = t
                        .data
                        .iter()
                        .map(|&v| if v != 0.0 { 1 } else { 0 })
                        .collect();
                    let logical = LogicalArray::new(bits, t.shape.clone())
                        .map_err(|e| getfield_flow(format!("getfield: {e}")))?;
                    Ok(Value::LogicalArray(logical))
                }
                other => Ok(other),
            }
        }
        Value::CharArray(array) => index_char_array(array, &resolved),
        Value::ComplexTensor(tensor) => index_complex_tensor(tensor, &resolved),
        Value::Tensor(_)
        | Value::StringArray(_)
        | Value::Cell(_)
        | Value::Num(_)
        | Value::Int(_) => perform_indexing(&value, &resolved_f64)
            .await
            .map_err(|err| remap_getfield_flow(err, Some("getfield: "))),
        Value::Bool(_) => {
            if resolved.len() == 1 && resolved[0] == 1 {
                Ok(value)
            } else {
                Err(getfield_flow("Index exceeds the number of array elements."))
            }
        }
        _ => Err(getfield_flow(
            "Struct contents reference from a non-struct array object.",
        )),
    }
}

fn resolve_indices(value: &Value, selector: &IndexSelector) -> BuiltinResult<Vec<usize>> {
    let dims = selector.components.len();
    let mut resolved = Vec::with_capacity(dims);
    for (dim_idx, component) in selector.components.iter().enumerate() {
        let index = match component {
            IndexComponent::Scalar(idx) => *idx,
            IndexComponent::End => dimension_length(value, dims, dim_idx)?,
        };
        resolved.push(index);
    }
    Ok(resolved)
}

fn dimension_length(value: &Value, dims: usize, dim_idx: usize) -> BuiltinResult<usize> {
    match value {
        Value::Tensor(tensor) => tensor_dimension_length(tensor, dims, dim_idx),
        Value::Cell(cell) => cell_dimension_length(cell, dims, dim_idx),
        Value::StringArray(sa) => string_array_dimension_length(sa, dims, dim_idx),
        Value::LogicalArray(logical) => logical_array_dimension_length(logical, dims, dim_idx),
        Value::CharArray(array) => char_array_dimension_length(array, dims, dim_idx),
        Value::ComplexTensor(tensor) => complex_tensor_dimension_length(tensor, dims, dim_idx),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            if dims == 1 {
                Ok(1)
            } else {
                Err(getfield_flow(
                    "getfield: indexing with more than one dimension is not supported for scalars",
                ))
            }
        }
        _ => Err(getfield_flow(
            "Struct contents reference from a non-struct array object.",
        )),
    }
}

fn tensor_dimension_length(tensor: &Tensor, dims: usize, dim_idx: usize) -> BuiltinResult<usize> {
    if dims == 1 {
        let total = tensor.data.len();
        if total == 0 {
            return Err(getfield_flow(
                "Index exceeds the number of array elements (0).",
            ));
        }
        return Ok(total);
    }
    if dims > 2 {
        return Err(getfield_flow(
            "getfield: indexing with more than two indices is not supported yet",
        ));
    }
    let len = if dim_idx == 0 {
        tensor.rows()
    } else {
        tensor.cols()
    };
    if len == 0 {
        return Err(getfield_flow(
            "Index exceeds the number of array elements (0).",
        ));
    }
    Ok(len)
}

fn cell_dimension_length(cell: &CellArray, dims: usize, dim_idx: usize) -> BuiltinResult<usize> {
    if dims == 1 {
        let total = cell.data.len();
        if total == 0 {
            return Err(getfield_flow(
                "Index exceeds the number of array elements (0).",
            ));
        }
        return Ok(total);
    }
    if dims > 2 {
        return Err(getfield_flow(
            "getfield: indexing with more than two indices is not supported yet",
        ));
    }
    let len = if dim_idx == 0 { cell.rows } else { cell.cols };
    if len == 0 {
        return Err(getfield_flow(
            "Index exceeds the number of array elements (0).",
        ));
    }
    Ok(len)
}

fn string_array_dimension_length(
    array: &runmat_builtins::StringArray,
    dims: usize,
    dim_idx: usize,
) -> BuiltinResult<usize> {
    if dims == 1 {
        let total = array.data.len();
        if total == 0 {
            return Err(getfield_flow(
                "Index exceeds the number of array elements (0).",
            ));
        }
        return Ok(total);
    }
    if dims > 2 {
        return Err(getfield_flow(
            "getfield: indexing with more than two indices is not supported yet",
        ));
    }
    let len = if dim_idx == 0 {
        array.rows()
    } else {
        array.cols()
    };
    if len == 0 {
        return Err(getfield_flow(
            "Index exceeds the number of array elements (0).",
        ));
    }
    Ok(len)
}

fn logical_array_dimension_length(
    logical: &LogicalArray,
    dims: usize,
    dim_idx: usize,
) -> BuiltinResult<usize> {
    if dims == 1 {
        let total = logical.data.len();
        if total == 0 {
            return Err(getfield_flow(
                "Index exceeds the number of array elements (0).",
            ));
        }
        return Ok(total);
    }
    if dims > 2 {
        return Err(getfield_flow(
            "getfield: indexing with more than two indices is not supported yet",
        ));
    }
    let len = if dim_idx == 0 {
        logical.shape.first().copied().unwrap_or(logical.data.len())
    } else {
        logical.shape.get(1).copied().unwrap_or(1)
    };
    if len == 0 {
        return Err(getfield_flow(
            "Index exceeds the number of array elements (0).",
        ));
    }
    Ok(len)
}

fn char_array_dimension_length(
    array: &CharArray,
    dims: usize,
    dim_idx: usize,
) -> BuiltinResult<usize> {
    if dims == 1 {
        let total = array.rows * array.cols;
        if total == 0 {
            return Err(getfield_flow(
                "Index exceeds the number of array elements (0).",
            ));
        }
        return Ok(total);
    }
    if dims > 2 {
        return Err(getfield_flow(
            "getfield: indexing with more than two indices is not supported yet",
        ));
    }
    let len = if dim_idx == 0 { array.rows } else { array.cols };
    if len == 0 {
        return Err(getfield_flow(
            "Index exceeds the number of array elements (0).",
        ));
    }
    Ok(len)
}

fn complex_tensor_dimension_length(
    tensor: &ComplexTensor,
    dims: usize,
    dim_idx: usize,
) -> BuiltinResult<usize> {
    if dims == 1 {
        let total = tensor.data.len();
        if total == 0 {
            return Err(getfield_flow(
                "Index exceeds the number of array elements (0).",
            ));
        }
        return Ok(total);
    }
    if dims > 2 {
        return Err(getfield_flow(
            "getfield: indexing with more than two indices is not supported yet",
        ));
    }
    let len = if dim_idx == 0 {
        tensor.rows
    } else {
        tensor.cols
    };
    if len == 0 {
        return Err(getfield_flow(
            "Index exceeds the number of array elements (0).",
        ));
    }
    Ok(len)
}

fn index_char_array(array: &CharArray, indices: &[usize]) -> BuiltinResult<Value> {
    if indices.is_empty() {
        return Err(getfield_flow(
            "getfield: at least one index is required for char arrays",
        ));
    }
    if indices.len() == 1 {
        let total = array.rows * array.cols;
        let idx = indices[0];
        if idx == 0 || idx > total {
            return Err(getfield_flow("Index exceeds the number of array elements."));
        }
        let linear = idx - 1;
        let rows = array.rows.max(1);
        let col = linear / rows;
        let row = linear % rows;
        let pos = row * array.cols + col;
        let ch = array
            .data
            .get(pos)
            .copied()
            .ok_or_else(|| getfield_flow("Index exceeds the number of array elements."))?;
        let out =
            CharArray::new(vec![ch], 1, 1).map_err(|e| getfield_flow(format!("getfield: {e}")))?;
        return Ok(Value::CharArray(out));
    }
    if indices.len() == 2 {
        let row = indices[0];
        let col = indices[1];
        if row == 0 || row > array.rows || col == 0 || col > array.cols {
            return Err(getfield_flow("Index exceeds the number of array elements."));
        }
        let pos = (row - 1) * array.cols + (col - 1);
        let ch = array
            .data
            .get(pos)
            .copied()
            .ok_or_else(|| getfield_flow("Index exceeds the number of array elements."))?;
        let out =
            CharArray::new(vec![ch], 1, 1).map_err(|e| getfield_flow(format!("getfield: {e}")))?;
        return Ok(Value::CharArray(out));
    }
    Err(getfield_flow(
        "getfield: indexing with more than two indices is not supported for char arrays",
    ))
}

fn index_complex_tensor(tensor: &ComplexTensor, indices: &[usize]) -> BuiltinResult<Value> {
    if indices.is_empty() {
        return Err(getfield_flow(
            "getfield: at least one index is required for complex tensors",
        ));
    }
    if indices.len() == 1 {
        let total = tensor.data.len();
        let idx = indices[0];
        if idx == 0 || idx > total {
            return Err(getfield_flow("Index exceeds the number of array elements."));
        }
        let (re, im) = tensor.data[idx - 1];
        return Ok(Value::Complex(re, im));
    }
    if indices.len() == 2 {
        let row = indices[0];
        let col = indices[1];
        if row == 0 || row > tensor.rows || col == 0 || col > tensor.cols {
            return Err(getfield_flow("Index exceeds the number of array elements."));
        }
        let pos = (row - 1) + (col - 1) * tensor.rows;
        let (re, im) = tensor
            .data
            .get(pos)
            .copied()
            .ok_or_else(|| getfield_flow("Index exceeds the number of array elements."))?;
        return Ok(Value::Complex(re, im));
    }
    Err(getfield_flow(
        "getfield: indexing with more than two indices is not supported for complex tensors",
    ))
}

#[async_recursion::async_recursion(?Send)]
async fn get_field_value(value: Value, name: &str) -> BuiltinResult<Value> {
    match value {
        Value::Struct(st) => get_struct_field(&st, name),
        Value::Object(obj) => get_object_field(&obj, name).await,
        Value::HandleObject(handle) => get_handle_field(&handle, name).await,
        Value::Listener(listener) => get_listener_field(&listener, name),
        Value::MException(ex) => get_exception_field(&ex, name),
        Value::Cell(cell) if is_struct_array(&cell) => {
            let Some(first) = struct_array_first(&cell)? else {
                return Err(getfield_flow(
                    "Struct contents reference from an empty struct array.",
                ));
            };
            get_field_value(first, name).await
        }
        _ => Err(getfield_flow(
            "Struct contents reference from a non-struct array object.",
        )),
    }
}

fn get_struct_field(struct_value: &StructValue, name: &str) -> BuiltinResult<Value> {
    struct_value
        .fields
        .get(name)
        .cloned()
        .ok_or_else(|| getfield_flow(format!("Reference to non-existent field '{}'.", name)))
}

async fn get_object_field(obj: &ObjectInstance, name: &str) -> BuiltinResult<Value> {
    if let Some((prop, _owner)) = runmat_builtins::lookup_property(&obj.class_name, name) {
        if prop.is_static {
            return Err(getfield_flow(format!(
                "You cannot access the static property '{}' through an instance of class '{}'.",
                name, obj.class_name
            )));
        }
        if prop.get_access == Access::Private {
            return Err(getfield_flow(format!(
                "You cannot get the '{}' property of '{}' class.",
                name, obj.class_name
            )));
        }
        if prop.is_dependent {
            let getter = format!("get.{name}");
            match call_builtin_async(&getter, &[Value::Object(obj.clone())]).await {
                Ok(value) => return Ok(value),
                Err(err) => {
                    if !is_undefined_function(&err) {
                        return Err(remap_getfield_flow(err, None));
                    }
                }
            }
            if let Some(val) = obj.properties.get(&format!("{name}_backing")) {
                return Ok(val.clone());
            }
        }
    }

    if let Some(value) = obj.properties.get(name) {
        return Ok(value.clone());
    }

    if let Some((prop, _owner)) = runmat_builtins::lookup_property(&obj.class_name, name) {
        if prop.get_access == Access::Private {
            return Err(getfield_flow(format!(
                "You cannot get the '{}' property of '{}' class.",
                name, obj.class_name
            )));
        }
        return Err(getfield_flow(format!(
            "No public property '{}' for class '{}'.",
            name, obj.class_name
        )));
    }

    Err(getfield_flow(format!(
        "Undefined property '{}' for class {}",
        name, obj.class_name
    )))
}

#[async_recursion::async_recursion(?Send)]
async fn get_handle_field(handle: &HandleRef, name: &str) -> BuiltinResult<Value> {
    if !handle.valid {
        return Err(getfield_flow(format!(
            "Invalid or deleted handle object '{}'.",
            handle.class_name
        )));
    }
    let target = unsafe { &*handle.target.as_raw() }.clone();
    get_field_value(target, name).await
}

fn get_listener_field(listener: &Listener, name: &str) -> BuiltinResult<Value> {
    match name {
        "Enabled" | "enabled" => Ok(Value::Bool(listener.enabled)),
        "Valid" | "valid" => Ok(Value::Bool(listener.valid)),
        "EventName" | "event_name" => Ok(Value::String(listener.event_name.clone())),
        "Callback" | "callback" => {
            let value = unsafe { &*listener.callback.as_raw() }.clone();
            Ok(value)
        }
        "Target" | "target" => {
            let value = unsafe { &*listener.target.as_raw() }.clone();
            Ok(value)
        }
        "Id" | "id" => Ok(Value::Int(runmat_builtins::IntValue::U64(listener.id))),
        other => Err(getfield_flow(format!(
            "getfield: unknown field '{}' on listener object",
            other
        ))),
    }
}

fn get_exception_field(exception: &MException, name: &str) -> BuiltinResult<Value> {
    match name {
        "message" => Ok(Value::String(exception.message.clone())),
        "identifier" => Ok(Value::String(exception.identifier.clone())),
        "stack" => exception_stack_to_value(&exception.stack),
        other => Err(getfield_flow(format!(
            "Reference to non-existent field '{}'.",
            other
        ))),
    }
}

fn exception_stack_to_value(stack: &[String]) -> BuiltinResult<Value> {
    if stack.is_empty() {
        return make_cell_with_shape(Vec::new(), vec![0, 1])
            .map_err(|e| getfield_flow(format!("getfield: {e}")));
    }
    let mut values = Vec::with_capacity(stack.len());
    for frame in stack {
        values.push(Value::String(frame.clone()));
    }
    make_cell_with_shape(values, vec![stack.len(), 1])
        .map_err(|e| getfield_flow(format!("getfield: {e}")))
}

fn is_struct_array(cell: &CellArray) -> bool {
    cell.data
        .iter()
        .all(|handle| matches!(unsafe { &*handle.as_raw() }, Value::Struct(_)))
}

fn struct_array_first(cell: &CellArray) -> BuiltinResult<Option<Value>> {
    if cell.data.is_empty() {
        return Ok(None);
    }
    let handle = cell.data.first().unwrap();
    let value = unsafe { &*handle.as_raw() };
    match value {
        Value::Struct(_) => Ok(Some(value.clone())),
        _ => Err(getfield_flow(
            "getfield: expected struct array elements to be structs",
        )),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_builtins::{
        Access, CellArray, CharArray, ClassDef, ComplexTensor, HandleRef, IntValue, Listener,
        MException, ObjectInstance, PropertyDef, StructValue,
    };
    use runmat_gc_api::GcPtr;

    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider as wgpu_backend;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::HostTensorView;

    use crate::builtins::common::test_support;

    fn error_message(err: crate::RuntimeError) -> String {
        err.message().to_string()
    }

    fn run_getfield(base: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(getfield_builtin(base, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getfield_scalar_struct() {
        let mut st = StructValue::new();
        st.fields.insert("answer".to_string(), Value::Num(42.0));
        let value = run_getfield(Value::Struct(st), vec![Value::from("answer")]).expect("getfield");
        assert_eq!(value, Value::Num(42.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getfield_nested_structs() {
        let mut inner = StructValue::new();
        inner.fields.insert("depth".to_string(), Value::Num(3.0));
        let mut outer = StructValue::new();
        outer
            .fields
            .insert("inner".to_string(), Value::Struct(inner));
        let result = run_getfield(
            Value::Struct(outer),
            vec![Value::from("inner"), Value::from("depth")],
        )
        .expect("nested getfield");
        assert_eq!(result, Value::Num(3.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getfield_struct_array_element() {
        let mut first = StructValue::new();
        first.fields.insert("name".to_string(), Value::from("Ada"));
        let mut second = StructValue::new();
        second
            .fields
            .insert("name".to_string(), Value::from("Grace"));
        let array = CellArray::new_with_shape(
            vec![Value::Struct(first), Value::Struct(second)],
            vec![1, 2],
        )
        .unwrap();
        let index =
            CellArray::new_with_shape(vec![Value::Int(IntValue::I32(2))], vec![1, 1]).unwrap();
        let result = run_getfield(
            Value::Cell(array),
            vec![Value::Cell(index), Value::from("name")],
        )
        .expect("struct array element");
        assert_eq!(result, Value::from("Grace"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getfield_object_property() {
        let mut obj = ObjectInstance::new("TestClass".to_string());
        obj.properties.insert("value".to_string(), Value::Num(7.0));
        let result = run_getfield(Value::Object(obj), vec![Value::from("value")]).expect("object");
        assert_eq!(result, Value::Num(7.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getfield_missing_field_errors() {
        let st = StructValue::new();
        let err = error_message(
            run_getfield(Value::Struct(st), vec![Value::from("missing")]).unwrap_err(),
        );
        assert!(err.contains("Reference to non-existent field 'missing'"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getfield_exception_fields() {
        let ex = MException::new("MATLAB:Test".to_string(), "failure".to_string());
        let msg = run_getfield(Value::MException(ex.clone()), vec![Value::from("message")])
            .expect("message");
        assert_eq!(msg, Value::String("failure".to_string()));
        let ident = run_getfield(Value::MException(ex), vec![Value::from("identifier")])
            .expect("identifier");
        assert_eq!(ident, Value::String("MATLAB:Test".to_string()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getfield_exception_stack_cell() {
        let mut ex = MException::new("MATLAB:Test".to_string(), "failure".to_string());
        ex.stack.push("demo.m:5".to_string());
        ex.stack.push("main.m:1".to_string());
        let stack = run_getfield(Value::MException(ex), vec![Value::from("stack")]).expect("stack");
        let Value::Cell(cell) = stack else {
            panic!("expected cell array");
        };
        assert_eq!(cell.rows, 2);
        assert_eq!(cell.cols, 1);
        let first = unsafe { &*cell.data[0].as_raw() }.clone();
        assert_eq!(first, Value::String("demo.m:5".to_string()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn indexing_missing_field_name_fails() {
        let mut outer = StructValue::new();
        outer.fields.insert("inner".to_string(), Value::Num(1.0));
        let index =
            CellArray::new_with_shape(vec![Value::Int(IntValue::I32(1))], vec![1, 1]).unwrap();
        let err = error_message(
            run_getfield(Value::Struct(outer), vec![Value::Cell(index)]).unwrap_err(),
        );
        assert!(err.contains("expected field name"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getfield_supports_end_index() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let mut st = StructValue::new();
        st.fields
            .insert("values".to_string(), Value::Tensor(tensor));
        let idx_cell =
            CellArray::new(vec![Value::CharArray(CharArray::new_row("end"))], 1, 1).unwrap();
        let result = run_getfield(
            Value::Struct(st),
            vec![Value::from("values"), Value::Cell(idx_cell)],
        )
        .expect("end index");
        assert_eq!(result, Value::Num(3.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getfield_struct_array_defaults_to_first() {
        let mut first = StructValue::new();
        first.fields.insert("name".to_string(), Value::from("Ada"));
        let mut second = StructValue::new();
        second
            .fields
            .insert("name".to_string(), Value::from("Grace"));
        let array = CellArray::new_with_shape(
            vec![Value::Struct(first), Value::Struct(second)],
            vec![1, 2],
        )
        .unwrap();
        let result =
            run_getfield(Value::Cell(array), vec![Value::from("name")]).expect("default index");
        assert_eq!(result, Value::from("Ada"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getfield_char_array_single_element() {
        let chars = CharArray::new_row("Ada");
        let mut st = StructValue::new();
        st.fields
            .insert("name".to_string(), Value::CharArray(chars));
        let index =
            CellArray::new_with_shape(vec![Value::Int(IntValue::I32(2))], vec![1, 1]).unwrap();
        let result = run_getfield(
            Value::Struct(st),
            vec![Value::from("name"), Value::Cell(index)],
        )
        .expect("char indexing");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 1);
                assert_eq!(ca.data, vec!['d']);
            }
            other => panic!("expected 1x1 CharArray, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getfield_complex_tensor_index() {
        let tensor =
            ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![2, 1]).expect("complex tensor");
        let mut st = StructValue::new();
        st.fields
            .insert("vals".to_string(), Value::ComplexTensor(tensor));
        let index =
            CellArray::new_with_shape(vec![Value::Int(IntValue::I32(2))], vec![1, 1]).unwrap();
        let result = run_getfield(
            Value::Struct(st),
            vec![Value::from("vals"), Value::Cell(index)],
        )
        .expect("complex index");
        assert_eq!(result, Value::Complex(3.0, 4.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getfield_dependent_property_invokes_getter() {
        let class_name = "runmat.unittest.GetfieldDependent";
        let mut def = ClassDef {
            name: class_name.to_string(),
            parent: None,
            properties: std::collections::HashMap::new(),
            methods: std::collections::HashMap::new(),
        };
        def.properties.insert(
            "p".to_string(),
            PropertyDef {
                name: "p".to_string(),
                is_static: false,
                is_dependent: true,
                get_access: Access::Public,
                set_access: Access::Public,
                default_value: None,
            },
        );
        runmat_builtins::register_class(def);

        let mut obj = ObjectInstance::new(class_name.to_string());
        obj.properties
            .insert("p_backing".to_string(), Value::Num(42.0));

        let result = run_getfield(Value::Object(obj), vec![Value::from("p")]).expect("dependent");
        assert_eq!(result, Value::Num(42.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getfield_invalid_handle_errors() {
        let target = unsafe { GcPtr::from_raw(Box::into_raw(Box::new(Value::Num(1.0)))) };
        let handle = HandleRef {
            class_name: "Demo".to_string(),
            target,
            valid: false,
        };
        let err = error_message(
            run_getfield(Value::HandleObject(handle), vec![Value::from("x")]).unwrap_err(),
        );
        assert!(err.contains("Invalid or deleted handle object 'Demo'"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getfield_listener_fields_resolved() {
        let target = unsafe { GcPtr::from_raw(Box::into_raw(Box::new(Value::Num(7.0)))) };
        let callback = unsafe {
            GcPtr::from_raw(Box::into_raw(Box::new(Value::FunctionHandle(
                "cb".to_string(),
            ))))
        };
        let listener = Listener {
            id: 9,
            target,
            event_name: "tick".to_string(),
            callback,
            enabled: true,
            valid: true,
        };
        let enabled = run_getfield(
            Value::Listener(listener.clone()),
            vec![Value::from("Enabled")],
        )
        .expect("enabled");
        assert_eq!(enabled, Value::Bool(true));
        let event_name = run_getfield(
            Value::Listener(listener.clone()),
            vec![Value::from("EventName")],
        )
        .expect("event name");
        assert_eq!(event_name, Value::String("tick".to_string()));
        let callback = run_getfield(Value::Listener(listener), vec![Value::from("Callback")])
            .expect("callback");
        assert!(matches!(callback, Value::FunctionHandle(_)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn getfield_gpu_tensor_indexing() {
        let _ = wgpu_backend::register_wgpu_provider(wgpu_backend::WgpuProviderOptions::default());
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");

        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");

        let mut st = StructValue::new();
        st.fields
            .insert("values".to_string(), Value::GpuTensor(handle.clone()));

        let direct = run_getfield(Value::Struct(st.clone()), vec![Value::from("values")])
            .expect("direct gpu field");
        match direct {
            Value::GpuTensor(out) => assert_eq!(out.buffer_id, handle.buffer_id),
            other => panic!("expected gpu tensor, got {other:?}"),
        }

        let idx_cell =
            CellArray::new(vec![Value::CharArray(CharArray::new_row("end"))], 1, 1).unwrap();
        let indexed = run_getfield(
            Value::Struct(st),
            vec![Value::from("values"), Value::Cell(idx_cell)],
        )
        .expect("gpu indexed field");
        match indexed {
            Value::Num(v) => assert_eq!(v, 3.0),
            other => panic!("expected numeric scalar, got {other:?}"),
        }
    }
}
