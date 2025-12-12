//! MATLAB-compatible `setfield` builtin with struct array and object support.
//!
//! Mirrors MATLAB's `setfield` semantics, including nested field creation, struct
//! array indexing via cell arguments, and property assignment on MATLAB-style
//! objects. The builtin performs all updates on host data; GPU-resident values are
//! gathered automatically before mutation. Updated tensors remain on the host.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::call_builtin;
use crate::gather_if_needed;
use runmat_builtins::{
    Access, CellArray, CharArray, ComplexTensor, HandleRef, LogicalArray, ObjectInstance,
    StructValue, Tensor, Value,
};
use runmat_gc_api::GcPtr;
use runmat_macros::runtime_builtin;
use std::convert::TryFrom;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "setfield",
        builtin_path = "crate::builtins::structs::core::setfield"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "setfield"
category: "structs/core"
keywords: ["setfield", "struct", "assignment", "struct array", "object property"]
summary: "Assign into struct fields, struct arrays, or MATLAB-style object properties."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Assignments run on the host. GPU tensors or handles embedded in structs are gathered to host memory before mutation."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::structs::core::setfield::tests"
  integration: "runmat_ignition::tests::functions::member_get_set_and_method_call_skeleton"
---

# What does the `setfield` function do in MATLAB / RunMat?
`S = setfield(S, field, value)` returns a copy of the struct (or object) with `field`
assigned to `value`. Additional field names and index cells let you update nested
structures, struct arrays, and array elements contained within fields.

## How does the `setfield` function behave in MATLAB / RunMat?
- Field names must be character vectors or string scalars. Provide as many field
  names as needed; each additional name drills deeper into nested structs, so
  `setfield(S,"outer","inner",value)` mirrors `S.outer.inner = value`.
- Missing struct fields are created automatically. If intermediary structs do not
  exist, RunMat allocates them so that the assignment completes successfully.
- Struct arrays require a leading cell array of one-based indices, e.g.
  `setfield(S,{2},"field",value)` or `setfield(S,{1,3},"field",value)`, and accept
  the keyword `end`.
- You can index into a field's contents before traversing deeper by placing a cell
  array of indices immediately after the field name:
  `setfield(S,"values",{1,2},"leaf",x)` matches `S.values{1,2}.leaf = x`.
- MATLAB-style objects honour property metadata: private setters raise access
  errors, static properties cannot be written through instances, and dependent
  properties forward to `set.<name>` methods when available.
- The function returns the updated struct or object. For value types the result is a
  new copy; handle objects still point at the same instance, and the handle is
  returned for chaining.

## `setfield` Function GPU Execution Behaviour
`setfield` executes entirely on the host. When fields contain GPU-resident tensors,
RunMat gathers those tensors to host memory before mutating them and stores the
resulting host tensor back into the struct or object. No GPU kernels are launched
for these assignments.

## Examples of using the `setfield` function in MATLAB / RunMat

### Assigning a new field in a scalar struct
```matlab
s = struct();
s = setfield(s, "answer", 42);
disp(s.answer);
```

Expected output:
```matlab
    42
```

### Creating nested structs automatically
```matlab
cfg = struct();
cfg = setfield(cfg, "solver", "name", "cg");
cfg = setfield(cfg, "solver", "tolerance", 1e-6);
disp(cfg.solver.tolerance);
```

Expected output:
```matlab
  1.0000e-06
```

### Updating an element of a struct array
```matlab
people = struct("name", {"Ada", "Grace"}, "id", {101, 102});
people = setfield(people, {2}, "id", 999);
disp(people(2).id);
```

Expected output:
```matlab
   999
```

### Assigning through a field that contains a cell array
```matlab
data = struct("samples", {{struct("value", 1), struct("value", 2)}} );
data = setfield(data, "samples", {2}, "value", 10);
disp(data.samples{2}.value);
```

Expected output:
```matlab
    10
```

### Setting an object property that honours access attributes
Save the following class definition as `Point.m`:
```matlab
classdef Point
    properties
        x double = 0;
    end
end
```

Then update the property from the command window:
```matlab
p = Point;
p = setfield(p, "x", 3);
disp(p.x);
```

Expected output:
```matlab
    3
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You do not have to move data explicitly when assigning into structs. If a field
contains a GPU tensor, `setfield` gathers it to host memory so the mutation can be
performed safely. Subsequent operations decide whether to migrate it back to the GPU.

## FAQ

### Does `setfield` modify the input in-place?
No. Like MATLAB, it returns a new struct (or object) with the requested update. In
Rust this entails cloning the source value and mutating the clone.

### Can I create nested structs in a single call?
Yes. Missing intermediate structs are created automatically when you provide multiple
field names, e.g. `setfield(S,"outer","inner",value)` builds `outer` when needed.

### How do I update a specific element of a struct array?
Supply an index cell before the first field name: `setfield(S,{row,col},"field",value)`
is the same as `S(row,col).field = value`.

### Does `setfield` work with handle objects?
Yes. Valid handle objects forward the assignment to the underlying instance. Deleted
or invalid handles raise the standard MATLAB-style error.

### Can I index into field contents before continuing?
Yes. Place a cell array of indices immediately after the field name. Each set of
indices uses MATLAB's one-based semantics and supports the keyword `end`.

### Why are GPU tensors gathered to the host?
Assignments require host-side mutation. Providers can re-upload the updated tensor on
subsequent GPU-aware operations; `setfield` itself never launches kernels.

## See Also
[getfield](./getfield), [fieldnames](./fieldnames), [struct](./struct), [gpuArray](./gpuarray), [gather](./gather)
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::structs::core::setfield")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "setfield",
    op_kind: GpuOpKind::Custom("setfield"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host-only metadata mutation; GPU tensors are gathered before assignment.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::structs::core::setfield")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "setfield",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Assignments terminate fusion and gather device data back to the host.",
};

#[runtime_builtin(
    name = "setfield",
    category = "structs/core",
    summary = "Assign into struct fields, struct arrays, or MATLAB-style object properties.",
    keywords = "setfield,struct,assignment,object property",
    builtin_path = "crate::builtins::structs::core::setfield"
)]
fn setfield_builtin(base: Value, rest: Vec<Value>) -> Result<Value, String> {
    let parsed = parse_arguments(rest)?;
    let ParsedArguments {
        leading_index,
        steps,
        value,
    } = parsed;
    assign_value(base, leading_index, steps, value)
}

struct ParsedArguments {
    leading_index: Option<IndexSelector>,
    steps: Vec<FieldStep>,
    value: Value,
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

fn parse_arguments(mut rest: Vec<Value>) -> Result<ParsedArguments, String> {
    if rest.len() < 2 {
        return Err("setfield: expected at least one field name and a value".to_string());
    }

    let value = rest
        .pop()
        .expect("rest contains at least two elements after early return");

    let mut parsed = ParsedArguments {
        leading_index: None,
        steps: Vec::new(),
        value,
    };

    if let Some(first) = rest.first() {
        if is_index_selector(first) {
            let selector = rest.remove(0);
            parsed.leading_index = Some(parse_index_selector(selector)?);
        }
    }

    if rest.is_empty() {
        return Err("setfield: expected field name arguments".to_string());
    }

    let mut iter = rest.into_iter().peekable();
    while let Some(arg) = iter.next() {
        let name = parse_field_name(arg)?;
        let mut step = FieldStep { name, index: None };
        if let Some(next) = iter.peek() {
            if is_index_selector(next) {
                let selector = iter.next().unwrap();
                step.index = Some(parse_index_selector(selector)?);
            }
        }
        parsed.steps.push(step);
    }

    if parsed.steps.is_empty() {
        return Err("setfield: expected field name arguments".to_string());
    }

    Ok(parsed)
}

fn assign_value(
    base: Value,
    leading_index: Option<IndexSelector>,
    steps: Vec<FieldStep>,
    rhs: Value,
) -> Result<Value, String> {
    if steps.is_empty() {
        return Err("setfield: expected field name arguments".to_string());
    }
    if let Some(selector) = leading_index {
        assign_with_leading_index(base, &selector, &steps, rhs)
    } else {
        assign_without_leading_index(base, &steps, rhs)
    }
}

fn assign_with_leading_index(
    base: Value,
    selector: &IndexSelector,
    steps: &[FieldStep],
    rhs: Value,
) -> Result<Value, String> {
    match base {
        Value::Cell(cell) => assign_into_struct_array(cell, selector, steps, rhs),
        other => Err(format!(
            "setfield: leading indices require a struct array, got {other:?}"
        )),
    }
}

fn assign_without_leading_index(
    base: Value,
    steps: &[FieldStep],
    rhs: Value,
) -> Result<Value, String> {
    match base {
        Value::Struct(struct_value) => assign_into_struct(struct_value, steps, rhs),
        Value::Object(object) => assign_into_object(object, steps, rhs),
        Value::Cell(cell) if is_struct_array(&cell) => {
            if cell.data.is_empty() {
                Err("setfield: struct array is empty; supply indices in a cell array".to_string())
            } else {
                let selector = IndexSelector {
                    components: vec![IndexComponent::Scalar(1)],
                };
                assign_into_struct_array(cell, &selector, steps, rhs)
            }
        }
        Value::HandleObject(handle) => assign_into_handle(handle, steps, rhs),
        Value::Listener(_) => {
            Err("setfield: listeners do not support direct field assignment".to_string())
        }
        other => Err(format!(
            "setfield unsupported on this value for field '{}': {other:?}",
            steps.first().map(|s| s.name.as_str()).unwrap_or_default()
        )),
    }
}

fn assign_into_struct_array(
    mut cell: CellArray,
    selector: &IndexSelector,
    steps: &[FieldStep],
    rhs: Value,
) -> Result<Value, String> {
    if selector.components.is_empty() {
        return Err("setfield: index cell must contain at least one element".to_string());
    }

    let resolved = resolve_indices(&Value::Cell(cell.clone()), selector)?;

    let position = match resolved.len() {
        1 => {
            let idx = resolved[0];
            if idx == 0 || idx > cell.data.len() {
                return Err("Index exceeds the number of array elements.".to_string());
            }
            idx - 1
        }
        2 => {
            let row = resolved[0];
            let col = resolved[1];
            if row == 0 || row > cell.rows || col == 0 || col > cell.cols {
                return Err("Index exceeds the number of array elements.".to_string());
            }
            (row - 1) * cell.cols + (col - 1)
        }
        _ => {
            return Err(
                "setfield: indexing with more than two indices is not supported yet".to_string(),
            );
        }
    };

    let handle = cell
        .data
        .get(position)
        .ok_or_else(|| "Index exceeds the number of array elements.".to_string())?
        .clone();

    let current = unsafe { &*handle.as_raw() }.clone();
    let updated = assign_into_value(current, steps, rhs)?;
    cell.data[position] = allocate_cell_handle(updated)?;
    Ok(Value::Cell(cell))
}

fn assign_into_value(value: Value, steps: &[FieldStep], rhs: Value) -> Result<Value, String> {
    if steps.is_empty() {
        return Ok(rhs);
    }
    match value {
        Value::Struct(struct_value) => assign_into_struct(struct_value, steps, rhs),
        Value::Object(object) => assign_into_object(object, steps, rhs),
        Value::Cell(cell) => assign_into_cell(cell, steps, rhs),
        Value::HandleObject(handle) => assign_into_handle(handle, steps, rhs),
        Value::Listener(_) => {
            Err("setfield: listeners do not support nested field assignment".to_string())
        }
        other => Err(format!(
            "Struct contents assignment to a {other:?} object is not supported."
        )),
    }
}

fn assign_into_struct(
    mut struct_value: StructValue,
    steps: &[FieldStep],
    rhs: Value,
) -> Result<Value, String> {
    let (first, rest) = steps
        .split_first()
        .expect("steps is non-empty when assign_into_struct is called");

    if rest.is_empty() {
        if let Some(selector) = &first.index {
            let current = struct_value
                .fields
                .get(&first.name)
                .cloned()
                .ok_or_else(|| format!("Reference to non-existent field '{}'.", first.name))?;
            let updated = assign_with_selector(current, selector, &[], rhs)?;
            struct_value.fields.insert(first.name.clone(), updated);
        } else {
            struct_value.fields.insert(first.name.clone(), rhs);
        }
        return Ok(Value::Struct(struct_value));
    }

    if let Some(selector) = &first.index {
        let current = struct_value
            .fields
            .get(&first.name)
            .cloned()
            .ok_or_else(|| format!("Reference to non-existent field '{}'.", first.name))?;
        let updated = assign_with_selector(current, selector, rest, rhs)?;
        struct_value.fields.insert(first.name.clone(), updated);
        return Ok(Value::Struct(struct_value));
    }

    let current = struct_value
        .fields
        .get(&first.name)
        .cloned()
        .unwrap_or_else(|| Value::Struct(StructValue::new()));
    let updated = assign_into_value(current, rest, rhs)?;
    struct_value.fields.insert(first.name.clone(), updated);
    Ok(Value::Struct(struct_value))
}

fn assign_into_object(
    mut object: ObjectInstance,
    steps: &[FieldStep],
    rhs: Value,
) -> Result<Value, String> {
    let (first, rest) = steps
        .split_first()
        .expect("steps is non-empty when assign_into_object is called");

    if first.index.is_some() {
        return Err(
            "setfield: indexing into object properties is not currently supported".to_string(),
        );
    }

    if rest.is_empty() {
        write_object_property(&mut object, &first.name, rhs)?;
        return Ok(Value::Object(object));
    }

    let current = read_object_property(&object, &first.name)?;
    let updated = assign_into_value(current, rest, rhs)?;
    write_object_property(&mut object, &first.name, updated)?;
    Ok(Value::Object(object))
}

fn assign_into_cell(cell: CellArray, steps: &[FieldStep], rhs: Value) -> Result<Value, String> {
    let (first, rest) = steps
        .split_first()
        .expect("steps is non-empty when assign_into_cell is called");

    let selector = first.index.as_ref().ok_or_else(|| {
        "setfield: cell array assignments require indices in a cell array".to_string()
    })?;
    if rest.is_empty() {
        assign_with_selector(Value::Cell(cell), selector, &[], rhs)
    } else {
        assign_with_selector(Value::Cell(cell), selector, rest, rhs)
    }
}

fn assign_with_selector(
    value: Value,
    selector: &IndexSelector,
    rest: &[FieldStep],
    rhs: Value,
) -> Result<Value, String> {
    let host_value = gather_if_needed(&value).map_err(|e| format!("setfield: {e}"))?;
    match host_value {
        Value::Cell(mut cell) => {
            let resolved = resolve_indices(&Value::Cell(cell.clone()), selector)?;
            let position = match resolved.len() {
                1 => {
                    let idx = resolved[0];
                    if idx == 0 || idx > cell.data.len() {
                        return Err("Index exceeds the number of array elements.".to_string());
                    }
                    idx - 1
                }
                2 => {
                    let row = resolved[0];
                    let col = resolved[1];
                    if row == 0 || row > cell.rows || col == 0 || col > cell.cols {
                        return Err("Index exceeds the number of array elements.".to_string());
                    }
                    (row - 1) * cell.cols + (col - 1)
                }
                _ => {
                    return Err(
                        "setfield: indexing with more than two indices is not supported yet"
                            .to_string(),
                    );
                }
            };

            let handle = cell
                .data
                .get(position)
                .ok_or_else(|| "Index exceeds the number of array elements.".to_string())?
                .clone();
            let existing = unsafe { &*handle.as_raw() }.clone();
            let new_value = if rest.is_empty() {
                rhs
            } else {
                assign_into_value(existing, rest, rhs)?
            };
            cell.data[position] = allocate_cell_handle(new_value)?;
            Ok(Value::Cell(cell))
        }
        Value::Tensor(mut tensor) => {
            if !rest.is_empty() {
                return Err(
                    "setfield: cannot traverse deeper fields after indexing into a numeric tensor"
                        .to_string(),
                );
            }
            assign_tensor_element(&mut tensor, selector, rhs)?;
            Ok(Value::Tensor(tensor))
        }
        Value::LogicalArray(mut logical) => {
            if !rest.is_empty() {
                return Err(
                    "setfield: cannot traverse deeper fields after indexing into a logical array"
                        .to_string(),
                );
            }
            assign_logical_element(&mut logical, selector, rhs)?;
            Ok(Value::LogicalArray(logical))
        }
        Value::StringArray(mut sa) => {
            if !rest.is_empty() {
                return Err(
                    "setfield: cannot traverse deeper fields after indexing into a string array"
                        .to_string(),
                );
            }
            assign_string_array_element(&mut sa, selector, rhs)?;
            Ok(Value::StringArray(sa))
        }
        Value::CharArray(mut ca) => {
            if !rest.is_empty() {
                return Err(
                    "setfield: cannot traverse deeper fields after indexing into a char array"
                        .to_string(),
                );
            }
            assign_char_array_element(&mut ca, selector, rhs)?;
            Ok(Value::CharArray(ca))
        }
        Value::ComplexTensor(mut tensor) => {
            if !rest.is_empty() {
                return Err(
                    "setfield: cannot traverse deeper fields after indexing into a complex tensor"
                        .to_string(),
                );
            }
            assign_complex_tensor_element(&mut tensor, selector, rhs)?;
            Ok(Value::ComplexTensor(tensor))
        }
        other => Err(format!(
            "Struct contents assignment to a {other:?} object is not supported."
        )),
    }
}

fn assign_tensor_element(
    tensor: &mut Tensor,
    selector: &IndexSelector,
    rhs: Value,
) -> Result<(), String> {
    let resolved = resolve_indices(&Value::Tensor(tensor.clone()), selector)?;
    let value = value_to_scalar(rhs)?;
    match resolved.len() {
        1 => {
            let idx = resolved[0];
            if idx == 0 || idx > tensor.data.len() {
                return Err("Index exceeds the number of array elements.".to_string());
            }
            tensor.data[idx - 1] = value;
            Ok(())
        }
        2 => {
            let row = resolved[0];
            let col = resolved[1];
            if row == 0 || row > tensor.rows() || col == 0 || col > tensor.cols() {
                return Err("Index exceeds the number of array elements.".to_string());
            }
            let pos = (row - 1) + (col - 1) * tensor.rows();
            tensor
                .data
                .get_mut(pos)
                .map(|slot| *slot = value)
                .ok_or_else(|| "Index exceeds the number of array elements.".to_string())
        }
        _ => Err("setfield: indexing with more than two indices is not supported yet".to_string()),
    }
}

fn assign_logical_element(
    logical: &mut LogicalArray,
    selector: &IndexSelector,
    rhs: Value,
) -> Result<(), String> {
    let resolved = resolve_indices(&Value::LogicalArray(logical.clone()), selector)?;
    let value = value_to_bool(rhs)?;
    match resolved.len() {
        1 => {
            let idx = resolved[0];
            if idx == 0 || idx > logical.data.len() {
                return Err("Index exceeds the number of array elements.".to_string());
            }
            logical.data[idx - 1] = if value { 1 } else { 0 };
            Ok(())
        }
        2 => {
            if logical.shape.len() < 2 {
                return Err("Index exceeds the number of array elements.".to_string());
            }
            let row = resolved[0];
            let col = resolved[1];
            let rows = logical.shape[0];
            let cols = logical.shape[1];
            if row == 0 || row > rows || col == 0 || col > cols {
                return Err("Index exceeds the number of array elements.".to_string());
            }
            let pos = (row - 1) + (col - 1) * rows;
            if pos >= logical.data.len() {
                return Err("Index exceeds the number of array elements.".to_string());
            }
            logical.data[pos] = if value { 1 } else { 0 };
            Ok(())
        }
        _ => Err("setfield: indexing with more than two indices is not supported yet".to_string()),
    }
}

fn assign_string_array_element(
    array: &mut runmat_builtins::StringArray,
    selector: &IndexSelector,
    rhs: Value,
) -> Result<(), String> {
    let resolved = resolve_indices(&Value::StringArray(array.clone()), selector)?;
    let text = String::try_from(&rhs)
        .map_err(|_| "setfield: string assignments require text-compatible values".to_string())?;
    match resolved.len() {
        1 => {
            let idx = resolved[0];
            if idx == 0 || idx > array.data.len() {
                return Err("Index exceeds the number of array elements.".to_string());
            }
            array.data[idx - 1] = text;
            Ok(())
        }
        2 => {
            let row = resolved[0];
            let col = resolved[1];
            if row == 0 || row > array.rows || col == 0 || col > array.cols {
                return Err("Index exceeds the number of array elements.".to_string());
            }
            let pos = (row - 1) + (col - 1) * array.rows;
            if pos >= array.data.len() {
                return Err("Index exceeds the number of array elements.".to_string());
            }
            array.data[pos] = text;
            Ok(())
        }
        _ => Err("setfield: indexing with more than two indices is not supported yet".to_string()),
    }
}

fn assign_char_array_element(
    array: &mut CharArray,
    selector: &IndexSelector,
    rhs: Value,
) -> Result<(), String> {
    let resolved = resolve_indices(&Value::CharArray(array.clone()), selector)?;
    let text = String::try_from(&rhs)
        .map_err(|_| "setfield: char assignments require text-compatible values".to_string())?;
    if text.chars().count() != 1 {
        return Err("setfield: char array assignments require single characters".to_string());
    }
    let ch = text.chars().next().unwrap();
    match resolved.len() {
        1 => {
            let idx = resolved[0];
            if idx == 0 || idx > array.data.len() {
                return Err("Index exceeds the number of array elements.".to_string());
            }
            array.data[idx - 1] = ch;
            Ok(())
        }
        2 => {
            let row = resolved[0];
            let col = resolved[1];
            if row == 0 || row > array.rows || col == 0 || col > array.cols {
                return Err("Index exceeds the number of array elements.".to_string());
            }
            let pos = (row - 1) * array.cols + (col - 1);
            if pos >= array.data.len() {
                return Err("Index exceeds the number of array elements.".to_string());
            }
            array.data[pos] = ch;
            Ok(())
        }
        _ => Err("setfield: indexing with more than two indices is not supported yet".to_string()),
    }
}

fn assign_complex_tensor_element(
    tensor: &mut ComplexTensor,
    selector: &IndexSelector,
    rhs: Value,
) -> Result<(), String> {
    let resolved = resolve_indices(&Value::ComplexTensor(tensor.clone()), selector)?;
    let (re, im) = match rhs {
        Value::Complex(r, i) => (r, i),
        Value::Num(n) => (n, 0.0),
        Value::Int(i) => (i.to_f64(), 0.0),
        other => {
            return Err(format!(
                "setfield: cannot assign {other:?} into a complex tensor element"
            ));
        }
    };
    match resolved.len() {
        1 => {
            let idx = resolved[0];
            if idx == 0 || idx > tensor.data.len() {
                return Err("Index exceeds the number of array elements.".to_string());
            }
            tensor.data[idx - 1] = (re, im);
            Ok(())
        }
        2 => {
            let row = resolved[0];
            let col = resolved[1];
            if row == 0 || row > tensor.rows || col == 0 || col > tensor.cols {
                return Err("Index exceeds the number of array elements.".to_string());
            }
            let pos = (row - 1) + (col - 1) * tensor.rows;
            if pos >= tensor.data.len() {
                return Err("Index exceeds the number of array elements.".to_string());
            }
            tensor.data[pos] = (re, im);
            Ok(())
        }
        _ => Err("setfield: indexing with more than two indices is not supported yet".to_string()),
    }
}

fn read_object_property(obj: &ObjectInstance, name: &str) -> Result<Value, String> {
    if let Some((prop, _owner)) = runmat_builtins::lookup_property(&obj.class_name, name) {
        if prop.is_static {
            return Err(format!(
                "You cannot access the static property '{}' through an instance of class '{}'.",
                name, obj.class_name
            ));
        }
        if prop.get_access == Access::Private {
            return Err(format!(
                "You cannot get the '{}' property of '{}' class.",
                name, obj.class_name
            ));
        }
        if prop.is_dependent {
            let getter = format!("get.{name}");
            match call_builtin(&getter, &[Value::Object(obj.clone())]) {
                Ok(value) => return Ok(value),
                Err(err) => {
                    if !err.contains("MATLAB:UndefinedFunction") {
                        return Err(err);
                    }
                }
            }
            if let Some(value) = obj.properties.get(&format!("{name}_backing")) {
                return Ok(value.clone());
            }
        }
    }

    if let Some(value) = obj.properties.get(name) {
        return Ok(value.clone());
    }

    if let Some((prop, _owner)) = runmat_builtins::lookup_property(&obj.class_name, name) {
        if prop.get_access == Access::Private {
            return Err(format!(
                "You cannot get the '{}' property of '{}' class.",
                name, obj.class_name
            ));
        }
        return Err(format!(
            "No public property '{}' for class '{}'.",
            name, obj.class_name
        ));
    }

    Err(format!(
        "Undefined property '{}' for class {}",
        name, obj.class_name
    ))
}

fn write_object_property(obj: &mut ObjectInstance, name: &str, rhs: Value) -> Result<(), String> {
    if let Some((prop, _owner)) = runmat_builtins::lookup_property(&obj.class_name, name) {
        if prop.is_static {
            return Err(format!(
                "Property '{}' is static; use classref('{}').{}",
                name, obj.class_name, name
            ));
        }
        if prop.set_access == Access::Private {
            return Err(format!("Property '{name}' is private"));
        }
        if prop.is_dependent {
            let setter = format!("set.{name}");
            if let Ok(value) = call_builtin(&setter, &[Value::Object(obj.clone()), rhs.clone()]) {
                if let Value::Object(updated) = value {
                    *obj = updated;
                    return Ok(());
                }
                return Err(format!(
                    "Dependent property setter for '{}' must return the updated object",
                    name
                ));
            }
            obj.properties.insert(format!("{name}_backing"), rhs);
            return Ok(());
        }
    }

    obj.properties.insert(name.to_string(), rhs);
    Ok(())
}

fn assign_into_handle(handle: HandleRef, steps: &[FieldStep], rhs: Value) -> Result<Value, String> {
    if steps.is_empty() {
        return Err(
            "setfield: expected at least one field name when assigning into a handle".to_string(),
        );
    }
    if !handle.valid {
        return Err(format!(
            "Invalid or deleted handle object '{}'.",
            handle.class_name
        ));
    }
    let current = unsafe { &*handle.target.as_raw() }.clone();
    let updated = assign_into_value(current, steps, rhs)?;
    let raw = unsafe { handle.target.as_raw_mut() };
    if raw.is_null() {
        return Err("setfield: handle target is null".to_string());
    }
    unsafe {
        *raw = updated;
    }
    Ok(Value::HandleObject(handle))
}

fn is_index_selector(value: &Value) -> bool {
    matches!(value, Value::Cell(_))
}

fn parse_index_selector(value: Value) -> Result<IndexSelector, String> {
    let Value::Cell(cell) = value else {
        return Err("setfield: indices must be provided in a cell array".to_string());
    };
    let mut components = Vec::with_capacity(cell.data.len());
    for handle in &cell.data {
        let entry = unsafe { &*handle.as_raw() };
        components.push(parse_index_component(entry)?);
    }
    Ok(IndexSelector { components })
}

fn parse_index_component(value: &Value) -> Result<IndexComponent, String> {
    match value {
        Value::CharArray(ca) => {
            let text: String = ca.data.iter().collect();
            parse_index_text(text.trim())
        }
        Value::String(s) => parse_index_text(s.trim()),
        Value::StringArray(sa) if sa.data.len() == 1 => parse_index_text(sa.data[0].trim()),
        _ => {
            let idx = parse_positive_scalar(value)
                .map_err(|e| format!("setfield: invalid index element ({e})"))?;
            Ok(IndexComponent::Scalar(idx))
        }
    }
}

fn parse_index_text(text: &str) -> Result<IndexComponent, String> {
    if text.eq_ignore_ascii_case("end") {
        return Ok(IndexComponent::End);
    }
    if text == ":" {
        return Err("setfield: ':' indexing is not currently supported".to_string());
    }
    if text.is_empty() {
        return Err("setfield: index elements must not be empty".to_string());
    }
    if let Ok(value) = text.parse::<usize>() {
        if value == 0 {
            return Err("setfield: index must be >= 1".to_string());
        }
        return Ok(IndexComponent::Scalar(value));
    }
    Err(format!("setfield: invalid index element '{}'", text))
}

fn parse_positive_scalar(value: &Value) -> Result<usize, String> {
    let number = match value {
        Value::Int(i) => i.to_i64() as f64,
        Value::Num(n) => *n,
        Value::Tensor(t) if t.data.len() == 1 => t.data[0],
        _ => {
            let repr = format!("{value:?}");
            return Err(format!("expected positive integer index, got {repr}"));
        }
    };

    if !number.is_finite() {
        return Err("index must be a finite number".to_string());
    }
    if number.fract() != 0.0 {
        return Err("index must be an integer".to_string());
    }
    if number <= 0.0 {
        return Err("index must be >= 1".to_string());
    }
    if number > usize::MAX as f64 {
        return Err("index exceeds platform limits".to_string());
    }
    Ok(number as usize)
}

fn parse_field_name(value: Value) -> Result<String, String> {
    match value {
        Value::String(s) => Ok(s),
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                Ok(sa.data[0].clone())
            } else {
                Err(
                    "setfield: field names must be scalar string arrays or character vectors"
                        .to_string(),
                )
            }
        }
        Value::CharArray(ca) => {
            if ca.rows == 1 {
                Ok(ca.data.iter().collect())
            } else {
                Err("setfield: field names must be 1-by-N character vectors".to_string())
            }
        }
        other => Err(format!("setfield: expected field name, got {other:?}")),
    }
}

fn resolve_indices(value: &Value, selector: &IndexSelector) -> Result<Vec<usize>, String> {
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

fn dimension_length(value: &Value, dims: usize, dim_idx: usize) -> Result<usize, String> {
    match value {
        Value::Tensor(tensor) => tensor_dimension_length(tensor, dims, dim_idx),
        Value::Cell(cell) => cell_dimension_length(cell, dims, dim_idx),
        Value::StringArray(array) => string_array_dimension_length(array, dims, dim_idx),
        Value::LogicalArray(logical) => logical_array_dimension_length(logical, dims, dim_idx),
        Value::CharArray(array) => char_array_dimension_length(array, dims, dim_idx),
        Value::ComplexTensor(tensor) => complex_tensor_dimension_length(tensor, dims, dim_idx),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            if dims == 1 {
                Ok(1)
            } else {
                Err(
                    "setfield: indexing with more than one dimension is not supported for scalars"
                        .to_string(),
                )
            }
        }
        other => Err(format!(
            "Struct contents assignment to a {other:?} object is not supported."
        )),
    }
}

fn tensor_dimension_length(tensor: &Tensor, dims: usize, dim_idx: usize) -> Result<usize, String> {
    if dims == 1 {
        let total = tensor.data.len();
        if total == 0 {
            return Err("Index exceeds the number of array elements (0).".to_string());
        }
        return Ok(total);
    }
    if dims > 2 {
        return Err(
            "setfield: indexing with more than two indices is not supported yet".to_string(),
        );
    }
    let len = if dim_idx == 0 {
        tensor.rows()
    } else {
        tensor.cols()
    };
    if len == 0 {
        return Err("Index exceeds the number of array elements (0).".to_string());
    }
    Ok(len)
}

fn cell_dimension_length(cell: &CellArray, dims: usize, dim_idx: usize) -> Result<usize, String> {
    if dims == 1 {
        let total = cell.data.len();
        if total == 0 {
            return Err("Index exceeds the number of array elements (0).".to_string());
        }
        return Ok(total);
    }
    if dims > 2 {
        return Err(
            "setfield: indexing with more than two indices is not supported yet".to_string(),
        );
    }
    let len = if dim_idx == 0 { cell.rows } else { cell.cols };
    if len == 0 {
        return Err("Index exceeds the number of array elements (0).".to_string());
    }
    Ok(len)
}

fn string_array_dimension_length(
    array: &runmat_builtins::StringArray,
    dims: usize,
    dim_idx: usize,
) -> Result<usize, String> {
    if dims == 1 {
        let total = array.data.len();
        if total == 0 {
            return Err("Index exceeds the number of array elements (0).".to_string());
        }
        return Ok(total);
    }
    if dims > 2 {
        return Err(
            "setfield: indexing with more than two indices is not supported yet".to_string(),
        );
    }
    let len = if dim_idx == 0 { array.rows } else { array.cols };
    if len == 0 {
        return Err("Index exceeds the number of array elements (0).".to_string());
    }
    Ok(len)
}

fn logical_array_dimension_length(
    array: &LogicalArray,
    dims: usize,
    dim_idx: usize,
) -> Result<usize, String> {
    if dims == 1 {
        let total = array.data.len();
        if total == 0 {
            return Err("Index exceeds the number of array elements (0).".to_string());
        }
        return Ok(total);
    }
    if dims > 2 {
        return Err(
            "setfield: indexing with more than two indices is not supported yet".to_string(),
        );
    }
    if array.shape.len() < dims {
        return Err("Index exceeds the number of array elements (0).".to_string());
    }
    let len = array.shape[dim_idx];
    if len == 0 {
        return Err("Index exceeds the number of array elements (0).".to_string());
    }
    Ok(len)
}

fn char_array_dimension_length(
    array: &CharArray,
    dims: usize,
    dim_idx: usize,
) -> Result<usize, String> {
    if dims == 1 {
        let total = array.data.len();
        if total == 0 {
            return Err("Index exceeds the number of array elements (0).".to_string());
        }
        return Ok(total);
    }
    if dims > 2 {
        return Err(
            "setfield: indexing with more than two indices is not supported yet".to_string(),
        );
    }
    let len = if dim_idx == 0 { array.rows } else { array.cols };
    if len == 0 {
        return Err("Index exceeds the number of array elements (0).".to_string());
    }
    Ok(len)
}

fn complex_tensor_dimension_length(
    tensor: &ComplexTensor,
    dims: usize,
    dim_idx: usize,
) -> Result<usize, String> {
    if dims == 1 {
        let total = tensor.data.len();
        if total == 0 {
            return Err("Index exceeds the number of array elements (0).".to_string());
        }
        return Ok(total);
    }
    if dims > 2 {
        return Err(
            "setfield: indexing with more than two indices is not supported yet".to_string(),
        );
    }
    let len = if dim_idx == 0 {
        tensor.rows
    } else {
        tensor.cols
    };
    if len == 0 {
        return Err("Index exceeds the number of array elements (0).".to_string());
    }
    Ok(len)
}

fn value_to_scalar(value: Value) -> Result<f64, String> {
    match value {
        Value::Num(n) => Ok(n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(if b { 1.0 } else { 0.0 }),
        Value::Tensor(t) if t.data.len() == 1 => Ok(t.data[0]),
        other => Err(format!(
            "setfield: cannot assign {other:?} into a numeric tensor element"
        )),
    }
}

fn value_to_bool(value: Value) -> Result<bool, String> {
    match value {
        Value::Bool(b) => Ok(b),
        Value::Num(n) => Ok(n != 0.0),
        Value::Int(i) => Ok(i.to_i64() != 0),
        Value::Tensor(t) if t.data.len() == 1 => Ok(t.data[0] != 0.0),
        other => Err(format!(
            "setfield: cannot assign {other:?} into a logical array element"
        )),
    }
}

fn allocate_cell_handle(value: Value) -> Result<GcPtr<Value>, String> {
    runmat_gc::gc_allocate(value)
        .map_err(|e| format!("setfield: failed to allocate cell element in GC: {e}"))
}

fn is_struct_array(cell: &CellArray) -> bool {
    cell.data
        .iter()
        .all(|handle| matches!(unsafe { &*handle.as_raw() }, Value::Struct(_)))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_builtins::{
        Access, CellArray, ClassDef, HandleRef, IntValue, ObjectInstance, PropertyDef, StructValue,
    };
    use runmat_gc::gc_allocate;

    use crate::builtins::common::test_support;

    #[test]
    fn setfield_creates_scalar_field() {
        let struct_value = StructValue::new();
        let updated = setfield_builtin(
            Value::Struct(struct_value),
            vec![Value::from("answer"), Value::Num(42.0)],
        )
        .expect("setfield");
        match updated {
            Value::Struct(st) => {
                assert_eq!(
                    st.fields.get("answer"),
                    Some(&Value::Num(42.0)),
                    "field should be inserted"
                );
            }
            other => panic!("expected struct result, got {other:?}"),
        }
    }

    #[test]
    fn setfield_creates_nested_structs() {
        let struct_value = StructValue::new();
        let updated = setfield_builtin(
            Value::Struct(struct_value),
            vec![
                Value::from("solver"),
                Value::from("name"),
                Value::from("cg"),
            ],
        )
        .expect("setfield");
        match updated {
            Value::Struct(st) => {
                let solver = st.fields.get("solver").expect("solver field");
                match solver {
                    Value::Struct(inner) => {
                        assert_eq!(
                            inner.fields.get("name"),
                            Some(&Value::from("cg")),
                            "inner field should exist"
                        );
                    }
                    other => panic!("expected inner struct, got {other:?}"),
                }
            }
            other => panic!("expected struct result, got {other:?}"),
        }
    }

    #[test]
    fn setfield_updates_struct_array_element() {
        let mut a = StructValue::new();
        a.fields
            .insert("id".to_string(), Value::Int(IntValue::I32(1)));
        let mut b = StructValue::new();
        b.fields
            .insert("id".to_string(), Value::Int(IntValue::I32(2)));
        let array = CellArray::new_with_shape(vec![Value::Struct(a), Value::Struct(b)], vec![1, 2])
            .unwrap();
        let indices =
            CellArray::new_with_shape(vec![Value::Int(IntValue::I32(2))], vec![1, 1]).unwrap();
        let updated = setfield_builtin(
            Value::Cell(array),
            vec![
                Value::Cell(indices),
                Value::from("id"),
                Value::Int(IntValue::I32(42)),
            ],
        )
        .expect("setfield");
        match updated {
            Value::Cell(cell) => {
                let second = unsafe { &*cell.data[1].as_raw() }.clone();
                match second {
                    Value::Struct(st) => {
                        assert_eq!(st.fields.get("id"), Some(&Value::Int(IntValue::I32(42))));
                    }
                    other => panic!("expected struct element, got {other:?}"),
                }
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[test]
    fn setfield_assigns_into_cell_then_struct() {
        let mut inner1 = StructValue::new();
        inner1.fields.insert("value".to_string(), Value::Num(1.0));
        let mut inner2 = StructValue::new();
        inner2.fields.insert("value".to_string(), Value::Num(2.0));
        let cell = CellArray::new_with_shape(
            vec![Value::Struct(inner1), Value::Struct(inner2)],
            vec![1, 2],
        )
        .unwrap();
        let mut root = StructValue::new();
        root.fields.insert("samples".to_string(), Value::Cell(cell));

        let index_cell =
            CellArray::new_with_shape(vec![Value::Int(IntValue::I32(2))], vec![1, 1]).unwrap();
        let updated = setfield_builtin(
            Value::Struct(root),
            vec![
                Value::from("samples"),
                Value::Cell(index_cell),
                Value::from("value"),
                Value::Num(10.0),
            ],
        )
        .expect("setfield");

        match updated {
            Value::Struct(st) => {
                let samples = st.fields.get("samples").expect("samples field");
                match samples {
                    Value::Cell(cell) => {
                        let value = unsafe { &*cell.data[1].as_raw() }.clone();
                        match value {
                            Value::Struct(inner) => {
                                assert_eq!(inner.fields.get("value"), Some(&Value::Num(10.0)));
                            }
                            other => panic!("expected struct, got {other:?}"),
                        }
                    }
                    other => panic!("expected cell array, got {other:?}"),
                }
            }
            other => panic!("expected struct, got {other:?}"),
        }
    }

    #[test]
    fn setfield_struct_array_with_end_index() {
        let mut first = StructValue::new();
        first
            .fields
            .insert("id".to_string(), Value::Int(IntValue::I32(1)));
        let mut second = StructValue::new();
        second
            .fields
            .insert("id".to_string(), Value::Int(IntValue::I32(2)));
        let array = CellArray::new_with_shape(
            vec![Value::Struct(first), Value::Struct(second)],
            vec![1, 2],
        )
        .unwrap();
        let index_cell = CellArray::new_with_shape(vec![Value::from("end")], vec![1, 1]).unwrap();
        let updated = setfield_builtin(
            Value::Cell(array),
            vec![
                Value::Cell(index_cell),
                Value::from("id"),
                Value::Int(IntValue::I32(99)),
            ],
        )
        .expect("setfield");
        match updated {
            Value::Cell(cell) => {
                let second = unsafe { &*cell.data[1].as_raw() }.clone();
                match second {
                    Value::Struct(st) => {
                        assert_eq!(st.fields.get("id"), Some(&Value::Int(IntValue::I32(99))));
                    }
                    other => panic!("expected struct element, got {other:?}"),
                }
            }
            other => panic!("expected cell array result, got {other:?}"),
        }
    }

    #[test]
    fn setfield_assigns_object_property() {
        let mut class_def = ClassDef {
            name: "Simple".to_string(),
            parent: None,
            properties: Default::default(),
            methods: Default::default(),
        };
        class_def.properties.insert(
            "x".to_string(),
            PropertyDef {
                name: "x".to_string(),
                is_static: false,
                is_dependent: false,
                get_access: Access::Public,
                set_access: Access::Public,
                default_value: None,
            },
        );
        runmat_builtins::register_class(class_def);

        let mut obj = ObjectInstance::new("Simple".to_string());
        obj.properties.insert("x".to_string(), Value::Num(0.0));

        let updated = setfield_builtin(Value::Object(obj), vec![Value::from("x"), Value::Num(5.0)])
            .expect("setfield");

        match updated {
            Value::Object(o) => {
                assert_eq!(o.properties.get("x"), Some(&Value::Num(5.0)));
            }
            other => panic!("expected object result, got {other:?}"),
        }
    }

    #[test]
    fn setfield_errors_when_indexing_missing_field() {
        let struct_value = StructValue::new();
        let index_cell =
            CellArray::new_with_shape(vec![Value::Int(IntValue::I32(1))], vec![1, 1]).unwrap();
        let err = setfield_builtin(
            Value::Struct(struct_value),
            vec![
                Value::from("missing"),
                Value::Cell(index_cell),
                Value::Num(1.0),
            ],
        )
        .expect_err("setfield should fail when field is missing");
        assert!(
            err.contains("Reference to non-existent field 'missing'."),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn setfield_errors_on_static_property_assignment() {
        let mut class_def = ClassDef {
            name: "StaticSetfield".to_string(),
            parent: None,
            properties: Default::default(),
            methods: Default::default(),
        };
        class_def.properties.insert(
            "version".to_string(),
            PropertyDef {
                name: "version".to_string(),
                is_static: true,
                is_dependent: false,
                get_access: Access::Public,
                set_access: Access::Public,
                default_value: None,
            },
        );
        runmat_builtins::register_class(class_def);

        let obj = ObjectInstance::new("StaticSetfield".to_string());
        let err = setfield_builtin(
            Value::Object(obj),
            vec![Value::from("version"), Value::Num(2.0)],
        )
        .expect_err("setfield should reject static property writes");
        assert!(
            err.contains("Property 'version' is static"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn setfield_updates_handle_target() {
        let mut inner = StructValue::new();
        inner.fields.insert("x".to_string(), Value::Num(0.0));
        let gc_ptr = gc_allocate(Value::Struct(inner)).expect("gc allocation");
        let handle_ptr = gc_ptr.clone();
        let handle = HandleRef {
            class_name: "PointHandle".to_string(),
            target: handle_ptr,
            valid: true,
        };

        let updated = setfield_builtin(
            Value::HandleObject(handle.clone()),
            vec![Value::from("x"), Value::Num(7.0)],
        )
        .expect("setfield handle update");

        match updated {
            Value::HandleObject(h) => assert!(h.valid),
            other => panic!("expected handle, got {other:?}"),
        }

        let pointee = unsafe { &*gc_ptr.as_raw() };
        match pointee {
            Value::Struct(st) => {
                assert_eq!(st.fields.get("x"), Some(&Value::Num(7.0)));
            }
            other => panic!("expected struct pointee, got {other:?}"),
        }
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn setfield_gpu_tensor_indexing_gathers_to_host() {
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };
        use runmat_accelerate_api::HostTensorView;

        if runmat_accelerate_api::provider().is_none()
            && register_wgpu_provider(WgpuProviderOptions::default()).is_err()
        {
            runmat_accelerate::simple_provider::register_inprocess_provider();
        }

        let provider = runmat_accelerate_api::provider().expect("accel provider");
        let data = [1.0, 2.0, 3.0, 4.0];
        let shape = [2usize, 2usize];
        let view = HostTensorView {
            data: &data,
            shape: &shape,
        };
        let handle = provider.upload(&view).expect("upload");

        let mut root = StructValue::new();
        root.fields
            .insert("values".to_string(), Value::GpuTensor(handle));

        let index_cell = CellArray::new_with_shape(
            vec![Value::Int(IntValue::I32(2)), Value::Int(IntValue::I32(2))],
            vec![1, 2],
        )
        .unwrap();

        let updated = setfield_builtin(
            Value::Struct(root),
            vec![
                Value::from("values"),
                Value::Cell(index_cell),
                Value::Num(99.0),
            ],
        )
        .expect("setfield gpu value");

        match updated {
            Value::Struct(st) => {
                let values = st.fields.get("values").expect("values field");
                match values {
                    Value::Tensor(tensor) => {
                        assert_eq!(tensor.shape, vec![2, 2]);
                        assert_eq!(tensor.data[3], 99.0);
                    }
                    other => panic!("expected tensor after gather, got {other:?}"),
                }
            }
            other => panic!("expected struct result, got {other:?}"),
        }
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
