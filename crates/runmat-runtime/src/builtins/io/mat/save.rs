//! MATLAB-compatible `save` builtin for RunMat.

use std::collections::HashSet;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use regex::Regex;
use runmat_builtins::{CharArray, StructValue, Tensor, Value};
use runmat_filesystem::File;
use runmat_macros::runtime_builtin;

use super::format::{
    MatArray, MatClass, MatData, FLAG_COMPLEX, FLAG_LOGICAL, MAT_HEADER_LEN, MI_DOUBLE, MI_INT32,
    MI_INT8, MI_MATRIX, MI_UINT16, MI_UINT32, MI_UINT8,
};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::gather_if_needed;
use crate::workspace;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "save",
        builtin_path = "crate::builtins::io::mat::save"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "save"
category: "io/mat"
keywords: ["save", "mat", "workspace", "io", "matlab save"]
summary: "Persist variables from the current workspace to a MATLAB-compatible MAT-file."
references:
  - https://www.mathworks.com/help/matlab/ref/save.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "save performs synchronous host I/O. GPU-resident variables are gathered prior to serialisation; no provider hooks are required."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::mat::save::tests"
  integration:
    - "builtins::io::mat::save::tests::save_gpu_tensor_roundtrip"
    - "builtins::io::mat::save::tests::save_wgpu_tensor_roundtrip"
---

# What does the `save` function do in MATLAB / RunMat?
`save` writes variables from the current workspace to a MAT-file on disk. RunMat mirrors MATLAB semantics for choosing variables, handling structs, pattern-based selection via `'-regexp'`, and processing options such as `'-struct'`.

## How does the `save` function behave in MATLAB / RunMat?
- `save` with no arguments writes every variable from the caller workspace to `matlab.mat` in the current working directory. Set `RUNMAT_SAVE_DEFAULT_PATH` to override the default target when no filename is supplied.
- `save filename` writes all workspace variables to `filename`. If the supplied name has no extension, `.mat` is added automatically. Paths are resolved relative to the current directory, and parent folders must already exist.
- `save(filename, 'A', 'B')` writes only the listed variables. String arrays or cell arrays of character vectors are accepted to specify multiple names.
- `save(filename, '-struct', 'S')` saves each field of struct `S` as a separate variable. Provide additional field names (`'field1'`, `'field2'`) to restrict the set.
- `save filename -regexp '^foo' 'bar$'` saves every variable whose name matches any of the supplied regular expressions.
- Existing files are overwritten unless `-append` is specified. RunMat currently reports an error when `-append` or other numeric/text format flags (for example `-ascii`, `-double`, `-v6`, `-v7.3`) are requested.
- Unsupported types (function handles, objects, opaque graphics handles) raise descriptive errors. Numeric, logical, character, string, cell, and struct data are stored using MATLAB Level-5 MAT-file layout.
- `save` returns `0` so scripts can treat it as a statement, matching MATLAB's void behaviour.

## `save` Function GPU Execution Behaviour
`save` acts as a residency sink. Before serialising, RunMat gathers any GPU-resident tensors through the active acceleration provider so the MAT-file contains host data. Fusion groups terminate at this builtin and providers do not require custom hooks.

## Examples of using the `save` function in MATLAB / RunMat

### Save the entire workspace
```matlab
x = 42;
y = magic(3);
save('session.mat');
```
Expected result: `session.mat` appears in the current folder containing both `x` and `y`.

### Save selected variables only
```matlab
a = rand(1, 3);
b = eye(2);
c = "ignore me";
save('selection.mat', 'a', 'b');
```
Expected result: the MAT-file stores `a` and `b`; `c` is omitted.

### Save struct fields as individual variables
```matlab
opts.output = y;
opts.answer = x;
save('opts.mat', '-struct', 'opts');
```
Expected result: the file contains variables `output` and `answer` populated from the struct fields.

### Select variables by regular expression
```matlab
result_acc = 1:3;
result_tmp = 4:6;
scratch = pi;
save('filtered.mat', '-regexp', '^result_', 'tmp$');
```
Expected result: only `result_acc` and `result_tmp` are saved because their names match the patterns.

### Save multiple names using a string array
```matlab
names = ["result_acc", "scratch"];
save('mixed.mat', names);
```
Expected result: both variables listed in `names` are written to `mixed.mat`.

### Save GPU arrays without manual gather
```matlab
G = gpuArray(magic(3));
save('gpu.mat', 'G');
```
Expected result: `gpu.mat` contains a standard double-precision matrix named `G`; RunMat gathered the `gpuArray` to host memory automatically.

### Error when a variable is missing
```matlab
save('bad.mat', 'missing');
```
Expected error:
```matlab
save: variable 'missing' was not found in the workspace
```

## GPU residency in RunMat (Do I need `gpuArray`?)
No manual action is required. `save` gathers `gpuArray` inputs automatically before writing the MAT-file. This matches MATLAB's behaviour when `gpuArray` variables are passed to `save`.

## FAQ

### Which MAT-file version is generated?
RunMat writes Level-5 (MATLAB 5.0) MAT-files, compatible with MATLAB R2006b and later as well as NumPy/Scipy readers. Structures and cell arrays are stored using standard `miMATRIX` elements.

### Is `-append` supported?
Not yet. The builtin currently reports `save: -append is not supported` to signal the limitation. Future releases will add incremental writing.

### Do text options like `-ascii` or `-double` work?
These legacy text/binary format switches are not supported. RunMat favours MATLAB's default MAT-file format.

### Are objects or function handles saved?
No. RunMat matches MATLAB by raising an error when unsupported types are encountered.

### Does `save` return a value?
Yes. The builtin returns `0`, which MATLAB treats as an empty output, so scripts can ignore the return value just as they do in MATLAB.

### How do I change the default filename used by bare `save`?
Set the environment variable `RUNMAT_SAVE_DEFAULT_PATH` before launching RunMat. When `save` is called without explicit filename arguments, the builtin writes to that path instead of `matlab.mat`.

### Does `save` create parent directories automatically?
No. Parent folders must already exist; otherwise the builtin raises an error from the host filesystem.

## See Also
[gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather), [fileread](../../io/filetext/fileread), [fopen](../../io/filetext/fopen)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/io/mat/save.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/mat/save.rs)
- Found a bug? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::mat::save")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "save",
    op_kind: GpuOpKind::Custom("io-save"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Performs synchronous host I/O; no GPU execution path is involved.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::mat::save")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "save",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Sink operation that terminates fusion graphs before serialisation.",
};

#[runtime_builtin(
    name = "save",
    category = "io/mat",
    summary = "Persist workspace variables to a MAT-file.",
    keywords = "save,mat,workspace",
    sink = true,
    builtin_path = "crate::builtins::io::mat::save"
)]
fn save_builtin(args: Vec<Value>) -> Result<Value, String> {
    let mut host_args = Vec::with_capacity(args.len());
    for value in &args {
        host_args.push(gather_if_needed(value)?);
    }

    let default_path = Value::from("matlab.mat");
    let (mut path_value, option_start, used_default) = match host_args.first() {
        Some(first) if option_token(first)?.is_some() => (default_path, 0usize, true),
        Some(first) => (first.clone(), 1usize, false),
        None => (default_path, 0usize, true),
    };

    if used_default {
        if let Ok(override_path) = std::env::var("RUNMAT_SAVE_DEFAULT_PATH") {
            path_value = Value::from(override_path);
        }
    }

    let request = parse_arguments(&host_args[option_start..])?;
    if request.append {
        return Err("save: -append is not supported yet".to_string());
    }

    let mut workspace_entries: Option<Vec<(String, Value)>> = None;
    let mut entries: Vec<(String, Value)> = Vec::new();

    if request.variables.is_empty()
        && request.structs.is_empty()
        && request.regex_patterns.is_empty()
    {
        let snapshot = ensure_workspace_entries(&mut workspace_entries)?;
        entries.extend(snapshot.iter().cloned());
    } else {
        for name in &request.variables {
            if let Some(snapshot) = workspace_entries.as_ref() {
                if let Some(value) = find_in_entries(snapshot, name) {
                    entries.push((name.clone(), value));
                    continue;
                }
            }
            let value = lookup_workspace(name)?;
            entries.push((name.clone(), value));
        }

        for struct_req in &request.structs {
            let value = if let Some(snapshot) = workspace_entries.as_ref() {
                find_in_entries(snapshot, &struct_req.source)
            } else {
                None
            };
            let value = match value {
                Some(val) => val,
                None => lookup_workspace(&struct_req.source)?,
            };

            let struct_value = match value {
                Value::Struct(s) => s,
                _ => {
                    return Err(format!(
                        "save: variable '{}' is not a struct",
                        struct_req.source
                    ))
                }
            };
            append_struct_fields(
                &struct_req.source,
                &struct_value,
                &struct_req.fields,
                &mut entries,
            )?;
        }

        if !request.regex_patterns.is_empty() {
            let snapshot = ensure_workspace_entries(&mut workspace_entries)?;
            let mut patterns = Vec::with_capacity(request.regex_patterns.len());
            for pattern in &request.regex_patterns {
                let regex = Regex::new(pattern).map_err(|err| {
                    format!("save: invalid regular expression '{pattern}': {err}")
                })?;
                patterns.push(regex);
            }
            let mut matched = 0usize;
            for (name, value) in snapshot.iter() {
                if patterns.iter().any(|regex| regex.is_match(name)) {
                    entries.push((name.clone(), value.clone()));
                    matched += 1;
                }
            }
            if matched == 0 {
                return Err("save: no variables matched '-regexp' patterns".to_string());
            }
        }
    }

    if entries.is_empty() {
        return Err("save: no variables selected".to_string());
    }

    // Deduplicate while preserving the last occurrence for MATLAB compatibility
    let mut seen = HashSet::new();
    let mut unique_entries = Vec::new();
    for (name, value) in entries.into_iter().rev() {
        if seen.insert(name.clone()) {
            unique_entries.push((name, value));
        }
    }
    unique_entries.reverse();

    let mut mat_vars = Vec::with_capacity(unique_entries.len());
    for (name, value) in unique_entries {
        mat_vars.push(MatVar {
            name,
            array: convert_value(&value)?,
        });
    }

    let path = normalise_path(&path_value)?;
    write_mat_file(&path, &mat_vars)?;

    Ok(Value::Num(0.0))
}

struct StructRequest {
    source: String,
    fields: Option<Vec<String>>,
}

struct SaveRequest {
    variables: Vec<String>,
    structs: Vec<StructRequest>,
    regex_patterns: Vec<String>,
    append: bool,
}

fn parse_arguments(values: &[Value]) -> Result<SaveRequest, String> {
    let mut variables = Vec::new();
    let mut structs = Vec::new();
    let mut regex_patterns = Vec::new();
    let mut append = false;

    let mut idx = 0;
    while idx < values.len() {
        if let Some(flag) = option_token(&values[idx])? {
            match flag.as_str() {
                "-append" => append = true,
                "-struct" => {
                    idx += 1;
                    if idx >= values.len() {
                        return Err("save: '-struct' requires a struct variable name".to_string());
                    }
                    let struct_names = extract_names(&values[idx])?;
                    if struct_names.len() != 1 {
                        return Err(
                            "save: '-struct' requires a single struct variable name".to_string()
                        );
                    }
                    let source = struct_names.into_iter().next().unwrap();
                    idx += 1;
                    let mut field_names = Vec::new();
                    while idx < values.len() {
                        if option_token(&values[idx])?.is_some() {
                            break;
                        }
                        let names = extract_names(&values[idx])?;
                        if names.is_empty() {
                            break;
                        }
                        field_names.extend(names);
                        idx += 1;
                    }
                    idx -= 1; // compensate for loop increment
                    let fields = if field_names.is_empty() {
                        None
                    } else {
                        Some(field_names)
                    };
                    structs.push(StructRequest { source, fields });
                }
                "-regexp" => {
                    idx += 1;
                    if idx >= values.len() {
                        return Err("save: '-regexp' requires at least one pattern".to_string());
                    }
                    let mut patterns = Vec::new();
                    while idx < values.len() {
                        if option_token(&values[idx])?.is_some() {
                            break;
                        }
                        let names = extract_names(&values[idx])?;
                        if names.is_empty() {
                            return Err(
                                "save: '-regexp' requires pattern strings or character rows"
                                    .to_string(),
                            );
                        }
                        patterns.extend(names);
                        idx += 1;
                    }
                    if patterns.is_empty() {
                        return Err("save: '-regexp' requires at least one pattern".to_string());
                    }
                    idx -= 1;
                    regex_patterns.extend(patterns);
                }
                other => {
                    return Err(format!("save: unsupported option '{other}'"));
                }
            }
        } else {
            let names = extract_names(&values[idx])?;
            variables.extend(names);
        }
        idx += 1;
    }

    Ok(SaveRequest {
        variables,
        structs,
        regex_patterns,
        append,
    })
}

fn ensure_workspace_entries(
    cache: &mut Option<Vec<(String, Value)>>,
) -> Result<&Vec<(String, Value)>, String> {
    if cache.is_none() {
        let entries = collect_workspace_entries()?;
        *cache = Some(entries);
    }
    Ok(cache.as_ref().unwrap())
}

fn collect_workspace_entries() -> Result<Vec<(String, Value)>, String> {
    let snapshot =
        workspace::snapshot().ok_or_else(|| "save: workspace state unavailable".to_string())?;
    let mut entries = Vec::with_capacity(snapshot.len());
    for (name, value) in snapshot {
        let gathered = gather_if_needed(&value)?;
        entries.push((name, gathered));
    }
    Ok(entries)
}

fn find_in_entries(entries: &[(String, Value)], name: &str) -> Option<Value> {
    entries
        .iter()
        .find(|(entry_name, _)| entry_name == name)
        .map(|(_, value)| value.clone())
}

fn option_token(value: &Value) -> Result<Option<String>, String> {
    if let Some(token) = value_to_string_scalar(value) {
        if token.starts_with('-') {
            return Ok(Some(token.to_ascii_lowercase()));
        }
    }
    Ok(None)
}

fn extract_names(value: &Value) -> Result<Vec<String>, String> {
    match value {
        Value::String(s) => Ok(vec![s.clone()]),
        Value::CharArray(ca) => {
            let rows = char_array_rows_as_strings(ca);
            if rows.is_empty() && ca.rows > 0 {
                return Err(
                    "save: character arrays used for variable names must contain non-empty rows"
                        .to_string(),
                );
            }
            Ok(rows)
        }
        Value::StringArray(sa) => {
            let mut names = Vec::with_capacity(sa.data.len());
            for s in &sa.data {
                names.push(s.clone());
            }
            Ok(names)
        }
        Value::Cell(ca) => {
            let mut names = Vec::with_capacity(ca.data.len());
            for handle in &ca.data {
                let inner = unsafe { &*handle.as_raw() };
                let text = value_to_string_scalar(inner)
                    .ok_or_else(|| "save: cell arrays must contain string scalars when specifying variable names".to_string())?;
                names.push(text);
            }
            Ok(names)
        }
        other => {
            let gathered = gather_if_needed(other)?;
            extract_names(&gathered)
        }
    }
}

fn value_to_string_scalar(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        _ => None,
    }
}

fn append_struct_fields(
    struct_name: &str,
    value: &StructValue,
    fields: &Option<Vec<String>>,
    out: &mut Vec<(String, Value)>,
) -> Result<(), String> {
    if let Some(ref names) = fields {
        for field in names {
            match value.fields.get(field) {
                Some(val) => {
                    let gathered = gather_if_needed(val)?;
                    out.push((field.clone(), gathered));
                }
                None => {
                    return Err(format!(
                        "save: struct '{}' does not have a field named '{}'",
                        struct_name, field
                    ));
                }
            }
        }
    } else {
        for (field, val) in &value.fields {
            let gathered = gather_if_needed(val)?;
            out.push((field.clone(), gathered));
        }
    }
    Ok(())
}

fn char_array_rows_as_strings(ca: &CharArray) -> Vec<String> {
    let mut rows = Vec::with_capacity(ca.rows);
    for row in 0..ca.rows {
        let mut buffer = String::with_capacity(ca.cols);
        for col in 0..ca.cols {
            let idx = row * ca.cols + col;
            buffer.push(ca.data[idx]);
        }
        let trimmed = buffer.trim_end_matches([' ', '\0']).to_string();
        if !trimmed.is_empty() {
            rows.push(trimmed);
        }
    }
    rows
}

fn lookup_workspace(name: &str) -> Result<Value, String> {
    workspace::lookup(name)
        .ok_or_else(|| format!("save: variable '{}' was not found in the workspace", name))
        .and_then(|value| gather_if_needed(&value))
}

fn normalise_path(path: &Value) -> Result<PathBuf, String> {
    let raw = value_to_string_scalar(path)
        .ok_or_else(|| "save: filename must be a character vector or string scalar".to_string())?;
    let mut path = PathBuf::from(raw);
    if path.extension().is_none() {
        path.set_extension("mat");
    }
    Ok(path)
}

struct MatVar {
    name: String,
    array: MatArray,
}

fn canonical_dims(shape: &[usize]) -> Vec<usize> {
    match shape.len() {
        0 => vec![1, 1],
        1 => vec![1, shape[0]],
        _ => shape.to_vec(),
    }
}

fn convert_value(value: &Value) -> Result<MatArray, String> {
    match value {
        Value::Num(n) => Ok(MatArray {
            class: MatClass::Double,
            dims: vec![1, 1],
            data: MatData::Double {
                real: vec![*n],
                imag: None,
            },
        }),
        Value::Int(i) => Ok(MatArray {
            class: MatClass::Double,
            dims: vec![1, 1],
            data: MatData::Double {
                real: vec![i.to_f64()],
                imag: None,
            },
        }),
        Value::Bool(b) => Ok(MatArray {
            class: MatClass::Logical,
            dims: vec![1, 1],
            data: MatData::Logical {
                data: vec![if *b { 1 } else { 0 }],
            },
        }),
        Value::Tensor(t) => Ok(MatArray {
            class: MatClass::Double,
            dims: canonical_dims(&t.shape),
            data: MatData::Double {
                real: t.data.clone(),
                imag: None,
            },
        }),
        Value::Complex(re, im) => Ok(MatArray {
            class: MatClass::Double,
            dims: vec![1, 1],
            data: MatData::Double {
                real: vec![*re],
                imag: Some(vec![*im]),
            },
        }),
        Value::ComplexTensor(t) => {
            let mut real = Vec::with_capacity(t.data.len());
            let mut imag = Vec::with_capacity(t.data.len());
            for (re, im) in &t.data {
                real.push(*re);
                imag.push(*im);
            }
            Ok(MatArray {
                class: MatClass::Double,
                dims: canonical_dims(&t.shape),
                data: MatData::Double {
                    real,
                    imag: Some(imag),
                },
            })
        }
        Value::LogicalArray(la) => Ok(MatArray {
            class: MatClass::Logical,
            dims: canonical_dims(&la.shape),
            data: MatData::Logical {
                data: la.data.clone(),
            },
        }),
        Value::CharArray(ca) => Ok(MatArray {
            class: MatClass::Char,
            dims: vec![ca.rows, ca.cols],
            data: MatData::Char {
                data: char_array_to_utf16(ca),
            },
        }),
        Value::String(s) => Ok(MatArray {
            class: MatClass::Char,
            dims: vec![1, s.chars().count()],
            data: MatData::Char {
                data: s.encode_utf16().collect(),
            },
        }),
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                return convert_value(&Value::String(sa.data[0].clone()));
            }
            let mut elements = Vec::with_capacity(sa.data.len());
            for text in &sa.data {
                elements.push(MatArray {
                    class: MatClass::Char,
                    dims: vec![1, text.chars().count()],
                    data: MatData::Char {
                        data: text.encode_utf16().collect(),
                    },
                });
            }
            Ok(MatArray {
                class: MatClass::Cell,
                dims: canonical_dims(&sa.shape),
                data: MatData::Cell { elements },
            })
        }
        Value::Cell(cell) => {
            let mut elements = Vec::with_capacity(cell.data.len());
            for col in 0..cell.cols {
                for row in 0..cell.rows {
                    let idx = row * cell.cols + col;
                    let element = unsafe { &*cell.data[idx].as_raw() };
                    let gathered = gather_if_needed(element)?;
                    elements.push(convert_value(&gathered)?);
                }
            }
            Ok(MatArray {
                class: MatClass::Cell,
                dims: vec![cell.rows, cell.cols],
                data: MatData::Cell { elements },
            })
        }
        Value::Struct(struct_value) => {
            let mut field_names: Vec<String> = struct_value.fields.keys().cloned().collect();
            field_names.sort();
            let mut field_values = Vec::with_capacity(field_names.len());
            for field in &field_names {
                let val = struct_value
                    .fields
                    .get(field)
                    .ok_or_else(|| format!("save: missing struct field '{field}'"))?;
                let gathered = gather_if_needed(val)?;
                field_values.push(convert_value(&gathered)?);
            }
            Ok(MatArray {
                class: MatClass::Struct,
                dims: vec![1, 1],
                data: MatData::Struct {
                    field_names,
                    field_values,
                },
            })
        }
        Value::GpuTensor(handle) => {
            let provider = runmat_accelerate_api::provider()
                .ok_or_else(|| "save: no acceleration provider registered".to_string())?;
            let host = provider.download(handle).map_err(|e| e.to_string())?;
            let tensor = Tensor::new(host.data, host.shape).map_err(|e| format!("save: {e}"))?;
            convert_value(&Value::Tensor(tensor))
        }
        unsupported => Err(format!(
            "save: value of type '{:?}' is not supported",
            unsupported
        )),
    }
}

fn char_array_to_utf16(ca: &CharArray) -> Vec<u16> {
    let mut data = Vec::with_capacity(ca.rows * ca.cols);
    for col in 0..ca.cols {
        for row in 0..ca.rows {
            let idx = row * ca.cols + col;
            data.push(ca.data[idx] as u16);
        }
    }
    data
}

fn write_mat_file(path: &Path, vars: &[MatVar]) -> Result<(), String> {
    let file = File::create(path)
        .map_err(|e| format!("save: failed to open '{}': {e}", path.display()))?;
    let mut writer = BufWriter::new(file);

    let mut header = [0u8; MAT_HEADER_LEN];
    let desc = b"MATLAB 5.0 MAT-file, RunMat save";
    for (i, byte) in desc.iter().enumerate() {
        header[i] = *byte;
    }
    header[124] = 0x00;
    header[125] = 0x01;
    header[126] = b'I';
    header[127] = b'M';
    writer
        .write_all(&header)
        .map_err(|e| format!("save: failed to write header: {e}"))?;

    for var in vars {
        let matrix_bytes = build_matrix_bytes(&var.array, Some(&var.name))?;
        write_tagged(&mut writer, MI_MATRIX, &matrix_bytes)?;
    }

    writer
        .flush()
        .map_err(|e| format!("save: flush failed: {e}"))
}

fn build_matrix_bytes(array: &MatArray, name: Option<&str>) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();

    let (flags0, flags1) = match &array.data {
        MatData::Double { imag, .. } => {
            let mut f0 = array.class.class_code();
            if imag.is_some() {
                f0 |= FLAG_COMPLEX;
            }
            (f0, 0u32)
        }
        MatData::Logical { .. } => ((array.class.class_code()) | FLAG_LOGICAL, 0u32),
        _ => (array.class.class_code(), 0u32),
    };

    let mut flags = Vec::with_capacity(8);
    flags.extend_from_slice(&flags0.to_le_bytes());
    flags.extend_from_slice(&flags1.to_le_bytes());
    write_subelement(&mut buf, MI_UINT32, &flags);

    let mut dims_bytes = Vec::with_capacity(array.dims.len() * 4);
    for &dim in &array.dims {
        dims_bytes.extend_from_slice(&((dim as i32).max(0)).to_le_bytes());
    }
    write_subelement(&mut buf, MI_INT32, &dims_bytes);

    let name_bytes = name.unwrap_or("").as_bytes();
    write_subelement(&mut buf, MI_INT8, name_bytes);

    match &array.data {
        MatData::Double { real, imag } => {
            let mut real_bytes = Vec::with_capacity(real.len() * 8);
            for v in real {
                real_bytes.extend_from_slice(&v.to_le_bytes());
            }
            write_subelement(&mut buf, MI_DOUBLE, &real_bytes);
            if let Some(imag) = imag {
                let mut imag_bytes = Vec::with_capacity(imag.len() * 8);
                for v in imag {
                    imag_bytes.extend_from_slice(&v.to_le_bytes());
                }
                write_subelement(&mut buf, MI_DOUBLE, &imag_bytes);
            }
        }
        MatData::Logical { data } => {
            write_subelement(&mut buf, MI_UINT8, data);
        }
        MatData::Char { data } => {
            let mut bytes = Vec::with_capacity(data.len() * 2);
            for code in data {
                bytes.extend_from_slice(&code.to_le_bytes());
            }
            write_subelement(&mut buf, MI_UINT16, &bytes);
        }
        MatData::Cell { elements } => {
            for elem in elements {
                let elem_bytes = build_matrix_bytes(elem, None)?;
                write_subelement(&mut buf, MI_MATRIX, &elem_bytes);
            }
        }
        MatData::Struct {
            field_names,
            field_values,
        } => {
            if array.dims != [1, 1] {
                return Err("save: struct arrays are not supported".to_string());
            }
            let max_len = field_names
                .iter()
                .map(|n| n.len())
                .max()
                .unwrap_or(0)
                .max(1);
            let len_bytes = (max_len as i32).to_le_bytes();
            write_subelement(&mut buf, MI_INT32, &len_bytes);
            let mut names_bytes = Vec::with_capacity(max_len * field_names.len());
            for name in field_names {
                let bytes = name.as_bytes();
                for i in 0..max_len {
                    let b = if i < bytes.len() { bytes[i] } else { 0 };
                    names_bytes.push(b);
                }
            }
            write_subelement(&mut buf, MI_INT8, &names_bytes);
            for value in field_values {
                let value_bytes = build_matrix_bytes(value, None)?;
                write_subelement(&mut buf, MI_MATRIX, &value_bytes);
            }
        }
    }

    Ok(buf)
}

fn write_tagged<W: Write>(writer: &mut W, data_type: u32, data: &[u8]) -> Result<(), String> {
    if data.len() > u32::MAX as usize {
        return Err("save: data too large for MAT-file".to_string());
    }
    writer
        .write_all(&data_type.to_le_bytes())
        .map_err(|e| format!("save: write failed: {e}"))?;
    writer
        .write_all(&(data.len() as u32).to_le_bytes())
        .map_err(|e| format!("save: write failed: {e}"))?;
    writer
        .write_all(data)
        .map_err(|e| format!("save: write failed: {e}"))?;
    let padding = (8 - (data.len() % 8)) % 8;
    if padding != 0 {
        let pad = [0u8; 8];
        writer
            .write_all(&pad[..padding])
            .map_err(|e| format!("save: write failed: {e}"))?;
    }
    Ok(())
}

fn write_subelement(buf: &mut Vec<u8>, data_type: u32, data: &[u8]) {
    buf.extend_from_slice(&data_type.to_le_bytes());
    buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
    buf.extend_from_slice(data);
    let padding = (8 - (data.len() % 8)) % 8;
    if padding != 0 {
        buf.extend(std::iter::repeat_n(0u8, padding));
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::workspace::WorkspaceResolver;
    use once_cell::sync::OnceCell;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::StringArray;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::sync::Mutex;
    use tempfile::tempdir;

    thread_local! {
        static TEST_WORKSPACE: RefCell<HashMap<String, Value>> = RefCell::new(HashMap::new());
    }

    fn ensure_test_resolver() {
        static INIT: OnceCell<()> = OnceCell::new();
        INIT.get_or_init(|| {
            workspace::register_workspace_resolver(WorkspaceResolver {
                lookup: |name| TEST_WORKSPACE.with(|slot| slot.borrow().get(name).cloned()),
                snapshot: || {
                    let mut entries: Vec<(String, Value)> =
                        TEST_WORKSPACE.with(|slot| slot.borrow().clone().into_iter().collect());
                    entries.sort_by(|a, b| a.0.cmp(&b.0));
                    entries
                },
                globals: || Vec::new(),
            });
        });
    }

    fn set_workspace(entries: &[(&str, Value)]) {
        TEST_WORKSPACE.with(|slot| {
            let mut map = HashMap::new();
            for (k, v) in entries {
                map.insert(k.to_string(), v.clone());
            }
            *slot.borrow_mut() = map;
        });
    }

    fn lock_env_override() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceCell<Mutex<()>> = OnceCell::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    struct EnvOverride {
        key: &'static str,
    }

    impl EnvOverride {
        fn set(key: &'static str, value: &str) -> Self {
            std::env::set_var(key, value);
            EnvOverride { key }
        }
    }

    impl Drop for EnvOverride {
        fn drop(&mut self) {
            std::env::remove_var(self.key);
        }
    }

    #[test]
    fn save_numeric_variable() {
        ensure_test_resolver();
        set_workspace(&[("A", Value::Num(42.0))]);
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_numeric.mat");
        let args = vec![
            Value::from(path.to_string_lossy().to_string()),
            Value::from("A"),
        ];
        save_builtin(args).unwrap();

        let file = File::open(&path).unwrap();
        let mat = matfile::MatFile::parse(file).unwrap();
        let array = mat.find_by_name("A").unwrap();
        match array.data() {
            matfile::NumericData::Double { real, .. } => {
                assert_eq!(real, &[42.0]);
            }
            _ => panic!("expected double array"),
        }
    }

    #[test]
    fn save_string_array_variable_names() {
        ensure_test_resolver();
        set_workspace(&[
            ("A", Value::Num(1.0)),
            ("B", Value::Num(2.0)),
            ("C", Value::Num(3.0)),
        ]);
        let dir = tempdir().unwrap();
        let path = dir.path().join("string_array.mat");
        let names = StringArray::new(vec!["A".into(), "B".into()], vec![1, 2]).unwrap();
        let args = vec![
            Value::from(path.to_string_lossy().to_string()),
            Value::StringArray(names),
        ];
        save_builtin(args).unwrap();

        let file = File::open(&path).unwrap();
        let mat = matfile::MatFile::parse(file).unwrap();
        assert!(mat.find_by_name("A").is_some());
        assert!(mat.find_by_name("B").is_some());
        assert!(mat.find_by_name("C").is_none());
    }

    #[test]
    fn save_char_matrix_variable_names() {
        ensure_test_resolver();
        set_workspace(&[
            ("foo", Value::Num(10.0)),
            ("bar", Value::Num(20.0)),
            ("baz", Value::Num(30.0)),
        ]);
        let dir = tempdir().unwrap();
        let path = dir.path().join("char_matrix.mat");
        let chars = CharArray::new("foobar".chars().collect(), 2, 3).unwrap();
        let args = vec![
            Value::from(path.to_string_lossy().to_string()),
            Value::CharArray(chars),
        ];
        save_builtin(args).unwrap();

        let file = File::open(&path).unwrap();
        let mat = matfile::MatFile::parse(file).unwrap();
        assert!(mat.find_by_name("foo").is_some());
        assert!(mat.find_by_name("bar").is_some());
        assert!(mat.find_by_name("baz").is_none());
    }

    #[test]
    fn save_struct_fields() {
        ensure_test_resolver();
        let mut opts_struct = StructValue::new();
        opts_struct
            .fields
            .insert("foo".to_string(), Value::Num(1.0));
        opts_struct
            .fields
            .insert("bar".to_string(), Value::Num(2.0));
        set_workspace(&[("opts", Value::Struct(opts_struct))]);
        let dir = tempdir().unwrap();
        let path = dir.path().join("struct.mat");
        let args = vec![
            Value::from(path.to_string_lossy().to_string()),
            Value::from("-struct"),
            Value::from("opts"),
        ];
        save_builtin(args).unwrap();

        let file = File::open(&path).unwrap();
        let mat = matfile::MatFile::parse(file).unwrap();
        let foo = mat.find_by_name("bar").unwrap();
        match foo.data() {
            matfile::NumericData::Double { real, .. } => assert_eq!(real, &[2.0]),
            _ => panic!("expected double"),
        }
    }

    #[test]
    fn save_struct_field_selection() {
        ensure_test_resolver();
        let mut opts_struct = StructValue::new();
        opts_struct
            .fields
            .insert("foo".to_string(), Value::Num(11.0));
        opts_struct
            .fields
            .insert("bar".to_string(), Value::Num(22.0));
        set_workspace(&[("opts", Value::Struct(opts_struct))]);
        let dir = tempdir().unwrap();
        let path = dir.path().join("struct_subset.mat");
        let args = vec![
            Value::from(path.to_string_lossy().to_string()),
            Value::from("-struct"),
            Value::from("opts"),
            Value::from("bar"),
        ];
        save_builtin(args).unwrap();

        let file = File::open(&path).unwrap();
        let mat = matfile::MatFile::parse(file).unwrap();
        assert!(mat.find_by_name("foo").is_none());
        let array = mat.find_by_name("bar").unwrap();
        match array.data() {
            matfile::NumericData::Double { real, .. } => assert_eq!(real, &[22.0]),
            _ => panic!("expected double array"),
        }
    }

    #[test]
    fn save_missing_variable_errors() {
        ensure_test_resolver();
        set_workspace(&[]);
        let result = save_builtin(vec![Value::from("missing.mat"), Value::from("x")]);
        assert!(result.unwrap_err().contains("variable 'x'"));
    }

    #[test]
    fn save_regex_variable_selection() {
        ensure_test_resolver();
        set_workspace(&[
            ("alpha", Value::Num(1.0)),
            ("beta", Value::Num(2.0)),
            ("gamma", Value::Num(3.0)),
        ]);
        let dir = tempdir().unwrap();
        let path = dir.path().join("regex.mat");
        let args = vec![
            Value::from(path.to_string_lossy().to_string()),
            Value::from("-regexp"),
            Value::from("^a"),
            Value::from("ma$"),
        ];
        save_builtin(args).unwrap();

        let file = File::open(&path).unwrap();
        let mat = matfile::MatFile::parse(file).unwrap();
        assert!(mat.find_by_name("alpha").is_some());
        assert!(mat.find_by_name("gamma").is_some());
        assert!(mat.find_by_name("beta").is_none());
    }

    #[test]
    fn save_regex_requires_pattern() {
        ensure_test_resolver();
        set_workspace(&[("foo", Value::Num(1.0))]);
        let result = save_builtin(vec![Value::from("-regexp")]);
        assert!(result
            .unwrap_err()
            .contains("'-regexp' requires at least one pattern"));
    }

    #[test]
    fn save_unsupported_option_errors() {
        ensure_test_resolver();
        set_workspace(&[("foo", Value::Num(1.0))]);
        let result = save_builtin(vec![
            Value::from("text.mat"),
            Value::from("-ascii"),
            Value::from("foo"),
        ]);
        assert!(result.unwrap_err().contains("unsupported option '-ascii'"));
    }

    #[test]
    fn save_defaults_to_matlab_mat() {
        ensure_test_resolver();
        let _lock = lock_env_override();
        set_workspace(&[("answer", Value::Num(7.0))]);
        let dir = tempdir().unwrap();
        let target = dir.path().join("matlab_default.mat");
        let target_str = target.to_string_lossy().to_string();
        let _env = EnvOverride::set("RUNMAT_SAVE_DEFAULT_PATH", &target_str);
        save_builtin(Vec::new()).unwrap();

        assert!(
            target.exists(),
            "expected {} to be created",
            target.display()
        );
        let file = File::open(&target).unwrap();
        let mat = matfile::MatFile::parse(file).unwrap();
        let array = mat.find_by_name("answer").unwrap();
        match array.data() {
            matfile::NumericData::Double { real, .. } => assert_eq!(real, &[7.0]),
            _ => panic!("expected double array"),
        }
    }

    #[test]
    fn save_struct_without_filename_defaults_to_matlab_mat() {
        ensure_test_resolver();
        let _lock = lock_env_override();
        let mut payload_struct = StructValue::new();
        payload_struct
            .fields
            .insert("alpha".to_string(), Value::Num(3.0));
        set_workspace(&[("payload", Value::Struct(payload_struct))]);
        let dir = tempdir().unwrap();
        let target = dir.path().join("matlab_struct.mat");
        let target_str = target.to_string_lossy().to_string();
        let _env = EnvOverride::set("RUNMAT_SAVE_DEFAULT_PATH", &target_str);
        save_builtin(vec![Value::from("-struct"), Value::from("payload")]).unwrap();

        assert!(
            target.exists(),
            "expected {} to be created",
            target.display()
        );
        let file = File::open(&target).unwrap();
        let mat = matfile::MatFile::parse(file).unwrap();
        let field = mat.find_by_name("alpha").unwrap();
        match field.data() {
            matfile::NumericData::Double { real, .. } => assert_eq!(real, &[3.0]),
            _ => panic!("expected double array"),
        }
    }

    #[test]
    fn save_gpu_tensor_roundtrip() {
        ensure_test_resolver();
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload tensor");
            set_workspace(&[("gpu_data", Value::GpuTensor(handle.clone()))]);

            let dir = tempdir().unwrap();
            let path = dir.path().join("gpu_roundtrip.mat");
            save_builtin(vec![
                Value::from(path.to_string_lossy().to_string()),
                Value::from("gpu_data"),
            ])
            .unwrap();

            let file = File::open(&path).unwrap();
            let mat = matfile::MatFile::parse(file).unwrap();
            let array = mat.find_by_name("gpu_data").unwrap();
            match array.data() {
                matfile::NumericData::Double { real, imag } => {
                    assert_eq!(real, &tensor.data);
                    assert!(imag.is_none());
                }
                _ => panic!("expected double array"),
            }
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn save_wgpu_tensor_roundtrip() {
        ensure_test_resolver();
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            return;
        }
        let Some(provider) = runmat_accelerate_api::provider() else {
            return;
        };

        let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload tensor");
        set_workspace(&[("wgpu_tensor", Value::GpuTensor(handle.clone()))]);

        let dir = tempdir().unwrap();
        let path = dir.path().join("wgpu_roundtrip.mat");
        save_builtin(vec![
            Value::from(path.to_string_lossy().to_string()),
            Value::from("wgpu_tensor"),
        ])
        .unwrap();

        let file = File::open(&path).unwrap();
        let mat = matfile::MatFile::parse(file).unwrap();
        let array = mat.find_by_name("wgpu_tensor").unwrap();
        match array.data() {
            matfile::NumericData::Double { real, imag } => {
                assert_eq!(real, &tensor.data);
                assert!(imag.is_none());
            }
            _ => panic!("expected double array"),
        }
    }

    #[test]
    fn doc_examples_present() {
        use crate::builtins::common::test_support;
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
