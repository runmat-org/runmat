//! MATLAB-compatible `load` builtin for RunMat.

use std::collections::HashMap;
use std::io::{BufReader, Cursor, Read};
use std::path::{Path, PathBuf};

use regex::Regex;
use runmat_builtins::{
    CharArray, ComplexTensor, LogicalArray, StringArray, StructValue, Tensor, Value,
};
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
use crate::{gather_if_needed, make_cell};

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "load",
        builtin_path = "crate::builtins::io::mat::load"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "load"
category: "io/mat"
keywords: ["load", "mat", "workspace", "io", "matlab load", "regex load"]
summary: "Load variables from a MATLAB-compatible MAT-file into the workspace or return them as a struct."
references:
  - https://www.mathworks.com/help/matlab/ref/load.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Files are read on the host. When auto-offload is enabled, planners may later promote tensors to the GPU; no provider hooks are required at load time."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::mat::load::tests"
  integration:
    - "builtins::io::mat::load::tests::load_selected_variables"
    - "builtins::io::mat::load::tests::load_regex_selection"
---

# What does the `load` function do in MATLAB / RunMat?
`load` reads variables from a MAT-file (Level-5 layout) and brings them into the current workspace. Like MATLAB, it can either populate variables directly or return a struct containing the loaded data.

## How does the `load` function behave in MATLAB / RunMat?
- `load filename` reads every variable stored in `filename.mat` and assigns them into the caller's workspace. When no extension is supplied, `.mat` is appended automatically. Set `RUNMAT_LOAD_DEFAULT_PATH` to override the default `matlab.mat` target when no filename argument is provided.
- `S = load(filename)` loads the file but returns a struct instead of modifying the workspace. The struct fields mirror the variables stored in the MAT-file.
- `load(filename, 'A', 'B')` restricts the operation to the listed variable names. String scalars, char vectors, string arrays, or cell arrays of character vectors are accepted.
- `load(filename, '-regexp', '^foo', 'bar$')` selects variables whose names match any of the supplied regular expressions.
- Repeated names are deduplicated so that the last occurrence wins, mirroring MATLAB's behavior.
- Unsupported data classes trigger descriptive errors. RunMat currently supports double and complex numeric arrays, logical arrays, character arrays, string arrays (stored as cell-of-char data), structs, and cells whose elements are composed of the supported types.
- Files saved on platforms that produce little-endian Level-5 MAT-files (MATLAB's default) are supported. Big-endian and compressed (`miCOMPRESSED`) files currently report an error.

## `load` Function GPU Execution Behaviour
`load` always reads data on the host. The resulting values start on the CPU. When RunMat Accelerate is active, auto-offload heuristics may later decide to promote tensors to the GPU if they participate in accelerated expressions, but no provider hooks are required during the `load` operation itself. GPU-resident variables that were saved earlier are gathered back to host memory as part of file serialisation, so loading them produces standard host values.

## Examples of using the `load` function in MATLAB / RunMat

### Load the entire file into the workspace
```matlab
load('results.mat');
disp(norm(weights));
```
Expected outcome: every variable contained in `results.mat` becomes available in the caller's workspace.

### Load a subset of variables by name
```matlab
load('sim_state.mat', 'state', 'time');
plot(time, state);
```
Only `state` and `time` are created; other variables in the file are ignored.

### Load variables using regular expressions
```matlab
load('checkpoint.mat', '-regexp', '^layer_\\d+$');
```
All variables whose names look like `layer_0`, `layer_1`, … are loaded.

### Capture loaded variables in a struct without altering the workspace
```matlab
S = load('snapshot.mat');
disp(fieldnames(S));
```
`S` contains one field per variable stored in `snapshot.mat`, leaving the workspace untouched.

### Combine explicit names and regex filters
```matlab
model = load('model.mat', 'config', '-regexp', '^weights_(conv|fc)');
```
The returned struct includes the `config` variable and every weight matrix whose name matches either `weights_conv` or `weights_fc`.

### Honour a custom default filename
```matlab
setenv('RUNMAT_LOAD_DEFAULT_PATH', fullfile(tempdir, 'autosave.mat'));
load();
```
With no arguments, `load` falls back to the file specified by `RUNMAT_LOAD_DEFAULT_PATH`.

### Load character and string data
```matlab
values = load('strings.mat', 'labels');
disp(values.labels(1));
```
String arrays saved by RunMat are reconstructed faithfully from the underlying MAT-file representation.

## GPU residency in RunMat (Do I need `gpuArray`?)
No manual action is required. `load` always creates host values. When the auto-offload planner decides that downstream computations benefit from GPU execution, it will promote tensors automatically. You can still call `gpuArray` on loaded variables explicitly if you want to pin them to the device immediately.

## FAQ

### Does `load` support ASCII text files?
No. RunMat (like MATLAB) restricts the `load` builtin in modern releases to MAT-files. Text and delimited files should be read using `readmatrix`, `readtable`, or other file I/O utilities such as `fileread`.

### How are structures handled?
Structure scalars are reconstructed as `struct` values whose fields match the MAT-file content. Nested structs, cells, logical arrays, and numeric data are all supported.

### Will `load` overwrite existing variables?
Yes. When you call `load` without capturing the output struct, any variables with matching names in the caller's workspace are overwritten with the values from the MAT-file.

### What happens if a requested variable is missing?
RunMat raises a descriptive error: `load: variable 'foo' was not found in the file`. This mirrors MATLAB's behavior.

### Can I load into a different workspace?
Use MATLAB-compatible functions such as `assignin` (when available) if you need to populate a different scope explicitly. The `load` builtin itself targets the caller workspace by default.

### How are GPU arrays handled?
GPU-resident values are serialised to host data when saved. Loading the resulting MAT-file produces standard host arrays. Downstream acceleration is handled automatically by RunMat Accelerate.

### How do I detect which variables were loaded?
Use the struct form: `info = load(filename);` and then inspect `fieldnames(info)` or `isfield` to programmatically check what was present in the MAT-file.

## See Also
[save](./save), [who](../../introspection/who), [fileread](../filetext/fileread), [matfile](https://www.mathworks.com/help/matlab/ref/matfile.html)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/io/mat/load.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/mat/load.rs)
- Found a bug? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::mat::load")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "load",
    op_kind: GpuOpKind::Custom("io-load"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Reads MAT-files on the host and produces CPU-resident values. Providers are not involved until accelerated code later promotes the results.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::mat::load")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "load",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "File I/O is not eligible for fusion. Registration exists for documentation completeness only.",
};

#[runtime_builtin(
    name = "load",
    category = "io/mat",
    summary = "Load variables from a MAT-file.",
    keywords = "load,mat,workspace",
    accel = "cpu",
    sink = true,
    builtin_path = "crate::builtins::io::mat::load"
)]
fn load_builtin(args: Vec<Value>) -> Result<Value, String> {
    let eval = evaluate(&args)?;
    Ok(eval.first_output())
}

#[derive(Clone, Debug)]
pub struct LoadEval {
    variables: Vec<(String, Value)>,
}

impl LoadEval {
    pub fn first_output(&self) -> Value {
        let mut st = StructValue::new();
        for (name, value) in &self.variables {
            st.fields.insert(name.clone(), value.clone());
        }
        Value::Struct(st)
    }

    pub fn variables(&self) -> &[(String, Value)] {
        &self.variables
    }

    pub fn into_variables(self) -> Vec<(String, Value)> {
        self.variables
    }
}

struct LoadRequest {
    variables: Vec<String>,
    regex_patterns: Vec<Regex>,
}

pub fn evaluate(args: &[Value]) -> Result<LoadEval, String> {
    let mut host_args = Vec::with_capacity(args.len());
    for arg in args {
        host_args.push(gather_if_needed(arg)?);
    }

    let invocation = parse_invocation(&host_args)?;

    let mut path_value = if let Some(path) = invocation.path_value {
        path
    } else {
        Value::from("matlab.mat")
    };

    if invocation.path_was_default {
        if let Ok(override_path) = std::env::var("RUNMAT_LOAD_DEFAULT_PATH") {
            path_value = Value::from(override_path);
        }
    }

    let mut regex_patterns = Vec::with_capacity(invocation.regex_tokens.len());
    for pattern in invocation.regex_tokens {
        let regex = Regex::new(&pattern)
            .map_err(|err| format!("load: invalid regular expression '{pattern}': {err}"))?;
        regex_patterns.push(regex);
    }

    let request = LoadRequest {
        variables: invocation.variables,
        regex_patterns,
    };
    let path = normalise_path(&path_value)?;
    let entries = read_mat_file(&path)?;

    let selected = select_variables(&entries, &request)?;
    Ok(LoadEval {
        variables: selected,
    })
}

struct ParsedInvocation {
    path_value: Option<Value>,
    path_was_default: bool,
    variables: Vec<String>,
    regex_tokens: Vec<String>,
}

fn parse_invocation(values: &[Value]) -> Result<ParsedInvocation, String> {
    let mut path_value = None;
    let mut path_was_default = false;
    let mut variables = Vec::new();
    let mut regex_tokens = Vec::new();
    let mut idx = 0usize;
    while idx < values.len() {
        if let Some(flag) = option_token(&values[idx])? {
            match flag.as_str() {
                "-mat" => {
                    idx += 1;
                    continue;
                }
                "-regexp" => {
                    idx += 1;
                    if idx >= values.len() {
                        return Err("load: '-regexp' requires at least one pattern".to_string());
                    }
                    while idx < values.len() {
                        if option_token(&values[idx])?.is_some() {
                            break;
                        }
                        let names = extract_names(&values[idx])?;
                        if names.is_empty() {
                            return Err(
                                "load: '-regexp' requires non-empty pattern strings".to_string()
                            );
                        }
                        regex_tokens.extend(names);
                        idx += 1;
                    }
                    continue;
                }
                other => {
                    return Err(format!("load: unsupported option '{other}'"));
                }
            }
        } else {
            if path_value.is_none() {
                path_value = Some(values[idx].clone());
                idx += 1;
                continue;
            }
            let names = extract_names(&values[idx])?;
            variables.extend(names);
            idx += 1;
        }
    }

    if path_value.is_none() {
        path_was_default = true;
    }

    Ok(ParsedInvocation {
        path_value,
        path_was_default,
        variables,
        regex_tokens,
    })
}

fn normalise_path(value: &Value) -> Result<PathBuf, String> {
    let raw = value_to_string_scalar(value)
        .ok_or_else(|| "load: filename must be a character vector or string scalar".to_string())?;
    let mut path = PathBuf::from(raw);
    if path.extension().is_none() {
        path.set_extension("mat");
    }
    Ok(path)
}

fn select_variables(
    entries: &[(String, Value)],
    request: &LoadRequest,
) -> Result<Vec<(String, Value)>, String> {
    if request.variables.is_empty() && request.regex_patterns.is_empty() {
        return Ok(entries.to_vec());
    }

    let mut by_name: HashMap<&str, &Value> = HashMap::with_capacity(entries.len());
    for (name, value) in entries {
        by_name.insert(name, value);
    }

    let mut selected = Vec::new();

    for name in &request.variables {
        let value = by_name
            .get(name.as_str())
            .ok_or_else(|| format!("load: variable '{name}' was not found in the file"))?;
        insert_or_replace(&mut selected, name, (*value).clone());
    }

    if !request.regex_patterns.is_empty() {
        let mut matched = 0usize;
        for (name, value) in entries {
            if request
                .regex_patterns
                .iter()
                .any(|regex| regex.is_match(name))
            {
                matched += 1;
                insert_or_replace(&mut selected, name, value.clone());
            }
        }
        if matched == 0 && request.variables.is_empty() {
            return Err("load: no variables matched '-regexp' patterns".to_string());
        }
    }

    if selected.is_empty() {
        return Err("load: no variables selected".to_string());
    }

    Ok(selected)
}

fn insert_or_replace(selected: &mut Vec<(String, Value)>, name: &str, value: Value) {
    if let Some(entry) = selected.iter_mut().find(|(existing, _)| existing == name) {
        entry.1 = value;
    } else {
        selected.push((name.to_string(), value));
    }
}

pub(crate) fn read_mat_file(path: &Path) -> Result<Vec<(String, Value)>, String> {
    let file = File::open(path)
        .map_err(|err| format!("load: failed to open '{}': {err}", path.display()))?;
    let mut reader = BufReader::new(file);

    let mut header = [0u8; MAT_HEADER_LEN];
    reader
        .read_exact(&mut header)
        .map_err(|err| format!("load: failed to read MAT-file header: {err}"))?;
    if header[126] != b'I' || header[127] != b'M' {
        return Err("load: file is not a MATLAB Level-5 MAT-file".to_string());
    }

    let mut variables = Vec::new();
    while let Some(tagged) = read_tagged(&mut reader, true)? {
        if tagged.data_type != MI_MATRIX {
            continue;
        }
        let parsed = parse_matrix(&tagged.data)?;
        let value = mat_array_to_value(parsed.array)?;
        variables.push((parsed.name, value));
    }
    Ok(variables)
}

struct ParsedMatrix {
    name: String,
    array: MatArray,
}

fn parse_matrix(buffer: &[u8]) -> Result<ParsedMatrix, String> {
    let mut cursor = Cursor::new(buffer);

    let flags = read_tagged(&mut cursor, false)?
        .ok_or_else(|| "load: matrix element missing array flags".to_string())?;
    if flags.data_type != MI_UINT32 || flags.data.len() < 8 {
        return Err("load: invalid array flags block".to_string());
    }
    let flags0 = u32::from_le_bytes(flags.data[0..4].try_into().unwrap());
    let class_code = flags0 & 0xFF;
    let mut class = MatClass::from_class_code(class_code)
        .ok_or_else(|| "load: unsupported MATLAB class".to_string())?;
    let is_logical = (flags0 & FLAG_LOGICAL) != 0;
    let has_imag = (flags0 & FLAG_COMPLEX) != 0;
    if matches!(class, MatClass::Double) && is_logical {
        class = MatClass::Logical;
    }

    let dims_elem = read_tagged(&mut cursor, false)?
        .ok_or_else(|| "load: matrix element missing dimensions".to_string())?;
    if dims_elem.data_type != MI_INT32 {
        return Err("load: dimension block must use MI_INT32".to_string());
    }
    if dims_elem.data.is_empty() || dims_elem.data.len() % 4 != 0 {
        return Err("load: malformed dimension block".to_string());
    }
    let mut dims = Vec::with_capacity(dims_elem.data.len() / 4);
    for chunk in dims_elem.data.chunks_exact(4) {
        let value = i32::from_le_bytes(chunk.try_into().unwrap());
        if value < 0 {
            return Err("load: negative dimensions are not supported".to_string());
        }
        dims.push(value as usize);
    }
    if dims.is_empty() {
        dims.push(1);
        dims.push(1);
    }

    let name_elem = read_tagged(&mut cursor, false)?
        .ok_or_else(|| "load: matrix element missing name".to_string())?;
    let name = match name_elem.data_type {
        MI_INT8 | MI_UINT8 => bytes_to_string(&name_elem.data),
        MI_UINT16 => {
            let mut bytes = Vec::with_capacity(name_elem.data.len());
            for chunk in name_elem.data.chunks_exact(2) {
                let code = u16::from_le_bytes(chunk.try_into().unwrap());
                if code == 0 {
                    break;
                }
                if let Some(ch) = char::from_u32(code as u32) {
                    bytes.push(ch);
                }
            }
            bytes.into_iter().collect()
        }
        _ => {
            return Err("load: unsupported array name encoding".to_string());
        }
    };

    let array = match class {
        MatClass::Double => parse_double_array(&mut cursor, dims, has_imag)?,
        MatClass::Logical => parse_logical_array(&mut cursor, dims)?,
        MatClass::Char => parse_char_array(&mut cursor, dims)?,
        MatClass::Cell => parse_cell_array(&mut cursor, dims)?,
        MatClass::Struct => parse_struct(&mut cursor, dims)?,
    };

    Ok(ParsedMatrix { name, array })
}

fn parse_double_array(
    cursor: &mut Cursor<&[u8]>,
    dims: Vec<usize>,
    has_imag: bool,
) -> Result<MatArray, String> {
    let real_elem = read_tagged(cursor, false)?
        .ok_or_else(|| "load: numeric array missing real component".to_string())?;
    if real_elem.data_type != MI_DOUBLE || real_elem.data.len() % 8 != 0 {
        return Err("load: numeric data must be stored as MI_DOUBLE".to_string());
    }
    let mut real = Vec::with_capacity(real_elem.data.len() / 8);
    for chunk in real_elem.data.chunks_exact(8) {
        real.push(f64::from_le_bytes(chunk.try_into().unwrap()));
    }

    let imag = if has_imag {
        let imag_elem = read_tagged(cursor, false)?
            .ok_or_else(|| "load: numeric array missing imaginary component".to_string())?;
        if imag_elem.data_type != MI_DOUBLE || imag_elem.data.len() % 8 != 0 {
            return Err("load: imaginary component must be MI_DOUBLE".to_string());
        }
        let mut imag = Vec::with_capacity(imag_elem.data.len() / 8);
        for chunk in imag_elem.data.chunks_exact(8) {
            imag.push(f64::from_le_bytes(chunk.try_into().unwrap()));
        }
        Some(imag)
    } else {
        None
    };

    Ok(MatArray {
        class: MatClass::Double,
        dims,
        data: MatData::Double { real, imag },
    })
}

fn parse_logical_array(cursor: &mut Cursor<&[u8]>, dims: Vec<usize>) -> Result<MatArray, String> {
    let elem = read_tagged(cursor, false)?
        .ok_or_else(|| "load: logical array missing data block".to_string())?;
    if elem.data_type != MI_UINT8 {
        return Err("load: logical arrays must be stored as MI_UINT8".to_string());
    }
    Ok(MatArray {
        class: MatClass::Logical,
        dims,
        data: MatData::Logical { data: elem.data },
    })
}

fn parse_char_array(cursor: &mut Cursor<&[u8]>, dims: Vec<usize>) -> Result<MatArray, String> {
    let elem = read_tagged(cursor, false)?
        .ok_or_else(|| "load: character array missing data block".to_string())?;
    if elem.data_type != MI_UINT16 {
        return Err("load: character data must be stored as MI_UINT16".to_string());
    }
    if elem.data.len() % 2 != 0 {
        return Err("load: malformed character data".to_string());
    }
    let mut data = Vec::with_capacity(elem.data.len() / 2);
    for chunk in elem.data.chunks_exact(2) {
        data.push(u16::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(MatArray {
        class: MatClass::Char,
        dims,
        data: MatData::Char { data },
    })
}

fn parse_cell_array(cursor: &mut Cursor<&[u8]>, dims: Vec<usize>) -> Result<MatArray, String> {
    let total: usize = dims
        .iter()
        .copied()
        .fold(1usize, |acc, d| acc.saturating_mul(d));
    let mut elements = Vec::with_capacity(total);
    for _ in 0..total {
        let elem = read_tagged(cursor, false)?
            .ok_or_else(|| "load: cell element missing matrix payload".to_string())?;
        if elem.data_type != MI_MATRIX {
            return Err("load: cell elements must be matrices".to_string());
        }
        let parsed = parse_matrix(&elem.data)?;
        elements.push(parsed.array);
    }
    Ok(MatArray {
        class: MatClass::Cell,
        dims,
        data: MatData::Cell { elements },
    })
}

fn parse_struct(cursor: &mut Cursor<&[u8]>, dims: Vec<usize>) -> Result<MatArray, String> {
    if dims.len() != 2 || dims[0] != 1 || dims[1] != 1 {
        return Err("load: struct arrays are not supported yet".to_string());
    }
    let len_elem = read_tagged(cursor, false)?
        .ok_or_else(|| "load: struct missing maximum field length specifier".to_string())?;
    if len_elem.data_type != MI_INT32 || len_elem.data.len() != 4 {
        return Err("load: struct field length must be MI_INT32".to_string());
    }
    let max_len = i32::from_le_bytes(len_elem.data[..4].try_into().unwrap());
    if max_len <= 0 {
        return Err("load: struct field length must be positive".to_string());
    }

    let names_elem = read_tagged(cursor, false)?
        .ok_or_else(|| "load: struct missing field name table".to_string())?;
    if names_elem.data_type != MI_INT8 && names_elem.data_type != MI_UINT8 {
        return Err("load: struct field names must be stored as MI_INT8/MI_UINT8".to_string());
    }
    if names_elem.data.len() % (max_len as usize) != 0 {
        return Err("load: malformed struct field name table".to_string());
    }
    let field_count = names_elem.data.len() / (max_len as usize);
    let mut field_names = Vec::with_capacity(field_count);
    for i in 0..field_count {
        let start = i * (max_len as usize);
        let end = start + (max_len as usize);
        let slice = &names_elem.data[start..end];
        field_names.push(bytes_to_string(slice));
    }

    let mut field_values = Vec::with_capacity(field_count);
    for _ in 0..field_count {
        let elem = read_tagged(cursor, false)?
            .ok_or_else(|| "load: struct field missing matrix payload".to_string())?;
        if elem.data_type != MI_MATRIX {
            return Err("load: struct fields must be matrices".to_string());
        }
        let parsed = parse_matrix(&elem.data)?;
        field_values.push(parsed.array);
    }

    Ok(MatArray {
        class: MatClass::Struct,
        dims,
        data: MatData::Struct {
            field_names,
            field_values,
        },
    })
}

fn mat_array_to_value(array: MatArray) -> Result<Value, String> {
    match array.data {
        MatData::Double { real, imag } => {
            let len = real.len();
            if let Some(imag) = imag {
                if imag.len() != len {
                    return Err("load: complex data has mismatched real/imag parts".to_string());
                }
                if len == 1 {
                    Ok(Value::Complex(real[0], imag[0]))
                } else {
                    let mut pairs = Vec::with_capacity(len);
                    for i in 0..len {
                        pairs.push((real[i], imag[i]));
                    }
                    let tensor = ComplexTensor::new(pairs, array.dims.clone())
                        .map_err(|e| format!("load: {e}"))?;
                    Ok(Value::ComplexTensor(tensor))
                }
            } else if len == 1 {
                Ok(Value::Num(real[0]))
            } else {
                let tensor =
                    Tensor::new(real, array.dims.clone()).map_err(|e| format!("load: {e}"))?;
                Ok(Value::Tensor(tensor))
            }
        }
        MatData::Logical { data } => {
            let total: usize = array
                .dims
                .iter()
                .copied()
                .fold(1usize, |acc, d| acc.saturating_mul(d));
            if data.len() != total {
                return Err("load: logical data length mismatch".to_string());
            }
            if total == 1 {
                Ok(Value::Bool(data.first().copied().unwrap_or(0) != 0))
            } else {
                let logical = LogicalArray::new(data, array.dims.clone())
                    .map_err(|e| format!("load: {e}"))?;
                Ok(Value::LogicalArray(logical))
            }
        }
        MatData::Char { data } => {
            let rows = array.dims.first().copied().unwrap_or(1);
            let cols = array.dims.get(1).copied().unwrap_or(1);
            let mut chars = Vec::with_capacity(rows.saturating_mul(cols));
            for code in data {
                let ch = char::from_u32(code as u32).unwrap_or('\u{FFFD}');
                chars.push(ch);
            }
            let char_array = CharArray::new(chars, rows, cols).map_err(|e| format!("load: {e}"))?;
            Ok(Value::CharArray(char_array))
        }
        MatData::Cell { elements } => {
            if let Some(strings) = cell_elements_to_strings(&elements) {
                let string_array = StringArray::new(strings, array.dims.clone())
                    .map_err(|e| format!("load: {e}"))?;
                return Ok(Value::StringArray(string_array));
            }
            if array.dims.len() != 2 {
                return Err(
                    "load: cell arrays with more than two dimensions are not supported yet"
                        .to_string(),
                );
            }
            let rows = array.dims[0];
            let cols = array.dims[1];
            let expected = rows.saturating_mul(cols);
            if elements.len() != expected {
                return Err("load: cell array element count mismatch".to_string());
            }
            let mut converted = Vec::with_capacity(elements.len());
            for elem in elements {
                converted.push(mat_array_to_value(elem)?);
            }
            let mut row_major = vec![Value::Num(0.0); expected];
            for col in 0..cols {
                for row in 0..rows {
                    let cm_idx = col * rows + row;
                    let rm_idx = row * cols + col;
                    row_major[rm_idx] = converted[cm_idx].clone();
                }
            }
            make_cell(row_major, rows, cols)
        }
        MatData::Struct {
            field_names,
            field_values,
        } => {
            if field_names.len() != field_values.len() {
                return Err("load: struct field metadata is inconsistent".to_string());
            }
            let mut st = StructValue::new();
            for (name, value) in field_names.into_iter().zip(field_values.into_iter()) {
                let converted = mat_array_to_value(value)?;
                st.fields.insert(name, converted);
            }
            Ok(Value::Struct(st))
        }
    }
}

fn cell_elements_to_strings(elements: &[MatArray]) -> Option<Vec<String>> {
    let mut strings = Vec::with_capacity(elements.len());
    for element in elements {
        if element.class != MatClass::Char {
            return None;
        }
        let rows = element.dims.first().copied().unwrap_or(1);
        if rows > 1 {
            return None;
        }
        match &element.data {
            MatData::Char { data } => strings.push(utf16_codes_to_string(data)),
            _ => return None,
        }
    }
    Some(strings)
}

fn utf16_codes_to_string(data: &[u16]) -> String {
    let mut chars: Vec<char> = data
        .iter()
        .map(|code| char::from_u32(*code as u32).unwrap_or('\u{FFFD}'))
        .collect();
    while matches!(chars.last(), Some(&'\0')) {
        chars.pop();
    }
    chars.into_iter().collect()
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
        Value::CharArray(ca) => Ok(char_array_rows_as_strings(ca)),
        Value::StringArray(sa) => Ok(sa.data.clone()),
        Value::Cell(ca) => {
            let mut names = Vec::with_capacity(ca.data.len());
            for handle in &ca.data {
                let inner = unsafe { &*handle.as_raw() };
                let text = value_to_string_scalar(inner).ok_or_else(|| {
                    "load: cell arrays used for variable selection must contain string scalars"
                        .to_string()
                })?;
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

fn char_array_rows_as_strings(ca: &CharArray) -> Vec<String> {
    let mut rows = Vec::with_capacity(ca.rows);
    for r in 0..ca.rows {
        let mut row = String::with_capacity(ca.cols);
        for c in 0..ca.cols {
            let idx = r * ca.cols + c;
            row.push(ca.data[idx]);
        }
        let trimmed = row.trim_end_matches([' ', '\0']).to_string();
        rows.push(trimmed);
    }
    rows
}

fn bytes_to_string(bytes: &[u8]) -> String {
    let trimmed = bytes
        .iter()
        .copied()
        .take_while(|b| *b != 0)
        .collect::<Vec<u8>>();
    String::from_utf8(trimmed).unwrap_or_default()
}

struct TaggedData {
    data_type: u32,
    data: Vec<u8>,
}

fn read_tagged<R: Read>(reader: &mut R, allow_eof: bool) -> Result<Option<TaggedData>, String> {
    let mut type_bytes = [0u8; 4];
    match reader.read_exact(&mut type_bytes) {
        Ok(()) => {}
        Err(err) => {
            if allow_eof && err.kind() == std::io::ErrorKind::UnexpectedEof {
                return Ok(None);
            }
            return Err(format!("load: failed to read MAT element header: {err}"));
        }
    }

    let type_field = u32::from_le_bytes(type_bytes);
    if (type_field & 0xFFFF0000) != 0 {
        let data_type = type_field & 0x0000FFFF;
        let num_bytes = ((type_field & 0xFFFF0000) >> 16) as usize;
        let mut inline = [0u8; 4];
        reader
            .read_exact(&mut inline)
            .map_err(|err| format!("load: failed to read compact MAT element: {err}"))?;
        let mut data = inline[..num_bytes.min(4)].to_vec();
        data.truncate(num_bytes.min(4));
        Ok(Some(TaggedData { data_type, data }))
    } else {
        let mut len_bytes = [0u8; 4];
        reader
            .read_exact(&mut len_bytes)
            .map_err(|err| format!("load: failed to read MAT element length: {err}"))?;
        let length = u32::from_le_bytes(len_bytes) as usize;
        let mut data = vec![0u8; length];
        reader
            .read_exact(&mut data)
            .map_err(|err| format!("load: failed to read MAT element body: {err}"))?;
        let padding = (8 - (length % 8)) % 8;
        if padding != 0 {
            let mut pad = vec![0u8; padding];
            reader
                .read_exact(&mut pad)
                .map_err(|err| format!("load: failed to read MAT padding: {err}"))?;
        }
        Ok(Some(TaggedData {
            data_type: type_field,
            data,
        }))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::workspace::WorkspaceResolver;
    use once_cell::sync::OnceCell;
    use runmat_builtins::StringArray;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use tempfile::tempdir;

    thread_local! {
        static TEST_WORKSPACE: RefCell<HashMap<String, Value>> = RefCell::new(HashMap::new());
    }

    fn ensure_test_resolver() {
        static INIT: OnceCell<()> = OnceCell::new();
        INIT.get_or_init(|| {
            crate::workspace::register_workspace_resolver(WorkspaceResolver {
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
            let mut map = slot.borrow_mut();
            map.clear();
            for (name, value) in entries {
                map.insert((*name).to_string(), value.clone());
            }
        });
    }

    #[test]
    fn load_roundtrip_numeric() {
        ensure_test_resolver();
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).unwrap();
        set_workspace(&[("A", Value::Tensor(tensor))]);

        let dir = tempdir().unwrap();
        let path = dir.path().join("numeric.mat");
        let save_arg = Value::from(path.to_string_lossy().to_string());
        crate::call_builtin("save", std::slice::from_ref(&save_arg)).unwrap();

        let eval =
            evaluate(&[Value::from(path.to_string_lossy().to_string())]).expect("load numeric");
        let struct_value = eval.first_output();
        match struct_value {
            Value::Struct(sv) => {
                assert!(sv.fields.contains_key("A"));
                match sv.fields.get("A").unwrap() {
                    Value::Tensor(t) => {
                        assert_eq!(t.shape, vec![2, 2]);
                        assert_eq!(t.data, vec![1.0, 4.0, 2.0, 5.0]);
                    }
                    other => panic!("expected tensor, got {other:?}"),
                }
            }
            other => panic!("expected struct, got {other:?}"),
        }
    }

    #[test]
    fn load_selected_variables() {
        ensure_test_resolver();
        set_workspace(&[("signal", Value::Num(42.0)), ("noise", Value::Num(5.0))]);
        let dir = tempdir().unwrap();
        let path = dir.path().join("selection.mat");
        let save_arg = Value::from(path.to_string_lossy().to_string());
        crate::call_builtin("save", std::slice::from_ref(&save_arg)).unwrap();

        let eval = evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("signal"),
        ])
        .expect("load selection");
        let vars = eval.variables();
        assert_eq!(vars.len(), 1);
        assert_eq!(vars[0].0, "signal");
        assert!(matches!(vars[0].1, Value::Num(42.0)));
    }

    #[test]
    fn load_regex_selection() {
        ensure_test_resolver();
        set_workspace(&[
            ("w1", Value::Num(1.0)),
            ("w2", Value::Num(2.0)),
            ("bias", Value::Num(3.0)),
        ]);
        let dir = tempdir().unwrap();
        let path = dir.path().join("regex.mat");
        let save_arg = Value::from(path.to_string_lossy().to_string());
        crate::call_builtin("save", std::slice::from_ref(&save_arg)).unwrap();

        let eval = evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("-regexp"),
            Value::from("^w\\d$"),
        ])
        .expect("load regex");
        let mut names: Vec<_> = eval.variables().iter().map(|(n, _)| n.clone()).collect();
        names.sort();
        assert_eq!(names, vec!["w1".to_string(), "w2".to_string()]);
    }

    #[test]
    fn load_missing_variable_errors() {
        ensure_test_resolver();
        set_workspace(&[("existing", Value::Num(7.0))]);
        let dir = tempdir().unwrap();
        let path = dir.path().join("missing.mat");
        let save_arg = Value::from(path.to_string_lossy().to_string());
        crate::call_builtin("save", std::slice::from_ref(&save_arg)).unwrap();

        let err = evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("missing"),
        ])
        .expect_err("expect missing variable error");
        assert!(err.contains("variable 'missing' was not found"));
    }

    #[test]
    fn load_string_array_roundtrip() {
        ensure_test_resolver();
        let strings = StringArray::new(vec!["foo".into(), "bar".into()], vec![1, 2]).unwrap();
        set_workspace(&[("labels", Value::StringArray(strings))]);
        let dir = tempdir().unwrap();
        let path = dir.path().join("strings.mat");
        let save_arg = Value::from(path.to_string_lossy().to_string());
        crate::call_builtin("save", std::slice::from_ref(&save_arg)).unwrap();

        let eval =
            evaluate(&[Value::from(path.to_string_lossy().to_string())]).expect("load strings");
        let struct_value = eval.first_output();
        match struct_value {
            Value::Struct(sv) => {
                let value = sv
                    .fields
                    .get("labels")
                    .expect("labels field missing in struct");
                match value {
                    Value::StringArray(sa) => {
                        assert_eq!(sa.shape, vec![1, 2]);
                        assert_eq!(sa.data, vec![String::from("foo"), String::from("bar")]);
                    }
                    other => panic!("expected string array, got {other:?}"),
                }
            }
            other => panic!("expected struct, got {other:?}"),
        }
    }

    #[test]
    fn load_option_before_filename() {
        ensure_test_resolver();
        set_workspace(&[("alpha", Value::Num(1.0)), ("beta", Value::Num(2.0))]);
        let dir = tempdir().unwrap();
        let path = dir.path().join("option_first.mat");
        let save_arg = Value::from(path.to_string_lossy().to_string());
        crate::call_builtin("save", std::slice::from_ref(&save_arg)).unwrap();

        let eval = evaluate(&[
            Value::from("-mat"),
            Value::from(path.to_string_lossy().to_string()),
            Value::from("beta"),
        ])
        .expect("load with option first");
        let vars = eval.variables();
        assert_eq!(vars.len(), 1);
        assert_eq!(vars[0].0, "beta");
        assert!(matches!(vars[0].1, Value::Num(2.0)));
    }

    #[test]
    fn load_char_array_names_trimmed() {
        ensure_test_resolver();
        set_workspace(&[("short", Value::Num(5.0)), ("longer", Value::Num(9.0))]);
        let dir = tempdir().unwrap();
        let path = dir.path().join("char_names.mat");
        let save_arg = Value::from(path.to_string_lossy().to_string());
        crate::call_builtin("save", std::slice::from_ref(&save_arg)).unwrap();

        let cols = 6;
        let mut data = Vec::new();
        for name in ["short", "longer"] {
            let mut chars: Vec<char> = name.chars().collect();
            while chars.len() < cols {
                chars.push(' ');
            }
            data.extend(chars);
        }
        let name_array = CharArray::new(data, 2, cols).unwrap();

        let eval = evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::CharArray(name_array),
        ])
        .expect("load with char array names");
        let vars = eval.variables();
        assert_eq!(vars.len(), 2);
        assert_eq!(vars[0].0, "short");
        assert!(matches!(vars[0].1, Value::Num(5.0)));
        assert_eq!(vars[1].0, "longer");
        assert!(matches!(vars[1].1, Value::Num(9.0)));
    }

    #[test]
    fn load_duplicate_names_last_wins() {
        ensure_test_resolver();
        set_workspace(&[("dup", Value::Num(11.0))]);
        let dir = tempdir().unwrap();
        let path = dir.path().join("duplicates.mat");
        let save_arg = Value::from(path.to_string_lossy().to_string());
        crate::call_builtin("save", std::slice::from_ref(&save_arg)).unwrap();

        let eval = evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("dup"),
            Value::from("dup"),
        ])
        .expect("load with duplicate names");
        let vars = eval.variables();
        assert_eq!(vars.len(), 1);
        assert_eq!(vars[0].0, "dup");
        assert!(matches!(vars[0].1, Value::Num(11.0)));
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn load_wgpu_tensor_roundtrip() {
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

        use runmat_accelerate_api::HostTensorView;

        let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload tensor");
        set_workspace(&[("gpu_var", Value::GpuTensor(handle))]);

        let dir = tempdir().unwrap();
        let path = dir.path().join("wgpu_load.mat");
        let save_args = vec![
            Value::from(path.to_string_lossy().to_string()),
            Value::from("gpu_var"),
        ];
        crate::call_builtin("save", &save_args).unwrap();

        let eval =
            evaluate(&[Value::from(path.to_string_lossy().to_string())]).expect("load wgpu file");
        let struct_value = eval.first_output();
        match struct_value {
            Value::Struct(sv) => match sv.fields.get("gpu_var") {
                Some(Value::Tensor(t)) => {
                    assert_eq!(t.shape, vec![2, 2]);
                    assert_eq!(t.data, tensor.data);
                }
                other => panic!("expected tensor, got {other:?}"),
            },
            other => panic!("expected struct, got {other:?}"),
        }
    }

    #[test]
    fn doc_examples_present() {
        let blocks = crate::builtins::common::test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
