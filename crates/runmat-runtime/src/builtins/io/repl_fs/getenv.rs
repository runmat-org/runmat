//! MATLAB-compatible `getenv` builtin for RunMat.
//!
//! Mirrors MATLAB semantics for querying environment variables. Supports scalar character
//! vectors, string scalars, string arrays, and cell arrays of character vectors. Calling
//! `getenv` with no arguments returns a struct containing every environment variable visible to
//! the current process.

use std::env;

use runmat_builtins::{CharArray, StringArray, StructValue, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{gather_if_needed, make_cell};

const ERR_TOO_MANY_INPUTS: &str = "getenv: too many input arguments";
const ERR_INVALID_TYPE: &str = "getenv: NAME must be a character vector, string scalar, string array, or cell array of character vectors";
const ERR_CHAR_MATRIX_CELL: &str =
    "getenv: cell array elements must be character vectors or string scalars";

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "getenv",
        builtin_path = "crate::builtins::io::repl_fs::getenv"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "getenv"
category: "io/repl_fs"
keywords: ["getenv", "environment variable", "env", "system variable", "process environment"]
summary: "Query environment variables as character vectors, strings, or structures."
references:
  - https://www.mathworks.com/help/matlab/ref/getenv.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Host-only operation. RunMat gathers GPU-resident arguments before querying the process environment; providers do not implement hooks for this builtin."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::repl_fs::getenv::tests"
  integration: "builtins::io::repl_fs::getenv::tests::getenv_returns_struct_with_all_variables"
---

# What does the `getenv` function do in MATLAB / RunMat?
`getenv` reads environment variables that are visible to the current process. When you call
`getenv("NAME")`, the builtin returns the value of the environment variable `NAME`, matching MATLAB
behaviour. Supplying no input arguments returns a struct where each field corresponds to an
environment variable.

## How does the `getenv` function behave in MATLAB / RunMat?
- Accepts character vectors (`'PATH'`) or string scalars (`"PATH"`). Character vector inputs return
  character vectors; string inputs return string scalars.
- Vectorised calls are supported: pass a string array to receive a string array of values, or a cell
  array of character vectors to receive a cell array of character vectors.
- When the requested variable is not defined, `getenv` returns an empty character vector or an empty
  string scalar (depending on the input type). This matches MATLAB’s “variable not found” semantics.
- Calling `getenv` with no arguments returns a scalar struct. Field names are the environment
  variable names; field values are character vectors containing each variable’s value.
- On Windows, environment variable lookups are case-insensitive, mirroring the operating system.
  On Unix-like systems, lookups are case-sensitive.
- Trailing spaces in character matrix inputs are ignored so that padded rows created with MATLAB’s
  string manipulation functions remain compatible.
- The builtin rejects non-text inputs (numeric arrays, logicals, GPU tensors, etc.) with a clear
  MATLAB-style diagnostic.

## `getenv` Function GPU Execution Behaviour
`getenv` always runs on the CPU. If the argument originates from the GPU—for example, a string
scalar produced by an accelerated builtin—RunMat gathers it to host memory before reading the
environment. No GPU kernels are launched, and acceleration providers do not need to implement
hooks for this builtin.

## GPU residency in RunMat (Do I need `gpuArray`?)
No. `getenv` has no GPU implementation and simply queries the host environment. Passing GPU-resident
values is supported (RunMat gathers them automatically), but there is no benefit to calling
`gpuArray` yourself.

## Examples of using the `getenv` function in MATLAB / RunMat

### Read A Single Environment Variable
```matlab
homeFolder = getenv("HOME");
disp(homeFolder);
```
Expected output (macOS/Linux):
```matlab
homeFolder =
    "/Users/alex"
```
Expected output (Windows):
```matlab
homeFolder =
    "C:\Users\alex"
```

### Handle Missing Environment Variables
```matlab
token = getenv("RUNMAT_TOKEN");
if token == ""
    warning("Token not configured.");
end
```
Expected output:
```matlab
Warning: Token not configured.
```

### Fetch Multiple Variables With A String Array
```matlab
vars = getenv(["PATH", "SHELL"]);
disp(vars);
```
Expected output:
```matlab
vars = 1x2 string
    "/usr/local/bin:/usr/bin:..."    "/bin/zsh"
```

### Use A Cell Array Of Character Vectors
```matlab
cells = getenv({'HOME', 'PATH'});
disp(cells{1});
```
Expected output:
```matlab
cells =
  1x2 cell array
    {'/Users/alex'}    {'/usr/local/bin:/usr/bin:...'}
```

### Inspect All Environment Variables
```matlab
env = getenv();
disp(env.USER);
```
Expected output:
```matlab
env =
  struct with fields:
    HOME: '/Users/alex'
    PATH: '/usr/local/bin:/usr/bin:...'
    USER: 'alex'
```

### Cache The Environment For Repeated Access
```matlab
env = getenv();
resultsFolder = fullfile(env.HOME, "results");
```
Expected output:
```matlab
resultsFolder =
    "/Users/alex/results"
```

## FAQ

### What does `getenv` return when the variable is missing?
An empty character vector (`''`) for character inputs, or an empty string scalar (`""`) for string
inputs. You can test for a missing variable with `isempty(value)` or `value == ""`.

### Does `getenv` support vectorised inputs?
Yes. Pass a string array or a cell array of character vectors to retrieve multiple values in one
call. The output mirrors the input container type.

### Are environment variable lookups case-sensitive?
RunMat follows the operating system. Lookups are case-insensitive on Windows and case-sensitive on
Unix-like systems, matching MATLAB.

### Does `getenv` modify the environment?
No. It is a read-only operation. Use the forthcoming `setenv` builtin to modify the environment once it is available.

### Can I call `getenv` from GPU-accelerated code?
Yes. Inputs originating on the GPU are gathered automatically. The builtin still executes on the
CPU, and the output lives on the host.

### How do I access all environment variables at once?
Call `getenv()` with no arguments. The result is a struct whose fields contain character vectors for
each environment variable.

### What happens if an environment variable name is not a valid struct field?
RunMat preserves the exact name as a struct field. Access it via dynamic field references when the
name contains characters that cannot be used in dot notation.

### Are trailing spaces in character inputs significant?
No. Trailing spaces introduced by padding character matrices are stripped before the lookup so that
MATLAB-style character arrays work as expected.

### Does `getenv` include inherited environment variables?
Yes. The builtin reports every variable visible to the RunMat process, including inherited values.

## See Also
[tempdir](./tempdir), [tempname](./tempname), [pwd](./pwd), [path](./path)

## Source & Feedback
- Source: [`crates/runmat-runtime/src/builtins/io/repl_fs/getenv.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/getenv.rs)
- Issues: [Open a GitHub ticket](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::repl_fs::getenv")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "getenv",
    op_kind: GpuOpKind::Custom("io"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host environment query with no GPU participation; providers do not implement hooks.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::repl_fs::getenv")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "getenv",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Environment lookups break fusion graphs and always execute on the CPU.",
};

#[runtime_builtin(
    name = "getenv",
    category = "io/repl_fs",
    summary = "Query environment variables as character vectors, strings, or structures.",
    keywords = "getenv,environment variable,env,system variable,process environment",
    accel = "cpu",
    builtin_path = "crate::builtins::io::repl_fs::getenv"
)]
fn getenv_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    match args.len() {
        0 => Ok(getenv_all()),
        1 => {
            let gathered = gather_if_needed(&args[0]).map_err(|err| format!("getenv: {err}"))?;
            getenv_one(gathered)
        }
        _ => Err(((ERR_TOO_MANY_INPUTS.to_string())).into()),
    }
}

fn getenv_all() -> Value {
    let mut st = StructValue::new();
    for (name, value) in env::vars() {
        st.fields
            .insert(name, Value::CharArray(CharArray::new_row(&value)));
    }
    Value::Struct(st)
}

fn getenv_one(value: Value) -> Result<Value, String> {
    match value {
        Value::CharArray(array) => getenv_from_char_array(array),
        Value::String(s) => Ok(Value::String(read_env_string(&s))),
        Value::StringArray(sa) => getenv_from_string_array(sa),
        Value::Cell(ca) => getenv_from_cell_array(ca),
        _ => Err(ERR_INVALID_TYPE.to_string()),
    }
}

fn getenv_from_char_array(array: CharArray) -> Result<Value, String> {
    if array.rows == 0 {
        return Ok(Value::CharArray(
            CharArray::new(Vec::new(), 0, array.cols)
                .map_err(|e| format!("getenv: unable to construct empty character array ({e})"))?,
        ));
    }

    if array.rows == 1 {
        let name = char_row_to_string(&array, 0);
        let value = CharArray::new_row(&read_env_string(&name));
        return Ok(Value::CharArray(value));
    }

    let mut rows = Vec::with_capacity(array.rows);
    for row in 0..array.rows {
        rows.push(read_env_string(&char_row_to_string(&array, row)));
    }
    let result = char_array_from_rows(&rows)
        .map_err(|err| format!("getenv: unable to build character matrix ({err})"))?;
    Ok(Value::CharArray(result))
}

fn getenv_from_string_array(array: StringArray) -> Result<Value, String> {
    let mut resolved = Vec::with_capacity(array.data.len());
    for name in &array.data {
        resolved.push(read_env_string(name));
    }
    let result =
        StringArray::new(resolved, array.shape.clone()).map_err(|err| format!("getenv: {err}"))?;
    Ok(Value::StringArray(result))
}

fn getenv_from_cell_array(array: runmat_builtins::CellArray) -> Result<Value, String> {
    let mut values: Vec<Value> = Vec::with_capacity(array.data.len());
    for cell in &array.data {
        let gathered = gather_if_needed(cell).map_err(|err| format!("getenv: {err}"))?;
        let resolved = match gathered {
            Value::CharArray(ca) => {
                if ca.rows != 1 {
                    return Err(ERR_CHAR_MATRIX_CELL.to_string());
                }
                Value::CharArray(CharArray::new_row(&read_env_string(&char_row_to_string(
                    &ca, 0,
                ))))
            }
            Value::String(s) => Value::String(read_env_string(&s)),
            _ => return Err(ERR_CHAR_MATRIX_CELL.to_string()),
        };
        values.push(resolved);
    }
    make_cell(values, array.rows, array.cols).map_err(|err| format!("getenv: {err}"))
}

fn read_env_string(name: &str) -> String {
    env::var(name).unwrap_or_default()
}

fn char_row_to_string(array: &CharArray, row: usize) -> String {
    let mut text = String::with_capacity(array.cols);
    for col in 0..array.cols {
        text.push(array.data[row * array.cols + col]);
    }
    while text.ends_with(' ') {
        text.pop();
    }
    text
}

fn char_array_from_rows(rows: &[String]) -> Result<CharArray, String> {
    if rows.is_empty() {
        return CharArray::new(Vec::new(), 0, 0);
    }

    let max_cols = rows
        .iter()
        .map(|row| row.chars().count())
        .max()
        .unwrap_or(0);
    if max_cols == 0 {
        return CharArray::new(Vec::new(), rows.len(), 0);
    }

    let mut data = Vec::with_capacity(rows.len() * max_cols);
    for row in rows {
        let mut chars = row.chars();
        for _ in 0..max_cols {
            if let Some(ch) = chars.next() {
                data.push(ch);
            } else {
                data.push(' ');
            }
        }
    }
    CharArray::new(data, rows.len(), max_cols)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK;
    use runmat_builtins::{CharArray, StringArray, Value};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getenv_char_existing_variable() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        env::set_var("RUNMAT_TEST_GETENV_CHAR", "char-value");
        let input = Value::CharArray(CharArray::new_row("RUNMAT_TEST_GETENV_CHAR"));
        let result = getenv_builtin(vec![input]).expect("getenv");
        match result {
            Value::CharArray(ca) => {
                let text: String = ca.data.iter().collect();
                assert_eq!(text, "char-value");
            }
            other => panic!("expected CharArray result, got {other:?}"),
        }
        env::remove_var("RUNMAT_TEST_GETENV_CHAR");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getenv_string_missing_variable_returns_empty_string() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        env::remove_var("RUNMAT_TEST_GETENV_MISSING");
        let input = Value::String("RUNMAT_TEST_GETENV_MISSING".to_string());
        let result = getenv_builtin(vec![input]).expect("getenv");
        match result {
            Value::String(s) => assert!(s.is_empty()),
            other => panic!("expected string output, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getenv_string_array_preserves_shape() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        env::set_var("RUNMAT_TEST_GETENV_A", "alpha");
        env::set_var("RUNMAT_TEST_GETENV_B", "beta");
        let data = vec![
            "RUNMAT_TEST_GETENV_A".to_string(),
            "RUNMAT_TEST_GETENV_B".to_string(),
        ];
        let sa = StringArray::new(data, vec![1, 2]).expect("string array");
        let result = getenv_builtin(vec![Value::StringArray(sa)]).expect("getenv");
        match result {
            Value::StringArray(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(out.data, vec!["alpha".to_string(), "beta".to_string()]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
        env::remove_var("RUNMAT_TEST_GETENV_A");
        env::remove_var("RUNMAT_TEST_GETENV_B");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getenv_char_matrix_handles_multiple_rows() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        env::set_var("RUN1", "first");
        env::set_var("RUN2", "second-value");
        let names = CharArray::new(vec!['R', 'U', 'N', '1', 'R', 'U', 'N', '2'], 2, 4)
            .expect("char matrix");
        let result = getenv_builtin(vec![Value::CharArray(names)]).expect("getenv");
        match result {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 2);
                assert_eq!(out.cols, "second-value".chars().count());
                let first = char_row_to_string(&out, 0);
                let second = char_row_to_string(&out, 1);
                assert_eq!(first.trim_end(), "first");
                assert_eq!(second.trim_end(), "second-value");
            }
            other => panic!("expected char matrix, got {other:?}"),
        }
        env::remove_var("RUN1");
        env::remove_var("RUN2");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getenv_char_input_missing_variable_returns_empty_char_vector() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        env::remove_var("RUNMAT_TEST_GETENV_EMPTY_CHAR");
        let input = Value::CharArray(CharArray::new_row("RUNMAT_TEST_GETENV_EMPTY_CHAR"));
        let result = getenv_builtin(vec![input]).expect("getenv");
        match result {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 1);
                assert_eq!(out.cols, 0, "expected empty character vector");
            }
            other => panic!("expected CharArray, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getenv_char_matrix_trims_trailing_spaces() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        env::set_var("RUNMAT_TEST_TRIM1", "value1");
        env::set_var("RUNMAT_TEST_TRIM2", "value-two");
        let names = char_array_from_rows(&[
            format!("{: <24}", "RUNMAT_TEST_TRIM1"),
            "RUNMAT_TEST_TRIM2".to_string(),
        ])
        .expect("char array from rows");
        let result = getenv_builtin(vec![Value::CharArray(names)]).expect("getenv");
        match result {
            Value::CharArray(out) => {
                let first = char_row_to_string(&out, 0);
                let second = char_row_to_string(&out, 1);
                assert_eq!(first.trim_end(), "value1");
                assert_eq!(second.trim_end(), "value-two");
            }
            other => panic!("expected CharArray result, got {other:?}"),
        }
        env::remove_var("RUNMAT_TEST_TRIM1");
        env::remove_var("RUNMAT_TEST_TRIM2");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getenv_cell_array_preserves_element_types() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        env::set_var("RUNMAT_TEST_CELL1", "one");
        env::set_var("RUNMAT_TEST_CELL2", "two");
        let cell_input = make_cell(
            vec![
                Value::CharArray(CharArray::new_row("RUNMAT_TEST_CELL1")),
                Value::String("RUNMAT_TEST_CELL2".to_string()),
            ],
            1,
            2,
        )
        .expect("cell creation");
        let result = getenv_builtin(vec![cell_input]).expect("getenv");
        match result {
            Value::Cell(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 2);
                unsafe {
                    match &*ca.data[0].as_raw() {
                        Value::CharArray(first) => {
                            let text: String = first.data.iter().collect();
                            assert_eq!(text, "one");
                        }
                        other => panic!("expected char array in first cell, got {other:?}"),
                    }
                    match &*ca.data[1].as_raw() {
                        Value::String(s) => assert_eq!(s, "two"),
                        other => panic!("expected string in second cell, got {other:?}"),
                    }
                }
            }
            other => panic!("expected cell array, got {other:?}"),
        }
        env::remove_var("RUNMAT_TEST_CELL1");
        env::remove_var("RUNMAT_TEST_CELL2");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getenv_cell_array_rejects_invalid_entries() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let invalid_cell = make_cell(
            vec![
                Value::CharArray(CharArray::new_row("RUNMAT_TEST_CELL_INVALID")),
                Value::Num(42.0),
            ],
            1,
            2,
        )
        .expect("cell creation");
        let err = getenv_builtin(vec![invalid_cell]).expect_err("expected error");
        assert!(
            err.contains("cell array elements"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getenv_returns_struct_with_all_variables() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        env::set_var("RUNMAT_TEST_STRUCT", "struct-value");
        let result = getenv_builtin(Vec::new()).expect("getenv");
        match result {
            Value::Struct(sv) => {
                let value = sv
                    .fields
                    .get("RUNMAT_TEST_STRUCT")
                    .expect("struct field missing");
                match value {
                    Value::CharArray(ca) => {
                        let text: String = ca.data.iter().collect();
                        assert_eq!(text, "struct-value");
                    }
                    other => panic!("expected char array field, got {other:?}"),
                }
            }
            other => panic!("expected struct result, got {other:?}"),
        }
        env::remove_var("RUNMAT_TEST_STRUCT");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getenv_invalid_input_errors() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let err =
            getenv_builtin(vec![Value::Num(std::f64::consts::PI)]).expect_err("expected error");
        assert!(
            err.contains("NAME must be"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getenv_too_many_arguments_errors() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let err = getenv_builtin(vec![
            Value::String("PATH".to_string()),
            Value::String("HOME".to_string()),
        ])
        .expect_err("expected error");
        assert!(
            err.contains("too many input arguments"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let examples = crate::builtins::common::test_support::doc_examples(DOC_MD);
        assert!(!examples.is_empty());
    }
}
