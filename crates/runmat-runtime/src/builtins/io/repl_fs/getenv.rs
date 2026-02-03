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
use crate::{build_runtime_error, gather_if_needed_async, make_cell, BuiltinResult, RuntimeError};

const ERR_TOO_MANY_INPUTS: &str = "getenv: too many input arguments";
const ERR_INVALID_TYPE: &str = "getenv: NAME must be a character vector, string scalar, string array, or cell array of character vectors";
const ERR_CHAR_MATRIX_CELL: &str =
    "getenv: cell array elements must be character vectors or string scalars";

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

const BUILTIN_NAME: &str = "getenv";

fn getenv_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let identifier = err.identifier().map(str::to_string);
    let mut builder = build_runtime_error(format!("{BUILTIN_NAME}: {}", err.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "getenv",
    category = "io/repl_fs",
    summary = "Query environment variables as character vectors, strings, or structures.",
    keywords = "getenv,environment variable,env,system variable,process environment",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::getenv_type),
    builtin_path = "crate::builtins::io::repl_fs::getenv"
)]
async fn getenv_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    match args.len() {
        0 => Ok(getenv_all()),
        1 => {
            let gathered = gather_if_needed_async(&args[0])
                .await
                .map_err(map_control_flow)?;
            getenv_one(gathered).await
        }
        _ => Err(getenv_error(ERR_TOO_MANY_INPUTS)),
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

async fn getenv_one(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::CharArray(array) => getenv_from_char_array(array),
        Value::String(s) => Ok(Value::String(read_env_string(&s))),
        Value::StringArray(sa) => getenv_from_string_array(sa),
        Value::Cell(ca) => getenv_from_cell_array(ca).await,
        _ => Err(getenv_error(ERR_INVALID_TYPE)),
    }
}

fn getenv_from_char_array(array: CharArray) -> BuiltinResult<Value> {
    if array.rows == 0 {
        return Ok(Value::CharArray(
            CharArray::new(Vec::new(), 0, array.cols).map_err(|e| {
                getenv_error(format!(
                    "getenv: unable to construct empty character array ({e})"
                ))
            })?,
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
        .map_err(|err| getenv_error(format!("getenv: unable to build character matrix ({err})")))?;
    Ok(Value::CharArray(result))
}

fn getenv_from_string_array(array: StringArray) -> BuiltinResult<Value> {
    let mut resolved = Vec::with_capacity(array.data.len());
    for name in &array.data {
        resolved.push(read_env_string(name));
    }
    let result = StringArray::new(resolved, array.shape.clone())
        .map_err(|err| getenv_error(format!("getenv: {err}")))?;
    Ok(Value::StringArray(result))
}

async fn getenv_from_cell_array(array: runmat_builtins::CellArray) -> BuiltinResult<Value> {
    let mut values: Vec<Value> = Vec::with_capacity(array.data.len());
    for cell in &array.data {
        let gathered = gather_if_needed_async(cell)
            .await
            .map_err(map_control_flow)?;
        let resolved = match gathered {
            Value::CharArray(ca) => {
                if ca.rows != 1 {
                    return Err(getenv_error(ERR_CHAR_MATRIX_CELL));
                }
                Value::CharArray(CharArray::new_row(&read_env_string(&char_row_to_string(
                    &ca, 0,
                ))))
            }
            Value::String(s) => Value::String(read_env_string(&s)),
            _ => return Err(getenv_error(ERR_CHAR_MATRIX_CELL)),
        };
        values.push(resolved);
    }
    make_cell(values, array.rows, array.cols).map_err(getenv_error)
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

fn char_array_from_rows(rows: &[String]) -> BuiltinResult<CharArray> {
    if rows.is_empty() {
        return CharArray::new(Vec::new(), 0, 0).map_err(getenv_error);
    }

    let max_cols = rows
        .iter()
        .map(|row| row.chars().count())
        .max()
        .unwrap_or(0);
    if max_cols == 0 {
        return CharArray::new(Vec::new(), rows.len(), 0).map_err(getenv_error);
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
    CharArray::new(data, rows.len(), max_cols).map_err(getenv_error)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK;
    use runmat_builtins::{CharArray, StringArray, Value};

    fn getenv_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::getenv_builtin(args))
    }

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
            err.message().contains("cell array elements"),
            "unexpected error message: {}",
            err.message()
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
            err.message().contains("NAME must be"),
            "unexpected error message: {}",
            err.message()
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
            err.message().contains("too many input arguments"),
            "unexpected error message: {}",
            err.message()
        );
    }
}
