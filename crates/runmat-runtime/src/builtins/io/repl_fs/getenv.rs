//! MATLAB-compatible `getenv` builtin for RunMat.
//!
//! Mirrors MATLAB semantics for querying environment variables. Supports scalar character
//! vectors, string scalars, string arrays, and cell arrays of character vectors. Calling
//! `getenv` with no arguments returns a struct containing every environment variable visible to
//! the current process.

use crate::builtins::common::env as runtime_env;
#[cfg(test)]
use std::env;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, StringArray, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, call_builtin_async, make_cell, BuiltinResult, RuntimeError};

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

const GETENV_OUTPUT_ALL: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "env",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Structure containing all visible environment variables.",
}];
const GETENV_OUTPUT_VALUE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Resolved environment value(s) matching NAME input shape/type.",
}];
const GETENV_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const GETENV_INPUTS_NAME: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "NAME",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Variable name query: char vector, string scalar/array, or cell array of char/string scalars.",
}];
const GETENV_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "env = getenv()",
        inputs: &GETENV_INPUTS_NONE,
        outputs: &GETENV_OUTPUT_ALL,
    },
    BuiltinSignatureDescriptor {
        label: "value = getenv(NAME)",
        inputs: &GETENV_INPUTS_NAME,
        outputs: &GETENV_OUTPUT_VALUE,
    },
];
const GETENV_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETENV.TOO_MANY_INPUTS",
    identifier: None,
    when: "More than one input argument is supplied.",
    message: "getenv: too many input arguments",
};
const GETENV_ERROR_INVALID_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETENV.INVALID_TYPE",
    identifier: None,
    when: "NAME input type is unsupported for getenv queries.",
    message: "getenv: NAME must be a character vector, string scalar, string array, or cell array of character vectors",
};
const GETENV_ERROR_CELL_ELEMENT_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETENV.CELL_ELEMENT_TYPE",
    identifier: None,
    when: "Cell NAME entries are not character vectors or string scalars.",
    message: "getenv: cell array elements must be character vectors or string scalars",
};
const GETENV_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETENV.INTERNAL",
    identifier: None,
    when: "Internal conversion of environment query outputs fails.",
    message: "getenv: internal conversion failure",
};
const GETENV_ERRORS: [BuiltinErrorDescriptor; 4] = [
    GETENV_ERROR_TOO_MANY_INPUTS,
    GETENV_ERROR_INVALID_TYPE,
    GETENV_ERROR_CELL_ELEMENT_TYPE,
    GETENV_ERROR_INTERNAL,
];
pub const GETENV_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GETENV_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GETENV_ERRORS,
};

pub const GETENV_CHAR_MATRIX_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "getenv-character-matrix-name",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "getenv with a multirow character-matrix name is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GetenvCharacterMatrixNameExtension"),
};

pub const GETENV_CELL_STRING_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "getenv-cell-string-name",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "getenv with string scalars inside a cell-array name is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GetenvCellStringNameExtension"),
};

pub const GETENV_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [GETENV_CHAR_MATRIX_EXTENSION, GETENV_CELL_STRING_EXTENSION];

pub const GETENV_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "getenv accepts host text names and returns text or a string dictionary. All eight integer classes and provider-resident numeric values reject without implicit text conversion, gather, provider access, or environment lookup.",
    };

fn getenv_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    getenv_error_with_message(error.message, error)
}

fn getenv_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
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
    summary = "Read environment variables.",
    keywords = "getenv,environment variable,env,system variable,process environment",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::getenv_type),
    descriptor(crate::builtins::io::repl_fs::getenv::GETENV_DESCRIPTOR),
    extensions(crate::builtins::io::repl_fs::getenv::GETENV_EXTENSIONS),
    integer_audit(crate::builtins::io::repl_fs::getenv::GETENV_INTEGER_AUDIT),
    builtin_path = "crate::builtins::io::repl_fs::getenv"
)]
async fn getenv_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    match args.len() {
        0 => getenv_all().await,
        1 => getenv_one(args.into_iter().next().expect("one argument")).await,
        _ => Err(getenv_error(&GETENV_ERROR_TOO_MANY_INPUTS)),
    }
}

async fn getenv_all() -> BuiltinResult<Value> {
    let mut entries: Vec<(String, String)> = runtime_env::vars();
    entries.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    let shape = vec![entries.len(), 1];
    let keys = StringArray::new(
        entries.iter().map(|(name, _)| name.clone()).collect(),
        shape.clone(),
    )
    .map_err(|err| {
        getenv_error_with_message(
            format!("{}: {err}", GETENV_ERROR_INTERNAL.message),
            &GETENV_ERROR_INTERNAL,
        )
    })?;
    let values = StringArray::new(entries.into_iter().map(|(_, value)| value).collect(), shape)
        .map_err(|err| {
            getenv_error_with_message(
                format!("{}: {err}", GETENV_ERROR_INTERNAL.message),
                &GETENV_ERROR_INTERNAL,
            )
        })?;
    call_builtin_async(
        "dictionary",
        &[Value::StringArray(keys), Value::StringArray(values)],
    )
    .await
    .map_err(map_control_flow)
}

async fn getenv_one(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::CharArray(array) => getenv_from_char_array(array),
        Value::String(s) => Ok(Value::CharArray(CharArray::new_row(&read_env_string(&s)))),
        Value::StringArray(sa) => getenv_from_string_array(sa),
        Value::Cell(ca) => getenv_from_cell_array(ca).await,
        _ => Err(getenv_error(&GETENV_ERROR_INVALID_TYPE)),
    }
}

fn getenv_from_char_array(array: CharArray) -> BuiltinResult<Value> {
    if array.rows == 0 {
        return Ok(Value::CharArray(
            CharArray::new(Vec::new(), 0, array.cols).map_err(|e| {
                getenv_error_with_message(
                    format!(
                        "{}: unable to construct empty character array ({e})",
                        GETENV_ERROR_INTERNAL.message
                    ),
                    &GETENV_ERROR_INTERNAL,
                )
            })?,
        ));
    }

    if array.rows == 1 {
        let name = char_row_to_string(&array, 0);
        let value = CharArray::new_row(&read_env_string(&name));
        return Ok(Value::CharArray(value));
    }

    crate::compatibility::ensure_builtin_extension_enabled(
        &GETENV_CHAR_MATRIX_EXTENSION,
        BUILTIN_NAME,
    )?;

    let mut rows = Vec::with_capacity(array.rows);
    for row in 0..array.rows {
        rows.push(read_env_string(&char_row_to_string(&array, row)));
    }
    let result = char_array_from_rows(&rows).map_err(|err| {
        getenv_error_with_message(
            format!(
                "{}: unable to build character matrix ({err})",
                GETENV_ERROR_INTERNAL.message
            ),
            &GETENV_ERROR_INTERNAL,
        )
    })?;
    Ok(Value::CharArray(result))
}

fn getenv_from_string_array(array: StringArray) -> BuiltinResult<Value> {
    let mut resolved = Vec::with_capacity(array.data.len());
    for name in &array.data {
        resolved.push(read_env_string(name));
    }
    let result = StringArray::new(resolved, array.shape.clone()).map_err(|err| {
        getenv_error_with_message(
            format!("{}: {err}", GETENV_ERROR_INTERNAL.message),
            &GETENV_ERROR_INTERNAL,
        )
    })?;
    Ok(Value::StringArray(result))
}

async fn getenv_from_cell_array(array: runmat_value::CellArray) -> BuiltinResult<Value> {
    for cell in &array.data {
        match cell {
            Value::CharArray(ca) if ca.rows == 1 => {}
            Value::String(_) => crate::compatibility::ensure_builtin_extension_enabled(
                &GETENV_CELL_STRING_EXTENSION,
                BUILTIN_NAME,
            )?,
            _ => return Err(getenv_error(&GETENV_ERROR_CELL_ELEMENT_TYPE)),
        }
    }
    let mut values: Vec<Value> = Vec::with_capacity(array.data.len());
    for cell in &array.data {
        let resolved = match cell {
            Value::CharArray(ca) => Value::CharArray(CharArray::new_row(&read_env_string(
                &char_row_to_string(ca, 0),
            ))),
            Value::String(s) => Value::String(read_env_string(s)),
            _ => unreachable!("cell entries validated before environment lookup"),
        };
        values.push(resolved);
    }
    make_cell(values, array.rows, array.cols).map_err(|err| {
        getenv_error_with_message(
            format!("{}: {err}", GETENV_ERROR_INTERNAL.message),
            &GETENV_ERROR_INTERNAL,
        )
    })
}

fn read_env_string(name: &str) -> String {
    runtime_env::var(name).unwrap_or_default()
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
        return CharArray::new(Vec::new(), 0, 0).map_err(|err| {
            getenv_error_with_message(
                format!("{}: {err}", GETENV_ERROR_INTERNAL.message),
                &GETENV_ERROR_INTERNAL,
            )
        });
    }

    let max_cols = rows
        .iter()
        .map(|row| row.chars().count())
        .max()
        .unwrap_or(0);
    if max_cols == 0 {
        return CharArray::new(Vec::new(), rows.len(), 0).map_err(|err| {
            getenv_error_with_message(
                format!("{}: {err}", GETENV_ERROR_INTERNAL.message),
                &GETENV_ERROR_INTERNAL,
            )
        });
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
    CharArray::new(data, rows.len(), max_cols).map_err(|err| {
        getenv_error_with_message(
            format!("{}: {err}", GETENV_ERROR_INTERNAL.message),
            &GETENV_ERROR_INTERNAL,
        )
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK;
    use runmat_value::{CharArray, IntValue, StringArray, Value};

    fn getenv_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::getenv_builtin(args))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn getenv_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = GETENV_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"env = getenv()"));
        assert!(labels.contains(&"value = getenv(NAME)"));
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
    fn getenv_string_scalar_returns_character_vector() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        env::remove_var("RUNMAT_TEST_GETENV_MISSING");
        let input = Value::String("RUNMAT_TEST_GETENV_MISSING".to_string());
        let result = getenv_builtin(vec![input]).expect("getenv");
        match result {
            Value::CharArray(array) => {
                assert_eq!((array.rows, array.cols), (1, 0));
                assert!(array.data.is_empty());
            }
            other => panic!("expected character-vector output, got {other:?}"),
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
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
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
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
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
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
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
                match &ca.data[0] {
                    Value::CharArray(first) => {
                        let text: String = first.data.iter().collect();
                        assert_eq!(text, "one");
                    }
                    other => panic!("expected char array in first cell, got {other:?}"),
                };
                match &ca.data[1] {
                    Value::String(s) => assert_eq!(s, "two"),
                    other => panic!("expected string in second cell, got {other:?}"),
                };
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
    fn getenv_no_argument_returns_string_dictionary() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        env::set_var("RUNMAT_TEST_STRUCT", "struct-value");
        let result = getenv_builtin(Vec::new()).expect("getenv");
        let Value::Object(dictionary) = result else {
            panic!("expected dictionary object");
        };
        assert!(dictionary.is_class("dictionary"));
        let Value::Cell(keys) = dictionary.properties.get("Keys").expect("dictionary keys") else {
            panic!("expected dictionary key cells");
        };
        let Value::Cell(values) = dictionary
            .properties
            .get("Values")
            .expect("dictionary values")
        else {
            panic!("expected dictionary value cells");
        };
        let index = keys
            .data
            .iter()
            .position(|value| value == &Value::String("RUNMAT_TEST_STRUCT".to_string()))
            .expect("environment key");
        assert_eq!(
            values.data[index],
            Value::String("struct-value".to_string())
        );
        env::remove_var("RUNMAT_TEST_STRUCT");
    }

    #[test]
    fn getenv_rejects_all_integer_name_classes_without_conversion() {
        let values = [
            IntValue::I8(1),
            IntValue::I16(1),
            IntValue::I32(1),
            IntValue::I64(1),
            IntValue::U8(1),
            IntValue::U16(1),
            IntValue::U32(1),
            IntValue::U64(1),
        ];
        for value in values {
            let err = getenv_builtin(vec![Value::Int(value)]).expect_err("integer name");
            assert!(err.message().contains("NAME must be"));
        }
        assert_eq!(
            GETENV_INTEGER_AUDIT.kind,
            BuiltinIntegerAuditKind::NotApplicable
        );
    }

    #[test]
    fn getenv_character_matrix_and_cell_string_extensions_are_mode_gated() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let matrix = CharArray::new(vec!['A', 'B'], 2, 1).expect("matrix");
        let err = getenv_builtin(vec![Value::CharArray(matrix)]).expect_err("matrix extension");
        assert_eq!(
            err.identifier(),
            GETENV_CHAR_MATRIX_EXTENSION.error_identifier
        );

        let cell = make_cell(vec![Value::String("PATH".to_string())], 1, 1).expect("cell");
        let err = getenv_builtin(vec![cell]).expect_err("cell string extension");
        assert_eq!(
            err.identifier(),
            GETENV_CELL_STRING_EXTENSION.error_identifier
        );
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
