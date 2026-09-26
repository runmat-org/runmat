//! MATLAB-compatible `readlines` and `writelines` helpers.

mod options;
use options::{parse_empty_line_rule, parse_newline};

use runmat_builtins::{BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind};
use std::path::PathBuf;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_filesystem as vfs;
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, StringArray, Value};

use crate::builtins::common::fs::expand_user_path;
use crate::builtins::common::shape::effective_rank;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

fn line_error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

const READLINES_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Text file to read.",
}];
const WRITELINES_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "lines",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Text lines to write.",
    },
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Destination text file.",
    },
];
const OUTPUT_VALUE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Result value.",
}];
const READLINES_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "lines = readlines(filename)",
    inputs: &READLINES_INPUTS,
    outputs: &OUTPUT_VALUE,
}];
pub const READLINES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &READLINES_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};
const WRITELINES_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "status = writelines(lines, filename)",
    inputs: &WRITELINES_INPUTS,
    outputs: &OUTPUT_VALUE,
}];
pub const WRITELINES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &WRITELINES_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};
pub const WRITELINES_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "writelines accepts textual lines, a host text filename, and text-valued options. Integer and resident numeric values are invalid and reject before provider or filesystem access.",
    };

fn scalar_text(value: &Value, name: &str, arg: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(array) if array.rows == 1 && effective_rank(&array.shape) <= 2 => {
            Ok(array.data.iter().collect())
        }
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        _ => Err(line_error(
            name,
            format!("{name}: {arg} must be a string scalar or character vector"),
        )),
    }
}

fn path_from_value(value: &Value, name: &str) -> BuiltinResult<PathBuf> {
    let text = scalar_text(value, name, "filename")?;
    Ok(PathBuf::from(
        expand_user_path(text.trim(), name).map_err(|err| line_error(name, err))?,
    ))
}

#[runtime_builtin(
    name = "readlines",
    category = "io/filetext",
    summary = "Read text file lines into a string array.",
    keywords = "readlines,text,file,string array,lines",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::readlines_type),
    descriptor(crate::builtins::io::filetext::lines::READLINES_DESCRIPTOR),
    builtin_path = "crate::builtins::io::filetext::lines"
)]
async fn readlines_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.is_empty() || args.len() > 3 {
        return Err(line_error("readlines", "readlines: filename is required"));
    }
    let path = path_from_value(&args[0], "readlines")?;
    let omit_empty = parse_empty_line_rule(&args[1..])?;
    let text = vfs::read_to_string_async(&path)
        .await
        .map_err(|err| line_error("readlines", format!("readlines: {err}")))?;
    let mut lines = split_lines_preserving_content(&text);
    if omit_empty {
        lines.retain(|line| !line.is_empty());
    }
    let rows = lines.len();
    Ok(Value::StringArray(
        StringArray::new(lines, vec![rows, 1]).map_err(|err| line_error("readlines", err))?,
    ))
}

fn split_lines_preserving_content(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    let mut lines: Vec<String> = normalized.split('\n').map(str::to_string).collect();
    if normalized.ends_with('\n') {
        lines.pop();
    }
    lines
}

#[runtime_builtin(
    name = "writelines",
    category = "io/filetext",
    summary = "Write strings or text lines to a file.",
    keywords = "writelines,text,file,string array,lines",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::num_type),
    descriptor(crate::builtins::io::filetext::lines::WRITELINES_DESCRIPTOR),
    integer_audit(crate::builtins::io::filetext::lines::WRITELINES_INTEGER_AUDIT),
    builtin_path = "crate::builtins::io::filetext::lines"
)]
async fn writelines_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.iter().any(|value| {
        crate::builtins::common::validation::value_contains_native_integer_class(value)
            || crate::value_contains_gpu(value)
    }) {
        return Err(line_error(
            "writelines",
            "writelines: lines, filename, and options must be host text values",
        ));
    }
    if args.len() < 2 {
        return Err(line_error(
            "writelines",
            "writelines: text and filename are required",
        ));
    }
    let lines = lines_from_value(&args[0])?;
    let path = path_from_value(&args[1], "writelines")?;
    let newline = parse_newline(&args[2..])?.unwrap_or("\n".to_string());
    let mut output = lines.join(&newline);
    output.push_str(&newline);
    vfs::write_async(&path, output.as_bytes())
        .await
        .map_err(|err| line_error("writelines", format!("writelines: {err}")))?;
    Ok(Value::Num(output.len() as f64))
}

fn lines_from_value(value: &Value) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(text) => Ok(vec![text.clone()]),
        Value::CharArray(array) if array.rows == 1 && effective_rank(&array.shape) <= 2 => {
            Ok(vec![array.data.iter().collect()])
        }
        Value::CharArray(array) if effective_rank(&array.shape) <= 2 => {
            let mut lines = Vec::with_capacity(array.rows);
            for row in 0..array.rows {
                let start = row * array.cols;
                lines.push(array.data[start..start + array.cols].iter().collect());
            }
            Ok(lines)
        }
        Value::StringArray(array) => Ok(array.data.clone()),
        Value::Cell(array) => cell_lines(array),
        _ => Err(line_error(
            "writelines",
            "writelines: text must be string, char, string array, or cell text",
        )),
    }
}

fn cell_lines(array: &CellArray) -> BuiltinResult<Vec<String>> {
    let mut out = Vec::with_capacity(array.data.len());
    for value in crate::object::cell::expand_all_cell_values(array)? {
        out.push(scalar_text(&value, "writelines", "cell element")?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK;

    fn run(value: impl std::future::Future<Output = BuiltinResult<Value>>) -> BuiltinResult<Value> {
        futures::executor::block_on(value)
    }

    #[test]
    fn readlines_returns_column_string_array() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let path = std::env::temp_dir().join("runmat_readlines_test.txt");
        std::fs::write(&path, "a\nb\n").unwrap();
        let value = run(readlines_builtin(vec![Value::String(
            path.to_string_lossy().into_owned(),
        )]))
        .unwrap();
        match value {
            Value::StringArray(array) => {
                assert_eq!(array.data, vec!["a", "b"]);
                assert_eq!(array.shape, vec![2, 1]);
            }
            other => panic!("unexpected value {other:?}"),
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn writelines_writes_each_string_array_element() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let path = std::env::temp_dir().join("runmat_writelines_test.txt");
        let lines = StringArray::new(vec!["a".to_string(), "b".to_string()], vec![2, 1]).unwrap();
        run(writelines_builtin(vec![
            Value::StringArray(lines),
            Value::String(path.to_string_lossy().into_owned()),
        ]))
        .unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "a\nb\n");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn writelines_rejects_integer_input_before_filesystem_access() {
        let error = run(writelines_builtin(vec![
            Value::Int(runmat_value::IntValue::U8(1)),
            Value::String("unused.txt".into()),
        ]))
        .expect_err("integer lines are outside the text-only surface");
        assert!(error.message().contains("must be host text values"));
    }

    #[test]
    fn invalid_read_options_precede_missing_file_errors() {
        for options in [
            vec![Value::String("EmptyLineRule".into())],
            vec![
                Value::String("Encoding".into()),
                Value::String("UTF-8".into()),
            ],
            vec![
                Value::String("EmptyLineRule".into()),
                Value::String("invalid".into()),
            ],
        ] {
            let mut args = vec![Value::String("/missing/readlines-admission-test".into())];
            args.extend(options);
            let error = run(readlines_builtin(args)).unwrap_err();
            assert!(
                error.message().contains("paired") || error.message().contains("unsupported"),
                "{error}"
            );
        }
    }

    #[test]
    fn invalid_write_options_do_not_modify_existing_file() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("unchanged.txt");
        std::fs::write(&path, "original").unwrap();
        for (name, value) in [
            ("WriteMode", "append"),
            ("LineEnding", "invalid"),
            ("Encoding", "UTF-8"),
        ] {
            assert!(run(writelines_builtin(vec![
                Value::String("new".into()),
                Value::String(path.to_string_lossy().into_owned()),
                Value::String(name.into()),
                Value::String(value.into())
            ]))
            .is_err());
            assert_eq!(std::fs::read_to_string(&path).unwrap(), "original");
        }
    }

    #[test]
    fn line_ending_aliases_and_empty_line_rules_are_explicit() {
        for alias in ["windows", "pc", "crlf", "PC"] {
            assert_eq!(
                parse_newline(&[
                    Value::String("LineEnding".into()),
                    Value::String(alias.into())
                ])
                .unwrap(),
                Some("\r\n".into())
            );
        }
        for alias in ["skip", "omit", "omitempty"] {
            assert!(parse_empty_line_rule(&[
                Value::String("EmptyLineRule".into()),
                Value::String(alias.into())
            ])
            .unwrap());
        }
        assert!(!parse_empty_line_rule(&[
            Value::String("EmptyLineRule".into()),
            Value::String("read".into())
        ])
        .unwrap());
        assert_eq!(
            split_lines_preserving_content("a\r\n\r\nb\rc\n"),
            vec!["a", "", "b", "c"]
        );
    }

    #[test]
    fn cell_text_uses_shared_matlab_linear_order() {
        let cell = CellArray::new(
            ["a", "b", "c", "d"]
                .into_iter()
                .map(|text| Value::String(text.into()))
                .collect(),
            2,
            2,
        )
        .unwrap();
        assert_eq!(cell_lines(&cell).unwrap(), vec!["a", "c", "b", "d"]);
    }
}
