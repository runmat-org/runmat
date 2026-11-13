//! MATLAB-compatible `readlines` and `writelines` helpers.

mod options;
use options::{parse_empty_line_rule, parse_newline};

use std::borrow::Cow;
use std::path::PathBuf;

use runmat_filesystem as vfs;
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, StringArray, Value};

use crate::builtins::common::fs::expand_user_path;
use crate::builtins::common::shape::effective_rank;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

fn line_error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(name)
        .with_identifier(format!("RunMat:{name}:RuntimeError"))
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}

pub use runmat_builtins::{
    READLINES_DESCRIPTOR, READLINES_ERROR_OUTPUTS, READLINES_INTEGER_AUDIT,
    WRITELINES_BYTE_COUNT_EXTENSION, WRITELINES_DESCRIPTOR, WRITELINES_ERROR_OUTPUTS,
    WRITELINES_INTEGER_AUDIT,
};

fn scalar_text<'a>(value: &'a Value, name: &str, arg: &str) -> BuiltinResult<Cow<'a, str>> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => Ok(Cow::Borrowed(text.as_str())),
        Value::CharArray(array) if array.rows == 1 && effective_rank(&array.shape) <= 2 => {
            crate::builtins::io::utf16::decode_utf16_with_error(array.data(), || {
                line_error(
                    name,
                    format!("{name}: {arg} contains malformed UTF-16 text"),
                )
            })
            .map(Cow::Owned)
        }
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Ok(Cow::Borrowed(
                array.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string")
                    .as_str(),
            ))
        }
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
    binding_variant = "default",
    builtin_path = "crate::builtins::io::filetext::lines"
)]
async fn readlines_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &READLINES_DESCRIPTOR,
        &READLINES_ERROR_OUTPUTS,
        "readlines",
    )?;
    if args.is_empty() || args.len() > 3 {
        return Err(line_error("readlines", "readlines: filename is required"));
    }
    let path = path_from_value(&args[0], "readlines")?;
    let omit_empty = parse_empty_line_rule(args.view().range(1..)?)?;
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
    binding_variant = "default",
    builtin_path = "crate::builtins::io::filetext::lines"
)]
async fn writelines_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &WRITELINES_DESCRIPTOR,
        &WRITELINES_ERROR_OUTPUTS,
        "writelines",
    )?;
    if crate::output_count::current_output_count().is_some_and(|count| count > 0) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &WRITELINES_BYTE_COUNT_EXTENSION,
            "writelines",
        )?;
    }
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
    let newline = parse_newline(args.view().range(2..)?)?.unwrap_or("\n".to_string());
    let mut output = lines.join(&newline);
    output.push_str(&newline);
    vfs::write_async(&path, output.as_bytes())
        .await
        .map_err(|err| line_error("writelines", format!("writelines: {err}")))?;
    Ok(Value::Num(output.len() as f64))
}

fn lines_from_value(value: &Value) -> BuiltinResult<Vec<Cow<'_, str>>> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => {
            Ok(vec![Cow::Borrowed(text.as_str())])
        }
        Value::CharArray(array) if array.rows == 1 && effective_rank(&array.shape) <= 2 => {
            crate::builtins::io::utf16::decode_utf16_with_error(array.data(), || {
                line_error("textscan", "text input contains malformed UTF-16")
            })
            .map(|text| vec![Cow::Owned(text)])
        }
        Value::CharArray(array) if effective_rank(&array.shape) <= 2 => {
            let mut lines = Vec::with_capacity(array.rows);
            for row in 0..array.rows {
                let start = row * array.cols;
                let text = crate::builtins::io::utf16::decode_utf16_with_error(
                    &array.data()[start..start + array.cols],
                    || line_error("textscan", "text input contains malformed UTF-16"),
                )?;
                lines.push(Cow::Owned(text));
            }
            Ok(lines)
        }
        Value::StringArray(array) if array.data().iter().all(|text| !text.is_missing()) => {
            Ok(array
                .data()
                .iter()
                .map(|text| Cow::Borrowed(text.as_str().expect("checked nonmissing string")))
                .collect())
        }
        Value::Cell(array) => cell_lines(array),
        _ => Err(line_error(
            "writelines",
            "writelines: text must be string, char, string array, or cell text",
        )),
    }
}

fn cell_lines(array: &CellArray) -> BuiltinResult<Vec<Cow<'_, str>>> {
    let mut out = Vec::with_capacity(array.data().len());
    for value in array.iter_column_major() {
        out.push(scalar_text(value, "writelines", "cell element")?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(value: impl std::future::Future<Output = BuiltinResult<Value>>) -> BuiltinResult<Value> {
        futures::executor::block_on(value)
    }

    #[test]
    fn readlines_returns_column_string_array() {
        let _guard = runmat_filesystem::provider_override_lock();
        let path = std::env::temp_dir().join("runmat_readlines_test.txt");
        std::fs::write(&path, "a\nb\n").unwrap();
        let value = run(readlines_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                path.to_string_lossy().into_owned().into(),
            )]),
        ))
        .unwrap();
        match value {
            Value::StringArray(array) => {
                assert_eq!(array.data(), vec!["a", "b"]);
                assert_eq!(array.shape, vec![2, 1]);
            }
            other => panic!("unexpected value {other:?}"),
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn writelines_writes_each_string_array_element() {
        let _guard = runmat_filesystem::provider_override_lock();
        let path = std::env::temp_dir().join("runmat_writelines_test.txt");
        let lines = StringArray::new(vec!["a".to_string(), "b".to_string()], vec![2, 1]).unwrap();
        run(writelines_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::StringArray(lines),
                Value::String(path.to_string_lossy().into_owned().into()),
            ]),
        ))
        .unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "a\nb\n");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn writelines_rejects_integer_input_before_filesystem_access() {
        let error = run(writelines_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Int(runmat_value::IntValue::U8(1)),
                Value::String("unused.txt".into()),
            ]),
        ))
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
            let error = run(readlines_builtin(
                crate::builtins::common::test_support::builtin_arguments(args),
            ))
            .unwrap_err();
            assert!(
                error.message().contains("paired") || error.message().contains("unsupported"),
                "{error}"
            );
        }
    }

    #[test]
    fn invalid_write_options_do_not_modify_existing_file() {
        let _guard = runmat_filesystem::provider_override_lock();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("unchanged.txt");
        std::fs::write(&path, "original").unwrap();
        for (name, value) in [
            ("WriteMode", "append"),
            ("LineEnding", "invalid"),
            ("Encoding", "UTF-8"),
        ] {
            assert!(run(writelines_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::String("new".into()),
                    Value::String(path.to_string_lossy().into_owned().into()),
                    Value::String(name.into()),
                    Value::String(value.into())
                ])
            ))
            .is_err());
            assert_eq!(std::fs::read_to_string(&path).unwrap(), "original");
        }
    }

    #[test]
    fn line_ending_aliases_and_empty_line_rules_are_explicit() {
        for alias in ["windows", "pc", "crlf", "PC"] {
            assert_eq!(
                parse_newline(crate::call::FunctionArgumentView::from_slice(&[
                    Value::String("LineEnding".into()),
                    Value::String(alias.into())
                ]))
                .unwrap(),
                Some("\r\n".into())
            );
        }
        for alias in ["skip", "omit", "omitempty"] {
            assert!(
                parse_empty_line_rule(crate::call::FunctionArgumentView::from_slice(&[
                    Value::String("EmptyLineRule".into()),
                    Value::String(alias.into())
                ]))
                .unwrap()
            );
        }
        assert!(
            !parse_empty_line_rule(crate::call::FunctionArgumentView::from_slice(&[
                Value::String("EmptyLineRule".into()),
                Value::String("read".into())
            ]))
            .unwrap()
        );
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

    #[test]
    fn text_shape_checks_preserve_trailing_singletons_but_reject_pages() {
        let row = Value::CharArray(
            runmat_value::CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 2, 1])
                .unwrap(),
        );
        let pages = Value::CharArray(
            runmat_value::CharArray::from_unicode_scalars_with_shape(
                vec!['a', 'b', 'c', 'd'],
                vec![1, 2, 2],
            )
            .unwrap(),
        );
        assert_eq!(scalar_text(&row, "readlines", "filename").unwrap(), "ab");
        assert_eq!(lines_from_value(&row).unwrap(), vec!["ab"]);
        assert!(scalar_text(&pages, "readlines", "filename").is_err());
        assert!(lines_from_value(&pages).is_err());
    }

    #[test]
    fn output_bounds_and_byte_count_gate_precede_filesystem_effects() {
        let _lock = runmat_filesystem::provider_override_lock();
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("lines.txt");
        let args = || {
            vec![
                Value::String("alpha".into()),
                Value::String(path.to_string_lossy().into_owned().into()),
            ]
        };
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        {
            let _outputs =
                crate::output_count::push_output_count(Some(2)).expect("output-count scope");
            assert_eq!(
                run(writelines_builtin(
                    crate::builtins::common::test_support::builtin_arguments(args())
                ))
                .unwrap_err()
                .identifier(),
                WRITELINES_ERROR_OUTPUTS.identifier
            );
            assert_eq!(
                run(readlines_builtin(
                    crate::builtins::common::test_support::builtin_arguments(vec![])
                ))
                .unwrap_err()
                .identifier(),
                READLINES_ERROR_OUTPUTS.identifier
            );
        }
        {
            let _outputs =
                crate::output_count::push_output_count(Some(1)).expect("output-count scope");
            assert_eq!(
                run(writelines_builtin(
                    crate::builtins::common::test_support::builtin_arguments(args())
                ))
                .unwrap_err()
                .identifier(),
                WRITELINES_BYTE_COUNT_EXTENSION.error_identifier
            );
            assert!(!path.exists());
        }
        for count in [None, Some(0)] {
            let _outputs =
                crate::output_count::push_output_count(count).expect("output-count scope");
            assert_eq!(
                run(writelines_builtin(
                    crate::builtins::common::test_support::builtin_arguments(args())
                ))
                .unwrap(),
                Value::Num(6.0)
            );
            assert_eq!(std::fs::read(&path).unwrap(), b"alpha\n");
        }
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let _outputs = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
        assert_eq!(
            run(writelines_builtin(
                crate::builtins::common::test_support::builtin_arguments(args())
            ))
            .unwrap(),
            Value::Num(6.0)
        );
    }
}
