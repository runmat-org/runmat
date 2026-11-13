use std::path::{Path, PathBuf};

use runmat_filesystem::OpenFileDialogFilter;
use runmat_value::{CellArray, Value};

use crate::{BuiltinResult, RuntimeError};

#[cfg(test)]
mod selected_context_tests;

pub(crate) type ErrorMapper = fn(String) -> RuntimeError;

pub(crate) fn parse_filter_spec(
    value: &Value,
    invalid_argument: ErrorMapper,
) -> BuiltinResult<Vec<OpenFileDialogFilter>> {
    match value {
        Value::Cell(cell) => parse_filter_cell(cell, invalid_argument),
        other => {
            let text = scalar_text(other, "filter", invalid_argument)?;
            Ok(vec![filter_from_pattern(&text, None)])
        }
    }
}

fn parse_filter_cell(
    cell: &CellArray,
    invalid_argument: ErrorMapper,
) -> BuiltinResult<Vec<OpenFileDialogFilter>> {
    if cell.data().is_empty() {
        return Ok(default_filters());
    }
    if crate::builtins::common::shape::effective_rank(&cell.shape) > 2
        || cell.cols == 0
        || cell.cols > 2
    {
        return Err(invalid_argument(
            "filter cell array must have one or two columns".to_string(),
        ));
    }
    let mut filters = Vec::with_capacity(cell.rows);
    for row in 0..cell.rows {
        let pattern_value = cell.get(row, 0).map_err(invalid_argument)?;
        let pattern = scalar_text(&pattern_value, "filter pattern", invalid_argument)?;
        let description = if cell.cols == 2 {
            let description_value = cell.get(row, 1).map_err(invalid_argument)?;
            Some(scalar_text(
                &description_value,
                "filter description",
                invalid_argument,
            )?)
        } else {
            None
        };
        filters.push(filter_from_pattern(&pattern, description));
    }
    Ok(filters)
}

fn filter_from_pattern(pattern: &str, description: Option<String>) -> OpenFileDialogFilter {
    let patterns = split_filter_patterns(pattern);
    OpenFileDialogFilter {
        patterns,
        description,
    }
}

fn split_filter_patterns(pattern: &str) -> Vec<String> {
    let mut patterns = pattern
        .split(';')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    if patterns.is_empty() {
        patterns.push("*.*".to_string());
    }
    patterns
}

pub(crate) fn default_filters() -> Vec<OpenFileDialogFilter> {
    vec![OpenFileDialogFilter {
        patterns: vec!["*.*".to_string()],
        description: Some("All Files".to_string()),
    }]
}

pub(crate) fn try_scalar_text(
    value: &Value,
    context: &str,
    invalid_argument: impl FnOnce(String) -> RuntimeError,
) -> BuiltinResult<Option<String>> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => {
            Ok(Some(text.as_str().to_owned()))
        }
        Value::CharArray(chars)
            if chars.rows == 1
                && crate::builtins::common::shape::effective_rank(&chars.shape) <= 2 =>
        {
            crate::builtins::io::utf16::decode_utf16_with_error(chars.data(), || {
                invalid_argument(format!("{context} contains malformed UTF-16 text"))
            })
            .map(Some)
        }
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Ok(Some(
                array.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string")
                    .as_str()
                    .to_owned(),
            ))
        }
        _ => Ok(None),
    }
}

pub(crate) fn scalar_text(
    value: &Value,
    context: &str,
    invalid_argument: ErrorMapper,
) -> BuiltinResult<String> {
    try_scalar_text(value, context, invalid_argument)?.ok_or_else(|| {
        invalid_argument(format!(
            "{context} must be a character vector or string scalar"
        ))
    })
}

pub(crate) struct SelectedPathParts {
    pub directory: String,
    pub file_name: String,
}

pub(crate) fn selected_path_parts(
    path: &Path,
    invalid_selection: ErrorMapper,
) -> BuiltinResult<SelectedPathParts> {
    let text = path.to_string_lossy();
    if text.is_empty() {
        return Err(invalid_selection(
            "selected path has no file name".to_string(),
        ));
    }

    let separator = text
        .char_indices()
        .rev()
        .find(|(_, ch)| *ch == '/' || *ch == '\\');

    let (directory, file_name) = match separator {
        Some((index, separator)) => {
            let file_start = index + separator.len_utf8();
            if file_start >= text.len() {
                return Err(invalid_selection(
                    "selected path has no file name".to_string(),
                ));
            }
            (
                text[..file_start].to_string(),
                text[file_start..].to_string(),
            )
        }
        None => (String::new(), text.into_owned()),
    };

    Ok(SelectedPathParts {
        directory,
        file_name,
    })
}

pub(crate) fn ensure_same_directory(
    paths: &[PathBuf],
    expected: &str,
    invalid_selection: ErrorMapper,
) -> BuiltinResult<()> {
    let expected = normalized_directory_key(expected);
    for path in paths.iter().skip(1) {
        let actual = selected_path_parts(path, invalid_selection)?.directory;
        if normalized_directory_key(&actual) != expected {
            return Err(invalid_selection(
                "multiple selected files must be in the same directory".to_string(),
            ));
        }
    }
    Ok(())
}

fn normalized_directory_key(directory: &str) -> String {
    let mut key = directory.replace('\\', "/");
    while key.len() > 1 && key.ends_with('/') {
        key.pop();
    }
    #[cfg(windows)]
    key.make_ascii_lowercase();
    key
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::CharArray;

    fn error(message: String) -> RuntimeError {
        crate::build_runtime_error(message).build()
    }

    #[test]
    fn scalar_text_distinguishes_pages_from_trailing_singletons() {
        let page = Value::CharArray(
            CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 1, 2]).unwrap(),
        );
        assert!(scalar_text(&page, "title", error).is_err());
        let row = Value::CharArray(
            CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 2, 1]).unwrap(),
        );
        assert_eq!(scalar_text(&row, "title", error).unwrap(), "ab");
        let malformed = Value::CharArray(
            CharArray::from_code_units(vec![runmat_value::Utf16CodeUnit::from_raw(0xd800)], 1, 1)
                .unwrap(),
        );
        let failure = try_scalar_text(&malformed, "title", error).unwrap_err();
        assert!(failure
            .source
            .as_deref()
            .unwrap()
            .downcast_ref::<runmat_value::Utf16DecodeError>()
            .is_some());
        assert!(try_scalar_text(&Value::Num(1.0), "title", error)
            .unwrap()
            .is_none());
    }

    #[test]
    fn filter_rows_preserve_pattern_description_pairs_and_reject_pages() {
        let values = vec![
            Value::from("*.m"),
            Value::from("Scripts"),
            Value::from("*.mat"),
            Value::from("Data"),
        ];
        let rows = CellArray::new_with_shape(values.clone(), vec![2, 2, 1]).unwrap();
        let filters = parse_filter_cell(&rows, error).unwrap();
        assert_eq!(filters[0].patterns, vec!["*.m"]);
        assert_eq!(filters[0].description.as_deref(), Some("Scripts"));
        assert_eq!(filters[1].patterns, vec!["*.mat"]);
        assert_eq!(filters[1].description.as_deref(), Some("Data"));
        let pages = CellArray::new_with_shape(values, vec![1, 2, 2]).unwrap();
        assert!(parse_filter_cell(&pages, error).is_err());
    }
}
