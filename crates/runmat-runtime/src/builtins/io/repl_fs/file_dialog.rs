use std::path::{Path, PathBuf};

use runmat_builtins::{CellArray, Value};
use runmat_filesystem::OpenFileDialogFilter;

use crate::{BuiltinResult, RuntimeError};

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
    if cell.data.is_empty() {
        return Ok(default_filters());
    }
    if cell.cols == 0 || cell.cols > 2 {
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

pub(crate) fn try_scalar_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::CharArray(chars) if chars.rows == 1 => Some(chars.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        _ => None,
    }
}

pub(crate) fn scalar_text(
    value: &Value,
    context: &str,
    invalid_argument: ErrorMapper,
) -> BuiltinResult<String> {
    try_scalar_text(value).ok_or_else(|| {
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
