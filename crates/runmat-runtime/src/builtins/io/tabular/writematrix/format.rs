use super::*;

pub(super) fn default_delimiter_for_path(path: &Path) -> String {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase());
    match ext.as_deref() {
        Some("csv") => ",".to_string(),
        Some("tsv") | Some("tab") => "\t".to_string(),
        Some("txt") | Some("dat") | Some("dlm") => " ".to_string(),
        _ => ",".to_string(),
    }
}

pub(super) fn format_numeric(
    value: runmat_value::NumericScalar,
    decimal_separator: char,
) -> String {
    // Keep this writer's existing explicit signed-zero normalization.
    let mut formatted = if value.is_zero() {
        "0".to_string()
    } else {
        crate::builtins::common::format::format_delimited_numeric(value)
    };
    if decimal_separator != '.' && formatted.contains('.') {
        formatted = formatted.replace('.', &decimal_separator.to_string());
    }
    formatted
}
