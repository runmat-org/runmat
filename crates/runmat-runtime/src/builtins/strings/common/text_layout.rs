//! Admission for row-wise text owners that do not implement character pages.
use runmat_value::Value;

/// String/cell array geometry is independent of character storage geometry.
/// Reject unsupported character pages, including inside cells, before any
/// gather or two-dimensional row extraction can silently drop later pages.
pub(crate) fn has_unsupported_character_pages(value: &Value) -> bool {
    match value {
        Value::CharArray(array) => array.shape().iter().skip(2).any(|extent| *extent != 1),
        Value::Cell(cell) => cell.data().iter().any(has_unsupported_character_pages),
        _ => false,
    }
}
