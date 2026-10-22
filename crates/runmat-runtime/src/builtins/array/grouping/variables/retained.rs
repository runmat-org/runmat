//! Borrow shared grouping inputs while the caller retains their root lease.

use runmat_value::{RetainedArrayValue, Value};

use super::{column::is_group_vector_list, numeric, text, GroupColumn, VariableResult};

pub(super) fn columns(
    name: &str,
    source: RetainedArrayValue,
    split_matrices: bool,
) -> VariableResult<Vec<GroupColumn>> {
    // Already-normalized columns need neither a new shape nor a cloned
    // collection. Retain the original value, including its child owners.
    let shape = match source.value() {
        Value::Tensor(value) => Some(value.shape()),
        Value::LogicalArray(value) => Some(value.shape.as_slice()),
        Value::StringArray(value) => Some(value.shape.as_slice()),
        _ => None,
    };
    if shape.is_some_and(|shape| matches!(shape, [_, 1])) {
        return GroupColumn::retained(name, source).map(|column| vec![column]);
    }
    match source.value() {
        Value::Cell(cell) if is_group_vector_list(cell) => {
            let mut columns = Vec::with_capacity(cell.data().len());
            for index in 0..cell.data().len() {
                let child = source
                    .project(|value| match value {
                        Value::Cell(cell) => cell.data().get(index),
                        _ => None,
                    })
                    .ok_or_else(|| {
                        super::VariableError::from(
                            "grouping child changed during immutable traversal",
                        )
                    })?;
                columns.extend(self::columns(
                    &format!("{name}{}", index + 1),
                    child,
                    false,
                )?);
            }
            Ok(columns)
        }
        // Numeric/logical buffer clones retain their physical payload. The
        // normalizers construct the requested column metadata and, for a
        // matrix split, only the actual output columns.
        Value::Tensor(value) => numeric::tensor_columns(name, value.clone(), split_matrices),
        Value::LogicalArray(value) => numeric::logical_columns(name, value.clone(), split_matrices),
        Value::StringArray(value) => text::string_columns(name, value.clone(), split_matrices),
        _ => GroupColumn::retained(name, source).map(|column| vec![column]),
    }
}
