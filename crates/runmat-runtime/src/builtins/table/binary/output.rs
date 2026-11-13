use runmat_value::ObjectInstance;

use crate::execution::{RootedValue, RootedValues};
use crate::BuiltinResult;

/// Publish a replacement table while every completed column remains rooted.
/// Names and values come from the same plan traversal and must remain paired.
pub(crate) fn finish_owned(
    source: &ObjectInstance,
    names: Vec<String>,
    columns: RootedValues,
) -> BuiltinResult<RootedValue> {
    super::super::table_replace_variables_like_owned(source, names, columns)
}
