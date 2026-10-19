use super::shape_scalar_query::define_shape_scalar_query_runtime;

define_shape_scalar_query_runtime!(
    builtin_fn: length_builtin,
    name: "length",
    query: runmat_builtins::ShapeScalarQuery::Length,
    entry: runmat_builtins::LENGTH_CATALOG_ENTRY,
    internal_error: runmat_builtins::LENGTH_ERROR_INTERNAL,
    output_error: runmat_builtins::LENGTH_ERROR_TOO_MANY_OUTPUTS,
    exact_error: Some(&runmat_builtins::LENGTH_ERROR_RESULT_NOT_EXACT),
    table_error: Some(&runmat_builtins::LENGTH_ERROR_UNSUPPORTED_TABLE),
    builtin_path: "crate::builtins::array::introspection::length"
);

#[cfg(test)]
#[path = "length/tests.rs"]
mod tests;
