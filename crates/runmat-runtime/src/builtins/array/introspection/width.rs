use super::shape_scalar_query::define_shape_scalar_query_runtime;

define_shape_scalar_query_runtime!(
    builtin_fn: width_builtin,
    name: "width",
    query: runmat_builtins::ShapeScalarQuery::Width,
    entry: runmat_builtins::WIDTH_CATALOG_ENTRY,
    internal_error: runmat_builtins::WIDTH_ERROR_INTERNAL,
    output_error: runmat_builtins::WIDTH_ERROR_TOO_MANY_OUTPUTS,
    exact_error: Some(&runmat_builtins::WIDTH_ERROR_RESULT_NOT_EXACT),
    table_error: None,
    builtin_path: "crate::builtins::array::introspection::width"
);

#[cfg(test)]
#[path = "width/tests.rs"]
mod tests;
