use super::shape_scalar_query::define_shape_scalar_query_runtime;

define_shape_scalar_query_runtime!(
    builtin_fn: height_builtin,
    name: "height",
    query: runmat_builtins::ShapeScalarQuery::Height,
    entry: runmat_builtins::HEIGHT_CATALOG_ENTRY,
    internal_error: runmat_builtins::HEIGHT_ERROR_INTERNAL,
    output_error: runmat_builtins::HEIGHT_ERROR_TOO_MANY_OUTPUTS,
    exact_error: Some(&runmat_builtins::HEIGHT_ERROR_RESULT_NOT_EXACT),
    table_error: None,
    builtin_path: "crate::builtins::array::introspection::height"
);

#[cfg(test)]
#[path = "height/tests.rs"]
mod tests;
