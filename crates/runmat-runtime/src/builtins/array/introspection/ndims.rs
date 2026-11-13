use super::shape_scalar_query::define_shape_scalar_query_runtime;

define_shape_scalar_query_runtime!(
    builtin_fn: ndims_builtin,
    name: "ndims",
    query: runmat_builtins::ShapeScalarQuery::Rank,
    entry: runmat_builtins::NDIMS_CATALOG_ENTRY,
    internal_error: runmat_builtins::NDIMS_ERROR_INTERNAL,
    output_error: runmat_builtins::NDIMS_ERROR_TOO_MANY_OUTPUTS,
    exact_error: None,
    table_error: None,
    builtin_path: "crate::builtins::array::introspection::ndims"
);

#[cfg(test)]
#[path = "ndims/tests.rs"]
mod tests;
