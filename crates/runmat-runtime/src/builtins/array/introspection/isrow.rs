//! Row-shape predicate.

super::shape_predicate::define_shape_predicate_runtime!(
    builtin_fn: isrow_builtin,
    name: "isrow",
    predicate: runmat_builtins::ShapePredicate::Row,
    entry: runmat_builtins::ISROW_CATALOG_ENTRY,
    internal_error: runmat_builtins::ISROW_ERROR_INTERNAL,
    output_error: runmat_builtins::ISROW_ERROR_TOO_MANY_OUTPUTS,
    builtin_path: "crate::builtins::array::introspection::isrow"
);

#[cfg(test)]
#[path = "isrow/tests.rs"]
mod tests;
