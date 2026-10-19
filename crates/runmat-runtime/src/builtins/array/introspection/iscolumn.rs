//! Column-shape predicate.

super::shape_predicate::define_shape_predicate_runtime!(
    builtin_fn: iscolumn_builtin,
    name: "iscolumn",
    predicate: runmat_builtins::ShapePredicate::Column,
    entry: runmat_builtins::ISCOLUMN_CATALOG_ENTRY,
    internal_error: runmat_builtins::ISCOLUMN_ERROR_INTERNAL,
    output_error: runmat_builtins::ISCOLUMN_ERROR_TOO_MANY_OUTPUTS,
    builtin_path: "crate::builtins::array::introspection::iscolumn"
);

#[cfg(test)]
#[path = "iscolumn/tests.rs"]
mod tests;
