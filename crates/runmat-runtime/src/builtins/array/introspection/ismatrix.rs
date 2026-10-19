//! Matrix-shape predicate.

super::shape_predicate::define_shape_predicate_runtime!(
    builtin_fn: ismatrix_builtin,
    name: "ismatrix",
    predicate: runmat_builtins::ShapePredicate::Matrix,
    entry: runmat_builtins::ISMATRIX_CATALOG_ENTRY,
    internal_error: runmat_builtins::ISMATRIX_ERROR_INTERNAL,
    output_error: runmat_builtins::ISMATRIX_ERROR_TOO_MANY_OUTPUTS,
    builtin_path: "crate::builtins::array::introspection::ismatrix"
);

#[cfg(test)]
#[path = "ismatrix/tests.rs"]
mod tests;
