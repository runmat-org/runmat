//! Scalar-shape predicate.

super::shape_predicate::define_shape_predicate_runtime!(
    builtin_fn: isscalar_builtin,
    name: "isscalar",
    predicate: runmat_builtins::ShapePredicate::Scalar,
    entry: runmat_builtins::ISSCALAR_CATALOG_ENTRY,
    internal_error: runmat_builtins::ISSCALAR_ERROR_INTERNAL,
    output_error: runmat_builtins::ISSCALAR_ERROR_TOO_MANY_OUTPUTS,
    builtin_path: "crate::builtins::array::introspection::isscalar"
);

#[cfg(test)]
#[path = "isscalar/tests.rs"]
mod tests;
