//! Vector-shape predicate.

super::shape_predicate::define_shape_predicate_runtime!(
    builtin_fn: isvector_builtin,
    name: "isvector",
    predicate: runmat_builtins::ShapePredicate::Vector,
    entry: runmat_builtins::ISVECTOR_CATALOG_ENTRY,
    internal_error: runmat_builtins::ISVECTOR_ERROR_INTERNAL,
    output_error: runmat_builtins::ISVECTOR_ERROR_TOO_MANY_OUTPUTS,
    builtin_path: "crate::builtins::array::introspection::isvector"
);

#[cfg(test)]
#[path = "isvector/tests.rs"]
mod tests;
