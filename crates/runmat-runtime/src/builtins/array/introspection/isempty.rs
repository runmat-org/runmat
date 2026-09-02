//! Empty-shape predicate.

super::shape_predicate::define_shape_predicate_runtime!(
    builtin_fn: isempty_builtin,
    name: "isempty",
    predicate: runmat_builtins::ShapePredicate::Empty,
    entry: runmat_builtins::ISEMPTY_CATALOG_ENTRY,
    internal_error: runmat_builtins::ISEMPTY_ERROR_INTERNAL,
    output_error: runmat_builtins::ISEMPTY_ERROR_TOO_MANY_OUTPUTS,
    builtin_path: "crate::builtins::array::introspection::isempty"
);

#[cfg(test)]
#[path = "isempty/tests.rs"]
mod tests;
