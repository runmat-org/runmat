use super::{fea, geometry};

/// Register immutable value classes that are part of the runtime itself.
///
/// Execution workers call this before encoding or decoding portable values so
/// class availability never depends on which builtins happened to run first in
/// that process.
pub(crate) fn ensure_builtin_value_classes_registered() {
    super::math::optim::integer_constraint::ensure_registered();
    fea::ensure_value_classes_registered();
    geometry::region::ensure_region_class_registered();
}
