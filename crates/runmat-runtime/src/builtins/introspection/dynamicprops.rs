//! MATLAB-compatible dynamic property support for handle objects.
pub const DYNAMICPROPS_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("dynamicprops");
pub const DYNAMIC_PROPERTY_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("matlab.metadata.DynamicProperty");
pub const STATIC_PROPERTY_METADATA_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("matlab.metadata.Property");

pub(super) const TARGET_FIELD: &str = "__runmat_dynamic_property_target__";
pub(super) const VALID_FIELD: &str = "__runmat_dynamic_property_valid__";

mod lifecycle;
mod metadata;
mod projection;
pub(crate) mod operations;
pub(crate) mod selected_graph;
#[cfg(test)]
mod tests;

pub(crate) use metadata::dynamic_property_assignment_access;
pub use metadata::{
    dynamic_property_assign, dynamic_property_exists, dynamic_property_get,
    dynamic_property_metadata_struct, dynamic_property_read, empty_metadata_result,
    metadata_assignment, object_dynamic_property_names,
};
pub(crate) use selected_graph::{checked_metadata_owner, selected_metadata_object};
