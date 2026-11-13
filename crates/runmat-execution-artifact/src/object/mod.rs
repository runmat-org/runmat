mod access;
mod catalog;
mod descriptor;
mod inventory;
mod policy;
mod project_runtime_policy;

pub use access::ArtifactAccess;
pub use catalog::ArtifactPublishedObjectSource;
pub(crate) use descriptor::validate_logical_name;
pub use descriptor::{LogicalObject, ObjectDescriptor, ObjectNamespace};
pub use inventory::{
    inventory_validation_scratch_bytes, validate_inventory, InventoryCapacityError,
    ObjectInventoryLimits, PLATFORM_ADDRESS_SPACE_BYTES, PLATFORM_MAXIMUM_OBJECT_BYTES,
};
pub use policy::{
    capture_typed_foreign_policy, decode_typed_foreign_policy, foreign_policy_descriptor,
    foreign_policy_object, foreign_policy_object_from_json_value, foreign_policy_ref,
    validate_foreign_policy_descriptor, validate_foreign_policy_object, MAX_FOREIGN_POLICY_BYTES,
};
pub use project_runtime_policy::{
    capture_typed_project_runtime_policy, decode_typed_project_runtime_policy,
    validate_project_runtime_policy_descriptor, validate_project_runtime_policy_object,
    MAX_PROJECT_RUNTIME_POLICY_BYTES, PROJECT_RUNTIME_POLICY_MEDIA_TYPE,
    PROJECT_RUNTIME_POLICY_SCHEMA_VERSION,
};
