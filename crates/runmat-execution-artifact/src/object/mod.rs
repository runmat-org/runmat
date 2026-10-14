mod descriptor;
mod inventory;
mod policy;

pub use descriptor::{LogicalObject, ObjectDescriptor, ObjectNamespace};
pub use inventory::{validate_inventory, ObjectInventoryLimits};
pub use policy::{
    capture_typed_foreign_policy, decode_typed_foreign_policy, foreign_policy_descriptor,
    foreign_policy_object, foreign_policy_object_from_json_value, foreign_policy_ref,
    validate_foreign_policy_descriptor, validate_foreign_policy_object, MAX_FOREIGN_POLICY_BYTES,
};
