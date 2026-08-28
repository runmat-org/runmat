mod descriptor;
mod inventory;

pub use descriptor::{LogicalObject, ObjectDescriptor, ObjectNamespace};
pub use inventory::{validate_inventory, ObjectInventoryLimits};
