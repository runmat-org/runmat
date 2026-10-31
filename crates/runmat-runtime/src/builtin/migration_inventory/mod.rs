mod build;
mod environment;
mod projection;
mod registration;
mod schema;
mod validation;

pub use build::{migration_inventory, migration_inventory_json};
pub use registration::registration_manifest;
pub use schema::*;

/// Compile-time feature facts for the runtime linked into this product.
pub fn runtime_feature_inventory() -> CrateFeatureInventory {
    environment::build_configuration().crate_feature_inventory
}

#[cfg(test)]
mod tests;
