//! Canonical path-level dataset contracts. Shared storage algorithms remain in runtime/data.

mod contract;
mod copy;
mod create;
mod delete;
mod descriptor;
mod documentation;
mod exists;
mod export;
mod import;
mod inference;
mod inspect;
mod list;
mod move_path;
mod open;
#[cfg(test)]
mod tests;

pub use contract::{DATASET_CLASS, DATA_NAMESPACE_LIFECYCLE_INTEGER_AUDIT};
pub use copy::DATA_COPY_CATALOG_ENTRY;
pub use create::DATA_CREATE_CATALOG_ENTRY;
pub use delete::DATA_DELETE_CATALOG_ENTRY;
pub use descriptor::{
    DATA_COPY_DESCRIPTOR, DATA_CREATE_DESCRIPTOR, DATA_CREATE_INTEGER_CAPABILITIES,
    DATA_DELETE_DESCRIPTOR, DATA_ERROR, DATA_EXISTS_DESCRIPTOR, DATA_EXPORT_DESCRIPTOR,
    DATA_IMPORT_DESCRIPTOR, DATA_INSPECT_DESCRIPTOR, DATA_LIST_DESCRIPTOR, DATA_MOVE_DESCRIPTOR,
    DATA_OPEN_DESCRIPTOR,
};
pub use exists::DATA_EXISTS_CATALOG_ENTRY;
pub use export::DATA_EXPORT_CATALOG_ENTRY;
pub use import::DATA_IMPORT_CATALOG_ENTRY;
pub use inspect::DATA_INSPECT_CATALOG_ENTRY;
pub use list::DATA_LIST_CATALOG_ENTRY;
pub use move_path::DATA_MOVE_CATALOG_ENTRY;
pub use open::DATA_OPEN_CATALOG_ENTRY;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &DATA_COPY_CATALOG_ENTRY,
    &DATA_CREATE_CATALOG_ENTRY,
    &DATA_DELETE_CATALOG_ENTRY,
    &DATA_EXISTS_CATALOG_ENTRY,
    &DATA_EXPORT_CATALOG_ENTRY,
    &DATA_IMPORT_CATALOG_ENTRY,
    &DATA_INSPECT_CATALOG_ENTRY,
    &DATA_LIST_CATALOG_ENTRY,
    &DATA_MOVE_CATALOG_ENTRY,
    &DATA_OPEN_CATALOG_ENTRY,
];
