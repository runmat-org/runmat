mod gpu_array;
mod transfer;

pub use gpu_array::*;
pub use transfer::*;

pub(super) const ENTRY_GROUPS: &[&[&crate::BuiltinCatalogEntry]] =
    &[gpu_array::ENTRIES, transfer::ENTRIES];
