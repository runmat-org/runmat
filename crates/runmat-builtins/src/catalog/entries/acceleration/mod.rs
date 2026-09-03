mod documentation;
mod gpu_array;
mod transfer;

pub use gpu_array::*;
pub use transfer::*;

pub(super) fn extend_entries(entries: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    super::extend_groups(entries, &[gpu_array::ENTRIES, transfer::ENTRIES]);
}
