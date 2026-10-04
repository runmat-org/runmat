//! Level-5 MAT persistence and its whole-variable object protocol.

mod contract;
mod documentation;
mod inference;
mod load;
mod matfile;
mod option;
mod save;
mod subsasgn;
mod subsref;
#[cfg(test)]
mod tests;

pub use contract::{
    LOAD_DESCRIPTOR, LOAD_INTEGER_CAPABILITIES, MATFILE_CLASS, MATFILE_DESCRIPTOR,
    MATFILE_INTEGER_CAPABILITIES, MATFILE_SUBSASGN_DESCRIPTOR,
    MATFILE_SUBSASGN_INTEGER_CAPABILITIES, MATFILE_SUBSREF_DESCRIPTOR,
    MATFILE_SUBSREF_INTEGER_CAPABILITIES, SAVE_DESCRIPTOR, SAVE_STATUS_EXTENSION,
};
pub use load::LOAD_CATALOG_ENTRY;
pub use matfile::MATFILE_CATALOG_ENTRY;
pub use option::{MatFileOptionName, MatLoadOption, MatSaveOption};
pub use save::SAVE_CATALOG_ENTRY;
pub use subsasgn::MATFILE_SUBSASGN_CATALOG_ENTRY;
pub use subsref::MATFILE_SUBSREF_CATALOG_ENTRY;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &LOAD_CATALOG_ENTRY,
    &MATFILE_CATALOG_ENTRY,
    &MATFILE_SUBSASGN_CATALOG_ENTRY,
    &MATFILE_SUBSREF_CATALOG_ENTRY,
    &SAVE_CATALOG_ENTRY,
];
