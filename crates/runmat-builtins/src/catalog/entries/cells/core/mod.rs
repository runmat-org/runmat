pub(in crate::catalog) mod cell2struct;

pub use cell2struct::{
    CELL2STRUCT_CATALOG_ENTRY, CELL2STRUCT_DESCRIPTOR, CELL2STRUCT_ERROR_INVALID_INPUT,
    CELL2STRUCT_ERROR_SHAPE, CELL2STRUCT_INTEGER_CAPABILITIES,
};

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = cell2struct::ENTRIES;
