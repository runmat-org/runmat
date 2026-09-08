pub(in crate::catalog) mod cell2struct;
pub(in crate::catalog) mod cellstr;

pub use cell2struct::{
    CELL2STRUCT_CATALOG_ENTRY, CELL2STRUCT_DESCRIPTOR, CELL2STRUCT_ERROR_INVALID_INPUT,
    CELL2STRUCT_ERROR_SHAPE, CELL2STRUCT_INTEGER_CAPABILITIES,
};
pub use cellstr::{
    CELLSTR_CATALOG_ENTRY, CELLSTR_CELL_INPUT_EXTENSION, CELLSTR_DESCRIPTOR,
    CELLSTR_ERROR_INTERNAL, CELLSTR_ERROR_INVALID_CONTENTS, CELLSTR_ERROR_INVALID_INPUT,
    CELLSTR_EXTENSIONS, CELLSTR_INTEGER_AUDIT, CELLSTR_SYMBOLIC_INPUT_EXTENSION,
};

pub(super) fn extend_entries(entries: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    super::super::extend_groups(entries, &[cell2struct::ENTRIES, cellstr::ENTRIES]);
}
