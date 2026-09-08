pub(in crate::catalog) mod cell;
pub(in crate::catalog) mod cell2struct;
pub(in crate::catalog) mod cellstr;
pub(in crate::catalog) mod num2cell;

pub use cell::{
    CELL_CATALOG_ENTRY, CELL_DESCRIPTOR, CELL_ERROR_INTERNAL, CELL_ERROR_INVALID_INPUT,
    CELL_ERROR_INVALID_SIZE, CELL_EXTENSIONS, CELL_GPU_SIZE_EXTENSION, CELL_INTEGER_CAPABILITIES,
    CELL_LIKE_EXTENSION,
};
pub use cell2struct::{
    CELL2STRUCT_CATALOG_ENTRY, CELL2STRUCT_DESCRIPTOR, CELL2STRUCT_ERROR_INVALID_INPUT,
    CELL2STRUCT_ERROR_SHAPE, CELL2STRUCT_INTEGER_CAPABILITIES,
};
pub use cellstr::{
    CELLSTR_CATALOG_ENTRY, CELLSTR_CELL_INPUT_EXTENSION, CELLSTR_DESCRIPTOR,
    CELLSTR_ERROR_INTERNAL, CELLSTR_ERROR_INVALID_CONTENTS, CELLSTR_ERROR_INVALID_INPUT,
    CELLSTR_EXTENSIONS, CELLSTR_INTEGER_AUDIT, CELLSTR_SYMBOLIC_INPUT_EXTENSION,
};
pub use num2cell::{
    NUM2CELL_CATALOG_ENTRY, NUM2CELL_DESCRIPTOR, NUM2CELL_ERROR_INTERNAL,
    NUM2CELL_ERROR_INVALID_INPUT, NUM2CELL_INTEGER_CAPABILITIES,
};

pub(super) fn extend_entries(entries: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    super::super::extend_groups(
        entries,
        &[
            cell::ENTRIES,
            cell2struct::ENTRIES,
            cellstr::ENTRIES,
            num2cell::ENTRIES,
        ],
    );
}
