pub(in crate::catalog) mod close;

pub use close::{
    CloseNetworkSelector, CLOSE_CATALOG_ENTRY, CLOSE_DESCRIPTOR, CLOSE_ERROR_INVALID_ARGUMENT,
    CLOSE_ERROR_TOO_MANY_OUTPUTS, CLOSE_EXTENSIONS, CLOSE_INTEGER_CAPABILITIES,
    CLOSE_INTEGER_FIGURE_NUMBER_EXTENSION, CLOSE_NETWORK_SELECTOR_EXTENSION,
    CLOSE_TCP_CELL_EXTENSION, CLOSE_VARIADIC_TARGETS_EXTENSION,
};

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend_from_slice(close::ENTRIES);
}
