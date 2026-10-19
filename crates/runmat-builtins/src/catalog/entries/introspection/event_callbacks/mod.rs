mod addlistener;
mod notify;
pub use addlistener::*;
pub use notify::*;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] =
    &[&ADDLISTENER_CATALOG_ENTRY, &NOTIFY_CATALOG_ENTRY];
