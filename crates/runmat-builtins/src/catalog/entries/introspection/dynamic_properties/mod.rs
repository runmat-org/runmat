mod addprop;
mod delete;
mod errors;
mod findprop;
pub use addprop::*;
pub use delete::*;
pub use errors::*;
pub use findprop::*;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &ADDPROP_CATALOG_ENTRY,
    &FINDPROP_CATALOG_ENTRY,
    &DYNAMIC_PROPERTY_DELETE_CATALOG_ENTRY,
];
