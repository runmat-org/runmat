mod find;
mod ind2sub;
mod sub2ind;

pub use find::*;
pub use ind2sub::*;
pub use sub2ind::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &FIND_CATALOG_ENTRY,
    &IND2SUB_CATALOG_ENTRY,
    &SUB2IND_CATALOG_ENTRY,
];
