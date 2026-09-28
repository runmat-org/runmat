mod bi2de;
mod de2bi;
mod inference;

pub use bi2de::*;
pub use de2bi::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] =
    &[&BI2DE_CATALOG_ENTRY, &DE2BI_CATALOG_ENTRY];
