mod ss;
mod tf;

pub use self::ss::*;
pub use self::tf::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&SS_CATALOG_ENTRY, &TF_CATALOG_ENTRY];
