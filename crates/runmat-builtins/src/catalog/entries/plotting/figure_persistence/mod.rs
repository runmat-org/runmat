mod contract;
mod hgload;
mod hgsave;
mod openfig;
mod saveas;
mod savefig;
#[cfg(test)]
mod tests;

use crate::BuiltinCatalogEntry;

pub use contract::*;
pub use hgload::*;
pub use hgsave::*;
pub use openfig::*;
pub use saveas::*;
pub use savefig::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &HGLOAD_CATALOG_ENTRY,
    &HGSAVE_CATALOG_ENTRY,
    &OPENFIG_CATALOG_ENTRY,
    &SAVEAS_CATALOG_ENTRY,
    &SAVEFIG_CATALOG_ENTRY,
];
