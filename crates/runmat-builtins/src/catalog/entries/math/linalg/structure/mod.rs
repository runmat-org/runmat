mod inference;
mod bandwidth;
mod ishermitian;
mod issymmetric;
mod isdiag;
mod istril;
mod istriu;
mod symrcm;

pub use bandwidth::*;
pub use isdiag::*;
pub use ishermitian::*;
pub use issymmetric::*;
pub use istril::*;
pub use istriu::*;
pub use symrcm::*;

pub(super) fn extend_entries(entries: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    entries.extend([
        &BANDWIDTH_CATALOG_ENTRY,
        &ISHERMITIAN_CATALOG_ENTRY,
        &ISSYMMETRIC_CATALOG_ENTRY,
        &ISDIAG_CATALOG_ENTRY,
        &ISTRIL_CATALOG_ENTRY,
        &ISTRIU_CATALOG_ENTRY,
        &SYMRCM_CATALOG_ENTRY,
    ]);
}
