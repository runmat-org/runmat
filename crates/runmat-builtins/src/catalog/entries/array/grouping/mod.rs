mod findgroups;
mod groupcounts;
mod grp2idx;
mod splitapply;

pub use findgroups::*;
pub use groupcounts::*;
pub use grp2idx::*;
pub use splitapply::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &FINDGROUPS_CATALOG_ENTRY,
    &GRP2IDX_CATALOG_ENTRY,
    &GROUPCOUNTS_CATALOG_ENTRY,
    &SPLITAPPLY_CATALOG_ENTRY,
];
