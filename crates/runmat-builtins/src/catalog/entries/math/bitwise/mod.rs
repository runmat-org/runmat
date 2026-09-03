mod binary;
mod bitcmp;
mod bitget;
mod bitset;
mod bitshift;
mod support;
mod swapbytes;

pub use binary::*;
pub use bitcmp::*;
pub use bitget::*;
pub use bitset::*;
pub use bitshift::*;
pub use support::*;
pub use swapbytes::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &BITAND_CATALOG_ENTRY,
    &BITCMP_CATALOG_ENTRY,
    &BITGET_CATALOG_ENTRY,
    &BITOR_CATALOG_ENTRY,
    &BITSET_CATALOG_ENTRY,
    &BITSHIFT_CATALOG_ENTRY,
    &BITXOR_CATALOG_ENTRY,
    &SWAPBYTES_CATALOG_ENTRY,
];
