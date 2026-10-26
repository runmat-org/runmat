//! Catalog identities of the `fea.RigidResponse` typed policy class.

use crate::*;

mod gupta;
mod lindley_yow;
#[cfg(test)]
mod tests;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &gupta::FEA_RIGID_RESPONSE_GUPTA_CATALOG_ENTRY,
    &lindley_yow::FEA_RIGID_RESPONSE_LINDLEY_YOW_CATALOG_ENTRY,
];
