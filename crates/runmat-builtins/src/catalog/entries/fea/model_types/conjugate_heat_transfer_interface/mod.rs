//! `fea.ConjugateHeatTransferInterface` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod conductance;
mod perfect;
mod resistance;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] =
    &[&perfect::ENTRY, &conductance::ENTRY, &resistance::ENTRY];
