//! `fea.PeriodicPairing` constructor family: one catalog identity per static method.

use crate::*;

mod geometric;
mod matching;

pub use crate::FEA_TYPED_CONSTRAINT_INTEGER_AUDIT as FEA_PERIODIC_PAIRING_INTEGER_AUDIT;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&matching::ENTRY, &geometric::ENTRY];
