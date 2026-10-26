//! `fea.PeriodicTransform` constructor family: one catalog identity per static method.

use crate::*;

mod identity;
mod rotation;
mod translation;

pub use crate::FEA_TYPED_CONSTRAINT_INTEGER_AUDIT as FEA_PERIODIC_TRANSFORM_INTEGER_AUDIT;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] =
    &[&identity::ENTRY, &translation::ENTRY, &rotation::ENTRY];
