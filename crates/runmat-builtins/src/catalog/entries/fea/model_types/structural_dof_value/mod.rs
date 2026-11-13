//! `fea.StructuralDofValue` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod rotation;
mod translation;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&translation::ENTRY, &rotation::ENTRY];
