//! `fea.ElectromagneticDomain` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod at_frequency;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&at_frequency::ENTRY];
