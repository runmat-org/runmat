//! `fea.FluidStructureInterface` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod no_slip;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&no_slip::ENTRY];
