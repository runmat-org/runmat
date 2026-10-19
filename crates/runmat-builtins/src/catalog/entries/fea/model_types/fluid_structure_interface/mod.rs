//! `fea.FluidStructureInterface` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod no_slip;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&no_slip::ENTRY];
