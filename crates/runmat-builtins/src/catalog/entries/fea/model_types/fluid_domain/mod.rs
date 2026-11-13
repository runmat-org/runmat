//! `fea.FluidDomain` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod newtonian;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&newtonian::ENTRY];
