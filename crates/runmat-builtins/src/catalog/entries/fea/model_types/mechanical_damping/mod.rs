//! `fea.MechanicalDamping` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod rayleigh;
mod undamped;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&undamped::ENTRY, &rayleigh::ENTRY];
