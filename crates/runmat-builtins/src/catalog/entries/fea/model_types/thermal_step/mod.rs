//! `fea.ThermalStep` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod steady;
mod transient;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&steady::ENTRY, &transient::ENTRY];
