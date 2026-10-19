//! `fea.ReferencePoint` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod from_coordinates;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&from_coordinates::ENTRY];
