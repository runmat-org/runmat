//! `fea.ContactInterface` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod coulomb;
mod frictionless;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&frictionless::ENTRY, &coulomb::ENTRY];
