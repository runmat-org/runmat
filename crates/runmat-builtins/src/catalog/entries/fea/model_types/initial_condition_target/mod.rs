//! `fea.InitialConditionTarget` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod whole_model;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&whole_model::ENTRY];
