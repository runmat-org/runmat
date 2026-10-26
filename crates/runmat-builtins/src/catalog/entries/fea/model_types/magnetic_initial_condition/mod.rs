//! `fea.MagneticInitialCondition` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod vector_potential;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&vector_potential::ENTRY];
