//! `fea.AcousticStep` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod harmonic;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&harmonic::ENTRY];
