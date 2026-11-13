//! `fea.AcousticBoundary` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod impedance;
mod radiation;
mod rigid_wall;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] =
    &[&rigid_wall::ENTRY, &radiation::ENTRY, &impedance::ENTRY];
