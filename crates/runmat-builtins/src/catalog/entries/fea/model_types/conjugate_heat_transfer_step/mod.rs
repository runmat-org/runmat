//! `fea.ConjugateHeatTransferStep` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod steady_state;
mod transient;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&steady_state::ENTRY, &transient::ENTRY];
