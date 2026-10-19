//! `fea.StepObjectActivation` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod new_activation;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&new_activation::ENTRY];
