//! `fea.StructuralStep` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod harmonic;
mod modal;
mod nonlinear;
mod static_step;
mod transient;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &static_step::ENTRY,
    &modal::ENTRY,
    &harmonic::ENTRY,
    &transient::ENTRY,
    &nonlinear::ENTRY,
];
