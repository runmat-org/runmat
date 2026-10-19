//! `fea.StepObjectRef` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod boundary;
mod connector;
mod constraint;
mod excitation;
mod initial_condition;
mod interface;
mod material_parameter;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &constraint::ENTRY,
    &boundary::ENTRY,
    &connector::ENTRY,
    &interface::ENTRY,
    &excitation::ENTRY,
    &initial_condition::ENTRY,
    &material_parameter::ENTRY,
];
