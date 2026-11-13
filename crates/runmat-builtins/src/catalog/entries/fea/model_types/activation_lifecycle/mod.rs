//! `fea.ActivationLifecycle` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod activate;
mod continue_from;
mod deactivate;
mod modify;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &activate::ENTRY,
    &continue_from::ENTRY,
    &deactivate::ENTRY,
    &modify::ENTRY,
];
