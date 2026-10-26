//! `fea.ExcitationCase` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod always;
mod during_steps;
mod mutually_exclusive;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &always::ENTRY,
    &during_steps::ENTRY,
    &mutually_exclusive::ENTRY,
];
