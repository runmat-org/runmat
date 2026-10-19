//! `fea.PropertyRegionTarget` constructor family: one catalog identity per static method.

use crate::*;

mod descriptor;
mod inference;
mod all_beams;
mod all_cells;
mod all_shells;
mod all_solids;
mod beam;
mod cell;
mod shell;
mod solid;
#[cfg(test)]
mod tests;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &beam::ENTRY,
    &shell::ENTRY,
    &solid::ENTRY,
    &cell::ENTRY,
    &all_beams::ENTRY,
    &all_shells::ENTRY,
    &all_solids::ENTRY,
    &all_cells::ENTRY,
];
