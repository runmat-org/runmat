use crate::*;

mod descriptor;
mod diagnostic;
mod field;
mod history;
mod inference;
mod monitor;
mod probe;
mod reduction;
#[cfg(test)]
mod tests;

use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &field::ENTRY,
    &history::ENTRY,
    &probe::ENTRY,
    &reduction::ENTRY,
    &diagnostic::ENTRY,
    &monitor::ENTRY,
];
