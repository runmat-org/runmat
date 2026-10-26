use crate::*;

mod continue_from_completed_state;
mod descriptor;
mod fresh;
mod inference;
mod resume_checkpoint;
#[cfg(test)]
mod tests;

use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &fresh::ENTRY,
    &continue_from_completed_state::ENTRY,
    &resume_checkpoint::ENTRY,
];
