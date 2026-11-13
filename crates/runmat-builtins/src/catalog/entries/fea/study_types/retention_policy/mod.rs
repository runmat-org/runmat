use crate::*;

mod canonical;
mod descriptor;
mod explicit_samples;
mod inference;
mod latest_samples;
mod until_dependents_complete;
#[cfg(test)]
mod tests;

use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &canonical::ENTRY,
    &until_dependents_complete::ENTRY,
    &latest_samples::ENTRY,
    &explicit_samples::ENTRY,
];
