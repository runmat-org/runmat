use crate::*;

mod accuracy_criterion_crossing;
mod contact_status_change;
mod convergence_failure;
mod descriptor;
mod first_yield;
mod inference;
mod peak_response;
#[cfg(test)]
mod tests;

use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &first_yield::ENTRY,
    &contact_status_change::ENTRY,
    &peak_response::ENTRY,
    &convergence_failure::ENTRY,
    &accuracy_criterion_crossing::ENTRY,
];
