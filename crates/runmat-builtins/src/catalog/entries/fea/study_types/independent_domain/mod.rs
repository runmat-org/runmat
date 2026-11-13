use crate::*;

mod descriptor;
mod eigenvalue;
mod frequency_hertz;
mod inference;
mod load_factor;
mod response_spectra;
mod static_domain;
mod study_parameter;
mod time_seconds;
#[cfg(test)]
mod tests;

use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &static_domain::ENTRY,
    &time_seconds::ENTRY,
    &frequency_hertz::ENTRY,
    &load_factor::ENTRY,
    &eigenvalue::ENTRY,
    &study_parameter::ENTRY,
    &response_spectra::ENTRY,
];
