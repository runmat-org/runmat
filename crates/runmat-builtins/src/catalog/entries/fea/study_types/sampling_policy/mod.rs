use crate::*;

mod adaptive;
mod descriptor;
mod event;
mod every_accepted_increment;
mod final_sample;
mod frequency_hertz;
mod frequency_interval_hertz;
mod inference;
mod load_factor_interval;
mod load_factors;
mod mode_indices;
mod study_parameter_values;
mod time_interval_seconds;
mod time_seconds;
#[cfg(test)]
mod tests;

use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &final_sample::ENTRY,
    &every_accepted_increment::ENTRY,
    &time_seconds::ENTRY,
    &time_interval_seconds::ENTRY,
    &frequency_hertz::ENTRY,
    &frequency_interval_hertz::ENTRY,
    &load_factors::ENTRY,
    &load_factor_interval::ENTRY,
    &study_parameter_values::ENTRY,
    &mode_indices::ENTRY,
    &event::ENTRY,
    &adaptive::ENTRY,
];
