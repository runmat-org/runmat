//! Catalog identities of the `fea.SpectrumAxis` typed policy class.

use crate::*;

mod frequency_hertz;
mod period_seconds;
#[cfg(test)]
mod tests;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &frequency_hertz::FEA_SPECTRUM_AXIS_FREQUENCY_HERTZ_CATALOG_ENTRY,
    &period_seconds::FEA_SPECTRUM_AXIS_PERIOD_SECONDS_CATALOG_ENTRY,
];
