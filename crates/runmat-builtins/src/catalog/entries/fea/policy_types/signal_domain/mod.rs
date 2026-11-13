//! Catalog identities of the `fea.SignalDomain` typed policy class.

use crate::*;

mod frequency;
mod study_parameter;
mod study_parameter_with_units;
mod time;
#[cfg(test)]
mod tests;

pub(super) const DOMAIN_ROLE: &str = concat!(
    "A signal domain names the independent variable a generated signal is a function of. ",
    "It fixes the unit of every sample coordinate: seconds for time, hertz for frequency,",
    " or the canonical value of a named study parameter. The value is immutable and ",
    "carries no function or samples of its own.",
);
pub(super) const DOMAIN_STUDY: &str = concat!(
    "Pass the domain as the third argument of a callable signal constructor such as ",
    "`fea.ForceSignal.fromFunction(id, callable, domain, sampling)`, together with an ",
    "`fea.SignalSampling` policy whose coordinates are in the domain's unit. Register the",
    " resulting signal with `fea.model(..., \"Signals\", {signal})` and reference it from ",
    "an excitation.",
);

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &time::FEA_SIGNAL_DOMAIN_TIME_CATALOG_ENTRY,
    &frequency::FEA_SIGNAL_DOMAIN_FREQUENCY_CATALOG_ENTRY,
    &study_parameter::FEA_SIGNAL_DOMAIN_STUDY_PARAMETER_CATALOG_ENTRY,
    &study_parameter_with_units::FEA_SIGNAL_DOMAIN_STUDY_PARAMETER_WITH_UNITS_CATALOG_ENTRY,
];
