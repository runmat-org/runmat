//! Catalog identities of the `fea.SignalSampling` typed policy class.

use crate::*;

mod adaptive;
mod canonical_points;
mod uniform;
#[cfg(test)]
mod tests;

pub(super) const SAMPLING_ROLE: &str = concat!(
    "A sampling policy says at which coordinates of the signal domain a generated ",
    "signal's function is evaluated. Coordinates are in the domain's unit (seconds, ",
    "hertz, or the study parameter's canonical value). Between samples the signal is ",
    "interpolated with the signal's interpolation policy. Validation happens before a ",
    "value is returned: every sampling policy must be finite, strictly increasing, and ",
    "contain at least two points, or the call fails with `RunMat:fea:InvalidInput`.",
);
pub(super) const SAMPLING_STUDY: &str = concat!(
    "Pass the policy as the fourth argument of a callable signal constructor such as ",
    "`fea.ForceSignal.fromFunction(id, callable, domain, sampling)`. For a frequency ",
    "domain the first sample must be at or above 0 Hz.",
);

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &uniform::FEA_SIGNAL_SAMPLING_UNIFORM_CATALOG_ENTRY,
    &adaptive::FEA_SIGNAL_SAMPLING_ADAPTIVE_CATALOG_ENTRY,
    &canonical_points::FEA_SIGNAL_SAMPLING_CANONICAL_POINTS_CATALOG_ENTRY,
];
