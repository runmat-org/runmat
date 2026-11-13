//! Catalog identities of the `fea.SeriesWeighting` typed policy class.

use crate::*;

mod trapezoidal_independent_domain;
mod uniform_samples;
#[cfg(test)]
mod tests;

pub(super) const ROLE: &str = concat!(
    "A series weighting states how the samples of a history contribute to a statistic ",
    "such as a mean, RMS, or variance.",
);
pub(super) const STUDY: &str =
    "Pass the weighting to `fea.SeriesReduction.mean`, `rootMeanSquare`, or `variance`.";
pub(super) const FAQ: BuiltinDocumentationFaq = BuiltinDocumentationFaq {
    question: "Which weighting should a transient history use?",
    answer: concat!(
        "Usually `trapezoidalIndependentDomain()`, so that adaptive or nonuniform time steps ",
        "do not bias the statistic.",
    ),
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &uniform_samples::FEA_SERIES_WEIGHTING_UNIFORM_SAMPLES_CATALOG_ENTRY,
    &trapezoidal_independent_domain::FEA_SERIES_WEIGHTING_TRAPEZOIDAL_INDEPENDENT_DOMAIN_CATALOG_ENTRY,
];
