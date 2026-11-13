//! Catalog identities of the `fea.SeriesReduction` typed policy class.

use crate::*;

mod absolute_maximum;
mod envelope;
mod maximum;
mod mean;
mod minimum;
mod none_method;
mod root_mean_square;
mod variance;
#[cfg(test)]
mod tests;

pub(super) const ROLE: &str = concat!(
    "A series reduction operates on a completed canonical history, after any spatial ",
    "reduction. It requires every produced sample, so a history retained with ",
    "`latestSamples` cannot be reduced, and event histories cannot request one. Extremal ",
    "results follow the extrema-tie policy.",
);
pub(super) const STUDY: &str = concat!(
    "Pass the series reduction as the second argument of ",
    "`fea.ReductionPolicy.new(spatial, series, extremaTies)`, and pass that policy as the",
    " twelfth argument of an observation constructor. `fea.Observation.field`, `history`,",
    " `probe`, and `reduction` take `(id, quantity, target, sourceLocation, ",
    "outputLocation, recovery, component, frame, sign, complexRepresentation, sampling, ",
    "reduction, storage, purpose, cost)`.",
);
pub(super) const FAQ: BuiltinDocumentationFaq = BuiltinDocumentationFaq {
    question: "Why does mean need a weighting argument?",
    answer: concat!(
        "So a nonuniform history is never silently averaged as if it were uniform. The ",
        "statistic owns its `fea.SeriesWeighting`.",
    ),
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &none_method::FEA_SERIES_REDUCTION_NONE_CATALOG_ENTRY,
    &minimum::FEA_SERIES_REDUCTION_MINIMUM_CATALOG_ENTRY,
    &maximum::FEA_SERIES_REDUCTION_MAXIMUM_CATALOG_ENTRY,
    &absolute_maximum::FEA_SERIES_REDUCTION_ABSOLUTE_MAXIMUM_CATALOG_ENTRY,
    &envelope::FEA_SERIES_REDUCTION_ENVELOPE_CATALOG_ENTRY,
    &root_mean_square::FEA_SERIES_REDUCTION_ROOT_MEAN_SQUARE_CATALOG_ENTRY,
    &mean::FEA_SERIES_REDUCTION_MEAN_CATALOG_ENTRY,
    &variance::FEA_SERIES_REDUCTION_VARIANCE_CATALOG_ENTRY,
];
