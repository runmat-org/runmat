//! Catalog identities of the `fea.SpatialReduction` typed policy class.

use crate::*;

mod average;
mod euclidean_norm;
mod integral;
mod maximum;
mod minimum;
mod moment;
mod none_method;
mod resultant;
mod root_mean_square;
#[cfg(test)]
mod tests;

pub(super) const ROLE: &str = concat!(
    "A spatial reduction collapses an observation's values over its target into one ",
    "global result before any series reduction. Integrals, averages, RMS values, and ",
    "norms use the admitted finite-element quadrature; partitioned evaluation integrates ",
    "owned cells and faces and combines them with a deterministic compensated reduction.",
);
pub(super) const STUDY: &str = concat!(
    "Pass the spatial reduction as the first argument of ",
    "`fea.ReductionPolicy.new(spatial, series, extremaTies)`, and pass that policy as the",
    " twelfth argument of an observation constructor. `fea.Observation.field`, `history`,",
    " `probe`, and `reduction` take `(id, quantity, target, sourceLocation, ",
    "outputLocation, recovery, component, frame, sign, complexRepresentation, sampling, ",
    "reduction, storage, purpose, cost)`.",
);
pub(super) const FAQ: BuiltinDocumentationFaq = BuiltinDocumentationFaq {
    question: "Is there an implicit origin for moments?",
    answer: concat!(
        "No. Use `fea.SpatialReduction.moment(about)` with an explicit `fea.Position`; RunMat",
        " never assumes the model origin.",
    ),
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &none_method::FEA_SPATIAL_REDUCTION_NONE_CATALOG_ENTRY,
    &integral::FEA_SPATIAL_REDUCTION_INTEGRAL_CATALOG_ENTRY,
    &average::FEA_SPATIAL_REDUCTION_AVERAGE_CATALOG_ENTRY,
    &root_mean_square::FEA_SPATIAL_REDUCTION_ROOT_MEAN_SQUARE_CATALOG_ENTRY,
    &euclidean_norm::FEA_SPATIAL_REDUCTION_EUCLIDEAN_NORM_CATALOG_ENTRY,
    &minimum::FEA_SPATIAL_REDUCTION_MINIMUM_CATALOG_ENTRY,
    &maximum::FEA_SPATIAL_REDUCTION_MAXIMUM_CATALOG_ENTRY,
    &resultant::FEA_SPATIAL_REDUCTION_RESULTANT_CATALOG_ENTRY,
    &moment::FEA_SPATIAL_REDUCTION_MOMENT_CATALOG_ENTRY,
];
