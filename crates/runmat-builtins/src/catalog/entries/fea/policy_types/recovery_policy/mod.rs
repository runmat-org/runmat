//! Catalog identities of the `fea.RecoveryPolicy` typed policy class.

use crate::*;

mod area_weighted_average;
mod conservative_projection;
mod extrapolate;
mod native;
mod shape_function_interpolation;
mod volume_weighted_average;
#[cfg(test)]
mod tests;

pub(super) const ROLE: &str = concat!(
    "A recovery policy states how an observation moves values from its source location to",
    " its output location. RunMat never chooses a transfer implicitly: an incompatible ",
    "pairing of recovery and locations fails study validation.",
);
pub(super) const STUDY: &str = concat!(
    "Pass the policy as the sixth argument of an observation constructor. ",
    "`fea.Observation.field`, `history`, `probe`, and `reduction` take `(id, quantity, ",
    "target, sourceLocation, outputLocation, recovery, component, frame, sign, ",
    "complexRepresentation, sampling, reduction, storage, purpose, cost)`.",
);
pub(super) const FAQ: BuiltinDocumentationFaq = BuiltinDocumentationFaq {
    question: "Why is my nodal stress observation rejected with native recovery?",
    answer: concat!(
        "Stress is computed at integration points. Reporting it at nodes needs an explicit ",
        "transfer such as `extrapolate` or `volumeWeightedAverage`.",
    ),
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &native::FEA_RECOVERY_POLICY_NATIVE_CATALOG_ENTRY,
    &extrapolate::FEA_RECOVERY_POLICY_EXTRAPOLATE_CATALOG_ENTRY,
    &volume_weighted_average::FEA_RECOVERY_POLICY_VOLUME_WEIGHTED_AVERAGE_CATALOG_ENTRY,
    &area_weighted_average::FEA_RECOVERY_POLICY_AREA_WEIGHTED_AVERAGE_CATALOG_ENTRY,
    &conservative_projection::FEA_RECOVERY_POLICY_CONSERVATIVE_PROJECTION_CATALOG_ENTRY,
    &shape_function_interpolation::FEA_RECOVERY_POLICY_SHAPE_FUNCTION_INTERPOLATION_CATALOG_ENTRY,
];
