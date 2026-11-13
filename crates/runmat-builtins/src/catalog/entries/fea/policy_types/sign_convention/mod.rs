//! Catalog identities of the `fea.SignConvention` typed policy class.

use crate::*;

mod compression_positive;
mod inward_positive;
mod native;
mod outward_positive;
mod tension_positive;
#[cfg(test)]
mod tests;

pub(super) const ROLE: &str = concat!(
    "A sign convention states which direction an observation reports as positive. A ",
    "convention that is not meaningful for the quantity fails study validation instead of",
    " silently flipping a sign.",
);
pub(super) const STUDY: &str = concat!(
    "Pass the convention as the ninth argument of an observation constructor. ",
    "`fea.Observation.field`, `history`, `probe`, and `reduction` take `(id, quantity, ",
    "target, sourceLocation, outputLocation, recovery, component, frame, sign, ",
    "complexRepresentation, sampling, reduction, storage, purpose, cost)`.",
);
pub(super) const FAQ: BuiltinDocumentationFaq = BuiltinDocumentationFaq {
    question: "Does a sign convention change stored solver data?",
    answer: "No. It applies only to the reported values of that observation.",
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &native::FEA_SIGN_CONVENTION_NATIVE_CATALOG_ENTRY,
    &tension_positive::FEA_SIGN_CONVENTION_TENSION_POSITIVE_CATALOG_ENTRY,
    &compression_positive::FEA_SIGN_CONVENTION_COMPRESSION_POSITIVE_CATALOG_ENTRY,
    &outward_positive::FEA_SIGN_CONVENTION_OUTWARD_POSITIVE_CATALOG_ENTRY,
    &inward_positive::FEA_SIGN_CONVENTION_INWARD_POSITIVE_CATALOG_ENTRY,
];
