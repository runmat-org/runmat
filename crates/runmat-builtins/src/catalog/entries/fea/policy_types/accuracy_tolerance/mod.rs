//! Catalog identities of the `fea.AccuracyTolerance` typed policy class.

use crate::*;

mod absolute;
mod relative;
#[cfg(test)]
mod tests;

pub(super) const TOLERANCE_ROLE: &str = concat!(
    "A tolerance is the acceptance bound of one `fea.AccuracyCriterion`. The constructor ",
    "checks only that the value is a finite real scalar and stores it as binary64. ",
    "Positivity is enforced when the study is validated: a zero, negative, or nonfinite ",
    "tolerance fails with an invalid accuracy plan error before any solve starts.",
);
pub(super) const TOLERANCE_STUDY: &str = concat!(
    "Pass the tolerance as the fourth argument of `fea.AccuracyCriterion.new`, together ",
    "with the target, error contribution, and required evidence it bounds.",
);

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &relative::FEA_ACCURACY_TOLERANCE_RELATIVE_CATALOG_ENTRY,
    &absolute::FEA_ACCURACY_TOLERANCE_ABSOLUTE_CATALOG_ENTRY,
];
