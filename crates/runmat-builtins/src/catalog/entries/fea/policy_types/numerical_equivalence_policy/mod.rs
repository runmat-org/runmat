//! Catalog identities of the `fea.NumericalEquivalencePolicy` typed policy class.

use crate::*;

mod absolute_tolerance;
mod deterministic;
mod relative_tolerance;
#[cfg(test)]
mod tests;

pub(super) const EQUIVALENCE_ROLE: &str = concat!(
    "The numerical-equivalence policy of an accuracy plan states how closely two ",
    "executions of the same step must agree: serial and partitioned evaluation, CPU and ",
    "GPU placement, a retry, or a remote run. It bounds execution variation only; it is ",
    "not a discretization tolerance and cannot loosen any criterion.",
);
pub(super) const EQUIVALENCE_STUDY: &str = concat!(
    "Pass the policy as the fifth argument of `fea.AccuracyPlan.new`. Criteria with the ",
    "`fea.AccuracyTarget.executionEquivalence()` target are assessed against it.",
);

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &deterministic::FEA_NUMERICAL_EQUIVALENCE_POLICY_DETERMINISTIC_CATALOG_ENTRY,
    &relative_tolerance::FEA_NUMERICAL_EQUIVALENCE_POLICY_RELATIVE_TOLERANCE_CATALOG_ENTRY,
    &absolute_tolerance::FEA_NUMERICAL_EQUIVALENCE_POLICY_ABSOLUTE_TOLERANCE_CATALOG_ENTRY,
];
