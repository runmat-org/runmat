//! Catalog identities of the `fea.DiagnosticKind` typed policy class.

use crate::*;

mod adaptation;
mod balance;
mod constraint;
mod contact;
mod convergence;
mod execution_equivalence;
mod resource;
#[cfg(test)]
mod tests;

pub(super) const ROLE: &str = concat!(
    "A diagnostic kind selects which typed diagnostic stream an observation records. ",
    "Diagnostics carry solver evidence, not a physical quantity, so they cannot be ",
    "accuracy targets.",
);
pub(super) const STUDY: &str = concat!(
    "Pass the kind as the second argument of `fea.Observation.diagnostic(id, kind, ",
    "sampling, retention)`.",
);
pub(super) const FAQ: BuiltinDocumentationFaq = BuiltinDocumentationFaq {
    question: "Can a diagnostic be used as an accuracy target?",
    answer: concat!(
        "No. Accuracy targets must name numeric field, history, probe, or reduction ",
        "observations.",
    ),
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &convergence::FEA_DIAGNOSTIC_KIND_CONVERGENCE_CATALOG_ENTRY,
    &balance::FEA_DIAGNOSTIC_KIND_BALANCE_CATALOG_ENTRY,
    &constraint::FEA_DIAGNOSTIC_KIND_CONSTRAINT_CATALOG_ENTRY,
    &contact::FEA_DIAGNOSTIC_KIND_CONTACT_CATALOG_ENTRY,
    &adaptation::FEA_DIAGNOSTIC_KIND_ADAPTATION_CATALOG_ENTRY,
    &resource::FEA_DIAGNOSTIC_KIND_RESOURCE_CATALOG_ENTRY,
    &execution_equivalence::FEA_DIAGNOSTIC_KIND_EXECUTION_EQUIVALENCE_CATALOG_ENTRY,
];
