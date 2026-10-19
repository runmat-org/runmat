//! Catalog identities of the `fea.ObservationPurpose` typed policy class.

use crate::*;

mod downstream_state;
mod publication;
mod solver_control;
mod visualization;
#[cfg(test)]
mod tests;

pub(super) const ROLE: &str = concat!(
    "An observation purpose records why a result is computed. It governs which storage ",
    "choices are admissible and is retained with the result.",
);
pub(super) const STUDY: &str = concat!(
    "Pass the purpose as the fourteenth argument of an observation constructor. ",
    "`fea.Observation.field`, `history`, `probe`, and `reduction` take `(id, quantity, ",
    "target, sourceLocation, outputLocation, recovery, component, frame, sign, ",
    "complexRepresentation, sampling, reduction, storage, purpose, cost)`.",
);
pub(super) const FAQ: BuiltinDocumentationFaq = BuiltinDocumentationFaq {
    question: "Does purpose change the numerical result?",
    answer: "No. It constrains storage and records intent; the recovered values are the same.",
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &publication::FEA_OBSERVATION_PURPOSE_PUBLICATION_CATALOG_ENTRY,
    &solver_control::FEA_OBSERVATION_PURPOSE_SOLVER_CONTROL_CATALOG_ENTRY,
    &visualization::FEA_OBSERVATION_PURPOSE_VISUALIZATION_CATALOG_ENTRY,
    &downstream_state::FEA_OBSERVATION_PURPOSE_DOWNSTREAM_STATE_CATALOG_ENTRY,
];
