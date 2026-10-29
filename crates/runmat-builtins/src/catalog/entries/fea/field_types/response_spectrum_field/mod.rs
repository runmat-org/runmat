//! `fea.ResponseSpectrumField.*` selectors: nodal response-spectrum peaks and
//! envelopes of linear structural quantities.

use crate::fea_contract::doc_links;
use crate::*;

pub(super) const RS_FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Can I add or subtract response-spectrum fields?", answer: "Not meaningfully. They are combined peak magnitudes, not a simultaneous state, so arithmetic between components or nodes does not give another peak." },
    BuiltinDocumentationFaq { question: "Which step do I read them from?", answer: "From the response-spectrum step of the run, selected with `run.step(stepId)` or `fea.stepResult`; the modal extraction step that precedes it does not hold them." },
];
pub(super) const LINKS: &[BuiltinDocumentationLink] =
    &[doc_links::OUTPUTS_AND_FIELDS, doc_links::STRUCTURAL_PHYSICS];
pub(super) const IMPLEMENTATION: &[BuiltinDocumentationLink] =
    &[doc_links::FEA_STATISTICAL_FIELD_SELECTORS_SOURCE];
pub(super) const VERIFICATION: &[BuiltinEvidenceReference] =
    &[doc_links::STATISTICAL_SELECTORS_TEST];

mod absolute_acceleration;
mod absolute_angular_acceleration;
mod envelope;
mod pseudo_acceleration;
mod pseudo_angular_acceleration;
mod pseudo_angular_velocity;
mod pseudo_velocity;
mod relative_acceleration;
mod relative_angular_acceleration;
mod relative_angular_velocity;
mod relative_displacement;
mod relative_velocity;
mod rotation;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &relative_displacement::ENTRY,
    &rotation::ENTRY,
    &relative_velocity::ENTRY,
    &relative_angular_velocity::ENTRY,
    &relative_acceleration::ENTRY,
    &relative_angular_acceleration::ENTRY,
    &absolute_acceleration::ENTRY,
    &absolute_angular_acceleration::ENTRY,
    &pseudo_velocity::ENTRY,
    &pseudo_angular_velocity::ENTRY,
    &pseudo_acceleration::ENTRY,
    &pseudo_angular_acceleration::ENTRY,
    &envelope::ENTRY,
];
