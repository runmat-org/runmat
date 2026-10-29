//! `fea.RandomResponseField.rms` and `.variance`: random-vibration statistics
//! of linear structural quantities.

use crate::fea_contract::doc_links;
use crate::*;

pub(super) const RR_FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "How are RMS and variance related?", answer: "The RMS is the square root of the variance for each component; units follow, so stress RMS is in Pa and stress variance in Pa^2." },
    BuiltinDocumentationFaq { question: "Why is von Mises stress rejected?", answer: "Random-response statistics are computed per component from linear modal transfers. Von Mises stress is a nonlinear combination, so its statistics cannot be formed that way; select `fea.StructuralField.stress()` instead." },
];
pub(super) const LINKS: &[BuiltinDocumentationLink] =
    &[doc_links::OUTPUTS_AND_FIELDS, doc_links::STRUCTURAL_PHYSICS];
pub(super) const IMPLEMENTATION: &[BuiltinDocumentationLink] =
    &[doc_links::FEA_STATISTICAL_FIELD_SELECTORS_SOURCE];
pub(super) const VERIFICATION: &[BuiltinEvidenceReference] =
    &[doc_links::STATISTICAL_SELECTORS_TEST];

mod rms;
mod variance;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&rms::ENTRY, &variance::ENTRY];
