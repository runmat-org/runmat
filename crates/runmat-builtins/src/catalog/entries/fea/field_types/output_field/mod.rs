//! `fea.OutputField.named`: select an authored `.fea` output by name.

use crate::fea_contract::doc_links;
use crate::*;

pub(super) const OUTPUT_FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Is the name the same as the field id?", answer: "No. It is the name after `output` in the `.fea` study. The study maps it to a field such as `structural.von_mises`." },
];
pub(super) const LINKS: &[BuiltinDocumentationLink] = &[doc_links::OUTPUTS_AND_FIELDS];
pub(super) const IMPLEMENTATION: &[BuiltinDocumentationLink] =
    &[doc_links::FEA_FIELD_SELECTORS_SOURCE];
pub(super) const VERIFICATION: &[BuiltinEvidenceReference] = &[doc_links::OUTPUT_SELECTOR_TEST];

mod named;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&named::ENTRY];
