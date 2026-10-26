//! Catalog identities of the `fea.PartialAccessPolicy` typed policy class.

use crate::*;

mod manifest_and_chunks;
mod whole_artifact;
#[cfg(test)]
mod tests;

pub(super) const ROLE: &str = concat!(
    "A partial-access policy states how an observation's stored result can be read back: ",
    "selectively, by manifest and chunk, or as a whole artifact.",
);
pub(super) const STUDY: &str = concat!(
    "Pass the policy as the fourth argument of `fea.ObservationStorage.new(precision, ",
    "compression, retention, partialAccess)`, and pass the storage as the thirteenth ",
    "argument of an observation constructor.",
);
pub(super) const FAQ: BuiltinDocumentationFaq = BuiltinDocumentationFaq {
    question: "Which policy should large histories use?",
    answer: concat!(
        "`manifestAndChunks()`, so that a consumer can read one sample without gathering the ",
        "rest.",
    ),
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &manifest_and_chunks::FEA_PARTIAL_ACCESS_POLICY_MANIFEST_AND_CHUNKS_CATALOG_ENTRY,
    &whole_artifact::FEA_PARTIAL_ACCESS_POLICY_WHOLE_ARTIFACT_CATALOG_ENTRY,
];
