//! Public reference resources shipped with this compiler on every host.
use sha2::{Digest, Sha256};
pub const COMPILER_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Clone, Copy)]
pub struct ReferenceDocument {
    pub id: &'static str,
    pub content: &'static str,
}
include!(concat!(env!("OUT_DIR"), "/fea_reference.rs"));

pub const FEA_LANGUAGE_REFERENCE_IDS: &[&str] = &[
    "fea/language/index.md",
    "fea/language/reference/declarations.md",
    "fea/language/reference/procedures.md",
    "fea/language/reference/operations-and-output-queries.md",
    "fea/language/diagnostics-and-tooling.md",
];

pub fn reference_document(id: &str) -> Option<&'static ReferenceDocument> {
    FEA_REFERENCE_DOCUMENTS.iter().find(|doc| doc.id == id)
}

pub fn reference_bundle_digest_bytes() -> [u8; 32] {
    static DIGEST: std::sync::OnceLock<[u8; 32]> = std::sync::OnceLock::new();
    *DIGEST.get_or_init(compute_reference_bundle_digest)
}

fn compute_reference_bundle_digest() -> [u8; 32] {
    let mut hash = Sha256::new();
    for document in FEA_REFERENCE_DOCUMENTS {
        hash.update((document.id.len() as u64).to_le_bytes());
        hash.update(document.id.as_bytes());
        hash.update((document.content.len() as u64).to_le_bytes());
        hash.update(document.content.as_bytes());
    }
    hash.finalize().into()
}

pub fn reference_bundle_digest() -> String {
    reference_bundle_digest_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn advertised_references_are_compiler_matched_and_present() {
        for id in FEA_LANGUAGE_REFERENCE_IDS {
            assert!(!reference_document(id)
                .expect("advertised reference")
                .content
                .is_empty());
        }
        assert_eq!(reference_bundle_digest().len(), 64);
        assert!(FEA_REFERENCE_DOCUMENTS
            .windows(2)
            .all(|pair| pair[0].id < pair[1].id));
    }
}
