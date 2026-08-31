mod content;
mod evidence;
mod example;

pub use content::*;
pub use evidence::*;
pub use example::*;

use serde::Serialize;

/// Canonical user-facing documentation for one builtin identity.
///
/// Machine-readable facts such as signatures, errors, placement, fusion,
/// compatibility, and required capabilities live in their dedicated catalog
/// contracts and are rendered alongside this content. Keeping them out of this
/// structure prevents prose metadata from becoming a second semantic registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinDocumentation {
    pub authority: BuiltinDocumentationAuthority,
    pub title: Option<&'static str>,
    pub slug: Option<&'static str>,
    pub summary: &'static str,
    pub description: &'static str,
    pub keywords: &'static [&'static str],
    pub related: &'static [&'static str],
    pub sections: &'static [BuiltinDocumentationSection],
    pub examples: &'static [BuiltinExample],
    pub example_exemption: Option<&'static str>,
    pub faqs: &'static [BuiltinDocumentationFaq],
    pub links: &'static [BuiltinDocumentationLink],
    pub media: &'static [BuiltinDocumentationMedia],
    pub evidence: BuiltinDocumentationEvidence,
    pub introduced: Option<&'static str>,
    pub status: Option<BuiltinDocumentationStatus>,
}

impl BuiltinDocumentation {
    /// Empty content is useful only while constructing tests and during the
    /// bounded catalog migration. Public catalog validation rejects incomplete
    /// documentation before an identity can be considered migrated.
    pub const EMPTY: Self = Self {
        authority: BuiltinDocumentationAuthority::LegacySidecar,
        title: None,
        slug: None,
        summary: "",
        description: "",
        keywords: &[],
        related: &[],
        sections: &[],
        examples: &[],
        example_exemption: None,
        faqs: &[],
        links: &[],
        media: &[],
        evidence: BuiltinDocumentationEvidence::EMPTY,
        introduced: None,
        status: None,
    };
}

/// Explicit ownership during the bounded C00–C07 migration. The legacy
/// variant is removed when the final sidecar reaches zero at R29.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinDocumentationAuthority {
    Catalog,
    LegacySidecar,
}
