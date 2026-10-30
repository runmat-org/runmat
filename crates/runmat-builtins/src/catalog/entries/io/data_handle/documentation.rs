//! Source-linked user documentation shared by dataset receiver entries.

use crate::*;

pub(super) const ATTRIBUTE_VALUE_BOUNDARY: &str = "Each stored attribute value must be a string scalar (including a one-element string array), character row vector, finite scalar double, fixed-width integer scalar, or scalar logical value. Numeric/logical array carriers, nonscalar string arrays, character matrices or pages, cells, and nested structs require explicit encoding and are rejected at JSON serialization. Typed missing strings store as JSON null and decode as missing strings. Empty strings and literal spellings such as `<missing>` remain ordinary text.";

const LINKS: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Datasets API",
    target: BuiltinDocumentationLinkTarget::Documentation("/docs/fs/datasets"),
}];
const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "Dataset receiver runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/data/dataset.rs") },
    BuiltinDocumentationLink { label: "DataArray receiver runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/data/array.rs") },
    BuiltinDocumentationLink { label: "DataTransaction runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/data/transaction.rs") },
    BuiltinDocumentationLink { label: "Shared data payload kernels", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/data/kernel/mod.rs") },
    BuiltinDocumentationLink { label: "Sealed filesystem owner", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/data/authority.rs") },
    BuiltinDocumentationLink { label: "Manifest and chunk storage", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/data/mod.rs") },
];

pub(super) const fn example(
    id: &'static str,
    title: &'static str,
    program: &'static str,
    assertions: &'static str,
) -> BuiltinExample {
    BuiltinExample {
        id,
        title,
        program,
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: assertions },
    }
}

pub(super) struct Content {
    pub name: &'static str,
    pub summary: &'static str,
    pub description: &'static str,
    pub keywords: &'static [&'static str],
    pub related: &'static [&'static str],
    pub sections: &'static [BuiltinDocumentationSection],
    pub examples: &'static [BuiltinExample],
    pub faqs: &'static [BuiltinDocumentationFaq],
    pub verification: &'static [BuiltinEvidenceReference],
}

pub(super) const fn documentation(content: Content) -> BuiltinDocumentation {
    BuiltinDocumentation {
        authority: BuiltinDocumentationAuthority::Catalog,
        title: Some(content.name),
        slug: Some(content.name),
        summary: content.summary,
        description: content.description,
        keywords: content.keywords,
        related: content.related,
        sections: content.sections,
        examples: content.examples,
        example_exemption: None,
        faqs: content.faqs,
        links: LINKS,
        media: &[],
        evidence: BuiltinDocumentationEvidence {
            implementation: IMPLEMENTATION,
            verification: content.verification,
            notes: &[],
        },
        introduced: None,
        status: Some(BuiltinDocumentationStatus::Stable),
    }
}
