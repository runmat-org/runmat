//! Shared presentation fields; each operation owns its public prose and assertions.

use crate::*;

const DATA_GUIDE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Datasets API",
    target: BuiltinDocumentationLinkTarget::Documentation("/docs/fs/datasets"),
};
const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "Dataset path runtime",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/data/path.rs"),
    },
    BuiltinDocumentationLink {
        label: "Captured filesystem and working directory",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/data/path_authority.rs"),
    },
    BuiltinDocumentationLink {
        label: "Manifest and chunk storage",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/data/mod.rs"),
    },
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
        // These programs create their own datasets in the native runner's
        // isolated filesystem workspace; no preloaded file is required.
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: assertions },
    }
}

pub(super) struct PathDocumentationContent {
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

pub(super) const fn documentation(content: PathDocumentationContent) -> BuiltinDocumentation {
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
        links: &[DATA_GUIDE],
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
