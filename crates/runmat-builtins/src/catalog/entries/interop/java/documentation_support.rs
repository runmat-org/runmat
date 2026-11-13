use crate::*;

pub(super) const JAVA_HOST: BuiltinExampleRequirements = BuiltinExampleRequirements {
    host: BuiltinExampleHostRequirement::NativeOnly,
    runtime: &[BuiltinRuntimeCapability::JavaVirtualMachine],
    ..BuiltinExampleRequirements::NONE
};
pub(super) const JAVA_DESKTOP: BuiltinExampleRequirements = BuiltinExampleRequirements {
    host: BuiltinExampleHostRequirement::DesktopHostOnly,
    runtime: &[BuiltinRuntimeCapability::JavaVirtualMachine],
    ..BuiltinExampleRequirements::NONE
};
pub(super) const JAVA_LINK: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Java interoperability",
    target: BuiltinDocumentationLinkTarget::Documentation("/docs/execution/java"),
};
pub(super) const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "Java builtin runtime forms",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/interop/java.rs",
        ),
    },
    BuiltinDocumentationLink {
        label: "Java session foreign adapter",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/foreign/java/adapter.rs",
        ),
    },
    BuiltinDocumentationLink {
        label: "Java builtin dispatch",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/interop/java/dispatch.rs",
        ),
    },
];

pub(super) const fn example(
    id: &'static str,
    title: &'static str,
    program: &'static str,
    assertions: &'static str,
    desktop: bool,
) -> BuiltinExample {
    BuiltinExample {
        id,
        title,
        program,
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: if desktop {
            BuiltinExampleHarness::InteractiveHost
        } else {
            BuiltinExampleHarness::NativeForeignRuntime
        },
        fixture: BuiltinExampleFixture::None,
        requirements: if desktop { JAVA_DESKTOP } else { JAVA_HOST },
        verification: BuiltinExampleVerification::Assertions { source: assertions },
    }
}

pub(super) struct JavaDocumentationContent {
    pub summary: &'static str,
    pub description: &'static str,
    pub keywords: &'static [&'static str],
    pub related: &'static [&'static str],
    pub sections: &'static [BuiltinDocumentationSection],
    pub examples: &'static [BuiltinExample],
    pub faqs: &'static [BuiltinDocumentationFaq],
    pub verification: &'static [BuiltinEvidenceReference],
}

pub(super) const fn documentation(
    name: &'static str,
    content: JavaDocumentationContent,
) -> BuiltinDocumentation {
    BuiltinDocumentation {
        authority: BuiltinDocumentationAuthority::Catalog,
        title: Some(name),
        slug: Some(name),
        summary: content.summary,
        description: content.description,
        keywords: content.keywords,
        related: content.related,
        sections: content.sections,
        examples: content.examples,
        example_exemption: None,
        faqs: content.faqs,
        links: &[JAVA_LINK],
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
