use crate::*;

pub(super) const VERLESSTHAN_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some(r####"verLessThan"####),
    slug: Some(r####"verLessThan"####),
    summary: r####"Compare RunMat's MATLAB compatibility version with a requested version."####,
    description: r####"`verLessThan('matlab', version)` compares RunMat's MATLAB compatibility version against `version` and returns a logical scalar."####,
    keywords: &[
        r####"verLessThan"####,
        r####"version"####,
        r####"compatibility"####,
    ],
    related: &[],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior and placement",
            paragraphs: &[
                r####"The product name `matlab` is recognized case-insensitively."####,
                r####"Unknown products return false, matching compatibility-branch use where non-MATLAB products are not installed."####,
                r####"Version components are compared numerically from left to right."####,
                r####"Both inputs are text controls. Numeric values, including every integer class and resident arrays, are rejected rather than converted to version text."####,
                r####"GPU support: Host metadata query."####,
                r####"The result is a host logical scalar."####,
                r####"No GPU kernels are launched."####,
            ],
        },
        BuiltinDocumentationSection {
            heading: "Legacy examples and displayed results",
            paragraphs: &[r####"Branching on MATLAB compatibility version
tf = verLessThan('matlab', '99.0')
Expected display: tf = logical(1)"####],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "typed-basic",
            title: "compare compatibility release",
            program: "tf = verLessThan('matlab','99.0')",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(tf);",
            },
        },
        BuiltinExample {
            id: "numeric-rejection",
            title: "Reject numeric product control",
            program: "verLessThan(1,'1.0')",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::ExpectedError {
                identifier: "RunMat:runtimeCompat:InvalidInput",
            },
        },
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: r####"Does this inspect installed toolbox versions?"####,
        answer: r####"No. This implementation supports the common MATLAB runtime compatibility branch and returns false for other products. The documented unknown-toolbox error remains a compatibility gap."####,
    }],
    links: &[BuiltinDocumentationLink {
        label: r####"version"####,
        target: BuiltinDocumentationLinkTarget::Documentation(r####"./version"####),
    }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: r####"`crates/runmat-runtime/src/builtins/introspection/runtime_compat.rs`"####,
            target: BuiltinDocumentationLinkTarget::Source(
                r####"https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/runtime_compat.rs"####,
            ),
        },
        BuiltinDocumentationLink { label: "Catalog contract", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-builtins/src/catalog/entries/introspection/runtime_compat/contract.rs") },
        BuiltinDocumentationLink { label: "Catalog inference", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-builtins/src/catalog/entries/introspection/runtime_compat/inference.rs") }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Legacy test reference",
            location: r####"builtins::introspection::runtime_compat::tests"####,
        }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
