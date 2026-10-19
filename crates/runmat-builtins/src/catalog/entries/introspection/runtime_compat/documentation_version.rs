use crate::*;

pub(super) const VERSION_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some(r####"version"####),
    slug: Some(r####"version"####),
    summary: r####"Return RunMat version and MATLAB-compatibility release metadata."####,
    description: r####"`version` returns the RunMat crate version. `version('-release')` returns RunMat's advertised MATLAB compatibility release string."####,
    keywords: &[
        r####"version"####,
        r####"release"####,
        r####"environment"####,
        r####"compatibility"####,
    ],
    related: &[],
    sections: &[BuiltinDocumentationSection {
        heading: "Behavior and placement",
        paragraphs: &[
            r####"No arguments returns the RunMat package version."####,
            r####"`'-release'` returns the centrally configured compatibility release for branching."####,
            r####"`'-description'` returns a human-readable RunMat description."####,
            r####"`'-java'` returns an empty string because RunMat does not embed MATLAB's Java runtime."####,
            r####"Options are text-only. Numeric and resident inputs are rejected without conversion or provider access."####,
            r####"The documented `'-date'` option and two-output `[v, d] = version` form are not implemented."####,
            r####"GPU support: Host metadata query."####,
            r####"The result is host-resident text."####,
            r####"No GPU kernels are launched."####,
        ],
    }],
    examples: &[
        BuiltinExample {
            id: "typed-basic",
            title: "query package and release",
            program: "v = version; r = version('-release')",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isstring(v)); assert(isstring(r));",
            },
        },
        BuiltinExample {
            id: "numeric-rejection",
            title: "Reject numeric version option",
            program: "version(1)",
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
        question: r####"Is this MATLAB's exact version?"####,
        answer: r####"No. It reports RunMat's runtime version plus a MATLAB compatibility release for common branching code."####,
    }],
    links: &[
        BuiltinDocumentationLink {
            label: r####"verLessThan"####,
            target: BuiltinDocumentationLinkTarget::Documentation(r####"./verLessThan"####),
        },
        BuiltinDocumentationLink {
            label: r####"isdeployed"####,
            target: BuiltinDocumentationLinkTarget::Documentation(r####"./isdeployed"####),
        },
        BuiltinDocumentationLink {
            label: r####"MATLAB language compatibility"####,
            target: BuiltinDocumentationLinkTarget::Documentation(
                r####"/docs/runtime/matlab-compatibility"####,
            ),
        },
    ],
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
