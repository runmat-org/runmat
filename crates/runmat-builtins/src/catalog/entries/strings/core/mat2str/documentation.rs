use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "All eight integer classes serialize directly from exact native storage without crossing binary64.",
        "The `'class'` form wraps the expression in the actual input constructor, such as `uint16([256 512])`.",
        "Typed integer precision controls and nonnumeric text input are explicitly gated RunMat extensions.",
    ] },
    BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &[
        "No GPU form is documented. Automatically resident values gather transparently; explicit gpuArray input is rejected.",
        "Not eligible for fusion; this builtin materializes host text or logical results.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "mat2str([1 2; 3 4]) returns the character vector '[1 2;3 4]'.",
        program: "text = mat2str([1 2; 3 4])",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(strcmp(text, '[1 2;3 4]'));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "mat2str(uint16([256 512]), 'class') returns 'uint16([256 512])'.",
        program: "text = mat2str(uint16([256 512]), 'class')",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(strcmp(text, 'uint16([256 512])'));",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") },
    BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/compat.rs") },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("mat2str"), slug: Some("mat2str"),
    summary: "Convert a numeric or logical matrix to MATLAB expression text.",
    description: "`mat2str(A)` converts a numeric or logical matrix to a character-vector expression. An optional positive precision controls significant digits, and a final `'class'` flag includes the input class constructor.",
    keywords: &["mat2str", "strings", "core", "matlab", "compatibility"],
    related: &["string", "char"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: FAQS, links: LINKS, media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/compat.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Existing identity family tests", location: "builtins::strings::core::compat::tests" },
        ],
        notes: &["Source-only canonical cutover; original JSON retained until compiled field and executable-example conservation."],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
