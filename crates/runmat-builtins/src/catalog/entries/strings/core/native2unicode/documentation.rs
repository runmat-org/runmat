use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "All eight integer classes are accepted, but every element must be exactly within 0 through 255; values are never clamped or converted through binary64.",
        "ASCII-width decoding preserves the input row or column vector shape, and character or string input passes through unchanged.",
        "Unsupported encoding names and explicit gpuArray inputs reject.",
    ] },
    BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &[
        "No GPU form is documented. Automatically resident byte vectors gather transparently; explicit gpuArray input is rejected.",
        "Not eligible for fusion; this builtin materializes host text or logical results.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "example-1",
    title: "native2unicode(uint16([104 105]), UTF-8) returns the character vector 'hi'.",
    program: "text = native2unicode(uint16([104 105]), \"UTF-8\")",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(text, 'hi'));",
    },
}];

const FAQS: &[BuiltinDocumentationFaq] = &[];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") },
    BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/compat.rs") },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("native2unicode"), slug: Some("native2unicode"),
    summary: "Decode native byte values into Unicode text.",
    description: "`native2unicode(bytes, encoding)` range-checks a numeric byte vector and decodes it to a host character vector using UTF-8 by default.",
    keywords: &["native2unicode", "strings", "core", "matlab", "compatibility"],
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
