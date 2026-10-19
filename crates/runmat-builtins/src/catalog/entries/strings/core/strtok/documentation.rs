use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Split text into first token and remainder.",
        "When multiple outputs are requested, returns both the first token and the remaining text.",
        "Leading delimiters are skipped before token selection; the returned remainder begins at the delimiter after the token.",
        "Integer arrays are not character-code input and are rejected without provider access or implicit text conversion.",
    ] },
    BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &[
        "Tokenization is host text processing. Numeric and provider-resident text or delimiter inputs reject before provider access.",
        "Not eligible for fusion; this builtin materializes host text or logical results.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "example-1",
    title: "strtok returns alpha and the unconsumed comma suffix",
    program: "[tok,rem] = strtok(\"  alpha,beta\", \" ,\")",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(tok, 'alpha')); assert(strcmp(rem, ',beta'));",
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
    title: Some("strtok"), slug: Some("strtok"),
    summary: "Split text into first token and remainder.",
    description: "`[tok, rem] = strtok(text, delimiters)` skips leading delimiter characters, returns the first token, and optionally returns the remaining text beginning at the delimiter after that token.",
    keywords: &["strtok", "strings", "core", "matlab", "compatibility"],
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
