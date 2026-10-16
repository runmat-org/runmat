use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Convert text containing numeric matrix literals to a numeric array.",
        "Parses bracketed numeric matrix text with whitespace, comma, and semicolon separators.",
        "When a second output is requested, returns a logical success flag instead of throwing for malformed numeric text.",
        "RunMat currently implements numeric matrix literal parsing. The compatibility target's `Evaluation` option and general expression-evaluation surface remain implementation gaps.",
    ] },
    BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &[
        "Literal numeric text parsing is host-only. Integer, numeric, and provider-resident inputs are not text and reject before provider access.",
        "Not eligible for fusion; this builtin materializes host numeric and logical results.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "example-1",
    title: "str2num returns a 2-by-2 numeric matrix and true",
    program: "[x,tf] = str2num(\"[1 2; 3 4]\")",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isequal(x, [1 2;3 4])); assert(tf);",
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
    title: Some("str2num"), slug: Some("str2num"),
    summary: "Convert text containing numeric matrix literals to a numeric array.",
    description: "`str2num(text)` parses a scalar string or character array as a numeric matrix expression. It is a text parser, not a numeric conversion function; use `string` to turn numeric data into text and `str2double` for element-wise text parsing.",
    keywords: &["str2num", "strings", "core", "matlab", "compatibility"],
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
