use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Convert scalar text containing a matrix expression using the normal RunMat parser and executor.",
        "The Evaluation option is 'all' by default, or 'restricted' for pure numeric expressions. Matrix spacing, commas, and semicolons follow the language's matrix syntax.",
        "Invalid expression text returns an empty matrix and a false logical status when the second output is requested. Cancellation, host-policy denial, and execution-service errors remain errors.",
        "All-mode execution has the host's existing dynamic-evaluation privileges and can invoke effectful functions. Do not use it for untrusted text; choose Evaluation='restricted' or str2double.",
    ] },
    BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &[
        "Evaluation is host-only. Integer, numeric, and provider-resident inputs are not text and reject before provider access.",
        "All text uses the session's captured source-evaluation authority and canonical parser in an isolated transactional frame; it does not publish implicit variables into the base workspace.",
        "Not eligible for fusion. Dynamic source evaluation may suspend and requires the host's source-evaluation capability.",
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
    summary: "Evaluate scalar text as a matrix expression.",
    description: "`str2num(text)` evaluates a scalar string or character array as a matrix expression. `Evaluation='restricted'` admits only pure numeric expressions; the default 'all' mode uses the host's dynamic-evaluation policy. Use `str2double` for element-wise text parsing.",
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
