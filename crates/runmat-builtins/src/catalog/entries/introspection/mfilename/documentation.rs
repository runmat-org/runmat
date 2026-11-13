use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("mfilename"), slug: Some("mfilename"),
    summary: "Return the lexical file name or current method class.",
    description: "`mfilename` inspects the currently executing source context without opening a file. RunMat returns a host string scalar; in a context without a lexical file or class, the result is empty.",
    keywords: &["mfilename", "source file", "fullpath", "class", "introspection"],
    related: &["which", "functions", "inputname"],
    sections: &[
        BuiltinDocumentationSection { heading: "Forms and source ownership", paragraphs: &[
            "`mfilename` returns the lexical file stem without its extension. `mfilename('fullpath')` returns the lexical full path without extension. Nested eval source retains its owning lexical file. A source without a file returns an empty string.",
            "`mfilename('class')` reads the current method's class-access context and returns its class name, or an empty string outside such a context. Option spelling is trimmed and case-insensitive. Unsupported or non-text options preserve the default name mode; more than one option raises `RunMat:TooManyInputs`.",
            "The result is a RunMat string scalar. It is source-context metadata, not a filesystem existence test or path-resolution operation. No operand payload is gathered and no GPU kernel is launched.",
        ] },
    ],
    examples: &[BuiltinExample {
        id: "empty-context", title: "Inspect an unbound source context",
        program: "name = mfilename(); full = mfilename('fullpath');",
        display_output: None, compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isstring(name)); assert(isstring(full));" },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Does `fullpath` read the file?", answer: "No. The result is derived from the lexical source record already attached to execution." }],
    links: &[BuiltinDocumentationLink { label: "which", target: BuiltinDocumentationLinkTarget::Documentation("./which") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime source-context query", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/mfilename.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Lexical source and nested context", location: "builtins::introspection::mfilename::tests" }],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
