use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("dictionary.subsasgn"), slug: Some("dictionary.subsasgn"),
    summary: "Insert, replace, or remove dictionary entries through an indexing path.",
    description: "`dictionary.subsasgn(d,S,rhs)` returns an updated dictionary. Parentheses and braces assign by key; an empty right-hand value removes matching keys.",
    keywords: &["dictionary", "subsasgn", "assignment", "key"],
    related: &["dictionary", "dictionary.subsref", "subsasgn"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "The returned dictionary carries the change; the original value is not mutated in place. A scalar right-hand value expands across several keys; non-scalar assignments must match the query shape.",
            "New values are normalized to the existing dictionary value class. Integer key comparison retains native precision, including values above the exact-double range.",
            "Single keys and values retain native binary32 class. Repeated inserted keys keep their first insertion position and last assigned value, including duplicates within one assignment.",
            "Chained paths update a selected child and write it back into the outer dictionary.",
        ] },
        BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &[
            "Dictionary assignment is host-side and a numeric fusion boundary. Nested values can have their own residency and indexing behavior.",
            "The method may suspend or throw during nested assignment; direct distributed dictionary handles are unsupported.",
            "Typed native edits preflight key/value conversions and shapes, admit native index scratch and retained outputs, then publish both properties atomically. The public standard-substruct adapter still rejects finite Managed execution before the shared path/component workspace is allocated.",
        ] },
    ],
    examples: &[BuiltinExample {
        id: "replace-entry", title: "Replace an integer value",
        program: "d = dictionary('a',uint64(3)); S = struct('type', '()', 'subs', {{'a'}}); d = subsasgn(d,S,uint64(9)); v = subsref(d,S);",
        display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isa(v,'uint64')); assert(v == uint64(9));" },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "How do I remove an entry?", answer: "Assign an empty right-hand value at its key and keep the returned dictionary." }],
    links: &[BuiltinDocumentationLink { label: "dictionary", target: BuiltinDocumentationLinkTarget::Builtin("dictionary") }, BuiltinDocumentationLink { label: "subsasgn", target: BuiltinDocumentationLinkTarget::Builtin("subsasgn") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime indexing", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/table/builtins/indexing.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Dictionary semantics", location: "crates/runmat-runtime/src/builtins/table/tests.rs" }],
        notes: &["Method-only identity; ordinary dictionary assignment dispatches through the class registry."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
