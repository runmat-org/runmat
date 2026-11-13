use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("dictionary.subsref"), slug: Some("dictionary.subsref"),
    summary: "Look up dictionary entries or inspect Keys and Values.",
    description: "`dictionary.subsref(d,S)` applies a standard subscript path to a dictionary. Parentheses or braces select entries by key; dot indexing reads the Keys or Values property.",
    keywords: &["dictionary", "subsref", "lookup", "key"],
    related: &["dictionary", "dictionary.subsasgn", "subsref"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "Each S element has a type and subs selector. Lookup converts queries to the configured key class, compares integer keys exactly, and reports missing keys as index errors.",
            "A scalar key returns its stored value. Several keys produce an array or cell container according to their value classes; the result depends on runtime contents.",
            "Single key conversion and homogeneous Single results preserve native binary32 class and payloads. Output shape follows the complete key-query geometry.",
            "Chained paths continue indexing into the retrieved value through the shared object protocol.",
        ] },
        BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &[
            "Dictionary key lookup is host-side and a numeric fusion boundary. Stored values can have their own residency; a retrieved value is not automatically gathered.",
            "The method may suspend or throw while processing a nested path. Direct distributed dictionary handles are unsupported.",
            "Typed native dictionary lookup admits its index scratch and retained output under the selected context. The public standard-substruct adapter still rejects finite Managed execution before its shared path/component workspace is allocated; this is a shared-ingress limit, not a class conversion or output fallback.",
        ] },
    ],
    examples: &[BuiltinExample {
        id: "exact-key-lookup", title: "Look up an exact integer-keyed value",
        program: "key = uint64(9007199254740992) + uint64(1); d = dictionary(key, uint64(7)); S = struct('type', '()', 'subs', {{key}}); v = subsref(d,S);",
        display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isa(v,'uint64')); assert(v == uint64(7));" },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "What happens if a key is absent?", answer: "Lookup reports an invalid-index error instead of adding an entry." }],
    links: &[BuiltinDocumentationLink { label: "dictionary", target: BuiltinDocumentationLinkTarget::Builtin("dictionary") }, BuiltinDocumentationLink { label: "subsref", target: BuiltinDocumentationLinkTarget::Builtin("subsref") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime indexing", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/table/builtins/indexing.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Dictionary semantics", location: "crates/runmat-runtime/src/builtins/table/tests.rs::categorical_categories_and_dictionary_lookup_have_semantics" }],
        notes: &["Method-only identity; ordinary dictionary lookup dispatches through the class registry."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
