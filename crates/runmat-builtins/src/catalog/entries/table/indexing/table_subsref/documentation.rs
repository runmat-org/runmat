use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("table.subsref"),
    slug: Some("table.subsref"),
    summary: "Read a table variable, a table subset, or selected variable contents.",
    description: "`table.subsref(T,S)` applies a standard indexing path to a table or timetable. Dot indexing reads a variable or Properties; parentheses return a table subset; braces return selected contents.",
    keywords: &["table", "timetable", "subsref", "indexing"],
    related: &["table", "table.subsasgn", "subsref"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "Each element of S supplies a type (`'.'`, `'()'`, or `'{}'`) and subs selector. Chained paths are processed in order through the object indexing protocol.",
            "A selected variable retains its stored numeric class. The result of a general path depends on table schema and selector contents, so static analysis keeps its value dynamic.",
            "Table and timetable receivers share the indexing implementation. A timetable's row-time variable can also be selected by its dimension name.",
        ] },
        BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &[
            "Table indexing is a host-side object operation and a numeric fusion boundary. Selected variables may themselves carry resident data; reading a variable does not imply a gather.",
            "The path can suspend or throw during nested indexing. Direct distributed table handles are not admitted by this method.",
        ] },
    ],
    examples: &[BuiltinExample {
        id: "variable-read", title: "Read a typed table variable",
        program: "T = table(uint64([1; 2]), 'VariableNames', {'Count'}); S = struct('type', '.', 'subs', 'Count'); v = subsref(T,S);",
        display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isa(v,'uint64')); assert(isequal(v,uint64([1; 2])));" },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Does table.subsref always return a table?", answer: "No. Parentheses return a table, while dot and brace indexing can return a stored variable or selected contents." }],
    links: &[BuiltinDocumentationLink { label: "table", target: BuiltinDocumentationLinkTarget::Builtin("table") }, BuiltinDocumentationLink { label: "subsref", target: BuiltinDocumentationLinkTarget::Builtin("subsref") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime indexing", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/table/builtins/indexing.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Table indexing tests", location: "crates/runmat-runtime/src/builtins/table/tests.rs" }],
        notes: &["Method-only identity; ordinary indexing syntax dispatches through the class registry."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
