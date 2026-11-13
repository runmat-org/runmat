use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("table.subsasgn"), slug: Some("table.subsasgn"),
    summary: "Assign through a table or timetable subscript path.",
    description: "`table.subsasgn(T,S,rhs)` returns an updated table or timetable. Dot assignment sets a variable or Properties; parentheses assign table rows and variables; braces assign selected variable contents.",
    keywords: &["table", "timetable", "subsasgn", "assignment"],
    related: &["table", "table.subsref", "subsasgn"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "The returned value carries the assignment. Callers must use the returned object; the original value is not mutated in place.",
            "Variable row counts and selected dimensions are checked before insertion. Chained paths assign into the selected child and write the updated child back into the outer table.",
            "Typed integer variables retain their class when the assignment is admissible. Static analysis preserves the receiver's table or timetable identity but invalidates its previous schema facts.",
        ] },
        BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &[
            "Assignment is a host-side object operation and a numeric fusion boundary. A selected nested value may have its own residency and indexing behavior.",
            "A chained assignment may suspend or throw. The runtime does not replay effects after a failed nested assignment.",
        ] },
    ],
    examples: &[BuiltinExample {
        id: "variable-assignment", title: "Replace a typed variable",
        program: "T = table(uint64([1; 2]), 'VariableNames', {'Count'}); S = struct('type', '.', 'subs', 'Count'); T = subsasgn(T,S,uint64([3; 4]));",
        display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(istable(T)); assert(isa(T.Count,'uint64')); assert(isequal(T.Count,uint64([3; 4])));" },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Does subsasgn mutate its input?", answer: "No. Assign the returned table to a variable to keep the change." }],
    links: &[BuiltinDocumentationLink { label: "table", target: BuiltinDocumentationLinkTarget::Builtin("table") }, BuiltinDocumentationLink { label: "subsasgn", target: BuiltinDocumentationLinkTarget::Builtin("subsasgn") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime indexing", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/table/builtins/indexing.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Table indexing tests", location: "crates/runmat-runtime/src/builtins/table/tests.rs" }],
        notes: &["Method-only identity; ordinary assignment syntax dispatches through the class registry."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
