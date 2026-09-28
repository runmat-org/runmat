use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["Parenthesis indexing accepts one scalar key; dot indexing exposes Count, KeyType, and ValueType. The key lookup result depends on the stored value and remains dynamic in static inference."] },
    BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &[
        "Map storage and outputs are host-resident. Supported resident inputs gather only after compatibility and argument admission; map operations do not use GPU kernels or numeric fusion.",
        "Map handles are local mutable identities backed by a runtime registry. Copying a handle is not a distributed transfer of the map.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "basic",
    title: "Exercise containers.Map.subsref",
    program:
        "m = containers.Map({'a'},{2});\ns = struct('type','()','subs',{{'a'}});\nv = subsref(m,s);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(v == 2);",
    },
}];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("containers.Map.subsref"),
    slug: Some("containers.map.subsref"),
    summary: "Read a key value or a read-only map property through a standard subscript path.",
    description: "Read a key value or a read-only map property through a standard subscript path. Parenthesis indexing accepts one scalar key; dot indexing exposes Count, KeyType, and ValueType. The key lookup result depends on the stored value and remains dynamic in static inference.",
    keywords: &["containers.Map", "map", "key", "value"],
    related: &["containers.Map"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "What happens for an unknown key?", answer: "Lookup raises a missing-key error." }],
    links: &[BuiltinDocumentationLink { label: "containers.Map", target: BuiltinDocumentationLinkTarget::Builtin("containers.Map") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/containers/map/containers.map.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Map semantic tests", location: "crates/runmat-runtime/src/builtins/containers/map/tests/general.rs::constructor_with_cells_lookup" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
