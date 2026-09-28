use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["Without a key set, values align with the ordering returned by keys. With a cell key set, the result keeps that cell array's shape; missing keys raise an error."] },
    BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &[
        "Map storage and outputs are host-resident. Supported resident inputs gather only after compatibility and argument admission; map operations do not use GPU kernels or numeric fusion.",
        "Map handles are local mutable identities backed by a runtime registry. Copying a handle is not a distributed transfer of the map.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "basic",
    title: "Exercise containers.Map.values",
    program: "m = containers.Map({'a','b'},{1,2});\nv = values(m,{'b','a'});",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isequal(v,{2,1}));",
    },
}];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("containers.Map.values"),
    slug: Some("containers.map.values"),
    summary: "Return a map's values as a host cell array, optionally selecting keys in a supplied cell-array shape.",
    description: "Return a map's values as a host cell array, optionally selecting keys in a supplied cell-array shape. Without a key set, values align with the ordering returned by keys. With a cell key set, the result keeps that cell array's shape; missing keys raise an error.",
    keywords: &["containers.Map", "map", "key", "value"],
    related: &["containers.Map"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Can a selected key be missing?", answer: "No. Selection raises a missing-key error." }],
    links: &[BuiltinDocumentationLink { label: "containers.Map", target: BuiltinDocumentationLinkTarget::Builtin("containers.Map") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/containers/map/containers.map.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Map semantic tests", location: "crates/runmat-runtime/src/builtins/containers/map/tests/general.rs::keys_values_iskey_remove" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
