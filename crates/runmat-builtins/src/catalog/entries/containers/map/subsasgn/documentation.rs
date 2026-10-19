use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["Assignment updates or inserts a key on the original shared handle and returns that handle. Declared ValueType and UniformValues admission still apply."] },
    BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &[
        "Map storage and outputs are host-resident. Supported resident inputs gather only after compatibility and argument admission; map operations do not use GPU kernels or numeric fusion.",
        "Map handles are local mutable identities backed by a runtime registry. Copying a handle is not a distributed transfer of the map.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "basic",
    title: "Exercise containers.Map.subsasgn",
    program: "m = containers.Map({'a'},{2});\ns = struct('type','()','subs',{{'a'}});\nm = subsasgn(m,s,3);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(m('a') == 3);",
    },
}];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("containers.Map.subsasgn"),
    slug: Some("containers.map.subsasgn"),
    summary: "Assign a value to one map key using a standard subscript path.",
    description: "Assign a value to one map key using a standard subscript path. Assignment updates or inserts a key on the original shared handle and returns that handle. Declared ValueType and UniformValues admission still apply.",
    keywords: &["containers.Map", "map", "key", "value"],
    related: &["containers.Map"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Does assigning through an alias update the original?", answer: "Yes. Both variables refer to the same handle." }],
    links: &[BuiltinDocumentationLink { label: "containers.Map", target: BuiltinDocumentationLinkTarget::Builtin("containers.Map") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/containers/map/containers.map.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Map semantic tests", location: "crates/runmat-runtime/src/builtins/containers/map/tests/general.rs::assignment_updates_and_inserts" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
