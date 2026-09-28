use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["The output preserves the native class of numeric keys, including uint64 keys beyond binary64 exactness. The result is host-resident."] },
    BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &[
        "Map storage and outputs are host-resident. Supported resident inputs gather only after compatibility and argument admission; map operations do not use GPU kernels or numeric fusion.",
        "Map handles are local mutable identities backed by a runtime registry. Copying a handle is not a distributed transfer of the map.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "basic",
    title: "Exercise containers.Map.keys",
    program: "m = containers.Map({'b','a'},{2,1});\nk = keys(m);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isequal(k,{'a','b'}));",
    },
}];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("containers.Map.keys"),
    slug: Some("containers.map.keys"),
    summary: "Return a row cell array of a map's keys in RunMat key order.",
    description: "Return a row cell array of a map's keys in RunMat key order. The output preserves the native class of numeric keys, including uint64 keys beyond binary64 exactness. The result is host-resident.",
    keywords: &["containers.Map", "map", "key", "value"],
    related: &["containers.Map"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Is insertion order preserved?", answer: "Do not rely on insertion order. RunMat returns keys in key order." }],
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
