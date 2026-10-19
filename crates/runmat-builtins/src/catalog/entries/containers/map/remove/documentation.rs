use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["Removal mutates the shared handle. Removing a missing key raises an error; resident key inputs require the RunMat compatibility extension."] },
    BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &[
        "Map storage and outputs are host-resident. Supported resident inputs gather only after compatibility and argument admission; map operations do not use GPU kernels or numeric fusion.",
        "Map handles are local mutable identities backed by a runtime registry. Copying a handle is not a distributed transfer of the map.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "basic",
    title: "Exercise containers.Map.remove",
    program: "m = containers.Map({'a','b'},{1,2});\nremove(m,{'a'});",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(~isKey(m,'a'));\nassert(m.Count == uint64(1));",
    },
}];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("containers.Map.remove"),
    slug: Some("containers.map.remove"),
    summary: "Remove existing keys from a containers.Map handle.",
    description: "Remove existing keys from a containers.Map handle. Removal mutates the shared handle. Removing a missing key raises an error; resident key inputs require the RunMat compatibility extension.",
    keywords: &["containers.Map", "map", "key", "value"],
    related: &["containers.Map"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Does remove return a new map?", answer: "No. It changes the existing handle and has no public output." }],
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
