use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    descriptors::FEA_STUDY_DESCRIPTOR,
    constructor_inference::{infer_constructor, FeaConstructorKind},
    integer::FEA_STRUCTURAL_INTEGER_AUDIT,
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_constructor(request, entry, FeaConstructorKind::Study)
}
const FIXTURE: BuiltinExampleFixture = BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
    id: BuiltinExampleFixtureId { local_name: "study-files" }, root: BuiltinFilesystemRoot::IsolatedWorkspace,
    entries: &[
        BuiltinFilesystemEntry::File { relative_path: "part.stl", content: BuiltinFixtureContent::Utf8("solid tri\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid tri\n") },
        BuiltinFilesystemEntry::File { relative_path: "bracket.fea", content: BuiltinFixtureContent::Utf8("version: 1\nkind: study\nid: bracket_static\ngeometry:\n  path: part.stl\n  units: meter\nmodel:\n  profile: linear_static_structural\nrun:\n  backend: cpu\n") },
    ],
});
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "construct-study", title: "Create a study from a geometry asset", program: "geom = geometry.load('part.stl');\nstudy = fea.study('bracket', geom, 'Profile', 'linear_static_structural', 'Backend', 'cpu');", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::NativeFilesystem, fixture: FIXTURE, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(study), 'fea.Study'));" } },
    BuiltinExample { id: "load-study", title: "Load a study from a document", program: "study = fea.study('bracket.fea');", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::NativeFilesystem, fixture: FIXTURE, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(study), 'fea.Study'));" } },
];
const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.study", summary: "Create or load a typed RunMat FEA study.",
    description: "`fea.study(path)` delegates to the `.fea` document loader and may return a `fea.Study` or `fea.Sweep`; `fea.study(id, geometry, Name, Value, ...)` assembles a typed `fea.Study`.",
    keywords: &["fea", "study", "model", "geometry"], related: &["fea.load", "fea.model", "fea.runOptions"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "The one-path form delegates to `fea.load`; its result class is determined by the document kind, including sweep. The constructor form requires a text id and `geometry.Asset`.",
        "Constructor options are RunKind/Kind, Profile, Backend, ModelId, Model, Frame, Defaults, Materials, MaterialAssignments/Assignments, BoundaryConditions/BCs, Loads/LoadCases, Steps, Domains, Interfaces, and RunOptions/Options.",
        "Profile is required; backend defaults to `cpu`, and defaults mode to `profile_scaffold`. Model id comes from an explicit model or sanitized `<study-id>_model`. Explicit run kind must agree with the profile-derived solver.",
        "Direct integer inputs have no role. Nested typed components and run options retain exact structural and binary64 physics contracts; public revision and count properties stay exact typed integers.",
        "Identity, option names, profile/run-kind agreement, and component classes are validated before construction. Existing STL/synthetic geometry needs no OCCT; topology CAD may require a configured importer.",
    ] }], examples: EXAMPLES,
    faqs: &[BuiltinDocumentationFaq { question: "Can numeric values stand in for profile or backend?", answer: "No. Structural identity fields are text or typed objects and are never numerically coerced." }],
    links: &[BuiltinDocumentationLink { label: "Using FEA", target: BuiltinDocumentationLinkTarget::Documentation("/docs/fea/using-fea") }],
    implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/study.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Study constructor and document", location: "crates/runmat-runtime/src/builtins/fea/tests/construction.rs" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "FEA VM execution", location: "crates/runmat-vm/tests/integer_builtin_fea_execution_and_feedback_file_status.rs" }],
    notes: &["An execution-owned distributed handle is not a local geometry asset or path; no prevalidated distributed constructor materialization is available."],
});
pub const FEA_STUDY_CATALOG_ENTRY: BuiltinCatalogEntry =
    catalog_entry::entry(ConstructorEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.study",
        documentation: DOCUMENTATION,
        descriptor: &FEA_STUDY_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &[],
        integer_audit: Some(&FEA_STRUCTURAL_INTEGER_AUDIT),
        filesystem_read: true,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_STUDY_CATALOG_ENTRY];
