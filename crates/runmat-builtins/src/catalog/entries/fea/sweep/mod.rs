use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    descriptors::FEA_SWEEP_DESCRIPTOR,
    constructor_inference::{infer_constructor, FeaConstructorKind},
    integer::FEA_STRUCTURAL_INTEGER_AUDIT,
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_constructor(request, entry, FeaConstructorKind::Sweep)
}
const FIXTURE: BuiltinExampleFixture = BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
    id: BuiltinExampleFixtureId { local_name: "triangle-stl" }, root: BuiltinFilesystemRoot::IsolatedWorkspace,
    entries: &[BuiltinFilesystemEntry::File { relative_path: "part.stl", content: BuiltinFixtureContent::Utf8("solid tri\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid tri\n") }],
});
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "two-studies", title: "Continue after failed studies", program: "geom = geometry.load('part.stl');\nstudyA = fea.study('variant_a', geom, 'Profile', 'linear_static_structural');\nstudyB = fea.study('variant_b', geom, 'Profile', 'linear_static_structural');\nsweep = fea.sweep('variants', {studyA, studyB}, 'FailFast', false);",
    display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: FIXTURE, requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(sweep), 'fea.Sweep'));" },
}];
const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.sweep", summary: "Create a deterministic FEA study sweep.",
    description: "`fea.sweep(id, studies, Name, Value, ...)` creates a `fea.Sweep` from one study or a cell array of studies.",
    keywords: &["fea", "sweep", "studies", "fail fast"], related: &["fea.study", "fea.validate", "fea.plan", "fea.run"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Required inputs are a text sweep id and one `fea.Study` or cell array of `fea.Study` objects.",
        "FailFast is the only option. It defaults to true and accepts a logical scalar; numeric and resident controls reject before construction.",
        "Direct integers have no study/id role and FailFast is logical-only. Public study counts and indexes stay exact typed integers.",
        "Construction is host-side and does not execute providers or intrinsically require native OCCT.",
    ] }], examples: EXAMPLES,
    faqs: &[BuiltinDocumentationFaq { question: "Does creating a sweep run its studies?", answer: "No. Pass the typed definition to `fea.validate`, `fea.plan`, or `fea.run`." }],
    links: &[BuiltinDocumentationLink { label: "FEA solves", target: BuiltinDocumentationLinkTarget::Documentation("/docs/fea/solves") }],
    implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/study.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Sweep constructor", location: "crates/runmat-runtime/src/builtins/fea/tests/construction.rs" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "FEA VM execution", location: "crates/runmat-vm/tests/integer_builtin_fea_execution_and_feedback_file_status.rs" }],
    notes: &["Study objects and their geometry ownership are not execution-owned distributed arrays; the constructor rejects distributed handles."],
});
pub const FEA_SWEEP_CATALOG_ENTRY: BuiltinCatalogEntry =
    catalog_entry::entry(ConstructorEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.sweep",
        documentation: DOCUMENTATION,
        descriptor: &FEA_SWEEP_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &[],
        integer_audit: Some(&FEA_STRUCTURAL_INTEGER_AUDIT),
        filesystem_read: false,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_SWEEP_CATALOG_ENTRY];
