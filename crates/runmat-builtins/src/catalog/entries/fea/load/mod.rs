use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    constructor_inference::{infer_constructor, FeaConstructorKind},
    descriptors::FEA_LOAD_DESCRIPTOR,
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_constructor(request, entry, FeaConstructorKind::LoadDocument)
}
const FIXTURE: BuiltinExampleFixture = BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
    id: BuiltinExampleFixtureId { local_name: "study-and-mesh" }, root: BuiltinFilesystemRoot::IsolatedWorkspace,
    entries: &[
        BuiltinFilesystemEntry::File { relative_path: "bracket.fea", content: BuiltinFixtureContent::Utf8("version: 1\nkind: study\nid: bracket_static\ngeometry:\n  path: part.stl\n  units: meter\nmodel:\n  profile: linear_static_structural\nrun:\n  backend: cpu\n") },
        BuiltinFilesystemEntry::File { relative_path: "part.stl", content: BuiltinFixtureContent::Utf8("solid tri\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid tri\n") },
    ],
});
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "load-study",
    title: "Load a study and its relative mesh",
    program: "study = fea.load('bracket.fea');",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: FIXTURE,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(study), 'fea.Study'));",
    },
}];
const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.load", summary: "Load a .fea study or sweep document.",
    description: "`fea.load(path)` reads a `.fea` document and resolves it to a typed `fea.Study` or `fea.Sweep` object. Relative geometry paths resolve against the document location.",
    keywords: &["fea", "study", "sweep", "load", "yaml"], related: &["fea.study", "fea.sweep", "fea.validate"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "The path must be host text. The document kind determines whether the returned object is a study or a sweep; static inference cannot promise one class from an unknown file.",
        "Reading, parsing, relative geometry resolution, and typed document validation occur before the loaded object is returned. Errors retain the FEA load/input/internal identifiers.",
        "This operation reads the filesystem and may suspend; no GPU operation or implicit resident-data gathering is performed.",
    ] }], examples: EXAMPLES,
    faqs: &[BuiltinDocumentationFaq { question: "Can a relative geometry path be used?", answer: "Yes. It is resolved relative to the `.fea` document; the example supplies both files in one isolated fixture." }],
    links: &[BuiltinDocumentationLink { label: "Using FEA", target: BuiltinDocumentationLinkTarget::Documentation("/docs/fea/using-fea") }],
    implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/document.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Document load and validation", location: "crates/runmat-runtime/src/builtins/fea/tests/construction.rs::fea_load_validate_and_plan_document_workflow" }],
    notes: &["The path is a local filesystem operand; an execution-owned distributed handle is not a path."],
});
pub const FEA_LOAD_CATALOG_ENTRY: BuiltinCatalogEntry =
    catalog_entry::entry(ConstructorEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.load",
        documentation: DOCUMENTATION,
        descriptor: &FEA_LOAD_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &[],
        integer_audit: None,
        filesystem_read: true,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_LOAD_CATALOG_ENTRY];
