use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition},
    descriptors::FEA_PLAN_DESCRIPTOR,
    integer::FEA_STRUCTURAL_INTEGER_AUDIT,
    operation_entry::{self, OperationEffectProfile, OperationEntryDefinition},
    operation_inference::{infer_evidence, FeaEvidenceKind},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_evidence(request, entry, FeaEvidenceKind::Plan)
}
const FIXTURE: BuiltinExampleFixture = BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
    id: BuiltinExampleFixtureId { local_name: "planning-study" }, root: BuiltinFilesystemRoot::IsolatedWorkspace,
    entries: &[
        BuiltinFilesystemEntry::File { relative_path: "bracket.fea", content: BuiltinFixtureContent::Utf8("version: 1\nkind: study\nid: bracket_static\ngeometry:\n  path: part.stl\n  units: meter\nmodel:\n  profile: linear_static_structural\nrun:\n  backend: cpu\n") },
        BuiltinFilesystemEntry::File { relative_path: "part.stl", content: BuiltinFixtureContent::Utf8("solid tri\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid tri\n") },
    ],
});
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "plan-study",
    title: "Plan without solving",
    program: "study = fea.load('bracket.fea');\nplan = fea.plan(study);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: FIXTURE,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source:
            "assert(strcmp(class(plan), 'fea.Plan'));\nassert(~isempty(plan.operation_sequence));",
    },
}];
const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.plan", summary: "Plan a typed FEA study or sweep without solving.",
    description: "`fea.plan(studyOrSweepOrPath)` validates and plans a typed study, sweep, or `.fea` document, then persists planning evidence.",
    keywords: &["fea", "plan", "study", "validation"], related: &["fea.validate", "fea.run", "fea.load"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Exactly one host `.fea` path, `fea.Study`, or `fea.Sweep` is required. Numeric arrays and resident values have no document role.",
        "Planning resolves model data, validates it, selects a versioned operation sequence, and writes evidence; the solver is not executed.",
        "A single study and sweep both return `fea.Plan`. Counts, indexes, and revisions remain exact structural integers; estimated physical and timing quantities keep their floating representation.",
        "Already loaded or STL geometry needs no OCCT. Path resolution of topology-bearing CAD may require the configured importer.",
    ] }], examples: EXAMPLES,
    faqs: &[BuiltinDocumentationFaq { question: "Does planning solve the model?", answer: "No. It records the selected operation sequence without running the solver." }],
    links: &[BuiltinDocumentationLink { label: "FEA solves", target: BuiltinDocumentationLinkTarget::Documentation("/docs/fea/solves") }],
    implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/entrypoints.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Document plan", location: "crates/runmat-runtime/src/builtins/fea/tests/construction.rs::fea_load_validate_and_plan_document_workflow" }],
    notes: &["Filesystem capability covers document/geometry resolution and planning evidence. Distributed object/path operands have no partition-local planner."],
});
pub const FEA_PLAN_CATALOG_ENTRY: BuiltinCatalogEntry =
    operation_entry::entry(OperationEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.plan",
        documentation: DOCUMENTATION,
        descriptor: &FEA_PLAN_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &[],
        integer_audit: Some(&FEA_STRUCTURAL_INTEGER_AUDIT),
        effects: OperationEffectProfile::EvidenceWrite,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_PLAN_CATALOG_ENTRY];
