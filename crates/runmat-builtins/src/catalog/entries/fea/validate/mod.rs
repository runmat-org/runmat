use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition},
    descriptors::FEA_VALIDATE_DESCRIPTOR,
    integer::FEA_STRUCTURAL_INTEGER_AUDIT,
    operation_entry::{self, OperationEffectProfile, OperationEntryDefinition},
    operation_inference::{infer_evidence, FeaEvidenceKind},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_evidence(request, entry, FeaEvidenceKind::Validate)
}
const FIXTURE: BuiltinExampleFixture = BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
    id: BuiltinExampleFixtureId { local_name: "validation-study" }, root: BuiltinFilesystemRoot::IsolatedWorkspace,
    entries: &[
        BuiltinFilesystemEntry::File { relative_path: "bracket.fea", content: BuiltinFixtureContent::Utf8("version: 1\nkind: study\nid: bracket_static\ngeometry:\n  path: part.stl\n  units: meter\nmodel:\n  profile: linear_static_structural\nrun:\n  backend: cpu\n") },
        BuiltinFilesystemEntry::File { relative_path: "part.stl", content: BuiltinFixtureContent::Utf8("solid tri\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid tri\n") },
    ],
});
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "validate-study",
    title: "Validate a study document",
    program: "study = fea.load('bracket.fea');\nvalidation = fea.validate(study);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: FIXTURE,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(validation), 'fea.Validation'));\nassert(validation.valid);",
    },
}];
const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.validate", summary: "Validate a typed FEA study or sweep without solving.",
    description: "`fea.validate(studyOrSweepOrPath)` validates one typed study, sweep, or `.fea` document and records validation evidence.",
    keywords: &["fea", "validate", "study", "sweep"], related: &["fea.load", "fea.plan", "fea.run"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Exactly one host `.fea` path, `fea.Study`, or `fea.Sweep` is required. Direct integer and resident numeric operands are not accepted.",
        "The `fea.Validation` object reports validity, issue codes and details, and an evidence path; sweep validation additionally reports per-study entries. These public fields are logical or text, not physical numeric measurements.",
        "Validation reads geometry and writes evidence but does not plan or solve. STL-backed and already resolved geometry need no OCCT; topology-bearing CAD may require the configured importer during path resolution.",
    ] }], examples: EXAMPLES,
    faqs: &[BuiltinDocumentationFaq { question: "Does validation allocate solver results?", answer: "No. It checks the study or sweep and persists validation evidence without solving." }],
    links: &[BuiltinDocumentationLink { label: "FEA validation", target: BuiltinDocumentationLinkTarget::Documentation("/docs/fea/validation") }],
    implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/entrypoints.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Document validation", location: "crates/runmat-runtime/src/builtins/fea/tests/construction.rs::fea_load_validate_and_plan_document_workflow" }],
    notes: &["Filesystem admission is required for path resolution and evidence persistence. FEA object/path operands are not distributed numeric partitions."],
});
pub const FEA_VALIDATE_CATALOG_ENTRY: BuiltinCatalogEntry =
    operation_entry::entry(OperationEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.validate",
        documentation: DOCUMENTATION,
        descriptor: &FEA_VALIDATE_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &[],
        integer_audit: Some(&FEA_STRUCTURAL_INTEGER_AUDIT),
        effects: OperationEffectProfile::EvidenceWrite,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_VALIDATE_CATALOG_ENTRY];
