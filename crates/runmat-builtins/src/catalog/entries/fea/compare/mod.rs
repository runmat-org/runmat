use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition},
    example_fixtures::CLOSED_CUBE_STUDY,
    integer::{FEA_COMPARE_DESCRIPTOR, FEA_STRUCTURAL_INTEGER_AUDIT},
    operation_entry::{self, OperationEffectProfile, OperationEntryDefinition},
    result_inference::{infer_result, FeaResultKind},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_result(request, entry, FeaResultKind::Compare)
}
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "identity-comparison", title: "Compare a persisted run with itself",
    program: "study = fea.load('cube.fea');\nrun = fea.run(study);\ncomparison = fea.compare(run.run_id, run.run_id);",
    display_output: None, compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem, fixture: CLOSED_CUBE_STUDY,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(comparison), 'fea.Compare'));" },
}];
const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.compare", summary: "Compare persisted FEA run evidence.",
    description: "`fea.compare(baselineRunId, candidateRunId)` compares summary, status, quality, iteration, and timing evidence from two persisted run identifiers.",
    keywords: &["fea", "compare", "results", "quality"], related: &["fea.run", "fea.results", "fea.trends"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Exactly two host text run ids are required; there are no Name, Value options. Numeric and resident inputs reject before artifact lookup.",
        "Both ids are resolved from the artifact store. Count and count-delta fields remain exact signed integers; timing deltas are binary64 measurements.",
        "The example compares one persisted run with itself to give a deterministic identity oracle. In ordinary use pass ids from distinct baseline and candidate runs.",
        "No solver or numeric provider runs during comparison, and no OCCT importer is needed.",
    ] }], examples: EXAMPLES,
    faqs: &[BuiltinDocumentationFaq { question: "Can options follow the two ids?", answer: "No. The current contract accepts exactly two run ids." }],
    links: &[BuiltinDocumentationLink { label: "FEA operation reference", target: BuiltinDocumentationLinkTarget::Documentation("/docs/fea/operation-reference") }],
    implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/comparison.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Comparison operation", location: "crates/runmat-runtime/src/analysis/tests.rs" }],
    notes: &["Run identifiers are host text and cannot be execution-owned distributed numeric arrays."],
});
pub const FEA_COMPARE_CATALOG_ENTRY: BuiltinCatalogEntry =
    operation_entry::entry(OperationEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.compare",
        documentation: DOCUMENTATION,
        descriptor: &FEA_COMPARE_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &[],
        integer_audit: Some(&FEA_STRUCTURAL_INTEGER_AUDIT),
        effects: OperationEffectProfile::ArtifactRead,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_COMPARE_CATALOG_ENTRY];
