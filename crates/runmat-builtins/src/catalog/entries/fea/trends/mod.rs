use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition},
    integer::{FEA_TRENDS_DESCRIPTOR, FEA_TRENDS_INTEGER_CAPABILITIES},
    operation_entry::{self, OperationEffectProfile, OperationEntryDefinition},
    operation_inference::{infer_evidence, FeaEvidenceKind},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_evidence(request, entry, FeaEvidenceKind::Trends)
}
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "recent-runs",
    title: "Summarize a bounded run window",
    program: "trends = fea.trends('WindowSize', uint32(32));",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(trends), 'fea.Trends'));",
    },
}];
const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.trends", summary: "Summarize recent persisted FEA run evidence.",
    description: "`fea.trends(Name, Value, ...)` queries persisted runs in a bounded per-solver-family window and returns `fea.Trends`.",
    keywords: &["fea", "trends", "window", "artifacts"], related: &["fea.run", "fea.compare", "fea.results"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "WindowSize is the only option; it defaults to 16. It must be a positive exact structural scalar. All eight integer classes and ordinary integral doubles are accepted; fractional, nonfinite, negative, out-of-range, nonscalar, and resident values reject before artifact lookup.",
        "The query reads existing run artifacts but performs no solver or numeric provider operation. An empty artifact set produces a typed trends summary with no family entries.",
        "Window and returned count metadata remain exact integers. Times, rates, ratios, and aggregate measurements retain binary64 representation. No OCCT importer is involved.",
    ] }], examples: EXAMPLES,
    faqs: &[BuiltinDocumentationFaq { question: "Is zero an unlimited window?", answer: "No. Omit WindowSize for the default 16; zero is invalid." }],
    links: &[BuiltinDocumentationLink { label: "FEA evidence", target: BuiltinDocumentationLinkTarget::Documentation("/docs/fea/evidence") }],
    implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/comparison.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Trends artifact query", location: "crates/runmat-runtime/src/analysis/tests.rs" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "FEA integer feedback", location: "crates/runmat-vm/tests/integer_builtin_fea_execution_and_feedback_file_status.rs" }],
    notes: &["The artifact store is process-local filesystem state; WindowSize preflight must precede lookup. Distributed WindowSize materialization would currently precede this validation."],
});
pub const FEA_TRENDS_CATALOG_ENTRY: BuiltinCatalogEntry =
    operation_entry::entry(OperationEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.trends",
        documentation: DOCUMENTATION,
        descriptor: &FEA_TRENDS_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &FEA_TRENDS_INTEGER_CAPABILITIES,
        integer_audit: None,
        effects: OperationEffectProfile::ArtifactRead,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_TRENDS_CATALOG_ENTRY];
