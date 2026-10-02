use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition},
    descriptors::FEA_RUN_DESCRIPTOR,
    example_fixtures::CLOSED_CUBE_STUDY,
    integer::FEA_STRUCTURAL_INTEGER_AUDIT,
    operation_entry::{self, OperationEffectProfile, OperationEntryDefinition},
    operation_inference::{infer_evidence, FeaEvidenceKind},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_evidence(request, entry, FeaEvidenceKind::Run)
}
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "run-closed-cube",
    title: "Run a closed-volume linear-static study",
    program: "study = fea.load('cube.fea');\nrun = fea.run(study);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: CLOSED_CUBE_STUDY,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(run), 'fea.RunResult'));\nassert(~isempty(run.run_id));",
    },
}];
const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.run", summary: "Execute a typed FEA study or sweep.",
    description: "`fea.run(studyOrSweepOrPath)` resolves and executes a typed study, sequential sweep, or `.fea` document, then records solver and mesh evidence.",
    keywords: &["fea", "run", "solve", "study", "sweep"], related: &["fea.study", "fea.validate", "fea.plan", "fea.results"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Exactly one host path, `fea.Study`, or `fea.Sweep` is required. Numeric and resident arguments are not silently gathered or treated as documents.",
        "A study returns `fea.RunResult` with run id, status, quality gates, provenance, selected operation, and evidence path. A sweep returns the same class with aggregate counts, per-study entries, and one-based failure indexes.",
        "The configured study backend may use a CPU or GPU solver; host-side input decoding and result serialization do not imply CPU-only solving. The solver can persist mesh, run, and evidence artifacts and may suspend.",
        "Discrete provenance, counts, indexes, fingerprints, and options remain exact typed integers; solver measurements, tolerances, rates, and physical quantities remain binary64.",
        "An already resolved study or STL document needs no OCCT. A topology-bearing CAD path may require the configured importer before solver work.",
    ] }], examples: EXAMPLES,
    faqs: &[BuiltinDocumentationFaq { question: "Does fea.run always use the CPU?", answer: "No. Solver backend selection is part of the study; the example requests CPU only for deterministic qualification." }],
    links: &[BuiltinDocumentationLink { label: "FEA solves", target: BuiltinDocumentationLinkTarget::Documentation("/docs/fea/solves") }],
    implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/entrypoints.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Closed-cube linear-static operation", location: "crates/runmat-runtime/src/analysis/tests.rs::analysis_run_study_executes_linear_static_path" }],
    notes: &["The public closed-cube STL fixture mirrors the tested operation topology but still requires executable catalog-example qualification. Distributed document/object admission lacks a prevalidated partition-local solver route."],
});
pub const FEA_RUN_CATALOG_ENTRY: BuiltinCatalogEntry =
    operation_entry::entry(OperationEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.run",
        documentation: DOCUMENTATION,
        descriptor: &FEA_RUN_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &[],
        integer_audit: Some(&FEA_STRUCTURAL_INTEGER_AUDIT),
        effects: OperationEffectProfile::EvidenceWrite,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_RUN_CATALOG_ENTRY];
