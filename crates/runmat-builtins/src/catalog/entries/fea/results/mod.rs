use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition},
    example_fixtures::CLOSED_CUBE_STUDY,
    integer::{FEA_RESULTS_DESCRIPTOR, FEA_RESULTS_INTEGER_CAPABILITIES},
    operation_entry::{self, OperationEffectProfile, OperationEntryDefinition},
    result_inference::{infer_result, FeaResultKind},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_result(request, entry, FeaResultKind::Results)
}
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "query-run", title: "Read persisted results from a run",
    program: "study = fea.load('cube.fea');\nrun = fea.run(study);\nresults = fea.results(run, 'IncludeDiagnostics', false);",
    display_output: None, compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem, fixture: CLOSED_CUBE_STUDY,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(results), 'fea.Results'));\nassert(strcmp(results.run_id, run.run_id));" },
}];
const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.results", summary: "Query a persisted FEA run for post-processing.",
    description: "`fea.results(runOrRunId, Name, Value, ...)` returns a typed `fea.Results` projection. A lone existing `fea.Results` object is returned unchanged.",
    keywords: &["fea", "results", "fields", "modes", "snapshots"], related: &["fea.run", "fea.field", "fea.plot"],
    sections: &[BuiltinDocumentationSection { heading: "Inputs and selectors", paragraphs: &[
        "The first input is a host run-id text scalar, one-run `fea.RunResult`, or an existing `fea.Results` with no additional options. A sweep result is queried through one of its per-study run ids.",
        "Fields/IncludeFields and DiagnosticCodes accept text scalars, string arrays, or text cells. Inclusion flags accept logical scalars or exact numeric zero/one; other values and resident controls reject before artifact lookup.",
        "ModeIndices and TransientSnapshotIndices are positive one-based structural vectors. All eight integer classes and integral doubles are accepted exactly; zero, negative, fractional, nonfinite, out-of-range, matrix-shaped, and resident selectors reject before lookup.",
        "Omitted lists mean no filtering; omitted mode/snapshot selectors include every available item; inclusion flags default true. Field values and other physics quantities stay binary64, while shapes, counts, indexes, byte sizes, iterations, and fingerprints retain typed integers.",
        "Stored device field references are returned as metadata, not executed or gathered. Artifact retrieval is host-side and needs no OCCT importer.",
    ] }, BuiltinDocumentationSection { heading: "Options", paragraphs: &[
        "Fields/IncludeFields, IncludeFieldValues/FieldValues, IncludeDiagnostics, DiagnosticCodes, IncludeModalResults, ModeIndices, IncludeTransientResults, TransientSnapshotIndices, IncludeNonlinearResults, and IncludeElectromagneticResults are supported.",
    ] }], examples: EXAMPLES,
    faqs: &[BuiltinDocumentationFaq { question: "Are result indexes zero-based?", answer: "No. Mode and snapshot selectors use one-based script indexes and are translated once at the host artifact boundary." }],
    links: &[BuiltinDocumentationLink { label: "FEA operation reference", target: BuiltinDocumentationLinkTarget::Documentation("/docs/fea/operation-reference") }],
    implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/results/constructors.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Result query and field", location: "crates/runmat-runtime/src/builtins/fea/tests/results_and_plotting.rs::fea_results_field_exposes_values_metadata_and_plot_context" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Exact selector boundary", location: "crates/runmat-vm/tests/integer_builtin_fea_execution_and_feedback_file_status.rs" }],
    notes: &["Result selectors are host structural or text values; execution-owned distributed arrays are not accepted as selectors."],
});
pub const FEA_RESULTS_CATALOG_ENTRY: BuiltinCatalogEntry =
    operation_entry::entry(OperationEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.results",
        documentation: DOCUMENTATION,
        descriptor: &FEA_RESULTS_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &FEA_RESULTS_INTEGER_CAPABILITIES,
        integer_audit: None,
        effects: OperationEffectProfile::ArtifactRead,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_RESULTS_CATALOG_ENTRY];
