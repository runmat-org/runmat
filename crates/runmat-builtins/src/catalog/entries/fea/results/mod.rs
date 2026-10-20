use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition},
    descriptors::FEA_RESULTS_DESCRIPTOR,
    example_fixtures::CLOSED_CUBE_STUDY,
    integer::FEA_RESULTS_INTEGER_CAPABILITIES,
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
    program: "geom = geometry.load('cube.stl');\nsteel = fea.material('steel', 'YoungsModulusPa', int64(200000000000), 'PoissonRatio', 0.3);\ntarget = fea.PropertyRegionTarget.allSolids(geom);\nassignment = fea.materialAssignment(target, steel);\nregion = geometry.region(geom, 'region_default');\nfixed = fea.StructuralConstraint.fixed('fixed', region);\nstep = fea.StructuralStep.static('static');\nmodel = fea.model('cube_model', geom, 'Materials', {steel}, 'MaterialAssignments', {assignment}, 'Constraints', {fixed}, 'Steps', {step});\nstudy = fea.study('closed_cube_static', geom, model);\nrun = fea.run(study);\nstepRun = run.step('static');\nresults = fea.results(stepRun, 'IncludeDiagnostics', false);",
    display_output: None, compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem, fixture: CLOSED_CUBE_STUDY,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(results), 'fea.Results'));\nassert(strcmp(results.run_id, stepRun.run_id));" },
}];
const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.results", summary: "Query persisted FEA run results.",
    description: "`fea.results(runOrRunId, Name, Value, ...)` accepts a host run-id, fea.StepRunResult, or a single-step fea.RunResult. A lone fea.Results is returned unchanged.",
    keywords: &["fea", "results"], related: &["fea.run", "fea.field", "fea.compare"],
    sections: &[BuiltinDocumentationSection { heading: "Typed boundary", paragraphs: &[
        "Select multi-step results with run.step(stepId) first. Exact one-based mode/snapshot selectors, text field/diagnostic filters, and logical or exact zero/one inclusion flags retain their current query contracts. Retrieval does not launch a solver.",
        "Host structural identities, revisions, shapes, and counts remain exact. Physics conversions occur only at their explicit owning boundaries; resident inputs are not implicitly gathered.",
    ] }], examples: EXAMPLES, faqs: &[],
    links: &[BuiltinDocumentationLink { label: "FEA API", target: BuiltinDocumentationLinkTarget::Documentation("/docs/fea/using-fea") }],
    implementation: &[BuiltinDocumentationLink { label: "Current implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/mod.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed Runtime FEA contracts", location: "crates/runmat-runtime/src/builtins/fea/mod.rs::tests::typed_constructors_build_full_study_and_sweep_objects" }],
    notes: &["Catalog examples require execution qualification; these declarations are not solver or numerical conformance receipts."],
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
