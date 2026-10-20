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
    program: "geom = geometry.load('cube.stl');\nsteel = fea.material('steel', 'YoungsModulusPa', int64(200000000000), 'PoissonRatio', 0.3);\ntarget = fea.PropertyRegionTarget.allSolids(geom);\nassignment = fea.materialAssignment(target, steel);\nregion = geometry.region(geom, 'region_default');\nfixed = fea.StructuralConstraint.fixed('fixed', region);\nstep = fea.StructuralStep.static('static');\nmodel = fea.model('cube_model', geom, 'Materials', {steel}, 'MaterialAssignments', {assignment}, 'Constraints', {fixed}, 'Steps', {step});\nstudy = fea.study('closed_cube_static', geom, model);\nrun = fea.run(study);\nstepRun = run.step('static');",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: CLOSED_CUBE_STUDY,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(run), 'fea.RunResult'));\nassert(~isempty(stepRun.run_id));",
    },
}];
const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.run", summary: "Execute a typed FEA study or sweep.",
    description: "`fea.run(studyOrSweep)` accepts exactly one typed host fea.Study or fea.Sweep, not a document path.",
    keywords: &["fea", "run"], related: &["fea.model", "fea.study", "fea.results"],
    sections: &[BuiltinDocumentationSection { heading: "Typed boundary", paragraphs: &[
        "A study returns fea.RunResult containing procedure-step results; a sweep returns fea.SweepRunResult. Select one persisted run with run.step(stepId), returning fea.StepRunResult. Actual execution honors the typed execution plan and persists its evidence.",
        "Host structural identities, revisions, shapes, and counts remain exact. Physics conversions occur only at their explicit owning boundaries; resident inputs are not implicitly gathered.",
    ] }], examples: EXAMPLES, faqs: &[],
    links: &[BuiltinDocumentationLink { label: "FEA API", target: BuiltinDocumentationLinkTarget::Documentation("/docs/fea/using-fea") }],
    implementation: &[BuiltinDocumentationLink { label: "Current implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/mod.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed Runtime FEA contracts", location: "crates/runmat-runtime/src/builtins/fea/mod.rs::tests::typed_constructors_build_full_study_and_sweep_objects" }],
    notes: &["Catalog examples require execution qualification; these declarations are not solver or numerical conformance receipts."],
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
