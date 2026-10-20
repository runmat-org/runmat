use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition},
    descriptors::FEA_FIELD_DESCRIPTOR,
    example_fixtures::CLOSED_CUBE_STUDY,
    integer::FEA_STRUCTURAL_INTEGER_AUDIT,
    operation_entry::{self, OperationEffectProfile, OperationEntryDefinition},
    result_inference::{infer_result, FeaResultKind},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_result(request, entry, FeaResultKind::Field)
}
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "von-mises", title: "Extract the solved stress field",
    program: "geom = geometry.load('cube.stl');\nsteel = fea.material('steel', 'YoungsModulusPa', int64(200000000000), 'PoissonRatio', 0.3);\ntarget = fea.PropertyRegionTarget.allSolids(geom);\nassignment = fea.materialAssignment(target, steel);\nregion = geometry.region(geom, 'region_default');\nfixed = fea.StructuralConstraint.fixed('fixed', region);\nstep = fea.StructuralStep.static('static');\nmodel = fea.model('cube_model', geom, 'Materials', {steel}, 'MaterialAssignments', {assignment}, 'Constraints', {fixed}, 'Steps', {step});\nstudy = fea.study('closed_cube_static', geom, model);\nrun = fea.run(study);\nstepRun = run.step('static');\nresults = fea.results(stepRun);\nstress = fea.field(results, fea.StructuralField.vonMises());",
    display_output: None, compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem, fixture: CLOSED_CUBE_STUDY,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(stress), 'fea.Field'));\nassert(strcmp(stress.field_id, 'structural.von_mises'));" },
}];
const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.field", summary: "Select a typed FEA result field.",
    description: "`fea.field(resultsOrRun, selector[, sample])` accepts typed fea.StructuralField, fea.ResponseSpectrumField, fea.RandomResponseField, or fea.OutputField selectors.",
    keywords: &["fea", "field"], related: &["fea.model", "fea.study", "fea.results"],
    sections: &[BuiltinDocumentationSection { heading: "Typed boundary", paragraphs: &[
        "An optional sample must be a positive finite fea.Frequency. Text field ids are not accepted. Inline host values may be retained; artifact-backed fields expose metadata and bounded values are read with field.page(start,count).",
        "Host structural identities, revisions, shapes, and counts remain exact. Physics conversions occur only at their explicit owning boundaries; resident inputs are not implicitly gathered.",
    ] }], examples: EXAMPLES, faqs: &[],
    links: &[BuiltinDocumentationLink { label: "FEA API", target: BuiltinDocumentationLinkTarget::Documentation("/docs/fea/using-fea") }],
    implementation: &[BuiltinDocumentationLink { label: "Current implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/mod.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed Runtime FEA contracts", location: "crates/runmat-runtime/src/builtins/fea/mod.rs::tests::typed_constructors_build_full_study_and_sweep_objects" }],
    notes: &["Catalog examples require execution qualification; these declarations are not solver or numerical conformance receipts."],
});
pub const FEA_FIELD_CATALOG_ENTRY: BuiltinCatalogEntry =
    operation_entry::entry(OperationEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.field",
        documentation: DOCUMENTATION,
        descriptor: &FEA_FIELD_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &[],
        integer_audit: Some(&FEA_STRUCTURAL_INTEGER_AUDIT),
        effects: OperationEffectProfile::ArtifactRead,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_FIELD_CATALOG_ENTRY];
