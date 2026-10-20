use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    constructor_inference::{infer_constructor, FeaConstructorKind},
    descriptors::FEA_SWEEP_DESCRIPTOR,
    example_fixtures::CLOSED_CUBE_STUDY,
    integer::FEA_STRUCTURAL_INTEGER_AUDIT,
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_constructor(request, entry, FeaConstructorKind::Sweep)
}
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "two-studies", title: "Continue after failed studies", program: "geom = geometry.load('cube.stl');\nsteel = fea.material('steel', 'YoungsModulusPa', int64(200000000000), 'PoissonRatio', 0.3);\ntarget = fea.PropertyRegionTarget.allSolids(geom);\nassignment = fea.materialAssignment(target, steel);\nregion = geometry.region(geom, 'region_default');\nfixed = fea.StructuralConstraint.fixed('fixed', region);\nstep = fea.StructuralStep.static('static');\nmodel = fea.model('cube_model', geom, 'Materials', {steel}, 'MaterialAssignments', {assignment}, 'Constraints', {fixed}, 'Steps', {step});\nstudy = fea.study('closed_cube_static', geom, model);\nstudyB = fea.study('variant_b', geom, model);\nsweep = fea.sweep('variants', {study, studyB}, 'FailFast', false);",
    display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: CLOSED_CUBE_STUDY, requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(sweep), 'fea.Sweep'));" },
}];
const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.sweep", summary: "Create a typed FEA study sweep.",
    description: "`fea.sweep(id, studies, Name, Value, ...)` accepts a fea.Study or a cell of studies with distinct identities.",
    keywords: &["fea", "sweep"], related: &["fea.model", "fea.study", "fea.results"],
    sections: &[BuiltinDocumentationSection { heading: "Typed boundary", paragraphs: &[
        "FailFast is the only option and requires a logical scalar. Construction does not execute the studies.",
        "Host structural identities, revisions, shapes, and counts remain exact. Physics conversions occur only at their explicit owning boundaries; resident inputs are not implicitly gathered.",
    ] }], examples: EXAMPLES, faqs: &[],
    links: &[BuiltinDocumentationLink { label: "FEA API", target: BuiltinDocumentationLinkTarget::Documentation("/docs/fea/using-fea") }],
    implementation: &[BuiltinDocumentationLink { label: "Current implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/mod.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed Runtime FEA contracts", location: "crates/runmat-runtime/src/builtins/fea/mod.rs::tests::typed_constructors_build_full_study_and_sweep_objects" }],
    notes: &["Catalog examples require execution qualification; these declarations are not solver or numerical conformance receipts."],
});
pub const FEA_SWEEP_CATALOG_ENTRY: BuiltinCatalogEntry =
    catalog_entry::entry(ConstructorEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.sweep",
        documentation: DOCUMENTATION,
        descriptor: &FEA_SWEEP_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &[],
        integer_audit: Some(&FEA_STRUCTURAL_INTEGER_AUDIT),
        filesystem_read: false,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_SWEEP_CATALOG_ENTRY];
