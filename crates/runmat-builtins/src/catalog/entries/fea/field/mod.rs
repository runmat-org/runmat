use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition},
    example_fixtures::CLOSED_CUBE_STUDY,
    integer::{FEA_FIELD_DESCRIPTOR, FEA_STRUCTURAL_INTEGER_AUDIT},
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
    program: "study = fea.load('cube.fea');\nrun = fea.run(study);\nresults = fea.results(run);\nstress = fea.field(results, 'structural.von_mises');",
    display_output: None, compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem, fixture: CLOSED_CUBE_STUDY,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(stress), 'fea.Field'));\nassert(strcmp(stress.field_id, 'structural.von_mises'));" },
}];
const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.field", summary: "Extract one named field from FEA results.",
    description: "`fea.field(resultsOrRun, fieldId)` resolves one named field and returns a typed `fea.Field` object.",
    keywords: &["fea", "field", "results", "post-processing"], related: &["fea.results", "fea.plot", "fea.run"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Exactly one host results/run object or run-id text and one host text field id are required. Numeric arrays have no run or field-id role.",
        "An exact id wins; a unique dotted-name suffix is accepted. Missing or ambiguous fields fail with a typed FEA input error.",
        "Host field values are binary64 tensors. A device field reference remains provider metadata and is not silently gathered. Shape, component-count, element-count, byte-size, and entity metadata remain typed integers.",
        "A `fea.Results` input can resolve from its embedded payload; a run id or `fea.RunResult` may read the artifact store first. No OCCT importer is involved.",
    ] }], examples: EXAMPLES,
    faqs: &[BuiltinDocumentationFaq { question: "Are shape and count fields physical data?", answer: "No. They are exact structural metadata; the values property contains the physical field data." }],
    links: &[BuiltinDocumentationLink { label: "Using FEA", target: BuiltinDocumentationLinkTarget::Documentation("/docs/fea/using-fea") }],
    implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/results/constructors.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Result field metadata", location: "crates/runmat-runtime/src/builtins/fea/tests/results_and_plotting.rs::fea_results_field_exposes_values_metadata_and_plot_context" }],
    notes: &["A field object may carry a device-reference descriptor; extracting it does not transfer or materialize device values."],
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
