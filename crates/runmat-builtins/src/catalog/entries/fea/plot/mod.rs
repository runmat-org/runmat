use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition},
    example_fixtures::CLOSED_CUBE_STUDY,
    integer::{FEA_PLOT_DESCRIPTOR, FEA_STRUCTURAL_INTEGER_AUDIT},
    operation_entry::{self, OperationEffectProfile, OperationEntryDefinition},
    result_inference::{infer_result, FeaResultKind},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_result(request, entry, FeaResultKind::Plot)
}
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "von-mises-figure", title: "Plot the solved stress field",
    program: "study = fea.load('cube.fea');\nrun = fea.run(study);\nfigureHandle = fea.plot(run, 'structural.von_mises');",
    display_output: None, compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem, fixture: CLOSED_CUBE_STUDY,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(figureHandle >= 1);" },
}];
const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.plot", summary: "Create a RunMat figure from a solved FEA field.",
    description: "`fea.plot` renders a contextual run, results object, field object, or explicit study/run pair using the `plot-core` runtime feature.",
    keywords: &["fea", "plot", "field", "mesh", "graphics"], related: &["fea.field", "fea.results", "fea.run"],
    sections: &[BuiltinDocumentationSection { heading: "Forms and effects", paragraphs: &[
        "Forms include plot(run, fieldId), plot(results, fieldId), plot(field), and plot(study, runId, fieldId), with optional Field/FieldId, Mesh, Overlay, and Deformed controls. Geometry context and a persisted run id are required.",
        "The result is a conventional double-valued figure handle, not a `fea.Field` object. It changes graphics state and reads run artifacts; it does not launch a numeric provider or re-import CAD.",
        "A requested device-backed field rejects with a targeted input error until explicitly materialized on the host; it is never silently gathered or omitted. Numeric arrays are not accepted as context or field ids.",
        "When `plot-core` is not built into the runtime, the identity is present but returns its typed operation error before plotting.",
    ] }], examples: EXAMPLES,
    faqs: &[BuiltinDocumentationFaq { question: "Does plotting re-import CAD?", answer: "No. It uses study geometry context carried by the run or result object." }],
    links: &[BuiltinDocumentationLink { label: "Using FEA", target: BuiltinDocumentationLinkTarget::Documentation("/docs/fea/using-fea") }],
    implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/plot.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Contextual field plots", location: "crates/runmat-runtime/src/builtins/fea/tests/results_and_plotting.rs::fea_plot_returns_figure_handle_for_contextual_run_results_and_fields" }],
    notes: &["The runtime must include `plot-core` and a graphics host. A device-backed field is not implicitly downloaded for plotting."],
});
pub const FEA_PLOT_CATALOG_ENTRY: BuiltinCatalogEntry =
    operation_entry::entry(OperationEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.plot",
        documentation: DOCUMENTATION,
        descriptor: &FEA_PLOT_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &[],
        integer_audit: Some(&FEA_STRUCTURAL_INTEGER_AUDIT),
        effects: OperationEffectProfile::Figure,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_PLOT_CATALOG_ENTRY];
