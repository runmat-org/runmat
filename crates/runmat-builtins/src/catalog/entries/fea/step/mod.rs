use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    constructor_inference::{infer_constructor, FeaConstructorKind},
    descriptors::FEA_STEP_DESCRIPTOR,
    integer::FEA_STRUCTURAL_INTEGER_AUDIT,
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_constructor(request, entry, FeaConstructorKind::Step)
}
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "transient-step",
    title: "Create a transient analysis step",
    program: "step = fea.step('transient_1', 'transient');",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(step), 'fea.Step'));",
    },
}];
const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.step", summary: "Create a typed FEA analysis step.",
    description: "`fea.step(id, kind)` creates a `fea.Step` from a text identifier and analysis kind.",
    keywords: &["fea", "step", "analysis", "solver"], related: &["fea.model", "fea.runOptions"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Exactly two host text scalar inputs are required; supported kinds are the analysis step kinds exposed by the selected model family.",
        "Direct integer and resident numeric inputs have no role and reject before object construction.",
        "This host object constructor invokes no solver, provider, or native OCCT geometry path.",
    ] }], examples: EXAMPLES,
    faqs: &[BuiltinDocumentationFaq { question: "Where are step counts configured?", answer: "Use solver-specific `fea.runOptions` count fields; `fea.step` only identifies a typed analysis step." }],
    links: &[BuiltinDocumentationLink { label: "FEA operation reference", target: BuiltinDocumentationLinkTarget::Documentation("/docs/fea/operation-reference") }],
    implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/load_step.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Step constructor", location: "crates/runmat-runtime/src/builtins/fea/tests/construction.rs" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "FEA VM execution", location: "crates/runmat-vm/tests/integer_builtin_fea_execution_and_feedback_file_status.rs" }],
    notes: &["A distributed array cannot be a step id or kind; no distributed materialization is performed."],
});
pub const FEA_STEP_CATALOG_ENTRY: BuiltinCatalogEntry =
    catalog_entry::entry(ConstructorEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.step",
        documentation: DOCUMENTATION,
        descriptor: &FEA_STEP_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &[],
        integer_audit: Some(&FEA_STRUCTURAL_INTEGER_AUDIT),
        filesystem_read: false,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_STEP_CATALOG_ENTRY];
