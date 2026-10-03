use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    constructor_inference::{infer_constructor, FeaConstructorKind},
    descriptors::FEA_LOAD_CASE_DESCRIPTOR,
    integer::FEA_LOAD_CASE_INTEGER_CAPABILITIES,
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_constructor(request, entry, FeaConstructorKind::LoadCase)
}
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "integer-force",
    title: "Create an integer-valued force",
    program:
        "load = fea.loadCase('tip_force', 'name:Tip_Load', 'force', 'Vector', int32([0 -1000 0]));",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(load), 'fea.LoadCase'));",
    },
}];
const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.loadCase", summary: "Create a typed RunMat FEA load or source.",
    description: "`fea.loadCase(id, region, kind, Name, Value, ...)` creates a force, moment, pressure, body-force, current-density, coil-current, or heat-source load.",
    keywords: &["fea", "load", "force", "moment", "pressure"], related: &["fea.model", "fea.boundaryCondition"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["Load id, region selector, and kind remain exact text. `torque` is an alias of canonical `moment`.", "Required vectors have exactly three finite real elements; scalar fields must be finite real numeric scalars. Kind-specific fields and shape are validated before construction; resident numeric fields are rejected.", "All eight integer classes remain exact until a single host-only binary64 model-storage conversion, so wide values can round."] },
        BuiltinDocumentationSection { heading: "Integer fields", paragraphs: &["Force, moment/torque, body-force, and current-density vectors; pressure magnitude; current-density phase and amplitude; coil current phase and amplitude; volumetric heat-source magnitude."] },
    ], examples: EXAMPLES,
    faqs: &[BuiltinDocumentationFaq { question: "When is a moment rejected for a solid region?", answer: "The constructor validates load syntax first. Model validation or assembly later rejects direct moments on regions without rotational degrees of freedom." }],
    links: &[BuiltinDocumentationLink { label: "FEA models", target: BuiltinDocumentationLinkTarget::Documentation("/docs/fea/models") }],
    implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/load_step.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Load-case constructor", location: "crates/runmat-runtime/src/builtins/fea/tests/construction.rs::fea_load_case_accepts_moment_and_torque_alias" }],
    notes: &["Distributed numeric option handles are rejected before any materialization; load kind and vector shape must be validated first."],
});
pub const FEA_LOAD_CASE_CATALOG_ENTRY: BuiltinCatalogEntry =
    catalog_entry::entry(ConstructorEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.loadCase",
        documentation: DOCUMENTATION,
        descriptor: &FEA_LOAD_CASE_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &FEA_LOAD_CASE_INTEGER_CAPABILITIES,
        integer_audit: None,
        filesystem_read: false,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_LOAD_CASE_CATALOG_ENTRY];
