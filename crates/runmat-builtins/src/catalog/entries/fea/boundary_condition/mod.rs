use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    inference::{infer_constructor, FeaConstructorKind},
    integer::{FEA_BOUNDARY_CONDITION_DESCRIPTOR, FEA_BOUNDARY_CONDITION_INTEGER_CAPABILITIES},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_constructor(request, entry, FeaConstructorKind::BoundaryCondition)
}
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "prescribed-rotation", title: "Prescribe integer-valued rotation", program: "bc = fea.boundaryCondition('rotation', 'tip', 'prescribedRotation', 'rx', int16(0), 'ry', int16(1), 'rz', int16(0));",
        display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(bc), 'fea.BoundaryCondition'));" } },
    BuiltinExample { id: "cfd-inlet", title: "Create a CFD inlet boundary", program: "bc = fea.boundaryCondition('inlet', 'inlet_face', 'cfdInletVelocity', 'velocityMPerS', uint8(12));",
        display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(bc), 'fea.BoundaryCondition'));" } },
];
const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.boundaryCondition", summary: "Create a typed RunMat FEA boundary condition.",
    description: "`fea.boundaryCondition(id, region, kind, Name, Value, ...)` creates a typed boundary-condition object for a RunMat FEA model.",
    keywords: &["fea", "boundary condition", "thermal", "acoustic", "cfd"], related: &["fea.model", "fea.loadCase"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["The first three inputs select the boundary-condition id, target region id, and boundary kind.", "Numeric fields must be finite real numeric scalars. Logical values, nonscalars, and unknown fields raise `RunMat:fea:InvalidInput`.", "All eight built-in integer classes cross exactly one IEEE-754 binary64 conversion at the host model-storage boundary, so wide values can round."] },
        BuiltinDocumentationSection { heading: "Integer fields", paragraphs: &["`prescribedRotation`: rx, ry, rz; `acousticImpedance`: specificImpedancePaSPerM; `thermalPrescribedTemperature`: temperatureK; `thermalHeatFlux`: heatFluxWPerM2; `thermalConvection`: ambientTemperatureK and coefficientWPerM2K; `cfdInletVelocity`: velocityMPerS; `cfdOutletPressure`: pressurePa."] },
    ], examples: EXAMPLES,
    faqs: &[BuiltinDocumentationFaq { question: "Are integer fields stored as integers?", answer: "No. They stay exact through argument parsing and cross one documented binary64 conversion into the model." }],
    links: &[BuiltinDocumentationLink { label: "Using FEA", target: BuiltinDocumentationLinkTarget::Documentation("/docs/fea/using-fea") }, BuiltinDocumentationLink { label: "FEA operation reference", target: BuiltinDocumentationLinkTarget::Documentation("/docs/fea/operation-reference") }],
    implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/boundary.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Boundary constructor", location: "crates/runmat-runtime/src/builtins/fea/tests/construction.rs::fea_boundary_condition" }],
    notes: &["Distributed numeric option handles are rejected before any materialization; the boundary kind and field schema must be validated first."],
});
pub const FEA_BOUNDARY_CONDITION_CATALOG_ENTRY: BuiltinCatalogEntry =
    catalog_entry::entry(ConstructorEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.boundaryCondition",
        documentation: DOCUMENTATION,
        descriptor: &FEA_BOUNDARY_CONDITION_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &FEA_BOUNDARY_CONDITION_INTEGER_CAPABILITIES,
        integer_audit: None,
        filesystem_read: false,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_BOUNDARY_CONDITION_CATALOG_ENTRY];
