//! Canonical documentation for fea.FluidStructureInterface.noSlip; original sidecar prose is preserved.
use super::contract;
use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.FluidStructureInterface.noSlip",
    summary: "Join fluid and structural regions with no-slip kinematics.",
    description: "Defines physical no-slip transfer from a fluid-side region to a structural-side region.",
    keywords: &["fea", "interface", "fluid", "structure", "fsi"],
    related: &[],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["The fluid side is primary and structure side secondary.", "Both regions must come from one geometry revision.", "Coupling relaxation and enforcement are step or solver policy."] }],
    examples: &[BuiltinExample {
        id: "author-fluid-structure-interface-no-slip",
        title: "Create an FSI wall",
        program: "geom = geometry.load('interface-patches.obj');\nfluidWall = geometry.region(geom, 'region_left');\nstructureWall = geometry.region(geom, 'region_right');\nfsi = fea.FluidStructureInterface.noSlip(\"wall-transfer\", fluidWall, structureWall);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: super::fixture::PATCHES,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(fsi), 'fea.FluidStructureInterface'));" },
    }],
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "FEA models", target: BuiltinDocumentationLinkTarget::Documentation("../../fea/models") }],
    implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/fea/interfaces.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/interfaces.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Geometry-bound interface constructor tests", location: "builtins::fea::interfaces::tests" }],
    notes: &[],
});
pub const FLUID_STRUCTURE_INTERFACE_CATALOG_ENTRY: BuiltinCatalogEntry = super::super::entry::entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "fea.FluidStructureInterface.noSlip",
    DOCUMENTATION,
    &contract::FLUID_STRUCTURE_INTERFACE_DESCRIPTOR,
    &[],
    Some(&contract::TYPED_INTERFACE_INTEGER_AUDIT),
);
