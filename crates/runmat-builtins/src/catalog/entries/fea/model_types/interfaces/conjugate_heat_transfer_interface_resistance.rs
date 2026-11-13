//! Canonical documentation for fea.ConjugateHeatTransferInterface.resistance; original sidecar prose is preserved.
use super::contract;
use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.ConjugateHeatTransferInterface.resistance",
    summary: "Join fluid and solid regions through a finite thermal resistance.",
    description: "Defines conjugate heat transfer using one positive typed area-normalized thermal resistance.",
    keywords: &["fea", "interface", "fluid", "thermal", "resistance"],
    related: &[],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["Requires `fea.ThermalContactResistance`.", "The transfer law is stored without a redundant conductance.", "Use `perfect` rather than zero resistance."] }],
    examples: &[BuiltinExample {
        id: "author-conjugate-heat-transfer-interface-resistance",
        title: "Create resistance-controlled transfer",
        program: "geom = geometry.load('interface-patches.obj');\nfluidWall = geometry.region(geom, 'region_left');\nsolidWall = geometry.region(geom, 'region_right');\nr = fea.ThermalContactResistance.squareMeterKelvinPerWatt(0.002);\ncht = fea.ConjugateHeatTransferInterface.resistance(\"wall-heat\", fluidWall, solidWall, r);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: super::fixture::PATCHES,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(cht), 'fea.ConjugateHeatTransferInterface'));" },
    }],
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "FEA models", target: BuiltinDocumentationLinkTarget::Documentation("../../fea/models") }],
    implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/fea/interfaces.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/interfaces.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Geometry-bound interface constructor tests", location: "builtins::fea::interfaces::tests" }],
    notes: &[],
});
pub const CHT_RESISTANCE_CATALOG_ENTRY: BuiltinCatalogEntry = super::super::entry::entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "fea.ConjugateHeatTransferInterface.resistance",
    DOCUMENTATION,
    &contract::CHT_RESISTANCE_DESCRIPTOR,
    &[],
    Some(&contract::TYPED_INTERFACE_INTEGER_AUDIT),
);
