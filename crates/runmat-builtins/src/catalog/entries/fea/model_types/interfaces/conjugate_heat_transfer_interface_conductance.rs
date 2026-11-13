//! Canonical documentation for fea.ConjugateHeatTransferInterface.conductance; original sidecar prose is preserved.
use super::contract;
use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.ConjugateHeatTransferInterface.conductance",
    summary: "Join fluid and solid regions through a finite thermal conductance.",
    description: "Defines conjugate heat transfer using one positive typed heat-transfer coefficient.",
    keywords: &["fea", "interface", "fluid", "thermal", "conductance"],
    related: &[],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["Requires `fea.HeatTransferCoefficient`.", "The transfer law is stored without a redundant resistance.", "Both regions must come from one geometry revision."] }],
    examples: &[BuiltinExample {
        id: "author-conjugate-heat-transfer-interface-conductance",
        title: "Create conductance-controlled transfer",
        program: "geom = geometry.load('interface-patches.obj');\nfluidWall = geometry.region(geom, 'region_left');\nsolidWall = geometry.region(geom, 'region_right');\nh = fea.HeatTransferCoefficient.wattsPerSquareMeterKelvin(750);\ncht = fea.ConjugateHeatTransferInterface.conductance(\"wall-heat\", fluidWall, solidWall, h);",
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
pub const CHT_CONDUCTANCE_CATALOG_ENTRY: BuiltinCatalogEntry = super::super::entry::entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "fea.ConjugateHeatTransferInterface.conductance",
    DOCUMENTATION,
    &contract::CHT_CONDUCTANCE_DESCRIPTOR,
    &[],
    Some(&contract::TYPED_INTERFACE_INTEGER_AUDIT),
);
