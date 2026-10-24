//! Canonical documentation for fea.ConjugateHeatTransferInterface.perfect; original sidecar prose is preserved.
use super::contract;
use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.ConjugateHeatTransferInterface.perfect",
    summary: "Join fluid and solid regions with perfect thermal contact.",
    description: "Defines temperature-continuous conjugate heat transfer without an added thermal resistance.",
    keywords: &["fea", "interface", "fluid", "thermal", "cht"],
    related: &[],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["The fluid side is primary and solid side secondary.", "No redundant conductance or resistance is authored.", "Numerical coupling policy is selected by the solve step."] }],
    examples: &[BuiltinExample {
        id: "author-conjugate-heat-transfer-interface-perfect",
        title: "Create perfect thermal contact",
        program: "geom = geometry.load('interface-patches.obj');\nfluidWall = geometry.region(geom, 'region_left');\nsolidWall = geometry.region(geom, 'region_right');\ncht = fea.ConjugateHeatTransferInterface.perfect(\"wall-heat\", fluidWall, solidWall);",
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
pub const CHT_PERFECT_CATALOG_ENTRY: BuiltinCatalogEntry = super::super::entry::entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "fea.ConjugateHeatTransferInterface.perfect",
    DOCUMENTATION,
    &contract::CHT_PERFECT_DESCRIPTOR,
    &[],
    Some(&contract::TYPED_INTERFACE_INTEGER_AUDIT),
);
