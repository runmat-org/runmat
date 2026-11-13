//! Canonical constructor documentation and the existing default binding.
use super::contract;
use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.CoordinateFrame.boundaryNormalTangent", summary: "Create a typed coordinate frame.", description: "Selects the immutable boundary-normal/tangent coordinate-frame policy for authoring FEA models and constraints.",
    keywords: &["fea", "coordinate", "frame", "typed"], related: &["fea.Direction.vector"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["The value does not select a geometry region or compute boundary normals by itself.", "The constructor takes no inputs and returns an immutable typed frame value."] }],
    examples: &[BuiltinExample { id: "author-boundary-normal-tangent", title: "Use a boundary normal-tangent frame", program: "frame = fea.CoordinateFrame.boundaryNormalTangent();", display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(frame), 'fea.BoundaryNormalTangentFrame'));" },
    }],
    faqs: &[], links: &[BuiltinDocumentationLink { label: "fea.Position.meters class-method implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/values.rs") }],
    implementation: &[BuiltinDocumentationLink { label: "crates/runmat-runtime/src/builtins/fea/constraints/support/frame.rs", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/constraints/support/frame.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Coordinate-frame validation and portable round-trip tests", location: "builtins::fea::constraints::tests::coordinate_frames_validate_and_round_trip_without_string_dispatch" }], notes: &[],
});

pub const FEA_COORDINATE_FRAME_BOUNDARY_CATALOG_ENTRY: BuiltinCatalogEntry =
    super::super::entry::entry(
        BuiltinCatalogProvenance::new(file!(), module_path!()),
        "fea.CoordinateFrame.boundaryNormalTangent",
        DOCUMENTATION,
        &contract::BOUNDARY_FRAME_DESCRIPTOR,
    );
