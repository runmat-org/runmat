//! Canonical constructor documentation and the existing default binding.
use super::contract;
use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.CoordinateFrame.cartesian", summary: "Create a typed coordinate frame.", description: "Creates an immutable named cartesian coordinate frame from a typed position and two typed directions.",
    keywords: &["fea", "coordinate", "frame", "typed"], related: &["fea.Direction.vector"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["The origin must be a finite fea.Position. xAxis and yAxis must be fea.Direction values.", "The identifier must not be empty; the two directions must be nonparallel. The constructor retains the supplied direction objects rather than changing them into an orthogonal basis."] }],
    examples: &[BuiltinExample { id: "author-cartesian", title: "Author a cartesian frame", program: "origin = fea.Position.meters([0.01, 0, 0]);\nxAxis = fea.Direction.vector([1, 0, 0]);\nyAxis = fea.Direction.vector([0, 1, 0]);\nframe = fea.CoordinateFrame.cartesian('local', origin, xAxis, yAxis);", display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(frame), 'fea.CartesianCoordinateFrame')); assert(strcmp(frame.frame_id, 'local')); assert(isequal(frame.origin.meters, [0.01, 0, 0])); assert(isequal(frame.x_axis.components, [1, 0, 0])); assert(isequal(frame.y_axis.components, [0, 1, 0]));" },
    }],
    faqs: &[], links: &[BuiltinDocumentationLink { label: "fea.Position.meters class-method implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/values.rs") }],
    implementation: &[BuiltinDocumentationLink { label: "crates/runmat-runtime/src/builtins/fea/constraints/support/frame.rs", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/constraints/support/frame.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Coordinate-frame validation and portable round-trip tests", location: "builtins::fea::constraints::tests::coordinate_frames_validate_and_round_trip_without_string_dispatch" }], notes: &[],
});

pub const FEA_COORDINATE_FRAME_CARTESIAN_CATALOG_ENTRY: BuiltinCatalogEntry =
    super::super::entry::entry(
        BuiltinCatalogProvenance::new(file!(), module_path!()),
        "fea.CoordinateFrame.cartesian",
        DOCUMENTATION,
        &contract::CARTESIAN_FRAME_DESCRIPTOR,
    );
