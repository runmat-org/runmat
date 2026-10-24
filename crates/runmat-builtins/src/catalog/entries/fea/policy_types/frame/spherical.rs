//! Canonical constructor documentation and the existing default binding.
use super::contract;
use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.CoordinateFrame.spherical", summary: "Create a typed coordinate frame.", description: "Creates an immutable named spherical coordinate frame from a typed position and two typed directions.",
    keywords: &["fea", "coordinate", "frame", "typed"], related: &["fea.Direction.vector"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["The origin must be a finite fea.Position. polarAxis and azimuthReference must be fea.Direction values.", "The identifier must not be empty; the two directions must be nonparallel. The value stores the spherical frame definition and does not convert a mesh or field."] }],
    examples: &[BuiltinExample { id: "author-spherical", title: "Author a spherical frame", program: "origin = fea.Position.meters([0.01, 0, 0]);\npolarAxis = fea.Direction.vector([0, 0, 1]);\nazimuthReference = fea.Direction.vector([1, 0, 0]);\nframe = fea.CoordinateFrame.spherical('local', origin, polarAxis, azimuthReference);", display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(frame), 'fea.SphericalCoordinateFrame')); assert(strcmp(frame.frame_id, 'local')); assert(isequal(frame.origin.meters, [0.01, 0, 0])); assert(isequal(frame.polar_axis.components, [0, 0, 1])); assert(isequal(frame.azimuth_reference.components, [1, 0, 0]));" },
    }],
    faqs: &[], links: &[BuiltinDocumentationLink { label: "fea.Position.meters class-method implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/values.rs") }],
    implementation: &[BuiltinDocumentationLink { label: "crates/runmat-runtime/src/builtins/fea/constraints/support/frame.rs", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/constraints/support/frame.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Coordinate-frame validation and portable round-trip tests", location: "builtins::fea::constraints::tests::coordinate_frames_validate_and_round_trip_without_string_dispatch" }], notes: &[],
});

pub const FEA_COORDINATE_FRAME_SPHERICAL_CATALOG_ENTRY: BuiltinCatalogEntry =
    super::super::entry::entry(
        BuiltinCatalogProvenance::new(file!(), module_path!()),
        "fea.CoordinateFrame.spherical",
        DOCUMENTATION,
        &contract::SPHERICAL_FRAME_DESCRIPTOR,
    );
