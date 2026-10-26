//! Canonical constructor documentation and the existing default binding.
mod contract;
use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.ReferencePoint.fromCoordinates", summary: "Create a typed reference point from a position.", description: "Creates an immutable reference point with a nonempty identifier and a typed position for FEA coupling and constraint authoring.",
    keywords: &["fea", "reference", "point", "position", "coupling", "typed"], related: &[],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["The identifier is validated before the position is stored. A raw numeric vector is not a substitute for fea.Position.", "The object retains reference_point_id and position. It does not select geometry or create a coupling constraint."] }],
    examples: &[BuiltinExample { id: "author-reference-point", title: "Place a named reference point", program: "position = fea.Position.meters([0.1, 0.2, 0.3]);\npoint = fea.ReferencePoint.fromCoordinates('mount', position);", display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(point), 'fea.ReferencePoint')); assert(strcmp(point.reference_point_id, 'mount')); assert(isequal(point.position.meters, [0.1, 0.2, 0.3]));" },
    }],
    faqs: &[], links: &[BuiltinDocumentationLink { label: "fea.Position.meters class-method implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/values.rs") }],
    implementation: &[BuiltinDocumentationLink { label: "crates/runmat-runtime/src/builtins/fea/constraints/support/reference_point.rs", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/constraints/support/reference_point.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed structural DOF and reference coupling tests", location: "builtins::fea::constraints::tests::prescribed_structural_dofs_and_reference_couplings_are_fully_typed" }], notes: &[],
});

pub const FEA_REFERENCE_POINT_FROM_COORDINATES_CATALOG_ENTRY: BuiltinCatalogEntry =
    super::entry::entry(
        BuiltinCatalogProvenance::new(file!(), module_path!()),
        "fea.ReferencePoint.fromCoordinates",
        DOCUMENTATION,
        &contract::REFERENCE_POINT_DESCRIPTOR,
    );

#[cfg(test)]
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] =
    &[&FEA_REFERENCE_POINT_FROM_COORDINATES_CATALOG_ENTRY];
