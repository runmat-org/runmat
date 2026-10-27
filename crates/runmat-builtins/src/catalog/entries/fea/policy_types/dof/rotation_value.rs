//! Canonical constructor documentation and the existing default binding.
use super::contract;
use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.StructuralDofValue.rotation", summary: "Bind a structural rotation DOF to a rotation value.", description: "Creates an immutable pairing of a structural rotation selector and a dimensioned rotation value.",
    keywords: &["fea", "structural", "dof", "rotation", "typed"], related: &["fea.StructuralDof.rz"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["dof must be a fea.StructuralRotationDof and value must be a fea.Rotation. Translation DOFs and displacement quantities are rejected.", "The object retains dof and value; creating it does not apply a constraint or execute a solver."] }],
    examples: &[BuiltinExample { id: "bind-rotation-value", title: "Prescribe a constant Z rotation", program: "dof = fea.StructuralDof.rz();\nvalue = fea.Rotation.radians(pi / 6);\nentry = fea.StructuralDofValue.rotation(dof, value);", display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(entry), 'fea.StructuralRotationDofValue')); assert(strcmp(class(entry.dof), 'fea.StructuralRotationDof')); assert(entry.dof.axis == 2); assert(entry.value.radians == pi / 6);" },
    }],
    faqs: &[], links: &[BuiltinDocumentationLink { label: "fea.Rotation.radians class-method implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/values.rs") }],
    implementation: &[BuiltinDocumentationLink { label: "crates/runmat-runtime/src/builtins/fea/constraints/support/dof.rs", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/constraints/support/dof.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed structural DOF and reference coupling tests", location: "builtins::fea::constraints::tests::prescribed_structural_dofs_and_reference_couplings_are_fully_typed" }], notes: &[],
});

pub const FEA_STRUCTURAL_DOF_VALUE_ROTATION_CATALOG_ENTRY: BuiltinCatalogEntry =
    super::super::entry::entry(
        BuiltinCatalogProvenance::new(file!(), module_path!()),
        "fea.StructuralDofValue.rotation",
        DOCUMENTATION,
        &contract::DOF_VALUE_DESCRIPTOR,
    );
