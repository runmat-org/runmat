//! Canonical constructor documentation and the existing default binding.
use super::contract;
use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.StructuralDof.uy", summary: "Create a typed structural degree of freedom.", description: "Creates the immutable Y-translation degree-of-freedom selector for structural constraint authoring.",
    keywords: &["fea", "structural", "dof", "typed"], related: &["fea.StructuralDofValue.translation"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["The constructor takes no inputs. The typed selector has a zero-based axis property of 1.", "Bind this selector to a fea.Displacement with fea.StructuralDofValue.translation; a rotation value belongs to the other DOF family.", "Creating a selector does not constrain a model or prescribe a motion history."] }],
    examples: &[BuiltinExample { id: "select-uy", title: "Select Y translation", program: "dof = fea.StructuralDof.uy();", display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(dof), 'fea.StructuralTranslationDof')); assert(dof.axis == 1);" },
    }],
    faqs: &[], links: &[],
    implementation: &[BuiltinDocumentationLink { label: "crates/runmat-runtime/src/builtins/fea/constraints/support/dof.rs", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/constraints/support/dof.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed structural DOF and reference coupling tests", location: "builtins::fea::constraints::tests::prescribed_structural_dofs_and_reference_couplings_are_fully_typed" }], notes: &[],
});

pub const FEA_STRUCTURAL_DOF_UY_CATALOG_ENTRY: BuiltinCatalogEntry = super::super::entry::entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "fea.StructuralDof.uy",
    DOCUMENTATION,
    &contract::DOF_DESCRIPTOR,
);
