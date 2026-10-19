use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`dof` must be `fea.StructuralDof.rx()`, `.ry()`, or `.rz()`; a translational DOF is rejected. `value` must be a `fea.Rotation`; `fea.Rotation.degrees(1)` stores about 0.01745 rad.",
            "All inputs are checked before anything is built: a wrong argument count, a value of the wrong class, an empty identifier, or an out-of-range value raises `RunMat:fea:InvalidInput` and no object is returned.",
            "The result is an immutable host value object, not a handle: assigning it to another variable copies the value, and its properties are read-only. It encodes portably, so it can cross an execution boundary unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Pass the entry to `fea.StructuralConstraint.prescribed`. Rotations constrain only formulations with rotational DOFs, such as beams and shells.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Errors",
        paragraphs: &[
            "`RunMat:fea:InvalidInput`: an argument has the wrong class or shape, an identifier is empty, or a value violates the constructor's range rules.",
            "`RunMat:fea:Internal`: the validated object could not be converted to a RunMat value (not expected in normal use).",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "rotation",
    title: "Bind a rotation value to a rotational structural DOF",
    program: "entry = fea.StructuralDofValue.rotation(fea.StructuralDof.rx(), fea.Rotation.degrees(0));",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(entry), 'fea.StructuralRotationDofValue'));\nassert(entry.value.radians == 0);\nassert(isequal(entry.dof.axis, uint8(0)));",
    },
}];

const FAQS: &[BuiltinDocumentationFaq] = &[];

const LINKS: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Structural constraints",
    target: BuiltinDocumentationLinkTarget::Documentation(
        "/docs/simulation/constraints/structural",
    ),
}];

const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/constraints/support/dof.rs"),
}];

const VERIFICATION: &[BuiltinEvidenceReference] = &[
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Prescribed DOFs and reference-point couplings are fully typed",
        location: "crates/runmat-runtime/src/builtins/fea/constraints.rs::tests::prescribed_structural_dofs_and_reference_couplings_are_fully_typed",
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    documentation(ConstructorDocumentationDefinition {
        title: "fea.StructuralDofValue.rotation",
        summary: "Bind a rotation value to a rotational structural DOF.",
        description: "`fea.StructuralDofValue.rotation(dof, value)` returns a `fea.StructuralRotationDofValue` pairing one rotation component with a prescribed rotation.",
        keywords: &[
            "fea",
            "structural",
            "dof",
            "rotation",
            "prescribed",
        ],
        related: &[
            "fea.StructuralDofValue.translation",
            "fea.StructuralDof.rx",
            "fea.StructuralConstraint.prescribed",
        ],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: FAQS,
        links: LINKS,
        implementation: IMPLEMENTATION,
        verification: VERIFICATION,
        notes: &[
        ],
    });
