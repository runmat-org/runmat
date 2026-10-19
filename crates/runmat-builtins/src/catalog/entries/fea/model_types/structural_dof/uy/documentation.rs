use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "The object stores `axis` as `uint8(1)` (0 for x, 1 for y, 2 for z). The axis is interpreted in the coordinate frame of the constraint that uses it, so in a cylindrical frame the first, second, and third axes are radial, tangential, and axial.",
            "All inputs are checked before anything is built: a wrong argument count, a value of the wrong class, an empty identifier, or an out-of-range value raises `RunMat:fea:InvalidInput` and no object is returned.",
            "The result is an immutable host value object, not a handle: assigning it to another variable copies the value, and its properties are read-only. It encodes portably, so it can cross an execution boundary unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Bind a value to it with `fea.StructuralDofValue.translation` and pass the result to `fea.StructuralConstraint.prescribed`. It is also the DOF argument of `fea.StructuralMotion.acceleration`.",
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
    id: "uy",
    title: "Select the y-translation structural degree of freedom",
    program: "dof = fea.StructuralDof.uy();\nentry = fea.StructuralDofValue.translation(dof, fea.Displacement.meters(0));",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(dof), 'fea.StructuralTranslationDof'));\nassert(isequal(dof.axis, uint8(1)));\nassert(isa(dof.axis, 'uint8'));",
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
        title: "fea.StructuralDof.uy",
        summary: "Select the y-translation structural degree of freedom.",
        description: "`fea.StructuralDof.uy()` returns a `fea.StructuralTranslationDof` naming the displacement component along the y axis.",
        keywords: &[
            "fea",
            "structural",
            "dof",
            "degree of freedom",
            "translation",
        ],
        related: &[
            "fea.StructuralDofValue.translation",
            "fea.StructuralConstraint.prescribed",
            "fea.StructuralDof.ux",
            "fea.StructuralDof.uz",
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
