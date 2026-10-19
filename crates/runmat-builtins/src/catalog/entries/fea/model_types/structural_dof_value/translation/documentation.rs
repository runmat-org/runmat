use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`dof` must be `fea.StructuralDof.ux()`, `.uy()`, or `.uz()`; a rotational DOF is rejected. `value` must be a `fea.Displacement`, for example `fea.Displacement.millimeters(-0.5)`. Both are stored unchanged as the `dof` and `value` properties.",
            "All inputs are checked before anything is built: a wrong argument count, a request for more than one output, a value of the wrong class, an empty identifier, or an out-of-range value raises `RunMat:fea:InvalidInput` and no object is returned.",
            "The result is an immutable host value object, not a handle: assigning it to another variable copies the value, and its properties are read-only. It encodes portably, so it can cross an execution boundary unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Pass one entry, or a cell array of entries for different DOFs, to `fea.StructuralConstraint.prescribed`. The constraint rejects duplicate DOFs.",
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
    id: "translation",
    title: "Bind a displacement value to a translational structural DOF",
    program: "entry = fea.StructuralDofValue.translation(fea.StructuralDof.uz(), fea.Displacement.millimeters(-0.5));",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(entry), 'fea.StructuralTranslationDofValue'));\nassert(abs(entry.value.meters + 5e-4) < 1e-15);\nassert(isequal(entry.dof.axis, uint8(2)));",
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
        title: "fea.StructuralDofValue.translation",
        summary: "Bind a displacement value to a translational structural DOF.",
        description: "`fea.StructuralDofValue.translation(dof, value)` returns a `fea.StructuralTranslationDofValue` pairing one translation component with a prescribed displacement.",
        keywords: &[
            "fea",
            "structural",
            "dof",
            "displacement",
            "prescribed",
        ],
        related: &[
            "fea.StructuralDofValue.rotation",
            "fea.StructuralDof.uz",
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
