use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "A reference point is a single node with translational and rotational degrees of freedom, defined by a global position in meters. It need not lie on the geometry: a bolt axis or the center of a bearing are typical locations.",
            "`id` must be nonempty text (whitespace only is rejected). `position` must be a `fea.Position`, for example `fea.Position.meters([0.25 0.05 0.05])` or `fea.Position.millimeters([250 50 50])`.",
            "All inputs are checked before anything is built: a wrong argument count, a value of the wrong class, an empty identifier, or an out-of-range value raises `RunMat:fea:InvalidInput` and no object is returned.",
            "The result is an immutable host value object, not a handle: assigning it to another variable copies the value, and its properties are read-only. It encodes portably, so it can cross an execution boundary unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Pass the point to `fea.StructuralConstraint.rigidCoupling` or `fea.StructuralConstraint.distributingCoupling`; the coupling stores the point together with the coupled region.",
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
    id: "fromCoordinates",
    title: "Create a named reference point at a position",
    program: "point = fea.ReferencePoint.fromCoordinates('hub', fea.Position.millimeters([250 50 50]));",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(point), 'fea.ReferencePoint'));\nassert(strcmp(point.reference_point_id, 'hub'));\nassert(strcmp(class(point.position), 'fea.Position'));",
    },
}];

const FAQS: &[BuiltinDocumentationFaq] = &[];

const LINKS: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Connectors and couplings",
    target: BuiltinDocumentationLinkTarget::Documentation(
        "/docs/simulation/constraints/connectors",
    ),
}];

const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/constraints/support/reference_point.rs"),
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
        title: "fea.ReferencePoint.fromCoordinates",
        summary: "Create a named reference point at a position.",
        description: "`fea.ReferencePoint.fromCoordinates(id, position)` builds a `fea.ReferencePoint` that couplings use as their driving point.",
        keywords: &[
            "fea",
            "reference point",
            "coupling",
            "RBE",
        ],
        related: &[
            "fea.StructuralConstraint.rigidCoupling",
            "fea.StructuralConstraint.distributingCoupling",
            "fea.Position.meters",
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
