use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::model_types::fixtures::BLOCK_GEOMETRY;
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Contact prevents interpenetration; while the surfaces stick, the tangential traction may not exceed mu times the contact pressure, and above that limit they slide. The coefficient is a single static/kinetic value.",
            "`id` names the interface. It must be nonempty text (a character vector or string scalar) and should be unique among objects of the same kind in one model; validation messages and run evidence report it. Step activations refer to it with `fea.StepObjectRef.interface(id)`.",
            "`primary` and `secondary` must be two distinct `geometry.Region` values from the same asset and revision; regions from different geometries, or one region given twice, are rejected. Both are recorded as boundary-face targets, the first as the interface's primary side.",
            "Each region must be a `geometry.Region` returned by `geometry.region(asset, selector)`. The object stores the region id together with the region's `geometry_id` and `geometry_revision` (exposed as read-only properties, the revision as `uint32`), so `fea.model` rejects it when the model is built on a different asset or a different revision of the same asset.",
            "`friction` must be a `fea.FrictionCoefficient` whose ratio is finite and strictly positive; a zero coefficient is rejected here (use `fea.ContactInterface.frictionless` instead). Typical dry steel on steel is 0.15 to 0.3.",
            "All inputs are checked before anything is built: a wrong argument count, a value of the wrong class, an empty identifier, or an out-of-range value raises `RunMat:fea:InvalidInput` and no object is returned.",
            "The result is an immutable host value object, not a handle: assigning it to another variable copies the value, and its properties are read-only. It encodes portably, so it can cross an execution boundary unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Pass the object to `fea.model` in the `Interfaces` option, either alone or in a cell array with others of the same family, for example `model = fea.model('part', block, 'Steps', {step}, 'Interfaces', {interface})`.",
            "Contact is state-dependent, so pair it with a structural procedure whose formulation admits contact, such as `fea.StructuralStep.nonlinear`. Do not also tie the same pair of regions; a tie suppresses the contact behavior.",
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
    id: "coulomb",
    title: "Create a Coulomb-friction contact interface between two face regions",
    program: "block = geometry.load('block.obj');\na = geometry.region(block, 'x_min');\nb = geometry.region(block, 'x_max');\ninterface = fea.ContactInterface.coulomb('flange', a, b, fea.FrictionCoefficient.ratio(0.3));",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: BLOCK_GEOMETRY,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(interface), 'fea.ContactInterface'));\nassert(interface.interface.kind.contact.friction.coulomb.coefficient.ratio == 0.3);",
    },
}];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Why is a friction coefficient of zero rejected?",
        answer: "A zero coefficient is frictionless contact, which has its own constructor, `fea.ContactInterface.frictionless`. The Coulomb form requires a positive coefficient.",
    },
    BuiltinDocumentationFaq {
        question: "Can I reuse the object with another geometry?",
        answer: "No. The object is bound to the `geometry_id` and `geometry_revision` of its region. Select the region again from the new asset and rebuild the object; `fea.model` rejects a mismatched binding.",
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Interfaces",
    target: BuiltinDocumentationLinkTarget::Documentation(
        "/docs/simulation/constraints/interfaces",
    ),
}];

const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/interfaces.rs"),
}];

const VERIFICATION: &[BuiltinEvidenceReference] = &[
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Concrete interfaces are geometry-bound, portable, and class-checked",
        location: "crates/runmat-runtime/src/builtins/fea/interfaces.rs::tests::concrete_interfaces_are_geometry_bound_portable_and_class_checked",
    },
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Interface construction rejects invalid identity and pairing",
        location: "crates/runmat-runtime/src/builtins/fea/interfaces.rs::tests::interface_construction_rejects_invalid_identity_and_pairing",
    },
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Model composition accepts matching interfaces and rejects foreign geometry",
        location: "crates/runmat-runtime/src/builtins/fea/interfaces.rs::tests::model_composition_accepts_matching_interfaces_and_rejects_foreign_geometry",
    },
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Interface parameter classes enforce semantic ranges",
        location: "crates/runmat-runtime/src/builtins/fea/interfaces.rs::tests::interface_parameter_classes_enforce_semantic_ranges",
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    documentation(ConstructorDocumentationDefinition {
        title: "fea.ContactInterface.coulomb",
        summary: "Create a Coulomb-friction contact interface between two face regions.",
        description: "`fea.ContactInterface.coulomb(id, primary, secondary, friction)` builds a `fea.ContactInterface` with Coulomb friction between two surfaces.",
        keywords: &[
            "fea",
            "interface",
            "contact",
            "friction",
            "Coulomb",
            "structural",
        ],
        related: &[
            "fea.ContactInterface.frictionless",
            "fea.FrictionCoefficient.ratio",
            "fea.model",
        ],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: FAQS,
        links: LINKS,
        implementation: IMPLEMENTATION,
        verification: VERIFICATION,
        notes: &[
            "Geometry examples load `block.obj`, a 0.2 m x 0.1 m x 0.1 m block whose six faces are named groups `x_min`, `x_max`, `y_min`, `y_max`, `z_min`, and `z_max`.",
        ],
    });
