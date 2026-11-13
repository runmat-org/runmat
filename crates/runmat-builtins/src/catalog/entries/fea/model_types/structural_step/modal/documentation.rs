use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "A modal step needs mass (material density) and stiffness. Loads are not used; supports are. Frequencies are reported in hertz, f = omega / (2 pi). An unsupported part has six rigid-body modes at zero frequency.",
            "The step is fixed to family `structural`, procedure `modal_extraction`, and the direct solution space. Its default independent domain is an eigenvalue domain requesting 6 modes; change it, and set activations, state transfer, observations, accuracy, and restart policy, with `fea.ProcedureStep.configure`.",
            "The step declares the state outputs `modal_basis` (mode shapes and frequencies), which later steps can consume. Every step starts with no activations, no state inputs, no observations, `restart` `'fresh'`, all excitation cases selected, and a default accuracy plan whose mandatory criterion is a relative algebraic residual of 1e-8 on global equilibrium, with resource ceilings of 8 GiB resident memory and one hour of wall time.",
            "`id` must be nonempty text (a character vector or string scalar). It names the step in the model, in `fea.ExcitationCase.duringSteps`, in initial conditions (`beforeStepId`), and in step activations (`fea.ActivationLifecycle.continueFrom`).",
            "All inputs are checked before anything is built: a wrong argument count, a request for more than one output, a value of the wrong class, an empty identifier, or an out-of-range value raises `RunMat:fea:InvalidInput` and no object is returned.",
            "The result is an immutable host value object, not a handle: assigning it to another variable copies the value, and its properties are read-only. It encodes portably, so it can cross an execution boundary unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Pass the step to `fea.model` in the `Steps` option, for example `model = fea.model('part', block, 'Steps', {step}, ...)`. A model requires at least one step; for several steps with explicit order and state transfer, also pass an `fea.AnalysisSequence` in the `Sequence` option.",
            "Its modal basis can feed a later response-spectrum or modal-superposition step.",
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
    id: "modal",
    title: "Create a structural modal (natural-frequency) extraction step",
    program: "step = fea.StructuralStep.modal('modal_1');",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(step), 'fea.StructuralStep'));\nassert(strcmp(step.step_id, 'modal_1'));\nassert(strcmp(step.procedure, 'modal_extraction'));\nassert(step.independent_domain.eigenvalue.requested_count == 6);",
    },
}];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "How do I request more than six modes?",
        answer: "Change the step's independent domain with `fea.ProcedureStep.configure` and an eigenvalue domain such as `fea.IndependentDomain.eigenvalue`.",
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "Studies, analyses, and state",
        target: BuiltinDocumentationLinkTarget::Documentation(
            "/docs/simulation/language/studies-analyses-and-state",
        ),
    },
    BuiltinDocumentationLink {
        label: "FEA models",
        target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/models"),
    },
    BuiltinDocumentationLink {
        label: "Structural physics",
        target: BuiltinDocumentationLinkTarget::Documentation(
            "/docs/simulation/physics/structural",
        ),
    },
    BuiltinDocumentationLink {
        label: "Modal analysis guide",
        target: BuiltinDocumentationLinkTarget::Documentation(
            "/docs/simulation/guides/modal-analysis",
        ),
    },
];

const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/steps.rs"),
}];

const VERIFICATION: &[BuiltinEvidenceReference] = &[
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Concrete step constructors cover every family and procedure",
        location: "crates/runmat-runtime/src/builtins/fea/steps.rs::tests::concrete_step_constructors_cover_every_current_family_and_procedure",
    },
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Model composition accepts concrete steps and rejects mislabeled ones",
        location: "crates/runmat-runtime/src/builtins/fea/steps.rs::tests::model_composition_accepts_concrete_steps_and_rejects_mislabeled_steps",
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    documentation(ConstructorDocumentationDefinition {
        title: "fea.StructuralStep.modal",
        summary: "Create a structural modal (natural-frequency) extraction step.",
        description: "`fea.StructuralStep.modal(id)` returns a `fea.StructuralStep` that solves the eigenproblem K phi = omega^2 M phi for natural frequencies and mode shapes.",
        keywords: &[
            "fea",
            "procedure",
            "step",
            "analysis",
            "modal",
            "natural frequency",
            "eigenvalue",
            "structural",
        ],
        related: &[
            "fea.StructuralStep.harmonic",
            "fea.ProcedureStep.configure",
            "fea.model",
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
