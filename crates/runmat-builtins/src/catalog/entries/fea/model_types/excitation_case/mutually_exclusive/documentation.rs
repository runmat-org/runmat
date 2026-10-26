use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::model_types::fixtures::BLOCK_GEOMETRY;
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Cases that share a `groupId` are alternatives, never applied together: for example wind from the north or wind from the east. Load-combination logic evaluates each member of the group separately. `groupId` must be nonempty text.",
            "`id` names the case and must be nonempty text. `actions` is one excitation action or a cell array of them (from the `fea.*Excitation` constructors); at least one is required. The case records only the action ids (`action_ids`); the actions themselves, and their signals, must also be passed to the model.",
            "All inputs are checked before anything is built: a wrong argument count, a request for more than one output, a value of the wrong class, an empty identifier, or an out-of-range value raises `RunMat:fea:InvalidInput` and no object is returned.",
            "The result is an immutable host value object, not a handle: assigning it to another variable copies the value, and its properties are read-only. It encodes portably, so it can cross an execution boundary unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Pass the case to `fea.model` in the `ExcitationCases` option, together with the actions in `Excitations` and their signals in `Signals`.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Errors",
        paragraphs: &[
            "`RunMat:fea:InvalidInput`: an argument has the wrong class or shape, an identifier is empty, or a value violates the constructor's range rules.",
            "`RunMat:fea:Internal`: the validated object could not be converted to a RunMat value (not expected in normal use).",
            "Errors are attributed to the method that raised them, for example `fea.ExcitationCase.mutuallyExclusive`, in the error's builtin context.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "mutuallyExclusive",
    title: "Put excitation actions into a mutually exclusive combination group",
    program: "block = geometry.load('block.obj');\ntip = geometry.region(block, 'x_max');\npull = fea.StructuralExcitation.force('tip_load', tip, fea.ForceSignal.constantNewtons('pull', [1200 0 0]));\nload_case = fea.ExcitationCase.mutuallyExclusive('wind_north', {pull}, 'wind_direction');",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: BLOCK_GEOMETRY,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(load_case), 'fea.ExcitationCase'));\nassert(strcmp(load_case.activation.mutually_exclusive_group.group_id, 'wind_direction'));",
    },
}];

const FAQS: &[BuiltinDocumentationFaq] = &[];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "Loads and excitations",
        target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/loads"),
    },
    BuiltinDocumentationLink {
        label: "Inputs, sweeps, and cases",
        target: BuiltinDocumentationLinkTarget::Documentation(
            "/docs/simulation/language/inputs-sweeps-and-cases",
        ),
    },
];

const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/excitations.rs"),
}];

const VERIFICATION: &[BuiltinEvidenceReference] = &[
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Force action and case preserve separate typed roles",
        location: "crates/runmat-runtime/src/builtins/fea/excitations.rs::tests::force_action_and_case_preserve_separate_typed_roles",
    },
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Actions reject the wrong signal quantity",
        location: "crates/runmat-runtime/src/builtins/fea/excitations.rs::tests::action_rejects_wrong_signal_quantity_and_cases_own_membership",
    },
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Model composition accepts separate signal, action, and case objects",
        location: "crates/runmat-runtime/src/builtins/fea/excitations.rs::tests::model_composition_accepts_the_separate_signal_action_and_case_objects",
    },
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Excitation-case errors name the failing method",
        location: "crates/runmat-runtime/src/builtins/fea/constraints/coverage_tests.rs::excitation_case_errors_name_the_failing_method",
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    documentation(ConstructorDocumentationDefinition {
        title: "fea.ExcitationCase.mutuallyExclusive",
        summary: "Put excitation actions into a mutually exclusive combination group.",
        description: "`fea.ExcitationCase.mutuallyExclusive(id, actions, groupId)` builds a `fea.ExcitationCase` that belongs to a named group in which at most one case is active at a time.",
        keywords: &[
            "fea",
            "excitation case",
            "load case",
            "activation",
        ],
        related: &[
            "fea.ExcitationCase.always",
            "fea.ExcitationCase.duringSteps",
            "fea.StructuralExcitation.force",
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
