use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::model_types::fixtures::BLOCK_GEOMETRY;
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "The force is a resultant in newtons. RunMat distributes it over the region so that the sum of nodal forces equals the signal value; the distribution follows the face area, not the node density. The moment of the distributed force is not separately enforced.",
            "The signal value is a resultant: it is projected onto the region so the integrated total is preserved (`distribution` `preserve_resultant`).",
            "`id` names the excitation. It must be nonempty text (a character vector or string scalar) and should be unique among objects of the same kind in one model; validation messages and run evidence report it. Step activations refer to it with `fea.StepObjectRef.excitation(id)`.",
            "`region` must be a `geometry.Region` returned by `geometry.region(asset, selector)`. The object stores the region id together with the region's `geometry_id` and `geometry_revision` (exposed as read-only properties, the revision as `uint32`), so `fea.model` rejects it when the model is built on a different asset or a different revision of the same asset.",
            "The region is recorded as a boundary-face target (`entity_dimension` `face`, cardinality one or more faces). Whether the selected region really is a set of faces is checked when the model is meshed and validated, not by the constructor.",
            "`signal` must be a typed signal of the required physical quantity; a signal of another quantity is rejected. The excitation stores only the signal id; the frame of the action is taken from the signal (global for constant signals). The action starts with no step restriction and no combination group: step activity and grouping are set by the excitation case. Build the signal with `fea.ForceSignal.constantNewtons` or the matching `fromTimeSamples`, `fromTimeKnots`, or `fromFunction` constructor.",
            "All inputs are checked before anything is built: a wrong argument count, a request for more than one output, a value of the wrong class, an empty identifier, or an out-of-range value raises `RunMat:fea:InvalidInput` and no object is returned.",
            "The result is an immutable host value object, not a handle: assigning it to another variable copies the value, and its properties are read-only. It encodes portably, so it can cross an execution boundary unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Pass the action to `fea.model` in the `Excitations` option and its signal in the `Signals` option, then group actions into load cases with `fea.ExcitationCase.always`, `.duringSteps`, or `.mutuallyExclusive` and pass the cases in the `ExcitationCases` option. The signal, the action, and the case are separate objects so one signal can drive several actions.",
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
    id: "force",
    title: "Apply a resultant force signal to a face region",
    program: "block = geometry.load('block.obj');\ntip = geometry.region(block, 'x_max');\npull = fea.ForceSignal.constantNewtons('pull', [1200 0 0]);\naction = fea.StructuralExcitation.force('tip_load', tip, pull);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: BLOCK_GEOMETRY,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(action), 'fea.StructuralExcitation'));\na = action.action.force;\nassert(strcmp(a.distribution, 'preserve_resultant'));\nassert(strcmp(a.target.region.region_id, 'region_x_max'));\nassert(isempty(a.active_step_ids));\nassert(strcmp(a.signal_id, 'pull'));",
    },
}];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Force or traction?",
        answer: "Use force when you know the total load and traction when you know the local stress (force per area). Force keeps the total fixed regardless of face area.",
    },
    BuiltinDocumentationFaq {
        question: "Can I reuse the object with another geometry?",
        answer: "No. The object is bound to the `geometry_id` and `geometry_revision` of its region. Select the region again from the new asset and rebuild the object; `fea.model` rejects a mismatched binding.",
    },
    BuiltinDocumentationFaq {
        question: "Why does the action not contain the value?",
        answer: "The value lives in the signal so that one time or frequency history can drive several actions and so its sampling policy is recorded once. The model must receive the signal in its `Signals` option.",
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "Loads and excitations",
        target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/loads"),
    },
    BuiltinDocumentationLink {
        label: "Structural loads",
        target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/loads/structural"),
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
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    documentation(ConstructorDocumentationDefinition {
        title: "fea.StructuralExcitation.force",
        summary: "Apply a resultant force signal to a face region.",
        description: "`fea.StructuralExcitation.force(id, region, signal)` builds a `fea.StructuralExcitation` that applies a total force vector to the selected faces.",
        keywords: &[
            "fea",
            "excitation",
            "load",
            "signal",
            "force",
            "structural",
        ],
        related: &[
            "fea.StructuralExcitation.traction",
            "fea.StructuralExcitation.moment",
            "fea.ForceSignal.constantNewtons",
            "fea.ExcitationCase.always",
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
