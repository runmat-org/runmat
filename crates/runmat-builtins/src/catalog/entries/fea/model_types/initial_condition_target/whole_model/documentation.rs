use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "The target stores `whole_model = true` and no geometry binding, so a condition built with it can be reused across geometries. To restrict an initial condition to part of the model, pass a `geometry.Region` as the target instead.",
            "All inputs are checked before anything is built: a wrong argument count, a request for more than one output, a value of the wrong class, an empty identifier, or an out-of-range value raises `RunMat:fea:InvalidInput` and no object is returned.",
            "The result is an immutable host value object, not a handle: assigning it to another variable copies the value, and its properties are read-only. It encodes portably, so it can cross an execution boundary unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Pass it as the `target` argument of any `fea.*InitialCondition` constructor, for example `fea.ThermalInitialCondition.temperature`.",
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
    id: "wholeModel",
    title: "Target an initial condition at the whole model",
    program: "target = fea.InitialConditionTarget.wholeModel();",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(target), 'fea.InitialConditionTarget'));\nassert(target.whole_model);",
    },
}];

const FAQS: &[BuiltinDocumentationFaq] = &[];

const LINKS: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Initial conditions",
    target: BuiltinDocumentationLinkTarget::Documentation(
        "/docs/simulation/constraints/initial-conditions",
    ),
}];

const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/initial_conditions.rs"),
}];

const VERIFICATION: &[BuiltinEvidenceReference] = &[
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Whole-model initial state is typed and portable",
        location: "crates/runmat-runtime/src/builtins/fea/initial_conditions.rs::tests::whole_model_initial_state_is_typed_and_portable",
    },
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Region initial state keeps its geometry binding and rejects the wrong quantity",
        location: "crates/runmat-runtime/src/builtins/fea/initial_conditions.rs::tests::region_initial_state_retains_geometry_and_rejects_wrong_quantity",
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    documentation(ConstructorDocumentationDefinition {
        title: "fea.InitialConditionTarget.wholeModel",
        summary: "Target an initial condition at the whole model.",
        description: "`fea.InitialConditionTarget.wholeModel()` returns a `fea.InitialConditionTarget` that applies an initial condition to every region of the model.",
        keywords: &[
            "fea",
            "initial condition",
            "target",
            "whole model",
        ],
        related: &[
            "fea.ThermalInitialCondition.temperature",
            "fea.StructuralInitialCondition.velocity",
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
