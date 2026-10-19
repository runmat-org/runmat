use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "The components are divided by their Euclidean length, so `[0 0 2]` and `[0 0 1]` give the same direction. The result stores a 1-by-3 `components` row of unit length in the global frame. A vector whose length is zero (to within machine epsilon) is rejected.",
            "`components` must be a real numeric 1-by-3 or 3-by-1 vector. Any built-in integer class is accepted; the values are validated exactly and then converted once to binary64 before normalization.",
            "All inputs are checked before anything is built: a wrong argument count, a value of the wrong class, an empty identifier, or an out-of-range value raises `RunMat:fea:InvalidInput` and no object is returned.",
            "The result is an immutable host value object, not a handle: assigning it to another variable copies the value, and its properties are read-only. It encodes portably, so it can cross an execution boundary unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "A direction is a building block: `fea.PeriodicTransform.rotation` uses it as the rotation axis, and `fea.CoordinateFrame.cartesian`, `.cylindrical`, and `.spherical` use two directions to orient the frame.",
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
    id: "vector",
    title: "Create a unit direction vector from three components",
    program: "direction = fea.Direction.vector([0 3 4]);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(direction), 'fea.Direction'));\nassert(max(abs(direction.components - [0 0.6 0.8])) < 1e-15);",
    },
}];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Do I have to pass a unit vector?",
        answer: "No. Any nonzero vector is accepted and normalized, so `[0 0 5]` is stored as `[0 0 1]`.",
    },
    BuiltinDocumentationFaq {
        question: "Can I pass integers?",
        answer: "Yes. All eight integer classes are accepted. `fea.Direction.vector(int32([0 0 5]))` gives the same direction as `[0 0 1]`.",
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Constraints overview",
    target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/constraints"),
}];

const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/constraints/support/direction.rs"),
}];

const VERIFICATION: &[BuiltinEvidenceReference] = &[
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Coordinate frames validate and round-trip",
        location: "crates/runmat-runtime/src/builtins/fea/constraints.rs::tests::coordinate_frames_validate_and_round_trip_without_string_dispatch",
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    documentation(ConstructorDocumentationDefinition {
        title: "fea.Direction.vector",
        summary: "Create a unit direction vector from three components.",
        description: "`fea.Direction.vector(components)` normalizes a three-component vector and returns it as a typed `fea.Direction`.",
        keywords: &[
            "fea",
            "direction",
            "axis",
            "unit vector",
        ],
        related: &[
            "fea.PeriodicTransform.rotation",
            "fea.CoordinateFrame.cartesian",
            "fea.CoordinateFrame.cylindrical",
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
