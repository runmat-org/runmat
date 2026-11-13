use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "All 15 arguments are required and positional; there are no defaults. `id` is non-empty text and the other arguments must be the typed values of their classes. The quantity fixes the canonical value shape (scalar, three-component vector, symmetric tensor, or tensor), which the request records as `value_shape`.",
        "A field request keeps the field unreduced: its `reduction` argument must be `fea.ReductionPolicy.none()`, and model validation rejects any `fea.ReductionPolicy.new(...)` here.",
        "The constructor only decodes and packages the request. Model validation then checks the whole combination against the step: the quantity must be produced by the step's physics family, the target must resolve in the model, the recovery must suit the source-to-output location transfer, the component must suit the value shape, the sign convention must be meaningful for the quantity, a complex representation needs a complex procedure, and sampling must match the step's independent domain.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Collect the request with the step's other observations in a cell array and pass it as the `observations` argument (6th) of `fea.ProcedureStep.configure`. After a run, `fea.results` and `fea.field` read the published result by the observation id.",
        "Field requests are the source for plotting: after the run, `fea.field` selects the published field and `plot` renders it. Observation requests decide what is computed and retained; plotting never triggers an unplanned full-field transfer.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when the argument count is not 15, `id` is empty or not text, or any argument is not the expected typed value.",
        "`RunMat:fea:Internal` if the request cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "displacement",
        title: "Request the static displacement field",
        program: concat!(
            "storage = fea.ObservationStorage.new( ...\n",
            "    fea.SignalPrecision.real64(), fea.CompressionPolicy.lossless(), ...\n",
            "    fea.RetentionPolicy.canonical(), fea.PartialAccessPolicy.manifestAndChunks());\n",
            "cost = fea.ObservationCost.new(25000, 400000, 24000, 24000, 0);\n",
            "displacement = fea.Observation.field(\"displacement\", ...\n",
            "    fea.ObservationQuantity.displacement(), fea.ObservationTarget.wholeModel(), ...\n",
            "    fea.FieldLocation.node(), fea.FieldLocation.node(), fea.RecoveryPolicy.native(), ...\n",
            "    fea.ObservationComponent.all(), fea.CoordinateFrame.globalFrame(), ...\n",
            "    fea.SignConvention.native(), fea.ComplexRepresentation.real(), ...\n",
            "    fea.SamplingPolicy.finalSample(), fea.ReductionPolicy.none(), ...\n",
            "    storage, fea.ObservationPurpose.visualization(), cost);",
        ),
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(displacement), 'fea.Observation'));\nassert(strcmp(displacement.value.field.observation_id, 'displacement'));\nassert(strcmp(displacement.value.field.value_shape, 'vector3'));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.Observation.field",
        summary: "Request a spatial result field from a procedure step.",
        description: "`fea.Observation.field(id, quantity, target, sourceLocation, outputLocation, recovery, component, frame, sign, complexRepresentation, sampling, reduction, storage, purpose, cost)` builds a `fea.Observation` that retains a spatial field, such as nodal displacement or element stress, over its target.",
        keywords: &["fea", "observation", "field", "displacement", "stress", "request"],
        related: &["fea.Observation.history", "fea.Observation.probe", "fea.Observation.reduction", "fea.ObservationTarget.wholeModel", "fea.SamplingPolicy.finalSample", "fea.ObservationStorage.new", "fea.ObservationCost.new", "fea.ProcedureStep.configure"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Do I need a field request to plot results?", answer: "Yes. Only requested observations are computed and retained, so declare a field for each quantity you want to plot." }, BuiltinDocumentationFaq { question: "Why is `ReductionPolicy.new` rejected here?", answer: "A field is unreduced by definition. Use `fea.Observation.reduction` to reduce a quantity." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Outputs, fields, and observations", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/outputs-fields-and-observations") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/observations/requests.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/observation/tests.rs::observation_constructors_follow_their_positional_contract" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed sequence round trip", location: "crates/runmat-runtime/src/builtins/fea/sequences.rs::tests::configured_sequence_round_trips_with_typed_study_contracts" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Authored displacement observation", location: "crates/runmat-analysis/fea/src/tests.rs::linear_static_executes_authored_displacement_observation" },
        ],
        notes: &[],
    });
