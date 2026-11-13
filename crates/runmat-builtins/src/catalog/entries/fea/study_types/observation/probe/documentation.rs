use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "All 15 arguments are required and positional; there are no defaults. `id` is non-empty text and the other arguments must be the typed values of their classes. The quantity fixes the canonical value shape (scalar, three-component vector, symmetric tensor, or tensor), which the request records as `value_shape`.",
        "Point and path probes interpolate inside the owning element: they require a nodal source location, `fea.FieldLocation.point()` output, and `fea.RecoveryPolicy.shapeFunctionInterpolation()`. A probe may carry a spatial reduction but not a series reduction.",
        "The constructor only decodes and packages the request. Model validation then checks the whole combination against the step: the quantity must be produced by the step's physics family, the target must resolve in the model, the recovery must suit the source-to-output location transfer, the component must suit the value shape, the sign convention must be meaningful for the quantity, a complex representation needs a complex procedure, and sampling must match the step's independent domain.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Collect the request with the step's other observations in a cell array and pass it as the `observations` argument (6th) of `fea.ProcedureStep.configure`. After a run, `fea.results` and `fea.field` read the published result by the observation id.",
        "Use probes for sensor-style values that are compared with test data.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when the argument count is not 15, `id` is empty or not text, or any argument is not the expected typed value.",
        "`RunMat:fea:Internal` if the request cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "sensor",
        title: "Temperature at a thermocouple location",
        program: concat!(
            "sensor = fea.ObservationTarget.point(fea.Position.meters([0.05 0.01 0]));\n",
            "storage = fea.ObservationStorage.new( ...\n",
            "    fea.SignalPrecision.real64(), fea.CompressionPolicy.lossless(), ...\n",
            "    fea.RetentionPolicy.canonical(), fea.PartialAccessPolicy.manifestAndChunks());\n",
            "cost = fea.ObservationCost.new(25000, 400000, 24000, 24000, 0);\n",
            "probe = fea.Observation.probe(\"sensor_temperature\", ...\n",
            "    fea.ObservationQuantity.temperature(), sensor, ...\n",
            "    fea.FieldLocation.node(), fea.FieldLocation.point(), fea.RecoveryPolicy.shapeFunctionInterpolation(), ...\n",
            "    fea.ObservationComponent.all(), fea.CoordinateFrame.globalFrame(), ...\n",
            "    fea.SignConvention.native(), fea.ComplexRepresentation.real(), ...\n",
            "    fea.SamplingPolicy.finalSample(), fea.ReductionPolicy.none(), ...\n",
            "    storage, fea.ObservationPurpose.publication(), cost);",
        ),
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(probe), 'fea.Observation'));\nassert(strcmp(probe.value.probe.observation_id, 'sensor_temperature'));\nassert(strcmp(probe.value.probe.value_shape, 'scalar'));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.Observation.probe",
        summary: "Request a value at a point, path, or named target.",
        description: "`fea.Observation.probe(id, quantity, target, sourceLocation, outputLocation, recovery, component, frame, sign, complexRepresentation, sampling, reduction, storage, purpose, cost)` builds a `fea.Observation` that samples a quantity at a point, path, connector, constraint, port, interface, or spatial target.",
        keywords: &["fea", "observation", "probe", "sensor", "point", "request"],
        related: &["fea.Observation.field", "fea.Observation.history", "fea.Observation.reduction", "fea.ObservationTarget.wholeModel", "fea.SamplingPolicy.finalSample", "fea.ObservationStorage.new", "fea.ObservationCost.new", "fea.ProcedureStep.configure"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Is the probe value taken from the nearest node?", answer: "No. It is interpolated with the owning element's shape functions at the exact position." }, BuiltinDocumentationFaq { question: "Can a probe record a history?", answer: "Use `fea.Observation.history` with a point target for values over the independent domain." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Outputs, fields, and observations", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/outputs-fields-and-observations") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/observations/requests.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/observation/tests.rs::observation_constructors_follow_their_positional_contract" },
        ],
        notes: &[],
    });
