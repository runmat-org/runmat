use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "All 15 arguments are required and positional; there are no defaults. `id` is non-empty text and the other arguments must be the typed values of their classes. The quantity fixes the canonical value shape (scalar, three-component vector, symmetric tensor, or tensor), which the request records as `value_shape`.",
        "A history is the only request that may carry a series reduction (`mean`, `rootMeanSquare`, `variance`, `minimum`, `maximum`, `absoluteMaximum`, or `envelope`). Series reductions need every produced sample, so they are rejected with `latestSamples` retention and with event sampling.",
        "The constructor only decodes and packages the request. Model validation then checks the whole combination against the step: the quantity must be produced by the step's physics family, the target must resolve in the model, the recovery must suit the source-to-output location transfer, the component must suit the value shape, the sign convention must be meaningful for the quantity, a complex representation needs a complex procedure, and sampling must match the step's independent domain.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Collect the request with the step's other observations in a cell array and pass it as the `observations` argument (6th) of `fea.ProcedureStep.configure`. After a run, `fea.results` and `fea.field` read the published result by the observation id.",
        "Long histories publish immutable checkpoints that bind the sample count, a rolling digest of every sample, and the retained suffix, so restarted and uninterrupted runs produce the same canonical history.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when the argument count is not 15, `id` is empty or not text, or any argument is not the expected typed value.",
        "`RunMat:fea:Internal` if the request cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "tip-deflection",
        title: "Tip deflection at four times",
        program: concat!(
            "tip = fea.ObservationTarget.point(fea.Position.meters([0.10 0 0]));\n",
            "storage = fea.ObservationStorage.new( ...\n",
            "    fea.SignalPrecision.real64(), fea.CompressionPolicy.lossless(), ...\n",
            "    fea.RetentionPolicy.canonical(), fea.PartialAccessPolicy.manifestAndChunks());\n",
            "cost = fea.ObservationCost.new(25000, 400000, 24000, 24000, 0);\n",
            "tipHistory = fea.Observation.history(\"tip_deflection\", ...\n",
            "    fea.ObservationQuantity.displacement(), tip, ...\n",
            "    fea.FieldLocation.node(), fea.FieldLocation.point(), fea.RecoveryPolicy.shapeFunctionInterpolation(), ...\n",
            "    fea.ObservationComponent.y(), fea.CoordinateFrame.globalFrame(), ...\n",
            "    fea.SignConvention.native(), fea.ComplexRepresentation.real(), ...\n",
            "    fea.SamplingPolicy.timeSeconds([0 0.1 0.4 1.0]), fea.ReductionPolicy.none(), ...\n",
            "    storage, fea.ObservationPurpose.publication(), cost);",
        ),
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(tipHistory), 'fea.Observation'));\nassert(strcmp(tipHistory.value.history.observation_id, 'tip_deflection'));\nassert(isequal(tipHistory.value.history.sampling.time_seconds, [0 0.1 0.4 1]));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.Observation.history",
        summary: "Request a result history over a step's independent domain.",
        description: "`fea.Observation.history(id, quantity, target, sourceLocation, outputLocation, recovery, component, frame, sign, complexRepresentation, sampling, reduction, storage, purpose, cost)` builds a `fea.Observation` that retains values over time, frequency, load factor, modes, or a study parameter.",
        keywords: &["fea", "observation", "history", "time", "series", "request"],
        related: &["fea.Observation.field", "fea.Observation.probe", "fea.Observation.reduction", "fea.ObservationTarget.wholeModel", "fea.SamplingPolicy.finalSample", "fea.ObservationStorage.new", "fea.ObservationCost.new", "fea.ProcedureStep.configure"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "How do I get the time-averaged value?", answer: "Use `fea.ReductionPolicy.new(fea.SpatialReduction.none(), fea.SeriesReduction.mean(fea.SeriesWeighting.trapezoidalIndependentDomain()), ...)` as the reduction." }, BuiltinDocumentationFaq { question: "What if a sampled event never happens?", answer: "The history is valid with zero samples." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Outputs, fields, and observations", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/outputs-fields-and-observations") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/observations/requests.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/observation/tests.rs::observation_constructors_follow_their_positional_contract" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Bounded histories end to end", location: "crates/runmat-analysis/language/tests/output_histories.rs::bounded_sampling_interpolation_and_series_reductions_are_typed_end_to_end" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Sampling and series domains", location: "crates/runmat-analysis/language/tests/output_histories.rs::sampling_and_series_domains_must_match" },
        ],
        notes: &[],
    });
