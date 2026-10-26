use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "All 15 arguments are required and positional; there are no defaults. `id` is non-empty text and the other arguments must be the typed values of their classes. The quantity fixes the canonical value shape (scalar, three-component vector, symmetric tensor, or tensor), which the request records as `value_shape`.",
        "The `reduction` argument must reduce something: model validation rejects `fea.ReductionPolicy.none()` and a policy whose spatial and series parts are both `none`. Series reductions require `fea.Observation.history` instead.",
        "Integrals, root-mean-square values, and L2 norms use the admitted element or face quadrature. Resultants integrate vector fields directly and convert boundary stress to Cauchy traction; minimum and maximum results keep the location where the value occurs.",
        "The constructor only decodes and packages the request. Model validation then checks the whole combination against the step: the quantity must be produced by the step's physics family, the target must resolve in the model, the recovery must suit the source-to-output location transfer, the component must suit the value shape, the sign convention must be meaningful for the quantity, a complex representation needs a complex procedure, and sampling must match the step's independent domain.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Collect the request with the step's other observations in a cell array and pass it as the `observations` argument (6th) of `fea.ProcedureStep.configure`. After a run, `fea.results` and `fea.field` read the published result by the observation id.",
        "Reductions are cheap to store and transfer, which makes them good acceptance metrics and `fea.Observation.monitor` sources when the result is scalar.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when the argument count is not 15, `id` is empty or not text, or any argument is not the expected typed value.",
        "`RunMat:fea:Internal` if the request cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "peak-stress",
        title: "Peak von Mises stress",
        program: concat!(
            "peak = fea.ReductionPolicy.new(fea.SpatialReduction.maximum(), ...\n",
            "    fea.SeriesReduction.none(), fea.ExtremaTiePolicy.firstCanonical());\n",
            "storage = fea.ObservationStorage.new( ...\n",
            "    fea.SignalPrecision.real64(), fea.CompressionPolicy.lossless(), ...\n",
            "    fea.RetentionPolicy.canonical(), fea.PartialAccessPolicy.manifestAndChunks());\n",
            "cost = fea.ObservationCost.new(25000, 400000, 24000, 24000, 0);\n",
            "peakStress = fea.Observation.reduction(\"peak_von_mises\", ...\n",
            "    fea.ObservationQuantity.vonMisesStress(), fea.ObservationTarget.wholeModel(), ...\n",
            "    fea.FieldLocation.element(), fea.FieldLocation.element(), fea.RecoveryPolicy.native(), ...\n",
            "    fea.ObservationComponent.all(), fea.CoordinateFrame.globalFrame(), ...\n",
            "    fea.SignConvention.native(), fea.ComplexRepresentation.real(), ...\n",
            "    fea.SamplingPolicy.finalSample(), peak, ...\n",
            "    storage, fea.ObservationPurpose.publication(), cost);",
        ),
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(peakStress), 'fea.Observation'));\nassert(strcmp(peakStress.value.reduction.reduction.spatial, 'maximum'));\nassert(strcmp(peakStress.value.reduction.quantity, 'von_mises_stress'));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.Observation.reduction",
        summary: "Request a governed spatial or series reduction of a quantity.",
        description: "`fea.Observation.reduction(id, quantity, target, sourceLocation, outputLocation, recovery, component, frame, sign, complexRepresentation, sampling, reduction, storage, purpose, cost)` builds a `fea.Observation` that reports a reduced value, such as a peak, integral, or resultant, instead of the full field.",
        keywords: &["fea", "observation", "reduction", "peak", "integral", "resultant"],
        related: &["fea.Observation.field", "fea.Observation.history", "fea.Observation.probe", "fea.ObservationTarget.wholeModel", "fea.SamplingPolicy.finalSample", "fea.ObservationStorage.new", "fea.ObservationCost.new", "fea.ProcedureStep.configure"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Where is the peak located?", answer: "Minimum and maximum reductions retain the canonical node, element, or interpolated point of the extreme value." }, BuiltinDocumentationFaq { question: "Is a parallel reduction deterministic?", answer: "Yes. Canonical compensated reduction fixes the fan-in order for the admitted partition plan." }],
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
