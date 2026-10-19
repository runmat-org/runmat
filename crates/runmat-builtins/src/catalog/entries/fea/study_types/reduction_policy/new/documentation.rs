use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`spatial` is a `fea.SpatialReduction` (none, integral, average, rootMeanSquare, euclideanNorm, minimum, maximum, resultant, or moment about a `fea.Position`). `series` is a `fea.SeriesReduction` (none, minimum, maximum, absoluteMaximum, mean, rootMeanSquare, variance, or envelope). `extremaTies` is `fea.ExtremaTiePolicy.firstCanonical()` or `allLocations()`.",
        "The spatial reduction runs first, then the series reduction runs on the completed canonical history. Minimum, maximum, and envelope results retain the exact node, element, or interpolated point and sample where the value occurs; ties follow `extremaTies` independently of worker count.",
        "Mean, rootMeanSquare, and variance require an explicit `fea.SeriesWeighting`. Trapezoidal weighting needs a continuous domain (time, frequency, load factor, or study parameter).",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass it as the 12th argument of `fea.Observation.reduction` (which must reduce something) or `fea.Observation.history` (series reductions require a history). Model validation rejects a reduction on a field request, a series reduction on a field, probe, or reduction request, a series reduction over event sampling, and a series reduction over `latestSamples` retention.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when an argument is not the expected typed reduction or tie object.",
        "`RunMat:fea:Internal` if the value cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "peak-stress",
        title: "Peak value over a target, first exact tie",
        program: "policy = fea.ReductionPolicy.new(fea.SpatialReduction.maximum(), ...\n    fea.SeriesReduction.none(), fea.ExtremaTiePolicy.firstCanonical());",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(policy), 'fea.ReductionPolicy'));\nassert(strcmp(policy.value.spatial, 'maximum'));\nassert(strcmp(policy.value.extrema_ties, 'first_canonical'));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.ReductionPolicy.new",
        summary: "Combine a spatial reduction, a series reduction, and an extrema tie rule.",
        description: "`fea.ReductionPolicy.new(spatial, series, extremaTies)` builds a `fea.ReductionPolicy` that reduces an observation over its target and, optionally, over its history.",
        keywords: &["fea", "observation", "reduction", "maximum", "integral", "resultant"],
        related: &["fea.ReductionPolicy.none", "fea.Observation.reduction", "fea.Observation.history"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "How do I take a moment of a resultant?", answer: "Use `fea.SpatialReduction.moment(fea.Position.meters([x y z]))`; a moment always names its reference point." }, BuiltinDocumentationFaq { question: "Are ties resolved differently on a cluster?", answer: "No. `firstCanonical` keeps the first exact tie in canonical target order and `allLocations` keeps all of them, regardless of partitioning." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Outputs, fields, and observations", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/outputs-fields-and-observations") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/observations/storage.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/reduction_policy/tests.rs::reduction_policy_constructors_follow_their_positional_contract" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Bounded histories end to end", location: "crates/runmat-analysis/language/tests/output_histories.rs::bounded_sampling_interpolation_and_series_reductions_are_typed_end_to_end" },
        ],
        notes: &[],
    });
