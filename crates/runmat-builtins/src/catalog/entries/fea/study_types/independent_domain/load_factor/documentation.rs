use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`start` and `end` are finite real scalars (dimensionless multipliers of the applied loads); integer classes are accepted.",
        "Model validation requires an ordered interval and a compatible procedure, typically nonlinear static. Load-factor sampling may include negative factors when the interval does.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass the domain as the second argument (`independentDomain`) of `fea.ProcedureStep.configure`. It replaces the step's default domain, and model validation checks it against the step's procedure and against every observation's sampling policy.",
        "`fea.SamplingPolicy.loadFactors`, `loadFactorInterval`, and solver events such as `firstYield` sample this domain.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when `start` or `end` is not a finite real numeric scalar.",
        "`RunMat:fea:Internal` if the value cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "ramp",
        title: "Ramp loads from zero to full",
        program: "domain = fea.IndependentDomain.loadFactor(0, 1);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(domain), 'fea.IndependentDomain'));\nassert(domain.value.load_factor.start == 0);" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.IndependentDomain.loadFactor",
        summary: "Declare a load-factor interval as a step's independent domain.",
        description: "`fea.IndependentDomain.loadFactor(start, end)` builds a load-factor `fea.IndependentDomain` that ramps the step's applied loads from `start` to `end`.",
        keywords: &["fea", "domain", "load", "factor", "nonlinear", "step"],
        related: &["fea.SamplingPolicy.loadFactors", "fea.ProcedureStep.configure", "fea.SamplingPolicy.finalSample"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Is the load factor a time?", answer: "No. It is a dimensionless multiplier; time-dependent loading uses `timeSeconds`." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Studies, analyses, and state", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/studies-analyses-and-state") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/sequences.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/independent_domain/tests.rs::independent_domain_constructors_follow_their_positional_contract" },
        ],
        notes: &[],
    });
