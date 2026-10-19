use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("random"), slug: Some("random"),
    summary: "Generate samples from a fitted or named probability distribution.",
    description: "`random(pd)` and `random(pd,sz...)` sample a fitted `ProbabilityDistribution` value object. `random(distname,params...,sz...)` samples a supported named distribution.",
    keywords: &["random", "fitdist", "probability distribution", "sampling", "statistics"],
    related: &["fitdist", "normrnd", "gamrnd"],
    sections: &[
        BuiltinDocumentationSection { heading: "Forms and output", paragraphs: &[
            "`random(pd)` returns one double scalar. Explicit scalar dimensions or a size vector specify the output shape. Named Normal, Exponential, Lognormal, Gamma, Weibull, and Poisson forms take their scalar family parameters before optional size arguments.",
            "Named family parameters and size controls are single or double in the documented MATLAB-compatible subset. The current RunMat evaluator also accepts logical named parameters as 0 or 1; logical size controls are not accepted. Typed-integer parameters and sizes are independently gated RunMat extensions. Parameters require exact binary64 representation; sizes are structural and decoded without floating conversion.",
            "The host RNG generates double samples. Host parameters produce host output. Resident named parameters select their exact owning provider for checked double-output restoration; size controls do not select residency. Explicit resident intent fails if restoration cannot preserve it.",
        ] },
        BuiltinDocumentationSection { heading: "Current scope", paragraphs: &[
            "Array-valued named parameters and additional distribution families are not yet implemented by this general API. Distribution-specific random functions remain available separately.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Draw a matrix from a fitted normal distribution", program: "pd = fitdist([1;2;3], \"Normal\"); r = random(pd, 2, 3)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(r),[2 3])); assert(strcmp(class(r),'double')); assert(all(isfinite(r(:))));" } },
        BuiltinExample { id: "example-2", title: "Draw from a named Weibull distribution", program: "r = random(\"Weibull\", 2, 3, 4, 5)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(r),[4 5])); assert(all(r(:)>=0));" } },
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Can random sample by distribution name?", answer: "Yes for Normal, Exponential, Lognormal, Gamma, Weibull, and Poisson with scalar family parameters. Specialized helpers such as normrnd are separate." }],
    links: &[
        BuiltinDocumentationLink { label: "fitdist", target: BuiltinDocumentationLinkTarget::Builtin("fitdist") },
        BuiltinDocumentationLink { label: "normrnd", target: BuiltinDocumentationLinkTarget::Builtin("normrnd") },
        BuiltinDocumentationLink { label: "gamrnd", target: BuiltinDocumentationLinkTarget::Builtin("gamrnd") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/fitted_distribution/sampling_entry.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Random sampling implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/fitted_distribution/sampling_entry.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Fitted distribution semantics", location: "crates/runmat-runtime/src/builtins/stats/summary/fitted_distribution/tests/random.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled distribution semantics", location: "crates/runmat-vm/tests/stats_ml.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
