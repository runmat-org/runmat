use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("cdf"), slug: Some("cdf"),
    summary: "Evaluate a fitted or named cumulative distribution function.",
    description: "`cdf(pd,x)` evaluates a `ProbabilityDistribution` object returned by `fitdist`; `cdf(distname,x,params...)` evaluates a supported named distribution directly.",
    keywords: &["cdf", "fitdist", "probability distribution", "cumulative", "statistics"],
    related: &["fitdist", "icdf", "normcdf"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "Named Normal, Exponential, Lognormal, Gamma, Weibull, and Poisson families are supported. Named parameters and evaluation points may be scalar or singleton-broadcast-compatible arrays; the result has their common shape.",
            "Appending `\"upper\"` evaluates the complementary probability directly, including stable extreme-tail formulas for supported families.",
            "Single and double evaluation points and named parameters are admitted in MATLAB-compatible mode. Typed-integer points and parameters are independent RunMat extensions; logical numeric inputs are another RunMat extension. Typed integers must be exactly representable as binary64.",
            "Any single numeric point or named parameter selects single output; otherwise the result is double.",
        ] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "Resident inputs use an owning-provider gather/evaluate/re-upload fallback. The output retains residency and the selected single/double class when the owner supports it; structural controls do not select an owner.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Evaluate a fitted normal CDF", program: "pd = fitdist([1;2;3], \"Normal\"); p = cdf(pd, 2)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(p-0.5)<1e-12);" } },
        BuiltinExample { id: "example-2", title: "Evaluate a named Poisson CDF", program: "p = cdf(\"Poisson\", 2, 3)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(p-exp(-3)*(1+3+9/2))<1e-12);" } },
        BuiltinExample { id: "example-3", title: "Single-precision upper tail with scalar expansion", program: "p = cdf(\"Normal\", single([0;2]), [0;1], 1, \"upper\")", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(p),[2 1])); assert(strcmp(class(p),'single')); assert(abs(double(p(1))-0.5)<1e-6);" } },
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "How do I compute inverse probabilities?", answer: "Use `icdf(pd,p)` with a fitted distribution object, or the named inverse form." }],
    links: &[
        BuiltinDocumentationLink { label: "fitdist", target: BuiltinDocumentationLinkTarget::Builtin("fitdist") },
        BuiltinDocumentationLink { label: "icdf", target: BuiltinDocumentationLinkTarget::Builtin("icdf") },
        BuiltinDocumentationLink { label: "normcdf", target: BuiltinDocumentationLinkTarget::Builtin("normcdf") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/fitted_distribution/cdf.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "CDF implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/fitted_distribution/cdf.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Fitted distribution semantics", location: "crates/runmat-runtime/src/builtins/stats/summary/fitted_distribution/tests/cdf.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled distribution semantics", location: "crates/runmat-vm/tests/stats_ml.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
