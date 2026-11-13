use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("fitdist"),
    slug: Some("fitdist"),
    summary: "Fit a probability distribution object to sample data.",
    description: "`fitdist(x,distname)` estimates distribution parameters from a numeric or logical vector and returns a `ProbabilityDistribution` value object accepted by `pdf`, `cdf`, `icdf`, and `random`.",
    keywords: &["fitdist", "probability distribution", "normal", "exponential", "lognormal", "gamma", "weibull", "poisson", "statistics"],
    related: &["pdf", "cdf", "icdf", "random"],
    sections: &[
        BuiltinDocumentationSection { heading: "Families and data", paragraphs: &[
            "Supported fitted families are Normal, Exponential, Lognormal, Gamma, Weibull, and Poisson. The short names norm, exp, logn, gam, wbl, and poiss are accepted.",
            "Observations must form a real numeric or logical vector. Logical values are evaluated as 0 or 1. NaN observations are omitted; infinite observations are rejected. Exponential data must be nonnegative, Lognormal, Gamma, and Weibull data positive, and Poisson data nonnegative integers.",
            "`fitdist(x,distname,\"Frequency\",w)` applies one finite nonnegative integer-valued count per observation. Logical Frequency values are evaluated as 0 or 1. Zero-frequency observations do not contribute to the fit.",
        ] },
        BuiltinDocumentationSection { heading: "Object and execution", paragraphs: &[
            "The value object exposes DistributionName, DistName, ParameterNames, ParameterValues, NumParameters, NumObservations, NLogL, IsTruncated, and applicable scalar parameter properties such as mu, sigma, lambda, a, or b.",
            "RunMat mode additionally admits typed-integer observations and Frequency arrays when values cross the binary64 fitting boundary exactly. These two extensions are gated independently.",
            "RunMat mode can gather resident observations and frequencies to fit on the host; MATLAB compatibility mode rejects that fallback before provider access. The fitted object and its floating parameters live on the host.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Fit a normal distribution and evaluate it", program: "pd = fitdist([1;2;3], \"Normal\"); y = pdf(pd, 2); p = cdf(pd, 2); x = icdf(pd, 0.5)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(pd.mu-2)<1e-12); assert(abs(y-1/sqrt(4*pi/3))<1e-12); assert(abs(p-0.5)<1e-12); assert(abs(x-2)<1e-12);" } },
        BuiltinExample { id: "example-2", title: "Fit with observation frequencies", program: "pd = fitdist([1;2], \"Exponential\", \"Frequency\", [1;3])", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(pd.mu-1.75)<1e-12); assert(pd.NumObservations==4);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Can fitdist fit typed-integer observations or Frequency arrays?", answer: "RunMat mode adds separate integer-data and integer-Frequency extensions for all eight integer classes; each value must be exactly representable at the binary64 boundary. MATLAB compatibility mode does not admit those extensions." },
        BuiltinDocumentationFaq { question: "Does fitdist support censoring or grouped fits?", answer: "No. Censoring, Options, and grouped By fits are rejected rather than ignored." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "pdf", target: BuiltinDocumentationLinkTarget::Builtin("pdf") },
        BuiltinDocumentationLink { label: "cdf", target: BuiltinDocumentationLinkTarget::Builtin("cdf") },
        BuiltinDocumentationLink { label: "icdf", target: BuiltinDocumentationLinkTarget::Builtin("icdf") },
        BuiltinDocumentationLink { label: "random", target: BuiltinDocumentationLinkTarget::Builtin("random") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/fitted_distribution/fit.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Fit implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/fitted_distribution/fit.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Fitted distribution semantics", location: "crates/runmat-runtime/src/builtins/stats/summary/fitted_distribution/tests/fit.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled distribution semantics", location: "crates/runmat-vm/tests/stats_ml.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
