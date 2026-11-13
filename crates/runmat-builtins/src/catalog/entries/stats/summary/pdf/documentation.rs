use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("pdf"), slug: Some("pdf"),
    summary: "Evaluate a fitted or named probability density or mass function.",
    description: "`pdf(pd,x)` evaluates a `ProbabilityDistribution` object returned by `fitdist`. `pdf(distname,x,params...)` evaluates a supported named family directly.",
    keywords: &["pdf", "fitdist", "probability distribution", "density", "mass", "statistics"],
    related: &["fitdist", "cdf", "normpdf"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "Named Normal, Exponential, Lognormal, Gamma, Weibull, and Poisson families are supported. Named parameters and evaluation points may be scalar or mutually broadcast-compatible numeric arrays.",
            "Continuous families return density values. Poisson returns probability mass at nonnegative integer support points and zero outside the support.",
            "Typed-integer evaluation points and typed-integer named parameters are independently gated RunMat extensions. Wide integers must cross the binary64 probability boundary exactly.",
            "Logical evaluation points and named parameters are evaluated numerically as 0 or 1 in the current RunMat implementation.",
            "Any single numeric evaluation point or named parameter selects single output; otherwise the result is double.",
        ] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "The current numeric evaluator runs on the host. Resident inputs use owning-provider gather and checked result restoration when the owner can represent the selected precision; explicit gpuArray intent fails rather than silently returning a host value when restoration is unavailable.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Evaluate a fitted normal density", program: "pd = fitdist([1;2;3], \"Normal\"); y = pdf(pd, 2)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(y-1/sqrt(4*pi/3))<1e-12);" } },
        BuiltinExample { id: "example-2", title: "Evaluate a named normal density", program: "y = pdf(\"Normal\", 0, 0, 1)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(y-1/sqrt(2*pi))<1e-12);" } },
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Can pdf evaluate by distribution name?", answer: "Yes. The named overload supports Normal, Exponential, Lognormal, Gamma, Weibull, and Poisson with their family parameters; specialized functions such as normpdf remain available." }],
    links: &[
        BuiltinDocumentationLink { label: "fitdist", target: BuiltinDocumentationLinkTarget::Builtin("fitdist") },
        BuiltinDocumentationLink { label: "cdf", target: BuiltinDocumentationLinkTarget::Builtin("cdf") },
        BuiltinDocumentationLink { label: "normpdf", target: BuiltinDocumentationLinkTarget::Builtin("normpdf") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/fitted_distribution/pdf.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "PDF implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/fitted_distribution/pdf.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Fitted distribution semantics", location: "crates/runmat-runtime/src/builtins/stats/summary/fitted_distribution/tests/pdf.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled distribution semantics", location: "crates/runmat-vm/tests/integer_numeric_transform_semantics.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
