use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("icdf"),
    slug: Some("icdf"),
    summary: "Evaluate inverse cumulative distribution functions by distribution name or fitted distribution object.",
    description: "`icdf(name,p,...)` evaluates inverse cumulative distribution functions for supported named distributions. `icdf(pd,p)` evaluates quantiles for a `ProbabilityDistribution` object returned by `fitdist`.",
    keywords: &["icdf", "inverse cdf", "normal", "student t", "weibull", "binomial", "chi-square", "gamma", "exponential", "poisson", "uniform", "lognormal", "beta", "f", "statistics"],
    related: &["norminv", "tinv", "wblinv", "fitdist", "binocdf", "chi2cdf"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "Supported distribution names include `\"Normal\"`, `\"t\"`, `\"Weibull\"`, `\"Chi-square\"`, `\"Binomial\"`, `\"Gamma\"`, `\"Exponential\"`, `\"Poisson\"`, `\"Uniform\"`, `\"Lognormal\"`, `\"Beta\"`, and `\"F\"`.",
            "Common aliases are accepted after case-insensitive normalization and removal of punctuation or spaces, including `\"norm\"`, `\"gaussian\"`, `\"student t\"`, `\"wbl\"`, `\"chi2\"`, `\"bino\"`, `\"gam\"`, `\"exp\"`, `\"poiss\"`, `\"unif\"`, `\"logn\"`, and `\"fdist\"`.",
            "`icdf(\"Normal\",p)` uses mean `0` and standard deviation `1`; `icdf(\"Normal\",p,mu,sigma)` uses supplied normal parameters.",
            "`icdf(\"t\",p,nu)` returns Student's t quantiles for degrees of freedom `nu`.",
            "`icdf(\"Weibull\",p)` uses scale `1` and shape `1`; `icdf(\"Weibull\",p,a,b)` uses supplied Weibull scale and shape.",
            "`icdf(\"Chi-square\",p,nu)` inverts the chi-square CDF with degrees of freedom `nu`.",
            "`icdf(\"Binomial\",p,n,prob)` returns the smallest integer outcome whose binomial CDF is at least `p`.",
            "For wide Binomial and Poisson parameters, RunMat returns the first integer-valued `double` representation whose CDF reaches `p`; the output cannot encode every integer above `flintmax`.",
            "`icdf(\"Gamma\",p,shape,scale)`, `icdf(\"Exponential\",p,mu)`, `icdf(\"Poisson\",p,lambda)`, `icdf(\"Beta\",p,a,b)`, and `icdf(\"F\",p,v1,v2)` require the shown distribution parameters.",
            "`icdf(\"Uniform\",p)` uses bounds `[0,1]`; `icdf(\"Uniform\",p,a,b)` uses supplied lower and upper bounds.",
            "`icdf(\"Lognormal\",p)` uses mean `0` and standard deviation `1` for the underlying normal distribution; `icdf(\"Lognormal\",p,mu)` and `icdf(\"Lognormal\",p,mu,sigma)` use supplied lognormal parameters.",
            "`icdf(pd,p)` supports fitted normal, exponential, lognormal, gamma, Weibull, and Poisson distribution objects returned by `fitdist`.",
            "Scalar and compatible array inputs broadcast to a common output shape. Invalid probabilities or distribution parameters produce `NaN` elements.",
            "Continuous distributions return endpoint values at `p = 0` and `Inf` where the upper tail is unbounded at `p = 1`; bounded distributions return their finite endpoints. The Poisson inverse returns `Inf` at `p = 1`.",
        ] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "Resident inputs are gathered through their owning provider for host numerical inversion. The result returns to that owner only when it can preserve the selected single or double class; incompatible automatic residency may remain on host and incompatible explicit gpuArray intent errors.",
            "This operation is a fusion boundary; the current implementation evaluates its numerical kernel on the host.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Normal median with parameters", program: "x = icdf(\"Normal\", 0.5, 10, 2)", display_output: Some("x = 10"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(x-10)<1e-12);" } },
        BuiltinExample { id: "example-2", title: "Discrete binomial quantile", program: "k = icdf(\"Binomial\", 0.75, 10, 0.5)", display_output: Some("k = 6"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(k==6);" } },
        BuiltinExample { id: "example-3", title: "Uniform quantile with bounds", program: "x = icdf(\"Uniform\", 0.25, 10, 20)", display_output: Some("x = 12.5"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(x-12.5)<1e-12);" } },
        BuiltinExample { id: "example-4", title: "Fitted distribution median", program: "pd = fitdist([1;2;3], \"Normal\"); x = icdf(pd, 0.5)", display_output: Some("x = 2"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(x-2)<1e-12);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "How is icdf related to distribution-specific inverse functions?", answer: "`icdf(\"Normal\",p,mu,sigma)` dispatches to the same numerical path as `norminv(p,mu,sigma)`, `icdf(\"t\",p,nu)` to `tinv`, and `icdf(\"Weibull\",p,a,b)` to `wblinv`. Other supported distribution names use matching runtime CDF or inverse helper implementations." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "norminv", target: BuiltinDocumentationLinkTarget::Builtin("norminv") },
        BuiltinDocumentationLink { label: "tinv", target: BuiltinDocumentationLinkTarget::Builtin("tinv") },
        BuiltinDocumentationLink { label: "wblinv", target: BuiltinDocumentationLinkTarget::Builtin("wblinv") },
        BuiltinDocumentationLink { label: "fitdist", target: BuiltinDocumentationLinkTarget::Builtin("fitdist") },
        BuiltinDocumentationLink { label: "binocdf", target: BuiltinDocumentationLinkTarget::Builtin("binocdf") },
        BuiltinDocumentationLink { label: "chi2cdf", target: BuiltinDocumentationLinkTarget::Builtin("chi2cdf") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/distributions/icdf.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "icdf implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/distributions/icdf.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Distribution semantics", location: "crates/runmat-runtime/src/builtins/stats/summary/distributions/tests/mod.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled distribution semantics", location: "crates/runmat-vm/tests/stats_ml.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
