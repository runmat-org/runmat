use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("wblinv"),
    slug: Some("wblinv"),
    summary: "Evaluate the inverse Weibull cumulative distribution function.",
    description: "`wblinv(p)` evaluates the unit Weibull inverse CDF. `wblinv(p,a,b)` uses scale parameter `a` and shape parameter `b`.",
    keywords: &["wblinv", "weibull", "inverse cdf", "statistics"],
    related: &["wblrnd", "tinv"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "`p`, `a`, and `b` support scalar expansion and matching array shapes.",
            "The documented input classes are single and double. Typed-integer probability, scale, and shape roles are independently gated RunMat extensions and must be exactly representable before binary64 Weibull evaluation.",
            "Output is single when any documented input is single and double otherwise.",
            "`p` must be in `[0,1]`; `a` and `b` must be positive. Invalid parameters produce `NaN` outputs.",
            "`p == 0` returns zero and `p == 1` returns `Inf`.",
            "Confidence-bound output forms that depend on fitted-parameter covariance are not implemented.",
        ] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "Automatic resident inputs use exact owner-aware gather fallback and restore the floating result when the owner can preserve its class. Explicit gpuArray input is a gated RunMat extension.",
            "This operation is a fusion boundary; the current implementation evaluates its numerical kernel on the host.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Weibull median", program: "x = wblinv(0.5,3,4)", display_output: Some("x = 2.7376"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(x-2.7376)<5e-5);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What is the formula?", answer: "RunMat computes `a * (-log(1-p))^(1/b)` for valid probabilities and positive parameters." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "wblrnd", target: BuiltinDocumentationLinkTarget::Builtin("wblrnd") },
        BuiltinDocumentationLink { label: "tinv", target: BuiltinDocumentationLinkTarget::Builtin("tinv") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/distributions/wblinv.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "wblinv implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/distributions/wblinv.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Distribution semantics", location: "crates/runmat-runtime/src/builtins/stats/summary/distributions/tests/mod.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled distribution semantics", location: "crates/runmat-vm/tests/stats_ml.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
