use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("chi2cdf"),
    slug: Some("chi2cdf"),
    summary: "Evaluate the chi-square cumulative distribution function.",
    description: "`chi2cdf(x,nu)` evaluates the chi-square CDF at `x` for degrees of freedom `nu`. `chi2cdf(x,nu,\"upper\")` returns the complementary upper-tail probability.",
    keywords: &["chi2cdf", "chi-square", "cdf", "upper tail", "statistics"],
    related: &["tcdf", "binocdf"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "`x` and `nu` support scalar expansion and matching array shapes.",
            "Typed-integer `x` and `nu` values from all eight classes are mode-gated RunMat extensions and must be exactly representable as binary64 values.",
            "Single-precision inputs produce single-precision output; otherwise output is double precision.",
            "`nu` must be positive; invalid parameters produce `NaN` outputs.",
            "Negative `x` values return zero for the lower tail and one for the upper tail.",
            "The computation uses regularized incomplete gamma functions.",
        ] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "Resident inputs use checked host evaluation. Output returns through the selected exact owner when it can preserve the required class; incompatible explicit gpuArray intent errors.",
            "This operation is a fusion boundary; the current implementation evaluates its numerical kernel on the host.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Evaluate chi-square CDF", program: "p = chi2cdf(3,5)", display_output: Some("p = 0.3000"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(p-0.3000)<5e-5);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Why use the upper-tail form?", answer: "The upper-tail form computes the complementary incomplete-gamma value directly, which is more stable for extreme upper tails than `1 - chi2cdf(...)`." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "tcdf", target: BuiltinDocumentationLinkTarget::Builtin("tcdf") },
        BuiltinDocumentationLink { label: "binocdf", target: BuiltinDocumentationLinkTarget::Builtin("binocdf") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/distributions/chi2cdf.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "chi2cdf implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/distributions/chi2cdf.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Distribution semantics", location: "crates/runmat-runtime/src/builtins/stats/summary/distributions/tests/mod.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled distribution semantics", location: "crates/runmat-vm/tests/stats_ml.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
