use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("normcdf"),
    slug: Some("normcdf"),
    summary: "Evaluate the normal cumulative distribution function.",
    description: "`normcdf(x)` evaluates the standard normal CDF. `normcdf(x,mu)` and `normcdf(x,mu,sigma)` evaluate a normal distribution with mean `mu` and standard deviation `sigma`. `normcdf(...,\"upper\")` returns the upper-tail probability.",
    keywords: &["normcdf", "normal", "gaussian", "cdf", "statistics"],
    related: &["normpdf", "norminv"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "RunMat broadcasts scalar and compatible singleton-sized `x`, `mu`, and `sigma` arrays to a common shape.",
            "MATLAB-compatible numeric inputs are single or double. Typed integers and logical values in any numeric role are independently gated RunMat extensions.",
            "Admitted integer values must be exactly representable as double. Any single floating input selects single output; otherwise output is double.",
            "Negative `sigma` values produce `NaN` elementwise.",
            "`sigma = 0` returns a step at `mu`.",
            "`\"upper\"` uses the complementary error-function path for better extreme-tail stability.",
        ] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "Resident inputs use checked host fallback and return through the exact owner only when it can preserve the required single/double class. Incompatible automatic residency may finish on host; incompatible explicit gpuArray intent errors.",
            "This operation is a fusion boundary; the current implementation evaluates its numerical kernel on the host.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Standard normal midpoint", program: "p = normcdf(0)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(p-0.5)<1e-12);" } },
        BuiltinExample { id: "example-2", title: "Upper tail", program: "p = normcdf(8, 0, 1, \"upper\")", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(p-0.5*erfc(8/sqrt(2)))<1e-20);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does normcdf support array inputs?", answer: "Yes. Scalar parameters broadcast over matching-size numeric arrays." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "normpdf", target: BuiltinDocumentationLinkTarget::Builtin("normpdf") },
        BuiltinDocumentationLink { label: "norminv", target: BuiltinDocumentationLinkTarget::Builtin("norminv") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/distributions/normcdf.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "normcdf implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/distributions/normcdf.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Distribution semantics", location: "crates/runmat-runtime/src/builtins/stats/summary/distributions/tests/mod.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled distribution semantics", location: "crates/runmat-vm/tests/integer_normal_distribution_and_regression_semantics.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
