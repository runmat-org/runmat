use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("norminv"),
    slug: Some("norminv"),
    summary: "Evaluate the inverse normal cumulative distribution function.",
    description: "`norminv(p)` returns the standard normal quantile. `norminv(p,mu)` and `norminv(p,mu,sigma)` return the quantile for a normal distribution with mean `mu` and standard deviation `sigma`.",
    keywords: &["norminv", "normal", "gaussian", "inverse", "cdf", "statistics"],
    related: &["normcdf", "erfcinv"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "RunMat broadcasts scalar and compatible singleton-sized `p`, `mu`, and `sigma` arrays to a common shape.",
            "MATLAB-compatible numeric inputs are single or double. Typed integers and logical values in any numeric role are independently gated RunMat extensions.",
            "Admitted integer values must be exactly representable as double. Any single floating input selects single output; otherwise output is double.",
            "`p` outside `[0,1]` or nonpositive `sigma` produces `NaN` elementwise.",
            "`p = 0` and `p = 1` return infinite tails.",
        ] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "Resident inputs use checked host fallback and return through the exact owner only when it can preserve the required single/double class. Incompatible automatic residency may finish on host; incompatible explicit gpuArray intent errors.",
            "This operation is a fusion boundary; the current implementation evaluates its numerical kernel on the host.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Median of the standard normal", program: "x = norminv(0.5)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(x)<1e-12);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "How is norminv related to erfcinv?", answer: "RunMat computes the normal inverse using the same inverse-complementary-error-function path as `erfcinv`." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "normcdf", target: BuiltinDocumentationLinkTarget::Builtin("normcdf") },
        BuiltinDocumentationLink { label: "erfcinv", target: BuiltinDocumentationLinkTarget::Builtin("erfcinv") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/distributions/norminv.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "norminv implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/distributions/norminv.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Distribution semantics", location: "crates/runmat-runtime/src/builtins/stats/summary/distributions/tests/mod.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled distribution semantics", location: "crates/runmat-vm/tests/integer_normal_distribution_and_regression_semantics.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
