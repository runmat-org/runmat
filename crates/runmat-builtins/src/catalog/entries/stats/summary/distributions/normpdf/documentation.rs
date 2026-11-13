use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("normpdf"),
    slug: Some("normpdf"),
    summary: "Evaluate the normal probability density function.",
    description: "`normpdf(x)` evaluates the standard normal density. `normpdf(x,mu)` and `normpdf(x,mu,sigma)` evaluate a normal distribution with mean `mu` and standard deviation `sigma`.",
    keywords: &["normpdf", "normal", "gaussian", "pdf", "statistics"],
    related: &["normcdf", "norminv"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "RunMat broadcasts scalar and compatible singleton-sized `x`, `mu`, and `sigma` arrays to a common shape.",
            "MATLAB-compatible numeric inputs are single or double. Typed integers and logical values in any numeric role are independently gated RunMat extensions.",
            "Admitted integer values must be exactly representable as double. Any single floating input selects single output; otherwise output is double.",
            "Nonpositive `sigma` values produce `NaN` elementwise.",
        ] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "Resident inputs use checked host fallback and return through the exact owner only when it can preserve the required single/double class. Incompatible automatic residency may finish on host; incompatible explicit gpuArray intent errors.",
            "This operation is a fusion boundary; the current implementation evaluates its numerical kernel on the host.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Standard normal density", program: "y = normpdf(0)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(y-1/sqrt(2*pi))<1e-12);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What is the default distribution?", answer: "The default is the standard normal distribution with `mu = 0` and `sigma = 1`." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "normcdf", target: BuiltinDocumentationLinkTarget::Builtin("normcdf") },
        BuiltinDocumentationLink { label: "norminv", target: BuiltinDocumentationLinkTarget::Builtin("norminv") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/distributions/normpdf.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "normpdf implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/distributions/normpdf.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Distribution semantics", location: "crates/runmat-runtime/src/builtins/stats/summary/distributions/tests/mod.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled distribution semantics", location: "crates/runmat-vm/tests/integer_normal_distribution_and_regression_semantics.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
