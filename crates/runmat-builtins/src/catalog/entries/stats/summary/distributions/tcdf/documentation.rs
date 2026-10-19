use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("tcdf"),
    slug: Some("tcdf"),
    summary: "Evaluate the Student's t cumulative distribution function.",
    description: "`tcdf(x,nu)` evaluates the Student's t cumulative distribution function. `tcdf(x,nu,\"upper\")` returns the complementary upper-tail probability.",
    keywords: &["tcdf", "student t", "cdf", "upper tail", "statistics", "distribution"],
    related: &["tpdf", "tinv", "trnd"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "`x` and `nu` may be scalars or compatible-size single- or double-precision arrays.",
            "RunMat also broadcasts compatible singleton dimensions to a common shape.",
            "Single-precision input produces single-precision output; otherwise the result is double precision.",
            "In RunMat mode, typed-integer and logical inputs are available as extensions. Integer values must be exactly representable as double before evaluation.",
            "Nonpositive or `NaN` degrees of freedom produce `NaN` elementwise.",
            "`nu = Inf` uses the standard normal distribution.",
        ] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "Resident inputs use checked host evaluation. Output returns through the exact owner when it can preserve the required class; incompatible explicit gpuArray intent errors.",
            "This operation is a fusion boundary; the current implementation evaluates its numerical kernel on the host.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Distribution midpoint", program: "p = tcdf(0, 10)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(p-0.5)<1e-12);" } },
        BuiltinExample { id: "example-2", title: "Upper tail", program: "p = tcdf(10, 99, \"upper\")", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(p-tcdf(-10,99))<1e-12);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does tcdf support upper-tail probabilities?", answer: "Yes. Pass `\"upper\"` as the final argument to compute the complementary tail." },
        BuiltinDocumentationFaq { question: "Can I use integer arrays with `tcdf`?", answer: "RunMat mode accepts all eight integer classes when every value is exactly representable as double. MATLAB-compatible mode accepts the documented single- and double-precision forms." },
        BuiltinDocumentationFaq { question: "What happens to a GPU result when RunMat uses a host fallback?", answer: "The host-computed result returns through the input's exact provider when that provider can preserve the required single or double class. Automatic residency may otherwise finish on the host; explicit gpuArray intent reports an error instead." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "tpdf", target: BuiltinDocumentationLinkTarget::Builtin("tpdf") },
        BuiltinDocumentationLink { label: "tinv", target: BuiltinDocumentationLinkTarget::Builtin("tinv") },
        BuiltinDocumentationLink { label: "trnd", target: BuiltinDocumentationLinkTarget::Builtin("trnd") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/distributions/tcdf.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "tcdf implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/distributions/tcdf.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Distribution semantics", location: "crates/runmat-runtime/src/builtins/stats/summary/distributions/tests/mod.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
