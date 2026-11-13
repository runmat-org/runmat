use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("tinv"),
    slug: Some("tinv"),
    summary: "Evaluate the inverse Student's t cumulative distribution function.",
    description: "`tinv(p,nu)` returns the Student's t quantile for probability `p` and degrees of freedom `nu`.",
    keywords: &["tinv", "student t", "inverse", "cdf", "statistics", "distribution"],
    related: &["tcdf", "tpdf", "trnd"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "`p` and `nu` may be single or double scalars or arrays with compatible scalar-expansion shapes.",
            "RunMat also broadcasts compatible singleton dimensions to a common shape.",
            "`p` outside `[0,1]` or nonpositive `nu` produces `NaN` elementwise.",
            "`p = 0` and `p = 1` return infinite tails.",
            "Single input selects single output; otherwise the result uses double precision.",
            "Native integer and logical inputs are mode-gated extensions. Integer values must be exactly representable at the binary64 inverse-CDF boundary.",
            "Resident input returns through the owning provider when owner-preserving restoration is available.",
        ] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "Resident inputs use owner-aware gather fallback and restore the result through the owning provider when possible.",
            "This operation is a fusion boundary; the current implementation evaluates its numerical kernel on the host.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "95th percentile", program: "x = tinv(0.95, 50)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(tcdf(x,50)-0.95)<1e-8);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "How is tinv computed?", answer: "RunMat solves against the Student's t CDF with a bounded bisection path, using symmetry around zero." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "tcdf", target: BuiltinDocumentationLinkTarget::Builtin("tcdf") },
        BuiltinDocumentationLink { label: "tpdf", target: BuiltinDocumentationLinkTarget::Builtin("tpdf") },
        BuiltinDocumentationLink { label: "trnd", target: BuiltinDocumentationLinkTarget::Builtin("trnd") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/distributions/tinv.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "tinv implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/distributions/tinv.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Distribution semantics", location: "crates/runmat-runtime/src/builtins/stats/summary/distributions/tests/mod.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
