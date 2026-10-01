use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("tpdf"),
    slug: Some("tpdf"),
    summary: "Evaluate the Student's t probability density function.",
    description: "`tpdf(x,nu)` evaluates the probability density of the Student's t distribution with `nu` degrees of freedom at `x`.",
    keywords: &["tpdf", "student t", "pdf", "statistics", "distribution"],
    related: &["tcdf", "tinv", "trnd"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "Documented `x` and `nu` inputs are single or double scalars and arrays. Inputs may have the same size, or either input may be scalar.",
            "The result is single when either documented floating input is single; otherwise it is double.",
            "RunMat mode additionally accepts logical and all eight native integer classes. Integer values must be exactly representable at the binary64 PDF boundary.",
            "RunMat also broadcasts compatible singleton dimensions to a common shape.",
            "Nonpositive or `NaN` degrees of freedom produce `NaN` elementwise.",
            "`nu = Inf` uses the standard normal density.",
        ] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "Resident inputs use checked host evaluation. Output returns through the exact owner when it can preserve the required class; incompatible explicit gpuArray intent errors.",
            "This operation is a fusion boundary; the current implementation evaluates its numerical kernel on the host.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Cauchy density at zero", program: "y = tpdf(0, 1)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(y-1/pi)<1e-12);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What is `nu`?", answer: "`nu` is the degrees-of-freedom parameter for the Student's t distribution." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "tcdf", target: BuiltinDocumentationLinkTarget::Builtin("tcdf") },
        BuiltinDocumentationLink { label: "tinv", target: BuiltinDocumentationLinkTarget::Builtin("tinv") },
        BuiltinDocumentationLink { label: "trnd", target: BuiltinDocumentationLinkTarget::Builtin("trnd") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/distributions/tpdf.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "tpdf implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/distributions/tpdf.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Distribution semantics", location: "crates/runmat-runtime/src/builtins/stats/summary/distributions/tests/mod.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
