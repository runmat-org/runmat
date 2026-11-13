use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("binocdf"),
    slug: Some("binocdf"),
    summary: "Evaluate the binomial cumulative distribution function.",
    description: "`binocdf(x,n,p)` returns the binomial lower-tail probability for `x`, number of trials `n`, and success probability `p`. `binocdf(x,n,p,\"upper\")` returns the complementary upper-tail probability.",
    keywords: &["binocdf", "binomial", "cdf", "upper tail", "statistics"],
    related: &["binornd", "chi2cdf"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "RunMat broadcasts scalar and compatible singleton-sized `x`, `n`, and `p` arrays to a common shape.",
            "MATLAB-compatible inputs are `single` or `double`; typed integers in any numeric input position and logical numeric inputs are independently gated RunMat extensions.",
            "Typed integers retain exact storage until the explicit binary64 distribution boundary and reject when a value is not exactly representable at that boundary.",
            "A native `single` tensor input produces `single` output; otherwise the probability output is `double`.",
            "`x` is floored to the nearest integer count before evaluation.",
            "`n` must be a nonnegative integer and `p` must be in `[0,1]`; invalid parameters produce `NaN` outputs.",
            "The `\"upper\"` form uses the complementary beta expression directly instead of subtracting from one.",
        ] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "Resident inputs use checked host evaluation. Output returns through the exact owner when it can preserve the required class; incompatible explicit gpuArray intent errors.",
            "This operation is a fusion boundary; the current implementation evaluates its numerical kernel on the host.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Binomial lower and upper tail probabilities", program: "p = binocdf(55,100,0.5)\nq = binocdf(55,100,0.5,\"upper\")", display_output: Some("p = 0.8644\nq = 0.1356"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(p-0.8644)<5e-5); assert(abs(q-0.1356)<5e-5); assert(abs(p+q-1)<1e-10);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does binocdf accept array parameters?", answer: "Yes. RunMat broadcasts compatible singleton dimensions to a common shape." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "binornd", target: BuiltinDocumentationLinkTarget::Builtin("binornd") },
        BuiltinDocumentationLink { label: "chi2cdf", target: BuiltinDocumentationLinkTarget::Builtin("chi2cdf") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/distributions/binocdf.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "binocdf implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/distributions/binocdf.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Distribution semantics", location: "crates/runmat-runtime/src/builtins/stats/summary/distributions/tests/mod.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled distribution semantics", location: "crates/runmat-vm/tests/stats_ml.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
