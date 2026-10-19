use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("polyder"),
    slug: Some("polyder"),
    summary: "Differentiate polynomial, product, and rational coefficient forms.",
    description: "`polyder(p)` differentiates one polynomial. `polyder(a,b)` with one output differentiates their product; `[num,den] = polyder(u,v)` returns numerator and denominator of the quotient-rule derivative.",
    keywords: &["polyder", "polynomial", "derivative", "product", "quotient"],
    related: &["polyval", "polyfit", "conv", "deconv", "polyint", "roots"],
    sections: &[
        BuiltinDocumentationSection { heading: "Coefficient forms", paragraphs: &[
            "Accepts real or complex scalar, row, column, empty, and singleton-extended vector coefficient arrays. Logical coefficients convert to double. Typed fixed-width integers are a RunMat-mode extension and cross to floating arithmetic only after exact-binary64 validation.",
            "Leading zeros in outputs are trimmed; an identically zero result retains one zero coefficient. Empty inputs represent the zero polynomial. The first input determines orientation for the one-output forms and quotient numerator; the quotient denominator follows the second input.",
            "One output with two input polynomials uses the product rule. Two outputs with two inputs use the quotient rule. Requested outputs after the second receive scalar zero padding; requesting a quotient output with only one input is an error.",
        ] },
        BuiltinDocumentationSection { heading: "Provider and precision behavior", paragraphs: &[
            "Compatible real provider-resident inputs use dedicated single, product, or quotient hooks when available and retain device residency. Complex, mixed-location, and unsupported provider cases gather through their owners and compute on the host; that fallback returns host output in the current implementation.",
            "Host arithmetic and its real/complex outputs use double precision. A supported real single-precision provider hook may retain single output storage. This coefficient operation is a fusion boundary.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "single", title: "Differentiate a cubic polynomial", program: "p = [3 -2 5 7];\ndp = polyder(p)", display_output: Some("dp = [9 -4 5]"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(dp,[9 -4 5]));" } },
        BuiltinExample { id: "product", title: "Apply the product rule", program: "p = [1 0 -2];\na = [1 1];\ndp = polyder(p,a)", display_output: Some("dp = [3 2 -2]"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(dp,[3 2 -2]));" } },
        BuiltinExample { id: "quotient", title: "Differentiate a rational function", program: "u = [1 0 -4];\nv = [1 -1];\n[num,den] = polyder(u,v)", display_output: Some("num = [1 -2 4]; den = [1 -2 1]"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(num,[1 -2 4])); assert(isequal(den,[1 -2 1]));" } },
        BuiltinExample { id: "column", title: "Preserve coefficient orientation", program: "p = [1;0;-3];\ndp = polyder(p)", display_output: Some("dp = [2;0]"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(dp,[2;0]));" } },
        BuiltinExample { id: "complex", title: "Differentiate complex coefficients", program: "p = [1+2i -3 4i];\ndp = polyder(p)", display_output: Some("dp = [2+4i -3]"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(max(abs(dp-[2+4i -3]))<1e-12);" } },
        BuiltinExample { id: "gpu", title: "Differentiate resident coefficients", program: "g = gpuArray([2 0 -5 4]);\ndp = polyder(g);\nresult = gather(dp)", display_output: Some("result = [6 0 -5]"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Wgpu, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(dp,'gpuArray')); assert(isequal(result,[6 0 -5]));" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What does an empty coefficient vector mean?", answer: "It represents the zero polynomial. A one-input derivative returns a single zero coefficient." },
        BuiltinDocumentationFaq { question: "Is column orientation preserved?", answer: "Yes for a nontrivial derivative. One-output forms use the first input's orientation; the quotient denominator uses the second input's orientation." },
        BuiltinDocumentationFaq { question: "What happens to leading zeros?", answer: "They are trimmed, except that a zero result keeps one zero coefficient." },
        BuiltinDocumentationFaq { question: "Are logical and integer coefficients accepted?", answer: "Logical values convert to double. Fixed-width integers require RunMat mode and exact representation at the floating-point boundary." },
        BuiltinDocumentationFaq { question: "How do I differentiate a ratio?", answer: "Request two outputs: `[num,den] = polyder(u,v)` returns the quotient-rule numerator and squared denominator coefficients." },
        BuiltinDocumentationFaq { question: "Do provider hooks run for complex coefficients?", answer: "The current hooks cover real coefficients. Complex work gathers and uses the host implementation." },
        BuiltinDocumentationFaq { question: "What if I request more than two outputs?", answer: "The first two are quotient numerator and denominator, and further requested slots receive scalar zeros." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "polyval", target: BuiltinDocumentationLinkTarget::Builtin("polyval") },
        BuiltinDocumentationLink { label: "polyfit", target: BuiltinDocumentationLinkTarget::Builtin("polyfit") },
        BuiltinDocumentationLink { label: "conv", target: BuiltinDocumentationLinkTarget::Builtin("conv") },
        BuiltinDocumentationLink { label: "deconv", target: BuiltinDocumentationLinkTarget::Builtin("deconv") },
        BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
        BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
        BuiltinDocumentationLink { label: "polyint", target: BuiltinDocumentationLinkTarget::Builtin("polyint") },
        BuiltinDocumentationLink { label: "roots", target: BuiltinDocumentationLinkTarget::Builtin("roots") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/poly/polyder.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Derivative runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/poly/polyder.rs") },
            BuiltinDocumentationLink { label: "Resident derivative hooks", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/poly/polyder/resident.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Scalar, product, quotient, complex, empty, and output-count forms", location: "builtins::math::poly::polyder::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "WGPU resident single, product, and quotient forms", location: "builtins::math::poly::polyder::tests" },
        ],
        notes: &["Legacy examples, links, and form descriptions are retained; integer compatibility, host versus provider precision, and fallback residency claims were reconciled with source. Standalone examples and device performance remain unqualified."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
