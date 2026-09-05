use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Step values",
        paragraphs: &[
            "`heaviside(X)` evaluates each real element independently. Negative values map to 0, positive values map to 1, and both positive and negative zero map to 0.5. Positive and negative infinity follow their signs, while NaN remains NaN.",
            "Dense `single` input returns `single`; dense `double` input returns `double`. In RunMat compatibility mode, fixed-width integers, logical values, and character code points are accepted and return real `double` output with the same shape. Symbolic scalar input remains a symbolic `heaviside(...)` expression. Complex, string, and sparse inputs are rejected.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "GPU and fused execution",
        paragraphs: &[
            "A real floating `gpuArray` can use its owning provider's unary step operation. RunMat validates output shape, storage, precision, owner, device, non-aliasing, and placement provenance. Only a typed unsupported-hook result enters exact-owner host fallback; other provider failures are terminal.",
            "Resident integer and logical extensions are classified after an exact-owner gather because the public result is `double`. A provider that can represent the result restores it to the source device; otherwise the result remains on the host. Fused floating execution uses the same zero and NaN rules as host execution.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "representative-values", title: "Evaluate representative step values", program: "values = heaviside([-2 0 3])", display_output: Some("values = [0 0.5 1]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(values, [0 0.5 1]));" } },
    BuiltinExample { id: "gate-sinusoid", title: "Gate a sinusoid with a unit step", program: "t = -1:0.01:1;\nx = sin(2*pi*t).*heaviside(t);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(all(x(t < 0) == 0));\nassert(abs(t(101)) < 1e-12);\nassert(abs(x(101)) < 1e-12);\nassert(isequal(size(x), size(t)));" } },
    BuiltinExample { id: "matrix-shape", title: "Preserve matrix shape", program: "A = [-1 0; 2 NaN];\nY = heaviside(A)", display_output: Some("Y = [0 0.5; 1 NaN]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(Y), [2 2]));\nassert(Y(1) == 0);\nassert(Y(2) == 1);\nassert(Y(3) == 0.5);\nassert(isnan(Y(4)));" } },
    BuiltinExample { id: "logical-extension", title: "Evaluate logical step inputs", program: "mask = [false true];\nY = heaviside(mask)", display_output: Some("Y = [0.5 1]"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(Y, 'double'));\nassert(isequal(Y, [0.5 1]));" } },
    BuiltinExample { id: "symbolic-expression", title: "Preserve a symbolic expression", program: "syms x;\nY = heaviside(x)", display_output: Some("Y = heaviside(x)"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(Y, 'sym'));\nassert(strcmp(char(Y), 'heaviside(x)'));" } },
    BuiltinExample { id: "gpu-residency", title: "Evaluate resident floating step values", program: "G = gpuArray(single([-1 0 1]));\nGy = heaviside(G);\nY = gather(Gy)", display_output: Some("Y = single([0 0.5 1])"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Wgpu, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(Gy, 'gpuArray'));\nassert(isa(Y, 'single'));\nassert(isequal(Y, single([0 0.5 1])));" } },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "What value does heaviside use at zero?", answer: "It returns 0.5 for both positive and negative zero." },
    BuiltinDocumentationFaq { question: "Does heaviside support complex numbers?", answer: "No. This builtin implements the real-valued step function and rejects complex input with `RunMat:heaviside:InvalidInput`." },
    BuiltinDocumentationFaq { question: "Does heaviside preserve NaN?", answer: "Yes. Host, provider, and fused floating paths all return NaN for a NaN input." },
    BuiltinDocumentationFaq { question: "Can a gpuArray result remain resident?", answer: "Real floating inputs remain resident after direct execution or a supported exact-owner fallback. Integer and logical extensions return double and remain resident only when the provider can represent that result." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "sign", target: BuiltinDocumentationLinkTarget::Builtin("sign") },
    BuiltinDocumentationLink { label: "times", target: BuiltinDocumentationLinkTarget::Builtin("times") },
    BuiltinDocumentationLink { label: "sin", target: BuiltinDocumentationLinkTarget::Builtin("sin") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/math/elementwise/heaviside") },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Heaviside runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/math/elementwise/heaviside") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Step values, classes, shapes, extensions, rejection, and symbolic behavior", location: "builtins::math::elementwise::heaviside::tests::semantics" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Direct execution, typed fallback, terminal errors, ownership, and metadata", location: "builtins::math::elementwise::heaviside::tests::provider" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU parity and residency", location: "builtins::math::elementwise::heaviside::tests::provider::wgpu_matches_host" },
    ],
    notes: &[],
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("heaviside"),
    slug: Some("heaviside"),
    summary: "Compute the elementwise Heaviside step function.",
    description: "`heaviside(X)` maps negative real values to 0, zero to 0.5, and positive values to 1 while preserving floating class and input shape.",
    keywords: &["heaviside", "unit step", "step function", "elementwise", "gpu"],
    related: &["sign", "times", "sin", "gpuArray", "gather"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: EVIDENCE,
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
