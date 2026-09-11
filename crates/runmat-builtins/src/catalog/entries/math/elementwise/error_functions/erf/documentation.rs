use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Error-function values",
        paragraphs: &[
            "`erf(X)` evaluates the Gaussian error function element by element for dense real single or double input. It preserves the input class and shape. NaN propagates, while positive and negative infinity map to 1 and -1.",
            "Fixed-width integer, logical, character, string, complex, object, and sparse inputs are rejected. RunMat does not silently apply a complex analytic continuation outside the documented real surface.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Accelerated and distributed execution",
        paragraphs: &[
            "A real floating provider input can use its owner's `unary_erf` operation. RunMat validates shape, storage, precision, owner, device, and non-aliasing, then restores the input's residency intent on the result. A typed unsupported result enters host fallback; provider failures and malformed outputs remain errors.",
            "Fallback gathers through the exact owner, evaluates the same real function, and restores the result to that owner with the original precision and residency intent. Distributed arrays use partition-local unary mapping.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Numerical use",
        paragraphs: &[
            "The error function is commonly used to express the cumulative distribution function of a normal random variable. For large positive values, a direct complementary-error implementation can retain a small tail more accurately than subtracting `erf(X)` from 1.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "scalar", title: "Evaluate a scalar", program: "Y = erf(1)", display_output: Some("Y = 0.8427"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(Y - 0.8427007929497149) < 1e-12);" } },
    BuiltinExample { id: "vector", title: "Evaluate a vector element by element", program: "X = [-0.5 0 1 3];\nY = erf(X)", display_output: Some("Y = [-0.5205 0 0.8427 1.0000]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(max(abs(Y - [-0.5204998778130465 0 0.8427007929497149 0.9999779095030014])) < 1e-12);" } },
    BuiltinExample { id: "matrix", title: "Preserve matrix shape", program: "A = [0.29 -0.11; 3.1 -2.9];\nB = erf(A)", display_output: Some("B is a 2-by-2 double matrix"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(B), [2 2]));\nassert(abs(B(1,1) - 0.3182834958609522) < 1e-12);\nassert(abs(B(1,2) + 0.1236228961994743) < 1e-12);" } },
    BuiltinExample { id: "normal-cdf", title: "Build a normal cumulative distribution", program: "X = -3:0.1:3;\nY = 0.5*(1 + erf(X/sqrt(2)));", display_output: Some("Y is a 1-by-61 double vector"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(Y), [1 61]));\nassert(abs(Y(31) - 0.5) < 1e-12);\nassert(Y(1) < Y(61));" } },
    BuiltinExample { id: "gpu-residency", title: "Keep a provider result resident", program: "G = gpuArray([-1 0 1]);\nGy = erf(G);\nY = gather(Gy)", display_output: Some("Y = [-0.8427 0 0.8427]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(Gy, 'gpuArray'));\nassert(max(abs(Y - [-0.8427007929497149 0 0.8427007929497149])) < 1e-6);" } },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does erf support complex input?", answer: "No. This builtin implements the documented dense real single/double surface and reports a structured input error for complex values." },
    BuiltinDocumentationFaq { question: "What class does erf return?", answer: "Single input returns single; double input returns double. The input shape is preserved." },
    BuiltinDocumentationFaq { question: "Can erf keep a provider input resident?", answer: "Yes. A valid direct result remains resident, and an unsupported provider operation uses exact-owner gather and restoration." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "erfcinv", target: BuiltinDocumentationLinkTarget::Builtin("erfcinv") },
    BuiltinDocumentationLink { label: "gamma", target: BuiltinDocumentationLinkTarget::Builtin("gamma") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/error_functions/erf") },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Real error-function runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/error_functions/erf") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Values, limits, shapes, classes, and rejected representations", location: "crates/runmat-runtime/src/builtins/math/elementwise/error_functions/erf.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Direct execution, fallback, ownership, and output validation", location: "crates/runmat-runtime/src/builtins/math/elementwise/error_functions/erf.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU real-function parity and residency", location: "crates/runmat-runtime/src/builtins/math/elementwise/error_functions/erf.rs" },
    ],
    notes: &[],
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("erf"),
    slug: Some("erf"),
    summary: "Evaluate the Gaussian error function element by element.",
    description: "`erf` evaluates dense real single or double values, preserves their class and shape, and supports validated provider-resident execution.",
    keywords: &["erf", "error function", "special function", "elementwise", "single", "double", "gpu"],
    related: &["erfcinv", "gamma", "gpuArray", "gather"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: EVIDENCE,
    introduced: Some("Before R2006a"),
    status: Some(BuiltinDocumentationStatus::Stable),
};
