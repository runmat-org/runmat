use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Gamma values and poles",
        paragraphs: &[
            "`gamma(X)` evaluates the Euler gamma function element by element. For a positive integer `n`, `gamma(n)` equals `(n-1)!`; `gamma(0.5)` equals `sqrt(pi)`. Analytic continuation defines values for negative nonintegers, while zero and negative integers are poles.",
            "Large positive inputs eventually overflow to infinity. The practical positive-integer thresholds are `gamma(172)` for double and `gamma(single(36))` for single because the next values exceed their respective finite ranges.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Classes and shapes",
        paragraphs: &[
            "Input must be dense, real single or double. Output preserves the input class and shape. Fixed-width integer, logical, character, string, complex, object, and sparse inputs are rejected in both compatibility modes. An ordinary whole-number literal such as `5` is double and remains valid.",
            "Scalar, vector, matrix, N-D, and empty arrays are evaluated element by element. NaN propagates and infinities follow the real gamma limits.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Accelerated and distributed execution",
        paragraphs: &[
            "A real floating provider input can use its owner's `unary_gamma` hook. RunMat accepts the result only when shape, real storage, precision, owner, device, provenance, and non-aliasing satisfy the contract. Only a typed unsupported result enters host fallback; provider failures and malformed outputs remain visible.",
            "Fallback gathers through the exact owner, evaluates the same host algorithm, and restores the result to that owner with the original precision and residency intent. Distributed arrays use the declared partition-local unary mapping.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "factorial-identity", title: "Relate gamma to factorial", program: "Y = gamma(5)", display_output: Some("Y = 24"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(Y - 24) < 1e-12);" } },
    BuiltinExample { id: "half", title: "Evaluate a half-integer", program: "Y = gamma(0.5)", display_output: Some("Y = 1.7725"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(Y - sqrt(pi)) < 1e-12);" } },
    BuiltinExample { id: "negative-half", title: "Evaluate a negative noninteger", program: "Y = gamma(-0.5)", display_output: Some("Y = -3.5449"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(Y + 2*sqrt(pi)) < 1e-11);" } },
    BuiltinExample { id: "array", title: "Evaluate an array element by element", program: "X = [1 2; 3 4];\nY = gamma(X)", display_output: Some("Y = [1 1; 2 6]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(max(abs(Y(:) - [1; 2; 1; 6])) < 1e-12);" } },
    BuiltinExample { id: "single", title: "Preserve single precision", program: "Y = gamma(single([0.5 1.5 2.5]))", display_output: Some("Y is a single row vector"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(Y, 'single'));\nassert(max(abs(double(Y) - [sqrt(pi) sqrt(pi)/2 3*sqrt(pi)/4])) < 2e-6);" } },
    BuiltinExample { id: "gpu-residency", title: "Evaluate gamma on resident input", program: "G = gpuArray([0.5 1.5 2.5]);\nGy = gamma(G);\nY = gather(Gy)", display_output: Some("Y = [1.7725 0.8862 1.3293]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(Gy, 'gpuArray'));\nassert(max(abs(Y - [sqrt(pi) sqrt(pi)/2 3*sqrt(pi)/4])) < 1e-6);" } },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "How is gamma related to factorial?", answer: "For each positive integer n, `gamma(n) = (n-1)!`, or equivalently `gamma(n+1) = factorial(n)`." },
    BuiltinDocumentationFaq { question: "What happens at zero and negative integers?", answer: "Those values are poles of the gamma function and produce infinities according to the real implementation's pole convention." },
    BuiltinDocumentationFaq { question: "Does gamma accept fixed-width integers?", answer: "No. Convert deliberately with `double` or `single`. An ordinary numeric literal such as `5` already has class double." },
    BuiltinDocumentationFaq { question: "Does gamma accept complex input?", answer: "No. This builtin implements the documented real single/double surface." },
    BuiltinDocumentationFaq { question: "What precision is returned?", answer: "Single input returns single; double input returns double. Shape is preserved." },
    BuiltinDocumentationFaq { question: "Can results remain provider-resident?", answer: "Yes. A valid native result remains resident, and unsupported hooks use exact-owner gather and restoration." },
];

const RELATED: &[&str] = &[
    "gammaln",
    "factorial",
    "log",
    "exp",
    "sqrt",
    "gpuArray",
    "gather",
    "abs",
    "angle",
    "conj",
    "double",
    "expm1",
    "hypot",
    "imag",
    "ldivide",
    "log10",
    "log1p",
    "log2",
    "minus",
    "plus",
    "pow2",
    "power",
    "rdivide",
    "real",
    "sign",
    "single",
    "times",
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "factorial", target: BuiltinDocumentationLinkTarget::Builtin("factorial") },
    BuiltinDocumentationLink { label: "gammaln", target: BuiltinDocumentationLinkTarget::Builtin("gammaln") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/gamma.rs") },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Real gamma runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/gamma.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Real values, poles, shapes, classes, errors, and rejection boundaries", location: "crates/runmat-runtime/src/builtins/math/elementwise/gamma.rs::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Exact-owner fallback preserves class, shape, and residency", location: "crates/runmat-runtime/src/builtins/math/elementwise/gamma.rs::tests::gpu_provider_roundtrip_preserves_residency" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU direct-execution parity", location: "crates/runmat-runtime/src/builtins/math/elementwise/gamma.rs::tests::wgpu_gamma_matches_host_for_real_inputs" },
    ],
    notes: &[],
};

pub(super) const GAMMA_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("gamma"), slug: Some("gamma"),
    summary: "Evaluate the real Euler gamma function element by element.",
    description: "`gamma` evaluates dense real single or double values, preserves their class and shape, and supports validated provider-resident execution.",
    keywords: &["gamma", "factorial", "special function", "poles", "single", "double", "gpu"],
    related: RELATED, sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: FAQS,
    links: LINKS, media: &[], evidence: EVIDENCE,
    introduced: Some("Before R2006a"), status: Some(BuiltinDocumentationStatus::Stable),
};
