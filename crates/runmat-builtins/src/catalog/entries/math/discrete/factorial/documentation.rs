use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Values, classes, and saturation",
        paragraphs: &[
            "`factorial(N)` multiplies the positive integers from 1 through each element of `N`. Input must be dense, real, finite, nonnegative, and integer-valued. Zero returns one; a negative, fractional, infinite, NaN, complex, sparse, or nonnumeric input produces an error.",
            "The result has the same size and numeric class as `N`. Double overflows to positive infinity at 171, single at 35, and each fixed-width integer class saturates at its maximum value when the mathematical result no longer fits.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "RunMat extensions",
        paragraphs: &[
            "RunMat compatibility mode also accepts logical input, returning double values, and provides `factorial(N, \"like\", prototype)`. The prototype selects host or provider residency only; it does not change the class or shape selected from `N`.",
            "A MATLAB compatibility pin rejects both extension forms. The one-input form, including all eight fixed-width integer classes and their saturating result rules, remains compatible.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Accelerated and distributed execution",
        paragraphs: &[
            "Provider-resident real single and double input uses the owning provider's factorial operation when available. The runtime validates the returned shape, storage, precision, owner, device, and non-aliasing before preserving the input's residency intent.",
            "A typed unsupported operation gathers through the exact owner, evaluates on the host, and restores the result when the provider can represent it. Other provider failures remain errors. Compatible GPU and distributed forms exclude 64-bit integer input; RunMat's explicit `like` form can request provider residency for a host result.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "scalar", title: "Compute a scalar factorial", program: "F = factorial(5)", display_output: Some("F = 120"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(F == 120);" } },
    BuiltinExample { id: "array", title: "Evaluate an array element by element", program: "N = [0 1 3; 4 5 6];\nF = factorial(N)", display_output: Some("F = [1 1 6; 24 120 720]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(F, [1 1 6; 24 120 720]));" } },
    BuiltinExample { id: "uint64", title: "Preserve exact uint64 results", program: "N = uint64([5 10 20]);\nF = factorial(N)", display_output: Some("F is a uint64 row vector"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(F, 'uint64'));\nassert(isequal(F, uint64([120 3628800 2432902008176640000])));" } },
    BuiltinExample { id: "integer-saturation", title: "Saturate at an integer class boundary", program: "F = factorial(uint8([5 6]))", display_output: Some("F = uint8([120 255])"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(F, 'uint8'));\nassert(isequal(F, uint8([120 255])));" } },
    BuiltinExample { id: "floating-overflow", title: "Observe floating-point overflow", program: "Fd = factorial(171);\nFs = factorial(single(35));", display_output: Some("Fd and Fs are positive infinity"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isinf(Fd) && Fd > 0);\nassert(isa(Fs, 'single'));\nassert(isinf(Fs) && Fs > 0);" } },
    BuiltinExample { id: "logical-extension", title: "Use logical input in RunMat mode", program: "F = factorial(logical([0 1 1]))", display_output: Some("F = [1 1 1]"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(F, 'double'));\nassert(isequal(F, [1 1 1]));" } },
    BuiltinExample { id: "gpu-input", title: "Keep compatible integer input resident", program: "G = gpuArray(uint16([3 4 5]));\nGf = factorial(G);\nF = gather(Gf)", display_output: Some("F = uint16([6 24 120])"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(Gf, 'gpuArray'));\nassert(isa(F, 'uint16'));\nassert(isequal(F, uint16([6 24 120])));" } },
    BuiltinExample { id: "like-residency", title: "Select output residency in RunMat mode", program: "prototype = gpuArray(0);\nGf = factorial([3 4], 'like', prototype);\nF = gather(Gf)", display_output: Some("F = [6 24]"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Wgpu, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(Gf, 'gpuArray'));\nassert(isa(F, 'double'));\nassert(isequal(F, [6 24]));" } },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Which inputs are valid?", answer: "Dense real single, double, and fixed-width integer values are supported when every element is finite, nonnegative, and integer-valued. RunMat mode also accepts logical values." },
    BuiltinDocumentationFaq { question: "Does factorial preserve integer classes?", answer: "Yes. Each fixed-width integer class is preserved, and a result beyond that class's range saturates at its maximum value." },
    BuiltinDocumentationFaq { question: "Why does factorial(171) return Inf?", answer: "The mathematical value of 171! exceeds the largest finite double. Single precision reaches positive infinity at 35!." },
    BuiltinDocumentationFaq { question: "What happens for a fractional or negative value?", answer: "`factorial` reports an invalid-input error. Use `gamma(X + 1)` when the gamma-function extension is the intended calculation." },
    BuiltinDocumentationFaq { question: "Does the like prototype choose the result class?", answer: "No. RunMat's `like` extension chooses host or provider residency. The input to `factorial` determines the result class and shape." },
    BuiltinDocumentationFaq { question: "Can a provider input remain resident?", answer: "Yes. A valid provider result remains resident, and typed unsupported operations use exact-owner gather and protected restoration when the output representation is supported." },
];

const RELATED: &[&str] = &[
    "gamma", "prod", "intmax", "power", "permute", "gpuArray", "gather",
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "gamma", target: BuiltinDocumentationLinkTarget::Builtin("gamma") },
    BuiltinDocumentationLink { label: "prod", target: BuiltinDocumentationLinkTarget::Builtin("prod") },
    BuiltinDocumentationLink { label: "intmax", target: BuiltinDocumentationLinkTarget::Builtin("intmax") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "Compatible factorial reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/factorial.html") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/discrete/factorial/mod.rs") },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Factorial runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/discrete/factorial/mod.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Domains, classes, saturation, shapes, options, and structured errors", location: "crates/runmat-runtime/src/builtins/math/discrete/factorial/tests.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Exact-owner execution, fallback, output validation, and residency intent", location: "crates/runmat-runtime/src/builtins/math/discrete/factorial/tests.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU floating and integer residency", location: "crates/runmat-runtime/src/builtins/math/discrete/factorial/tests.rs" },
    ],
    notes: &[],
};

pub(super) const FACTORIAL_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("factorial"),
    slug: Some("factorial"),
    summary: "Compute elementwise factorial values while preserving numeric class and shape.",
    description: "`factorial` evaluates nonnegative integer-valued input, preserves floating or fixed-width integer storage, applies documented saturation, and supports validated provider-resident execution.",
    keywords: &[
        "factorial",
        "combinatorics",
        "n!",
        "permutations",
        "integer",
        "saturation",
        "gpu",
        "gpuArray",
        "like",
    ],
    related: RELATED,
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
