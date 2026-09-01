use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Forms and behavior",
        paragraphs: &[
            "`round(X)` rounds floating values to the nearest integer-valued result, with exact halfway cases rounded away from zero. `round(X, N)` rounds to `N` decimal places; negative `N` selects positions to the left of the decimal point. `round(X, N, \"significant\")` retains `N` significant digits and requires positive `N`.",
            "Double and single inputs retain their class and shape. Complex floating values are rounded component by component. `NaN`, positive infinity, and negative infinity propagate unchanged. Logical and character inputs produce double values; characters use their Unicode code points.",
            "All eight real fixed-width integer classes are already integral, so `round(integer_X)` is an exact identity that retains class, shape, bits, and supported residency. Multi-input forms reject integer `X`. In RunMat compatibility mode, a typed-integer `N` is decoded exactly and range-checked without conversion through binary64.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Modes and compatibility",
        paragraphs: &[
            "The documented mode tokens are `\"decimals\"` and `\"significant\"`. Omitting the mode selects decimal-place rounding. RunMat mode also accepts `\"decimal\"` as an explicit convenience alias; MATLAB compatibility mode rejects that alias and typed-integer `N` controls.",
            "`N` must be a finite integer scalar in the supported signed range. Significant-digit rounding additionally requires `N > 0`. Invalid controls fail before data execution.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "GPU execution and fusion",
        paragraphs: &[
            "Resident floating input asks its owning provider to execute `unary_round` or the digit-aware operation. Returned handles must be distinct and preserve shape, storage, precision, owner, and device before RunMat accepts them. A typed unsupported response gathers once, computes at the input precision, and restores the result to the same owner; other provider errors remain visible.",
            "Resident integer input in the one-input form returns the same exact handle. Logical input follows conversion to double. Compatible one-input floating expressions may fuse with neighboring elementwise operations; digit-aware forms remain explicit provider operations rather than using the one-input fusion template.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "nearest",
        title: "Round halfway values away from zero",
        program: "x = [-3.5 -2.2 -0.5 0 0.5 1.7];\ny = round(x)",
        display_output: Some("y = [-4 -2 -1 0 1 2]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(y, [-4 -2 -1 0 1 2]));",
        },
    },
    BuiltinExample {
        id: "decimal-places",
        title: "Round to decimal places",
        program: "values = [21.456 19.995 22.501];\ny = round(values, 2)",
        display_output: Some("y = [21.46 20.00 22.50]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(max(abs(y - [21.46 20 22.5])) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "negative-decimal-places",
        title: "Round to hundreds",
        program: "values = [1234 5678 91011];\ny = round(values, -2)",
        display_output: Some("y = [1200 5700 91000]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(y, [1200 5700 91000]));",
        },
    },
    BuiltinExample {
        id: "significant-digits",
        title: "Round to three significant digits",
        program: "values = [0.001234 12.3456 98765];\ny = round(values, 3, 'significant')",
        display_output: Some("y = [0.00123 12.3 98800]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(max(abs(y - [0.00123 12.3 98800])) < 1e-10);",
        },
    },
    BuiltinExample {
        id: "complex",
        title: "Round complex components independently",
        program: "z = [1.2 - 3.6i, -2.5 + 0.5i];\ny = round(z)",
        display_output: Some("y = [1 - 4i, -3 + 1i]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(y, [1 - 4i, -3 + 1i]));",
        },
    },
    BuiltinExample {
        id: "typed-integer",
        title: "Retain exact wide integers",
        program: "x = uint64([0x0020000000000001u64, 0xFFFFFFFFFFFFFFFFu64]);\ny = round(x)",
        display_output: Some("y retains class uint64 and every input bit"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(y, 'uint64')); assert(isequal(y, x));",
        },
    },
    BuiltinExample {
        id: "typed-digits",
        title: "Use an exact typed-integer digits control",
        program: "x = [1234 5678];\ny = round(x, int8(-2))",
        display_output: Some("y = [1200 5700]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(y, [1200 5700]));",
        },
    },
    BuiltinExample {
        id: "gpu-residency",
        title: "Round resident values and gather the result",
        program: "g = gpuArray([-2.5 -0.5 0.5 2.5]);\ny = round(g);\nresult = gather(y)",
        display_output: Some("result = [-3 -1 1 3]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(result, [-3 -1 1 3]));",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "How are halfway values rounded?", answer: "Halfway values round away from zero: `round(0.5)` is `1`, and `round(-0.5)` is `-1`." },
    BuiltinDocumentationFaq { question: "How do decimal places differ from significant digits?", answer: "`round(X, N)` uses a fixed decimal position. `round(X, N, \"significant\")` chooses the decimal position separately for each magnitude so that `N` significant digits remain." },
    BuiltinDocumentationFaq { question: "Can N be negative?", answer: "Yes in decimal mode, where negative `N` rounds to tens, hundreds, and larger powers of ten. Significant mode requires positive `N`." },
    BuiltinDocumentationFaq { question: "How are complex values handled?", answer: "The real and imaginary components are rounded independently while the floating class and shape are retained." },
    BuiltinDocumentationFaq { question: "What happens to NaN and infinities?", answer: "They propagate unchanged in every mode." },
    BuiltinDocumentationFaq { question: "What happens to typed integer X?", answer: "The one-input form returns it unchanged because every value is already integral. Digit-aware forms reject typed integer X." },
    BuiltinDocumentationFaq { question: "Can N itself be a typed integer?", answer: "Yes in RunMat mode. The scalar is decoded from native integer storage and range-checked exactly. MATLAB compatibility mode restricts this RunMat extension." },
    BuiltinDocumentationFaq { question: "Does provider fallback lose residency?", answer: "No. Unsupported provider hooks use one owner-specific download and restore the computed value to that owner. Other provider failures are reported." },
    BuiltinDocumentationFaq { question: "Can round be fused?", answer: "The one-input floating form is eligible for elementwise fusion. Digit-aware forms use their explicit provider or host operation." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "floor", target: BuiltinDocumentationLinkTarget::Builtin("floor") },
    BuiltinDocumentationLink { label: "ceil", target: BuiltinDocumentationLinkTarget::Builtin("ceil") },
    BuiltinDocumentationLink { label: "fix", target: BuiltinDocumentationLinkTarget::Builtin("fix") },
    BuiltinDocumentationLink { label: "mod", target: BuiltinDocumentationLinkTarget::Builtin("mod") },
    BuiltinDocumentationLink { label: "rem", target: BuiltinDocumentationLinkTarget::Builtin("rem") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/rounding/round.rs") },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Rounding runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/rounding/round.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Forms, controls, classes, complex values, errors, and special values", location: "crates/runmat-runtime/src/builtins/math/rounding/round.rs::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Validated owner-preserving provider execution and fallback", location: "crates/runmat-runtime/src/builtins/math/rounding/round.rs::tests::round_gpu_provider_roundtrip" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU one-input and digit-aware parity", location: "crates/runmat-runtime/src/builtins/math/rounding/round.rs::tests" },
    ],
    notes: &[],
};

pub(crate) const ROUND_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("round"),
    slug: Some("round"),
    summary: "Round values to nearest integers, decimal places, or significant digits.",
    description: "`round` applies nearest-value rounding with explicit decimal-place and significant-digit forms while preserving floating class, shape, and supported residency.",
    keywords: &["round", "rounding", "decimal places", "significant digits", "integers", "complex", "gpu"],
    related: &["floor", "ceil", "fix", "mod", "rem", "gpuArray", "gather"],
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
