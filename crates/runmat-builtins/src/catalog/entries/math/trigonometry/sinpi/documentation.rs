use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`sinpi(X)` evaluates `sin(pi*X)` without first rounding the product `pi*X`. Integer values return exact zero, and odd half-integers return exact positive or negative one. Scalar, vector, matrix, empty, and N-D shapes are preserved; non-finite real inputs produce `NaN`.",
            "Real and complex `single` inputs return `single`; real and complex `double` inputs return `double`. Complex values use the analytic continuation of `sin(pi*z)` while retaining exact real-axis sine and cosine factors where possible.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "RunMat extensions",
        paragraphs: &[
            "In `runmat` compatibility mode, all eight real fixed-width integer classes return exact shape-preserving double zeros directly from their native storage. This includes `int64` and `uint64` values that cannot be represented exactly as binary64.",
            "Logical and character arrays are also RunMat extensions. Logical values and Unicode character code points produce shape-preserving `double` results. Strings, sparse arrays, and typed complex integers are rejected.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution placement",
        paragraphs: &[
            "Provider-resident input gathers through its owning provider and runs through the host implementation. The result is host-resident. This deliberate boundary preserves exact integer and half-integer behavior instead of lowering the operation to an approximate `sin(X*pi)` provider expression.",
            "Fusion is disabled for the same reason: replacing `sinpi(X)` with multiplication followed by ordinary sine would change its exactness contract. Manual `gpuArray` input remains accepted for compatibility, but this implementation does not launch a provider kernel.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "integer-and-half-integer",
        title: "Evaluate integer and half-integer multiples",
        program: "Y = sinpi([0 1/2 1 3/2 2])",
        display_output: Some("Y = [0 1 0 -1 0]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(Y, [0 1 0 -1 0]));",
        },
    },
    BuiltinExample {
        id: "avoid-product-drift",
        title: "Avoid product rounding at an integer multiple",
        program: "exactValue = sinpi(1);\napproximateValue = sin(pi)",
        display_output: Some("exactValue = 0; approximateValue is close to zero"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(exactValue == 0);\nassert(approximateValue ~= 0);",
        },
    },
    BuiltinExample {
        id: "wide-integer-extension",
        title: "Evaluate wide integers without floating conversion",
        program: "values = [0x0020000000000001u64 0xFFFFFFFFFFFFFFFFu64];\nY = sinpi(values)",
        display_output: Some("Y = [0 0]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(Y, \"double\"));\nassert(isequal(Y, [0 0]));",
        },
    },
    BuiltinExample {
        id: "complex-single",
        title: "Evaluate complex single input",
        program: "z = complex(single(0.5), single(1));\nY = sinpi(z)",
        display_output: Some("Y is a complex single scalar"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = complex(cosh(pi), 0);\nassert(abs(double(Y) - expected) < 1e-4);",
        },
    },
    BuiltinExample {
        id: "provider-gather",
        title: "Gather provider input for exact host evaluation",
        program: "G = gpuArray([0 0.5 1]);\nY = sinpi(G)",
        display_output: Some("Y = [0 1 0] on the host"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(~isa(Y, \"gpuArray\"));\nassert(isequal(Y, [0 1 0]));",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Why use sinpi instead of sin(X*pi)?", answer: "`pi` is not exactly representable in binary floating point. `sinpi` handles integer and half-integer arguments directly, avoiding product-rounding noise in common exact results." },
    BuiltinDocumentationFaq { question: "Does sinpi preserve single precision?", answer: "Yes. Real and complex `single` input returns `single`; corresponding `double` input returns `double`." },
    BuiltinDocumentationFaq { question: "Can sinpi accept integers?", answer: "All eight real fixed-width integer classes are accepted in RunMat mode and return exact double zeros without conversion through binary64." },
    BuiltinDocumentationFaq { question: "Does sinpi support complex input?", answer: "Yes for floating complex input. It evaluates the analytic continuation and preserves the floating class." },
    BuiltinDocumentationFaq { question: "Does sinpi run on a provider?", answer: "Provider-resident input is supported, but currently gathers through its owner for exact host evaluation and returns a host value." },
    BuiltinDocumentationFaq { question: "Why is sinpi not fused?", answer: "Lowering it to multiplication by `pi` followed by ordinary sine would lose the exact integer and half-integer guarantees." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "GPU execution",
        target: BuiltinDocumentationLinkTarget::Documentation("/docs/runtime/gpu"),
    },
    BuiltinDocumentationLink {
        label: "Implementation",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/sinpi.rs"),
    },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Exact pi-scaled sine runtime",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/sinpi.rs"),
    }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Exact, typed, complex, and error behavior", location: "crates/runmat-runtime/src/builtins/math/trigonometry/sinpi.rs::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Owner-directed provider gather", location: "crates/runmat-runtime/src/builtins/math/trigonometry/sinpi.rs::tests::gpu_input_is_gathered" },
    ],
    notes: &[],
};

pub(crate) const SINPI_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("sinpi"),
    slug: Some("sinpi"),
    summary: "Compute sin(pi*X) with exact integer and half-integer results.",
    description:
        "`sinpi` evaluates pi-scaled sine without first materializing the rounded product `pi*X`.",
    keywords: &[
        "sinpi",
        "sine",
        "pi",
        "trigonometry",
        "exact",
        "elementwise",
        "gpu",
    ],
    related: &["cospi", "sin", "sind", "gpuArray", "gather"],
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
