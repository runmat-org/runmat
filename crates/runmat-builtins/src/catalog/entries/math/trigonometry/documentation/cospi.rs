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
            "`cospi(X)` evaluates `cos(pi*X)` without first rounding the product `pi*X`. Integer values return exact positive or negative one, and odd half-integers return exact zero. Scalar, vector, matrix, empty, and N-D shapes are preserved; non-finite real inputs produce `NaN`.",
            "Real and complex `single` inputs return `single`; real and complex `double` inputs return `double`. Complex values use the analytic continuation of `cos(pi*z)` while retaining exact real-axis sine and cosine factors where possible.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "RunMat extensions",
        paragraphs: &[
            "In `runmat` compatibility mode, all eight real fixed-width integer classes return exact shape-preserving double results computed directly from native parity. This includes `int64` and `uint64` values that cannot be represented exactly as binary64.",
            "Logical and character arrays are also RunMat extensions. Logical values and Unicode character code points produce shape-preserving `double` results. Strings, sparse arrays, typed complex integers, tables, timetables, and tall containers are rejected. Execution-owned distributed arrays use the catalog's elementwise mapping contract rather than entering this host value adapter directly.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution placement",
        paragraphs: &[
            "Provider-resident input gathers through its owning provider and runs through the host implementation. RunMat uploads the result to that same provider, preserving the input's residency and owner. The host boundary preserves exact integer and half-integer behavior instead of lowering the operation to an approximate `cos(X*pi)` provider expression.",
            "Fusion is disabled for the same reason: replacing `cospi(X)` with multiplication followed by ordinary cosine would change its exactness contract.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "integer-and-half-integer",
        title: "Evaluate integer and half-integer multiples",
        program: "Y = cospi([0 1/2 1 3/2 2])",
        display_output: Some("Y = [1 0 -1 0 1]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(Y, [1 0 -1 0 1]));",
        },
    },
    BuiltinExample {
        id: "avoid-product-drift",
        title: "Avoid product rounding at a half-integer",
        program: "exactValue = cospi(1/2);\napproximateValue = cos(pi/2)",
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
        program: "values = [0x0020000000000000u64 0xFFFFFFFFFFFFFFFFu64];\nY = cospi(values)",
        display_output: Some("Y = [1 -1]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(Y, \"double\"));\nassert(isequal(Y, [1 -1]));",
        },
    },
    BuiltinExample {
        id: "complex-single",
        title: "Evaluate complex single input",
        program: "z = complex(single(0.5), single(1));\nY = cospi(z)",
        display_output: Some("Y is a complex single scalar"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(Y, \"single\"));\nassert(real(Y) == 0);\nassert(abs(double(imag(Y)) + sinh(pi)) < 1e-5);",
        },
    },
    BuiltinExample {
        id: "provider-restoration",
        title: "Restore exact results to the input provider",
        program: "G = gpuArray([0 0.5 1]);\nY = cospi(G);\nhostY = gather(Y)",
        display_output: Some("hostY = [1 0 -1]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(Y, \"gpuArray\"));\nassert(isequal(hostY, [1 0 -1]));",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Why use cospi instead of cos(X*pi)?", answer: "`pi` is not exactly representable in binary floating point. `cospi` handles integer and half-integer arguments directly, avoiding product-rounding noise in common exact results." },
    BuiltinDocumentationFaq { question: "Does cospi preserve single precision?", answer: "Yes. Real and complex `single` input returns `single`; corresponding `double` input returns `double`." },
    BuiltinDocumentationFaq { question: "Can cospi accept integers?", answer: "All eight real fixed-width integer classes are accepted in RunMat mode and return exact double results from native parity without conversion through binary64." },
    BuiltinDocumentationFaq { question: "Does cospi support complex input?", answer: "Yes for floating complex input. It evaluates the analytic continuation and preserves the floating class." },
    BuiltinDocumentationFaq { question: "Does cospi run on a provider?", answer: "Provider-resident input gathers through its owner for exact host evaluation, and the result is uploaded back to the same provider." },
    BuiltinDocumentationFaq { question: "Why is cospi not fused?", answer: "Lowering it to multiplication by `pi` followed by ordinary cosine would lose the exact integer and half-integer guarantees." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "MATLAB cospi documentation",
        target: BuiltinDocumentationLinkTarget::External(
            "https://www.mathworks.com/help/matlab/ref/double.cospi.html",
        ),
    },
    BuiltinDocumentationLink {
        label: "GPU execution",
        target: BuiltinDocumentationLinkTarget::Documentation("/docs/runtime/gpu"),
    },
    BuiltinDocumentationLink {
        label: "sinpi",
        target: BuiltinDocumentationLinkTarget::Builtin("sinpi"),
    },
    BuiltinDocumentationLink {
        label: "cos",
        target: BuiltinDocumentationLinkTarget::Builtin("cos"),
    },
    BuiltinDocumentationLink {
        label: "cosd",
        target: BuiltinDocumentationLinkTarget::Builtin("cosd"),
    },
    BuiltinDocumentationLink {
        label: "Implementation",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/cospi.rs"),
    },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Exact pi-scaled cosine runtime",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/cospi.rs"),
    }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Exact, typed, complex, and error behavior", location: "crates/runmat-runtime/src/builtins/math/trigonometry/cospi.rs::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Owner-directed gather and restoration", location: "crates/runmat-runtime/src/builtins/math/trigonometry/cospi.rs::tests::gpu_fallback_restores_output_to_owner" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled wide-integer parity", location: "crates/runmat-vm/tests/integer_cosine_semantics.rs::compiled_cospi_keeps_wide_uint64_parity_exact" },
    ],
    notes: &[],
};

pub(crate) const COSPI_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("cospi"),
    slug: Some("cospi"),
    summary: "Compute cos(pi*X) with exact integer and half-integer results.",
    description:
        "`cospi` evaluates pi-scaled cosine without first materializing the rounded product `pi*X`.",
    keywords: &[
        "cospi",
        "cosine",
        "pi",
        "trigonometry",
        "exact",
        "elementwise",
        "gpu",
    ],
    related: &["sinpi", "cos", "cosd", "gpuArray", "gather"],
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
