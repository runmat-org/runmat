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
            "`asin(X)` computes the principal inverse sine of each element in radians. Real values in `[-1, 1]` return real results. A real value outside that interval produces a complex result; if any element of a real array requires promotion, the whole result uses complex storage.",
            "Real and complex `single` input returns single-precision storage, while real and complex `double` input returns double precision. The operation preserves scalar, vector, matrix, empty, and N-D shape. Complex input follows the principal branch of the analytic inverse sine. NaN and infinity follow the corresponding floating and complex arithmetic.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "RunMat extensions",
        paragraphs: &[
            "RunMat mode accepts all eight real fixed-width integer classes, logical arrays, and character arrays. These forms cross an explicit binary64 computation boundary and return double or complex double. Character elements are interpreted as Unicode scalar values; values above one therefore produce complex results.",
            "Real provider-resident input that requires a complex result is also a RunMat extension. MATLAB compatibility modes require potentially complex resident input to be explicitly complex. Strings, sparse arrays, typed complex integers, direct table/timetable overloads, and tall containers are rejected. Execution-owned distributed arrays use the catalog's unary mapping policy.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "GPU execution and fusion",
        paragraphs: &[
            "For floating real input, RunMat checks the domain with provider reductions. Values proven to remain in `[-1, 1]` can use the provider's unary inverse-sine operation and stay resident. Missing hooks and results that require complex promotion gather through the input's owner, compute on the host, and return to that same owner and device.",
            "Fusion is disabled because a real-only shader expression would return `NaN` where the language requires complex promotion. Provider execution remains available for calls whose runtime domain probe proves that every value is in `[-1, 1]`.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "scalar",
        title: "Compute inverse sine of a scalar",
        program: "y = asin(0.5)",
        display_output: Some("y = 0.5236"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(y - pi/6) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "complex-promotion",
        title: "Promote real input outside the unit interval",
        program: "z = asin(1.2)",
        display_output: Some("z = 1.5708 - 0.6224i"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(real(z) - pi/2) < 1e-12);\nassert(abs(imag(z) + 0.622362503714779) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "matrix",
        title: "Apply inverse sine elementwise",
        program: "A = [0 -0.5 0.75; 1 0.25 -0.8];\nY = asin(A)",
        display_output: Some(
            "Y =\n    0          -0.5236    0.8481\n    1.5708      0.2527   -0.9273",
        ),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(Y), size(A)));\nassert(max(abs(sin(Y(:)) - A(:))) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "logical-extension",
        title: "Evaluate logical values in RunMat mode",
        program: "mask = logical([0 1 1 0]);\nangles = asin(mask)",
        display_output: Some("angles = [0 1.5708 1.5708 0]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(angles, \"double\"));\nassert(isequal(angles, [0 pi/2 pi/2 0]));",
        },
    },
    BuiltinExample {
        id: "provider-resident",
        title: "Keep an in-domain result on its provider",
        program: "G = gpuArray(linspace(-1, 1, 5));\nresultGpu = asin(G);\nresult = gather(resultGpu)",
        display_output: Some("result = [-1.5708 -0.5236 0 0.5236 1.5708]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = asin(linspace(-1, 1, 5));\nassert(isa(resultGpu, \"gpuArray\"));\nassert(max(abs(result - expected)) < 1e-6);",
        },
    },
    BuiltinExample {
        id: "complex-input",
        title: "Evaluate complex input on the principal branch",
        program: "values = [1+2i, -0.5+0.75i];\nw = asin(values)",
        display_output: Some("w is a 1-by-2 complex row vector"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(w), [1 2]));\nassert(max(abs(sin(w) - values)) < 1e-12);",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Why does asin sometimes return complex numbers?",
        answer: "The principal inverse sine is defined on the complex plane. A real input outside `[-1, 1]` has no real inverse-sine value, so the result is complex.",
    },
    BuiltinDocumentationFaq {
        question: "Does asin support GPU execution?",
        answer: "Yes. In-domain floating input can use a provider's inverse-sine and reduction operations. Other supported resident forms use owner-preserving host fallback.",
    },
    BuiltinDocumentationFaq {
        question: "How does asin treat logical or integer input?",
        answer: "RunMat mode accepts them as explicit extensions. Their values convert to double at the inverse-sine algorithm boundary, and the result is double or complex double.",
    },
    BuiltinDocumentationFaq {
        question: "What happens with NaN or Inf?",
        answer: "NaN propagates. Positive and negative infinity produce the principal complex results defined by the floating complex implementation.",
    },
    BuiltinDocumentationFaq {
        question: "Can a complex result remain provider-resident?",
        answer: "In RunMat mode, yes. The runtime gathers real input that needs promotion, computes the complex result on the host, and uploads it to the original owner and device.",
    },
    BuiltinDocumentationFaq {
        question: "Why does asin of a character array return numeric data?",
        answer: "Character input is a RunMat extension that applies inverse sine to each Unicode scalar value. The output is therefore double or complex double rather than character data.",
    },
    BuiltinDocumentationFaq {
        question: "Does asin fuse with surrounding operations?",
        answer: "No. The current fusion contract keeps `asin` as a boundary because real input can require complex output. In-domain provider execution is still available outside fusion.",
    },
    BuiltinDocumentationFaq {
        question: "Can CPU and GPU results differ slightly?",
        answer: "They can differ within the normal error of the selected floating precision and provider implementation, especially near the domain endpoints.",
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "sin",
        target: BuiltinDocumentationLinkTarget::Builtin("sin"),
    },
    BuiltinDocumentationLink {
        label: "acos",
        target: BuiltinDocumentationLinkTarget::Builtin("acos"),
    },
    BuiltinDocumentationLink {
        label: "atan",
        target: BuiltinDocumentationLinkTarget::Builtin("atan"),
    },
    BuiltinDocumentationLink {
        label: "gpuArray",
        target: BuiltinDocumentationLinkTarget::Builtin("gpuArray"),
    },
    BuiltinDocumentationLink {
        label: "gather",
        target: BuiltinDocumentationLinkTarget::Builtin("gather"),
    },
    BuiltinDocumentationLink {
        label: "acosh",
        target: BuiltinDocumentationLinkTarget::Builtin("acosh"),
    },
    BuiltinDocumentationLink {
        label: "asinh",
        target: BuiltinDocumentationLinkTarget::Builtin("asinh"),
    },
    BuiltinDocumentationLink {
        label: "atan2",
        target: BuiltinDocumentationLinkTarget::Builtin("atan2"),
    },
    BuiltinDocumentationLink {
        label: "atanh",
        target: BuiltinDocumentationLinkTarget::Builtin("atanh"),
    },
    BuiltinDocumentationLink {
        label: "cos",
        target: BuiltinDocumentationLinkTarget::Builtin("cos"),
    },
    BuiltinDocumentationLink {
        label: "cosh",
        target: BuiltinDocumentationLinkTarget::Builtin("cosh"),
    },
    BuiltinDocumentationLink {
        label: "sinh",
        target: BuiltinDocumentationLinkTarget::Builtin("sinh"),
    },
    BuiltinDocumentationLink {
        label: "tan",
        target: BuiltinDocumentationLinkTarget::Builtin("tan"),
    },
    BuiltinDocumentationLink {
        label: "tanh",
        target: BuiltinDocumentationLinkTarget::Builtin("tanh"),
    },
    BuiltinDocumentationLink {
        label: "Implementation",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/asin.rs",
        ),
    },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Inverse-sine runtime",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/asin.rs",
        ),
    }],
    verification: &[
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Real, complex, typed, extension, shape, and error behavior",
            location: "crates/runmat-runtime/src/builtins/math/trigonometry/asin.rs::tests",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::ProviderTest,
            label: "Resident unary execution and owner-preserving complex fallback",
            location: "crates/runmat-runtime/src/builtins/math/trigonometry/asin.rs::tests::asin_gpu_provider_roundtrip",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::WgpuTest,
            label: "Actual WGPU inverse-sine parity",
            location: "crates/runmat-runtime/src/builtins/math/trigonometry/asin.rs::tests::asin_wgpu_matches_cpu_elementwise",
        },
    ],
    notes: &[],
};

pub(crate) const ASIN_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("asin"),
    slug: Some("asin"),
    summary: "Compute elementwise principal inverse sine in radians with value-dependent complex promotion.",
    description: "`asin` preserves floating precision and shape, promotes out-of-domain real values to the principal complex result, and retains supported provider residency.",
    keywords: &[
        "asin",
        "inverse sine",
        "arcsine",
        "trigonometry",
        "complex",
        "principal branch",
        "elementwise",
        "gpu",
    ],
    related: &[
        "sin", "acos", "atan", "gpuArray", "gather", "acosh", "asinh", "atan2", "atanh",
        "cos", "cosh", "sinh", "tan", "tanh",
    ],
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
