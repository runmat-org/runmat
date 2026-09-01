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
            "`cos(X)` evaluates the cosine of each element in radians. Scalar, vector, matrix, empty, and N-D shapes are preserved. Real and complex `single` inputs return `single`; real and complex `double` inputs return `double`.",
            "Complex inputs use `cos(a + bi) = cos(a)cosh(b) - i sin(a)sinh(b)`. The operation retains complex output and propagates non-finite components according to floating-point arithmetic.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "RunMat extensions",
        paragraphs: &[
            "In `runmat` compatibility mode, real fixed-width integers, logical arrays, and character arrays are accepted. Integer values must be exactly representable at the binary64 transcendental boundary and return `double`; logical values and Unicode character code points also return shape-preserving `double` results.",
            "The `cos(X, \"like\", P)` form is also a RunMat extension. The prototype selects host or provider residency and real or complex representation. It does not silently override the computation class established by `X`.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution placement",
        paragraphs: &[
            "When acceleration is available, RunMat may place eligible work on the active provider without an explicit `gpuArray` call. Real floating provider-resident inputs use the provider's unary cosine operation when available, and the fusion planner can combine `cos` with neighboring elementwise operations to avoid intermediate transfers.",
            "Unsupported provider representations gather through their owning provider. The host result returns to the source provider when that provider can represent its class; an explicit `\"like\"` prototype instead applies the requested host/provider and real/complex representation after evaluation. Manual `gpuArray` and `gather` calls remain available when a program needs an explicit residency boundary.",
            "Sparse, string, table, timetable, tall, and unsupported object inputs are not accepted by this implementation.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "scalar",
        title: "Compute the cosine of zero",
        program: "y = cos(0)",
        display_output: Some("y = 1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(y == 1);",
        },
    },
    BuiltinExample {
        id: "vector",
        title: "Evaluate evenly spaced angles",
        program: "theta = linspace(0, 2*pi, 5);\nvalues = cos(theta)",
        display_output: Some("values = [1 0 -1 0 1]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(max(abs(values - [1 0 -1 0 1])) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "complex",
        title: "Evaluate a complex angle",
        program: "z = cos(1 + 2i)",
        display_output: Some("z = 2.0327 - 3.0519i"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = cos(1) * cosh(2) - 1i * sin(1) * sinh(2);\nassert(abs(z - expected) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "matrix",
        title: "Evaluate a matrix elementwise",
        program: "A = reshape(0:5, [3 2]);\nresult = cos(A)",
        display_output: Some("result is a 3-by-2 matrix"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(result), [3 2]));\nassert(max(abs(result(:) - cos((0:5)'))) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "gpu-like",
        title: "Request provider residency with a prototype",
        program: "prototype = gpuArray(single(0));\ndeviceResult = cos(single([0 pi/2 pi]), \"like\", prototype);\nresult = gather(deviceResult)",
        display_output: Some("result = single([1 0 -1])"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(result, \"single\"));\nassert(max(abs(double(result) - [1 0 -1])) < 1e-5);",
        },
    },
    BuiltinExample {
        id: "host-like",
        title: "Request host output for provider input",
        program: "G = gpuArray([0 1 2]);\nhostResult = cos(G, \"like\", zeros(1, \"double\"))",
        display_output: Some("hostResult = [1 0.5403 -0.4161]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(hostResult, \"double\"));\nassert(max(abs(hostResult - cos([0 1 2]))) < 1e-12);",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "When should I use cos?", answer: "Use `cos` for elementwise cosine calculations on angles expressed in radians." },
    BuiltinDocumentationFaq { question: "Does cos use radians or degrees?", answer: "`cos` uses radians. Use `cosd` for degree input or convert with `deg2rad`." },
    BuiltinDocumentationFaq { question: "Does cos preserve single precision?", answer: "Yes. Real and complex `single` inputs return `single`; `double` inputs return `double`." },
    BuiltinDocumentationFaq { question: "Does cos support complex values?", answer: "Yes. Floating complex values use the analytic cosine and preserve their floating class." },
    BuiltinDocumentationFaq { question: "Can cos accept integer input?", answer: "In RunMat compatibility mode, all eight real integer classes are accepted when every value is exactly representable as binary64. The result is `double`." },
    BuiltinDocumentationFaq { question: "Can cos accept logical or character arrays?", answer: "Yes in RunMat compatibility mode. Logical values and Unicode character code points produce shape-preserving `double` results." },
    BuiltinDocumentationFaq { question: "What if a provider lacks unary cosine?", answer: "RunMat gathers through the owning provider, evaluates the host implementation, and applies any explicit `\"like\"` placement request." },
    BuiltinDocumentationFaq { question: "What does the like prototype control?", answer: "The RunMat-only form controls host/provider residency and real/complex representation. The input determines the computation class." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "GPU execution",
        target: BuiltinDocumentationLinkTarget::Documentation("/docs/runtime/gpu"),
    },
    BuiltinDocumentationLink {
        label: "Implementation",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/cos.rs"),
    },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Cosine runtime and provider fallback",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/cos.rs"),
    }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Host, complex, integer, character, and prototype behavior", location: "crates/runmat-runtime/src/builtins/math/trigonometry/cos.rs::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Provider ownership and fallback behavior", location: "crates/runmat-runtime/src/builtins/math/trigonometry/cos.rs provider tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "WGPU elementwise parity", location: "crates/runmat-runtime/src/builtins/math/trigonometry/cos.rs::tests::cos_wgpu_matches_cpu_elementwise" },
    ],
    notes: &[],
};

pub(crate) const COS_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("cos"),
    slug: Some("cos"),
    summary: "Compute elementwise cosine values in radians.",
    description: "`cos` evaluates the radian cosine of supported real or complex values while preserving shape and floating-point class.",
    keywords: &["cos", "cosine", "trigonometry", "radians", "gpu", "elementwise", "like"],
    related: &[
        "sin", "tan", "cosd", "deg2rad", "gpuArray", "gather", "acos", "acosh", "asin",
        "asinh", "atan", "atan2", "atanh", "cosh", "sinh", "tanh",
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
