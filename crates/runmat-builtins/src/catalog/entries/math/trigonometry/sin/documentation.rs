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
            "`sin(X)` evaluates the sine of each element in radians. Scalar, vector, matrix, empty, and N-D shapes are preserved. Real and complex `single` inputs return `single`; real and complex `double` inputs return `double`.",
            "Complex inputs use `sin(a + bi) = sin(a)cosh(b) + i cos(a)sinh(b)`. The operation retains complex output and propagates non-finite components according to floating-point arithmetic.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "RunMat extensions",
        paragraphs: &[
            "In `runmat` compatibility mode, real fixed-width integers, logical arrays, and character arrays are accepted. Integer values must be exactly representable at the binary64 transcendental boundary and return `double`; logical values and Unicode character code points also return shape-preserving `double` results.",
            "The `sin(X, \"like\", P)` form is also a RunMat extension. The prototype selects host or provider residency and real or complex representation. It does not silently override the computation class established by `X`.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution placement",
        paragraphs: &[
            "When acceleration is available, RunMat may place eligible work on the active provider without an explicit `gpuArray` call. Real floating provider-resident inputs use the provider's unary sine operation when available, and the fusion planner can combine `sin` with neighboring elementwise operations to avoid intermediate transfers.",
            "Unsupported provider representations gather through their owning provider. The host result returns to the source provider when that provider can represent its class; an explicit `\"like\"` prototype instead applies the requested host/provider and real/complex representation after evaluation. Manual `gpuArray` and `gather` calls remain available when a program needs an explicit residency boundary.",
            "Sparse, string, table, timetable, tall, and unsupported object inputs are not accepted by this implementation.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "scalar",
        title: "Compute the sine of a scalar",
        program: "y = sin(pi / 2)",
        display_output: Some("y = 1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(y - 1) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "vector",
        title: "Evaluate several angles in radians",
        program: "angles = [0 pi/6 pi/4 pi/3];\nvalues = sin(angles)",
        display_output: Some("values = [0 0.5000 0.7071 0.8660]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = [0 0.5 sqrt(0.5) sqrt(3)/2];\nassert(max(abs(values - expected)) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "complex",
        title: "Evaluate a complex angle",
        program: "z = sin(1 + 2i)",
        display_output: Some("z = 3.1658 + 1.9596i"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = sin(1) * cosh(2) + 1i * cos(1) * sinh(2);\nassert(abs(z - expected) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "gpu-matrix",
        title: "Evaluate a single-precision matrix on a provider",
        program: "G = gpuArray(single([0 pi/2; pi 3*pi/2]));\nresult = gather(sin(G))",
        display_output: Some("result is a 2-by-2 single matrix"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(result, \"single\"));\nassert(max(abs(double(result(:)) - [0; 0; 1; -1])) < 1e-5);",
        },
    },
    BuiltinExample {
        id: "gpu-like",
        title: "Request provider residency with a prototype",
        program: "prototype = gpuArray(single(0));\ndeviceResult = sin(single([0 pi/2]), \"like\", prototype);\nresult = gather(deviceResult)",
        display_output: Some("result = single([0 1])"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(result, \"single\"));\nassert(max(abs(double(result) - [0 1])) < 1e-5);",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "When should I use sin?", answer: "Use `sin` for elementwise periodic calculations in signals, controls, geometry, simulation, and other numerical work expressed in radians." },
    BuiltinDocumentationFaq { question: "Does sin use radians or degrees?", answer: "`sin` uses radians. Use `sind` for degree input or convert with `deg2rad`." },
    BuiltinDocumentationFaq { question: "Does sin preserve single precision?", answer: "Yes. Real and complex `single` inputs return `single`; `double` inputs return `double`." },
    BuiltinDocumentationFaq { question: "Does sin support complex values?", answer: "Yes. Floating complex values use the analytic sine and preserve their floating class." },
    BuiltinDocumentationFaq { question: "Can sin accept integer input?", answer: "In RunMat compatibility mode, all eight real integer classes are accepted when every value is exactly representable as binary64. The result is `double`." },
    BuiltinDocumentationFaq { question: "Can sin accept logical or character arrays?", answer: "Yes in RunMat compatibility mode. Logical values and Unicode character code points produce shape-preserving `double` results." },
    BuiltinDocumentationFaq { question: "What if a provider lacks unary sine?", answer: "RunMat gathers through the owning provider, evaluates the host implementation, and applies any explicit `\"like\"` placement request." },
    BuiltinDocumentationFaq { question: "What does the like prototype control?", answer: "The RunMat-only form controls host/provider residency and real/complex representation. The input determines the computation class." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "GPU execution",
        target: BuiltinDocumentationLinkTarget::Documentation("/docs/runtime/gpu"),
    },
    BuiltinDocumentationLink {
        label: "Implementation",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/sin.rs"),
    },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Sine runtime and provider fallback",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/sin.rs"),
    }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Host, complex, integer, character, and prototype behavior", location: "crates/runmat-runtime/src/builtins/math/trigonometry/sin.rs::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Provider ownership and fallback behavior", location: "crates/runmat-runtime/src/builtins/math/trigonometry/sin.rs provider tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "WGPU elementwise parity", location: "crates/runmat-runtime/src/builtins/math/trigonometry/sin.rs::tests::sin_wgpu_matches_cpu_elementwise" },
    ],
    notes: &[],
};

pub(crate) const SIN_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("sin"),
    slug: Some("sin"),
    summary: "Compute elementwise sine values in radians.",
    description: "`sin` evaluates the radian sine of supported real or complex values while preserving shape and floating-point class.",
    keywords: &["sin", "sine", "trigonometry", "radians", "gpu", "elementwise"],
    related: &[
        "cos", "tan", "sind", "deg2rad", "gpuArray", "gather", "acos", "acosh", "asin",
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
