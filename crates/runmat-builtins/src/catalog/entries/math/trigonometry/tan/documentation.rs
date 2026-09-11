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
            "`tan(X)` evaluates the tangent of each element in radians. Scalar, vector, matrix, empty, and N-D shapes are preserved. Real and complex `single` inputs return `single`; real and complex `double` inputs return `double`.",
            "Complex inputs use `tan(a + bi) = sin(2a)/(cos(2a) + cosh(2b)) + i sinh(2b)/(cos(2a) + cosh(2b))`. Values near odd multiples of `pi/2` can have very large magnitudes because floating-point `pi/2` is an approximation to a pole; non-finite components follow floating-point arithmetic.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "RunMat extensions",
        paragraphs: &[
            "In `runmat` compatibility mode, real fixed-width integers, logical arrays, and character arrays are accepted. Integer values must be exactly representable at the binary64 transcendental boundary and return `double`; logical values and Unicode character code points also return shape-preserving `double` results.",
            "The `tan(X, \"like\", P)` form is also a RunMat extension. The prototype selects host or provider residency and real or complex representation. It does not silently override the computation class established by `X`.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution placement",
        paragraphs: &[
            "When acceleration is available, RunMat may place eligible work on the active provider without an explicit `gpuArray` call. Real floating provider-resident inputs use the provider's unary tangent operation when available, and the fusion planner can combine `tan` with neighboring elementwise operations to avoid intermediate transfers.",
            "Unsupported provider representations gather through their owning provider. The host result returns to the source provider when that provider can represent its class; an explicit `\"like\"` prototype instead applies the requested host/provider and real/complex representation after evaluation. Manual `gpuArray` and `gather` calls remain available when a program needs an explicit residency boundary.",
            "Sparse, string, table, timetable, tall, and unsupported object inputs are not accepted by this implementation.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "quarter-turn",
        title: "Compute the tangent of pi over four",
        program: "y = tan(pi/4)",
        display_output: Some("y = 1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(y - 1) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "vector",
        title: "Evaluate a vector away from the poles",
        program: "theta = linspace(-pi/2 + 0.1, pi/2 - 0.1, 5);\nwave = tan(theta)",
        display_output: Some("wave = [-9.9666 -0.9047 0 0.9047 9.9666]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(wave), [1 5]));\nassert(max(abs(wave + fliplr(wave))) < 1e-12);\nassert(abs(wave(3)) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "gpu-matrix",
        title: "Evaluate a matrix on a provider",
        program: "G = gpuArray([0 pi/6; pi/4 pi/3]);\nT = tan(G);\nresult = gather(T)",
        display_output: Some("result = [0 0.5774; 1.0000 1.7321]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = [0 1/sqrt(3); 1 sqrt(3)];\nassert(max(abs(result(:) - expected(:))) < 1e-10);",
        },
    },
    BuiltinExample {
        id: "complex",
        title: "Evaluate a complex angle",
        program: "z = 1 + 0.5i;\ntz = tan(z)",
        display_output: Some("tz is a complex scalar"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "denominator = cos(2) + cosh(1);\nexpected = sin(2)/denominator + 1i*sinh(1)/denominator;\nassert(abs(tz - expected) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "small-angle",
        title: "Compare tangent with a small-angle approximation",
        program: "angles = [-1e-6 0 1e-6];\napprox = tan(angles)",
        display_output: Some("approx = [-1e-6 0 1e-6] to displayed precision"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(max(abs(approx - angles)) < 1e-15);",
        },
    },
    BuiltinExample {
        id: "degrees",
        title: "Convert degrees before evaluating tangent",
        program: "anglesInDegrees = [0 30 60 89];\nradians = deg2rad(anglesInDegrees);\nresult = tan(radians)",
        display_output: Some("result = [0 0.5774 1.7321 57.2900]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(result(1)) < 1e-12);\nassert(abs(result(2) - 1/sqrt(3)) < 1e-12);\nassert(abs(result(3) - sqrt(3)) < 1e-12);\nassert(result(4) > 57);",
        },
    },
    BuiltinExample {
        id: "gpu-like",
        title: "Request provider-resident single output",
        program: "prototype = gpuArray(single(0));\nangles = gpuArray(single([0 pi/6 pi/4]));\ndeviceResult = tan(angles, \"like\", prototype);\nresult = gather(deviceResult)",
        display_output: Some("result = single([0 0.5774 1.0000])"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(result, \"single\"));\nassert(max(abs(double(result) - [0 1/sqrt(3) 1])) < 1e-5);",
        },
    },
    BuiltinExample {
        id: "characters",
        title: "Evaluate Unicode character code points",
        program: "codes = tan('ABC')",
        display_output: Some("codes is a 1-by-3 double array"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = tan(double('ABC'));\nassert(isa(codes, \"double\"));\nassert(isequal(size(codes), [1 3]));\nassert(max(abs(codes - expected)) < 1e-12);",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "When should I use tan?", answer: "Use `tan` for elementwise tangent calculations on angles expressed in radians." },
    BuiltinDocumentationFaq { question: "What happens near odd multiples of pi over two?", answer: "Those angles are poles of tangent. Floating-point inputs near a pole produce large finite magnitudes or non-finite results according to the represented value." },
    BuiltinDocumentationFaq { question: "Does tan support complex values?", answer: "Yes. Floating complex values use the analytic tangent and preserve their floating class." },
    BuiltinDocumentationFaq { question: "Can tan keep its result on a provider?", answer: "Yes. A provider unary operation or fused kernel can keep the result resident. Unsupported hooks gather through the owner, compute on the host, and restore the declared placement." },
    BuiltinDocumentationFaq { question: "Does tan use radians or degrees?", answer: "`tan` uses radians. Convert degree input with `deg2rad`, or use `tand` where appropriate." },
    BuiltinDocumentationFaq { question: "Does tan preserve single precision?", answer: "Yes. Real and complex `single` inputs return `single`; `double` inputs return `double`. RunMat-only integer, logical, and character inputs return `double`." },
    BuiltinDocumentationFaq { question: "Can tan accept logical or character arrays?", answer: "Yes in RunMat compatibility mode. Logical values and Unicode character code points produce shape-preserving `double` results." },
    BuiltinDocumentationFaq { question: "Can tan be fused with neighboring operations?", answer: "Yes. Eligible elementwise expressions can use a fused provider kernel, avoiding intermediate host transfers and allocations." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "GPU execution",
        target: BuiltinDocumentationLinkTarget::Documentation("/docs/runtime/gpu"),
    },
    BuiltinDocumentationLink {
        label: "Implementation",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/tan.rs"),
    },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Tangent runtime and provider fallback",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/tan.rs"),
    }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Host, complex, integer, character, and prototype behavior", location: "crates/runmat-runtime/src/builtins/math/trigonometry/tan.rs::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Provider ownership and fallback behavior", location: "crates/runmat-runtime/src/builtins/math/trigonometry/tan.rs provider tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "WGPU elementwise parity", location: "crates/runmat-runtime/src/builtins/math/trigonometry/tan.rs::tests::tan_wgpu_matches_cpu_elementwise" },
    ],
    notes: &[],
};

pub(crate) const TAN_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("tan"),
    slug: Some("tan"),
    summary: "Compute elementwise tangent values in radians.",
    description: "`tan` evaluates the radian tangent of supported real or complex values while preserving shape and floating-point class.",
    keywords: &["tan", "tangent", "trigonometry", "radians", "gpu", "elementwise", "like"],
    related: &[
        "sin", "cos", "tand", "deg2rad", "gpuArray", "gather", "acos", "acosh", "asin",
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
