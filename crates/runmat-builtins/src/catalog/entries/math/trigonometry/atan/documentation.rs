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
            "`atan(X)` computes the principal inverse tangent of each element in radians. Real input always returns real output in the interval `[-pi/2, pi/2]`; complex floating input follows the principal analytic continuation.",
            "Real and complex `single` input returns single-precision storage, while real and complex `double` input returns double precision. Scalar, vector, matrix, empty, and N-D shape is preserved. NaN propagates and real positive or negative infinity approaches positive or negative `pi/2`.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "RunMat extensions",
        paragraphs: &[
            "RunMat mode accepts all eight real fixed-width integer classes, logical arrays, and character arrays. These forms cross an explicit binary64 computation boundary and return real double output with the input shape. Character elements are interpreted as Unicode scalar values.",
            "RunMat mode also accepts `atan(X, \"like\", P)`. A real or complex host prototype selects the corresponding result representation and host placement; a real provider-resident prototype requests provider placement. A real prototype cannot represent a complex result, and complex provider prototypes are not currently supported. Strings, sparse arrays, typed complex integers, direct table/timetable overloads, and tall containers are rejected.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "GPU execution and fusion",
        paragraphs: &[
            "Supported real floating input can use the owning provider's unary inverse-tangent operation. An unsupported hook gathers through the input owner, computes on the host, and restores the result to that owner and device. Integer and logical resident extensions use the same owner-preserving fallback.",
            "Real inverse tangent has no value-dependent complex promotion, so its real floating expression can participate in elementwise fusion. Complex input and output-template conversion remain ordinary runtime boundaries rather than using that real-only fused expression.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "scalar",
        title: "Compute inverse tangent of a scalar",
        program: "y = atan(1)",
        display_output: Some("y = 0.7854"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(y - pi/4) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "vector",
        title: "Apply inverse tangent elementwise to a vector",
        program: "x = linspace(-2, 2, 5);\nangles = atan(x)",
        display_output: Some("angles = [-1.1071 -0.7854 0 0.7854 1.1071]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(angles), size(x)));\nassert(max(abs(tan(angles) - x)) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "matrix",
        title: "Apply inverse tangent elementwise to a matrix",
        program: "A = [-2 -1 0; 1 2 3];\nY = atan(A)",
        display_output: Some(
            "Y =\n   -1.1071   -0.7854         0\n    0.7854    1.1071    1.2490",
        ),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(Y), size(A)));\nassert(max(abs(tan(Y(:)) - A(:))) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "complex-input",
        title: "Evaluate a complex value on the principal branch",
        program: "z = 1 + 2i;\nw = atan(z)",
        display_output: Some("w = 1.33897 + 0.40236i"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(tan(w) - z) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "provider-like-extension",
        title: "Select provider placement with a prototype",
        program: "prototype = gpuArray(zeros(1, 1));\nG = gpuArray([-3 -1 0 1 3]);\ndeviceResult = atan(G, \"like\", prototype);\nresult = gather(deviceResult)",
        display_output: Some("result = [-1.2490 -0.7854 0 0.7854 1.2490]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = atan([-3 -1 0 1 3]);\nassert(isa(deviceResult, \"gpuArray\"));\nassert(max(abs(result - expected)) < 1e-6);",
        },
    },
    BuiltinExample {
        id: "character-extension",
        title: "Evaluate character code points in RunMat mode",
        program: "codes = atan('RUN')",
        display_output: Some("codes is a 1-by-3 double row vector"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = atan(double('RUN'));\nassert(isa(codes, \"double\"));\nassert(isequal(size(codes), [1 3]));\nassert(max(abs(codes - expected)) < 1e-12);",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "When should I use atan?",
        answer: "Use `atan` for elementwise inverse tangent when one coordinate or ratio is available. Use `atan2(y, x)` when the signs of two coordinates must determine the quadrant.",
    },
    BuiltinDocumentationFaq {
        question: "How is atan different from atan2?",
        answer: "`atan` takes one input and real results lie in `[-pi/2, pi/2]`. `atan2` takes `y` and `x`, resolves the quadrant from both signs, and returns a wider angular range.",
    },
    BuiltinDocumentationFaq {
        question: "Does atan support complex numbers?",
        answer: "Yes. Complex single and double input follow the principal analytic continuation and preserve their floating precision.",
    },
    BuiltinDocumentationFaq {
        question: "Can atan results remain provider-resident?",
        answer: "Yes. Supported real floating input uses the provider hook, while unsupported resident forms gather and restore through the same owner. RunMat's `\"like\"` form can independently select real provider placement.",
    },
    BuiltinDocumentationFaq {
        question: "What happens with NaN or infinity?",
        answer: "Real NaN propagates. Real positive and negative infinity return positive and negative `pi/2`, respectively.",
    },
    BuiltinDocumentationFaq {
        question: "Does atan fuse with neighboring operations?",
        answer: "The real floating one-input form can participate in elementwise fusion because real inverse tangent does not require value-dependent complex promotion. Other forms remain runtime boundaries.",
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "atan2", target: BuiltinDocumentationLinkTarget::Builtin("atan2") },
    BuiltinDocumentationLink { label: "tan", target: BuiltinDocumentationLinkTarget::Builtin("tan") },
    BuiltinDocumentationLink { label: "asin", target: BuiltinDocumentationLinkTarget::Builtin("asin") },
    BuiltinDocumentationLink { label: "acos", target: BuiltinDocumentationLinkTarget::Builtin("acos") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "acosh", target: BuiltinDocumentationLinkTarget::Builtin("acosh") },
    BuiltinDocumentationLink { label: "asinh", target: BuiltinDocumentationLinkTarget::Builtin("asinh") },
    BuiltinDocumentationLink { label: "atanh", target: BuiltinDocumentationLinkTarget::Builtin("atanh") },
    BuiltinDocumentationLink { label: "cos", target: BuiltinDocumentationLinkTarget::Builtin("cos") },
    BuiltinDocumentationLink { label: "cosh", target: BuiltinDocumentationLinkTarget::Builtin("cosh") },
    BuiltinDocumentationLink { label: "sin", target: BuiltinDocumentationLinkTarget::Builtin("sin") },
    BuiltinDocumentationLink { label: "sinh", target: BuiltinDocumentationLinkTarget::Builtin("sinh") },
    BuiltinDocumentationLink { label: "tanh", target: BuiltinDocumentationLinkTarget::Builtin("tanh") },
    BuiltinDocumentationLink {
        label: "Implementation",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/atan.rs",
        ),
    },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Inverse-tangent runtime",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/atan.rs",
        ),
    }],
    verification: &[
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Real, complex, typed, extension, template, shape, and error behavior",
            location: "crates/runmat-runtime/src/builtins/math/trigonometry/atan.rs::tests",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::ProviderTest,
            label: "Resident unary execution, template placement, and owner preservation",
            location: "crates/runmat-runtime/src/builtins/math/trigonometry/atan.rs::tests::atan_gpu_provider_roundtrip",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::WgpuTest,
            label: "Actual WGPU inverse-tangent parity",
            location: "crates/runmat-runtime/src/builtins/math/trigonometry/atan.rs::tests::atan_wgpu_matches_cpu_elementwise",
        },
    ],
    notes: &[],
};

pub(crate) const ATAN_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("atan"),
    slug: Some("atan"),
    summary: "Compute elementwise principal inverse tangent in radians with optional RunMat output templating.",
    description: "`atan` preserves floating precision and shape, keeps real input real, supports principal complex results, and retains supported provider placement.",
    keywords: &["atan", "arctangent", "inverse tangent", "trigonometry", "complex", "like", "elementwise", "gpu"],
    related: &["atan2", "tan", "asin", "acos", "gpuArray", "gather", "acosh", "asinh", "atanh", "cos", "cosh", "sin", "sinh", "tanh"],
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
