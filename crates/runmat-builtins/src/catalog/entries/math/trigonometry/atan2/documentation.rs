use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Angles, quadrants, and signed zero",
        paragraphs: &[
            "`atan2(Y, X)` computes the four-quadrant inverse tangent of the coordinate pair `(X, Y)` element by element. Results are radians in the closed interval `[-pi, pi]`; using both coordinate signs distinguishes quadrants that `atan(Y ./ X)` cannot.",
            "RunMat follows MATLAB's signed-zero convention. `atan2(0, -0)` and `atan2(-0, -0)` return positive zero, while `atan2(-0, 0)` retains negative zero. NaN propagates and infinite coordinate pairs follow the corresponding quadrant limits.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Classes, shapes, and containers",
        paragraphs: &[
            "Documented inputs are real single, double, table, or timetable values. Scalar, vector, matrix, and N-D numeric inputs use implicit expansion. A single input on either side selects single output; otherwise the numeric result is double. Complex and sparse inputs are not accepted.",
            "Tables and timetables apply `atan2` to corresponding variables and retain their container metadata. The two containers must have the same identity, variable names, and variable order; a compatible numeric scalar or array can supply the other operand.",
            "RunMat mode also accepts all eight real fixed-width integer classes, logical arrays, and character arrays. Integer storage remains authoritative until the floating-point angle calculation. Logical values contribute zero or one and characters contribute their Unicode code points. These forms are rejected in MATLAB compatibility mode.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Accelerated, fused, and distributed execution",
        paragraphs: &[
            "Shape-matched real floating operands owned by one provider can use its direct `elem_atan2` operation. The returned handle is accepted only when its shape, real storage, precision, owner, device, and aliasing satisfy the contract. Only a typed unsupported result enters host fallback; other provider failures remain visible.",
            "Broadcasting, mixed physical types, and admitted integer or logical storage gather through the exact owner, compute with the canonical host rule, and restore the floating result to that owner when possible. Real floating expressions may also participate in elementwise fusion, including the signed-zero correction.",
            "Distributed inputs currently use the declared materialization path before the same canonical operation is applied. This preserves behavior without claiming a partition-local binary mapping primitive that the execution service does not yet provide.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "point",
        title: "Find the angle of a point",
        program: "theta = atan2(4, 3)",
        display_output: Some("theta = 0.9273"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(theta - 0.9272952180016122) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "quadrants",
        title: "Resolve quadrants from coordinate signs",
        program: "Y = [-1 0 1];\nX = [-1 -1 -1];\nangles = atan2(Y, X)",
        display_output: Some("angles = [-2.3562 3.1416 2.3562]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = [-3*pi/4, pi, 3*pi/4];\nassert(max(abs(angles - expected)) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "implicit-expansion",
        title: "Expand a denominator across a matrix",
        program: "A = [1 2 3; 4 5 6];\nangles = atan2(A, 2)",
        display_output: Some("angles has the same 2-by-3 shape as A"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(angles), [2 3]));\nassert(max(abs(tan(angles(:)) - A(:)/2)) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "signed-zero",
        title: "Apply the signed-zero convention",
        program: "negativeZero = -0.0;\ntheta = atan2([0 negativeZero negativeZero], [negativeZero negativeZero 0])",
        display_output: Some("theta contains positive zero, positive zero, and negative zero"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(theta(1) == 0 && theta(2) == 0 && theta(3) == 0);\nassert(1/theta(1) > 0 && 1/theta(2) > 0 && 1/theta(3) < 0);",
        },
    },
    BuiltinExample {
        id: "single-class",
        title: "Retain single precision",
        program: "Y = single([1 -1]);\nP = atan2(Y, -2)",
        display_output: Some("P is a 1-by-2 single row vector"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(P, 'single'));\nassert(max(abs(double(P) - atan2(double(Y), -2))) < 1e-6);",
        },
    },
    BuiltinExample {
        id: "table",
        title: "Apply atan2 to table variables",
        program: "Y = table([1; -1], [0; 1], 'VariableNames', {'A', 'B'});\nP = atan2(Y, -1)",
        display_output: Some("P retains variables A and B"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(istable(P));\nassert(max(abs(P.A - [3*pi/4; -3*pi/4])) < 1e-12);\nassert(max(abs(P.B - [pi; 3*pi/4])) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "character-extension",
        title: "Use character code points in RunMat mode",
        program: "theta = atan2('A', 100)",
        display_output: Some("theta = 0.5764"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(theta - atan2(65, 100)) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "gpu-residency",
        title: "Compute four-quadrant angles on resident values",
        program: "Gy = gpuArray([1 1; -1 -1]);\nGx = gpuArray([1 -1; 1 -1]);\nanglesDevice = atan2(Gy, Gx);\nangles = gather(anglesDevice)",
        display_output: Some("angles = [pi/4 3*pi/4; -pi/4 -3*pi/4]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = [pi/4 3*pi/4; -pi/4 -3*pi/4];\nassert(isa(anglesDevice, 'gpuArray'));\nassert(max(abs(angles(:) - expected(:))) < 1e-6);",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "How is atan2 different from atan?", answer: "`atan2(Y, X)` uses both coordinate signs to select a quadrant and returns angles in `[-pi, pi]`. `atan(Y ./ X)` loses that sign information and is limited to `[-pi/2, pi/2]`." },
    BuiltinDocumentationFaq { question: "Can atan2 accept complex input?", answer: "No. Both operands must be real. Use `angle(Z)` or `atan2(imag(Z), real(Z))` for a complex value." },
    BuiltinDocumentationFaq { question: "What happens when X is zero?", answer: "A positive Y returns `pi/2`, a negative Y returns `-pi/2`, and zero pairs follow the documented signed-zero convention." },
    BuiltinDocumentationFaq { question: "Does atan2 preserve shape?", answer: "The output has the shape produced by MATLAB-style implicit expansion of Y and X. Table and timetable forms retain their container identity and variable layout." },
    BuiltinDocumentationFaq { question: "What class does atan2 return?", answer: "A single input on either side selects single output; otherwise numeric output is double. RunMat integer, logical, and character extensions return double unless paired with single." },
    BuiltinDocumentationFaq { question: "Can the result remain provider-resident?", answer: "Yes. A valid direct provider result remains resident. Unsupported or broadcast forms gather through the exact owner and restore the result when its representation is supported." },
    BuiltinDocumentationFaq { question: "How do I convert the result to degrees?", answer: "Use `rad2deg(atan2(Y, X))` or multiply by `180/pi`." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "atan", target: BuiltinDocumentationLinkTarget::Builtin("atan") },
    BuiltinDocumentationLink { label: "angle", target: BuiltinDocumentationLinkTarget::Builtin("angle") },
    BuiltinDocumentationLink { label: "hypot", target: BuiltinDocumentationLinkTarget::Builtin("hypot") },
    BuiltinDocumentationLink { label: "rad2deg", target: BuiltinDocumentationLinkTarget::Builtin("rad2deg") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "acos", target: BuiltinDocumentationLinkTarget::Builtin("acos") },
    BuiltinDocumentationLink { label: "acosh", target: BuiltinDocumentationLinkTarget::Builtin("acosh") },
    BuiltinDocumentationLink { label: "asin", target: BuiltinDocumentationLinkTarget::Builtin("asin") },
    BuiltinDocumentationLink { label: "asinh", target: BuiltinDocumentationLinkTarget::Builtin("asinh") },
    BuiltinDocumentationLink { label: "atanh", target: BuiltinDocumentationLinkTarget::Builtin("atanh") },
    BuiltinDocumentationLink { label: "cos", target: BuiltinDocumentationLinkTarget::Builtin("cos") },
    BuiltinDocumentationLink { label: "cosh", target: BuiltinDocumentationLinkTarget::Builtin("cosh") },
    BuiltinDocumentationLink { label: "sin", target: BuiltinDocumentationLinkTarget::Builtin("sin") },
    BuiltinDocumentationLink { label: "sinh", target: BuiltinDocumentationLinkTarget::Builtin("sinh") },
    BuiltinDocumentationLink { label: "tan", target: BuiltinDocumentationLinkTarget::Builtin("tan") },
    BuiltinDocumentationLink { label: "tanh", target: BuiltinDocumentationLinkTarget::Builtin("tanh") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/atan2.rs") },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Four-quadrant angle runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/atan2.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Quadrants, signed zero, classes, broadcasting, tables, extensions, and errors", location: "crates/runmat-runtime/src/builtins/math/trigonometry/atan2/tests.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Direct execution, owner restoration, and output validation", location: "crates/runmat-runtime/src/builtins/math/trigonometry/atan2/tests.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU angle and signed-zero parity", location: "crates/runmat-runtime/src/builtins/math/trigonometry/atan2/tests.rs::atan2_wgpu_matches_cpu_elementwise" },
    ],
    notes: &[],
};

pub(super) const ATAN2_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("atan2"),
    slug: Some("atan2"),
    summary: "Compute a real four-quadrant inverse tangent in radians.",
    description: "`atan2` combines both coordinate signs, applies implicit expansion, preserves supported floating precision and container identity, and validates provider-resident execution.",
    keywords: &["atan2", "arctangent", "inverse tangent", "quadrant", "signed zero", "table", "gpu"],
    related: &["atan", "angle", "hypot", "rad2deg", "sin", "cos", "tan", "gpuArray", "gather"],
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
