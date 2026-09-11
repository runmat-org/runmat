use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Definition and signs",
        paragraphs: &[
            "`rem(A, B)` computes `A - B .* fix(A ./ B)` element by element. The result is zero or has the dividend's sign. This differs from `mod` whenever a nonzero result combines operands with different signs.",
            "For floating operands, a zero divisor produces `NaN`. A finite dividend and infinite divisor return the dividend; an infinite dividend with a finite divisor and `NaN` inputs produce `NaN`. Fixed-width integer storage cannot represent `NaN`, so an integer zero-divisor element returns integer zero.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Classes, shapes, and containers",
        paragraphs: &[
            "Real scalar, vector, matrix, and N-D operands use implicit expansion. Double and single operands retain their floating class, with single taking precedence when the floating inputs differ. Logical and character operands enter the double domain; characters contribute their Unicode code points. Complex and sparse inputs are not accepted.",
            "Matching fixed-width integer operands retain their class and use exact native storage. A documented scalar-double pairing is accepted where the integer class permits it. Wide integers are not converted through binary64.",
            "Tables and timetables apply `rem` to corresponding variables and retain their container metadata. A numeric scalar can be applied to every variable. Duration operands are measured in days for the calculation and return a duration with the source display format.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Accelerated and distributed execution",
        paragraphs: &[
            "Compatible resident operands use the exact provider that owns both handles. Provider results are accepted only when shape, storage, precision, owner, device, and aliasing satisfy the catalog contract. A typed unsupported response uses one host fallback and restores the result to that owner when its physical representation can be preserved; other provider errors remain visible.",
            "Compatible expressions may use the elementwise fusion definition. Distributed inputs currently use the declared materialization path before the same canonical operation is applied; this preserves values and classes without claiming a partition-local binary map that the execution service does not yet provide.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "positive",
        title: "Compute a positive remainder",
        program: "r = rem(17, 5)",
        display_output: Some("r = 2"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(r == 2);",
        },
    },
    BuiltinExample {
        id: "dividend-sign",
        title: "Follow the dividend's sign",
        program: "values = [-7 -3 4 9];\nr = rem(values, 4)",
        display_output: Some("r = [-3 -3 0 1]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(r, [-3 -3 0 1]));",
        },
    },
    BuiltinExample {
        id: "negative-divisor",
        title: "Use a negative divisor",
        program: "r = rem(7, -4)",
        display_output: Some("r = 3"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(r == 3);",
        },
    },
    BuiltinExample {
        id: "implicit-expansion",
        title: "Expand a column against a row",
        program: "A = [-5; 8];\nB = [3 4 5];\nR = rem(A, B)",
        display_output: Some("R = [-2 -1 0; 2 0 3]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(R, [-2 -1 0; 2 0 3]));",
        },
    },
    BuiltinExample {
        id: "zero-divisor",
        title: "Observe a floating zero divisor",
        program: "R = rem([2 0 -2], 0)",
        display_output: Some("R = [NaN NaN NaN]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(all(isnan(R)));",
        },
    },
    BuiltinExample {
        id: "integer-class",
        title: "Retain exact integer storage",
        program:
            "A = uint64([0x0020000000000001u64, 0xFFFFFFFFFFFFFFFFu64]);\nR = rem(A, uint64(2))",
        display_output: Some("R is uint64([1 1])"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(R, 'uint64')); assert(isequal(R, uint64([1 1])));",
        },
    },
    BuiltinExample {
        id: "single-class",
        title: "Retain single precision",
        program: "A = single([-3.5 4.5]);\nR = rem(A, 2)",
        display_output: Some("R is single([-1.5 0.5])"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(R, 'single')); assert(isequal(R, single([-1.5 0.5])));",
        },
    },
    BuiltinExample {
        id: "table",
        title: "Apply remainder to table variables",
        program: "T = table([5; 8], [7; 11], 'VariableNames', {'A', 'B'});\nR = rem(T, 3)",
        display_output: Some("R retains A and B with values [2;2] and [1;2]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source:
                "assert(istable(R)); assert(isequal(R.A, [2; 2])); assert(isequal(R.B, [1; 2]));",
        },
    },
    BuiltinExample {
        id: "duration",
        title: "Compute remainders of durations",
        program: "A = hours([25 50]);\nR = rem(A, hours(24))",
        display_output: Some("R = [01:00:00 02:00:00]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isduration(R)); assert(max(abs(hours(R) - [1 2])) < 1e-10);",
        },
    },
    BuiltinExample {
        id: "gpu-residency",
        title: "Compute on resident values",
        program: "G = gpuArray(-5:5);\nR = gather(rem(G, 4))",
        display_output: Some("R = [-1 0 -3 -2 -1 0 1 2 3 0 1]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(R, [-1 0 -3 -2 -1 0 1 2 3 0 1]));",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "How does rem differ from mod?", answer: "`rem` rounds the quotient toward zero, so a nonzero result has the dividend's sign. `mod` rounds the quotient toward negative infinity, so its nonzero result has the divisor's sign." },
    BuiltinDocumentationFaq { question: "What does a zero divisor return?", answer: "Floating operands produce `NaN`. Exact fixed-width integer storage returns integer zero because its class has no `NaN` representation." },
    BuiltinDocumentationFaq { question: "Does rem accept complex values?", answer: "No. Both operands must be real." },
    BuiltinDocumentationFaq { question: "Are integer values converted to double?", answer: "No. Supported fixed-width integer forms retain their integer class and use exact native storage." },
    BuiltinDocumentationFaq { question: "How are tables handled?", answer: "The operation is applied variable by variable. Table and timetable identity, variable order, and timetable row-time metadata are retained." },
    BuiltinDocumentationFaq { question: "What unit is used for duration and numeric combinations?", answer: "Numeric values are interpreted as 24-hour days. The result is returned as a duration when either operand is a duration." },
    BuiltinDocumentationFaq { question: "Will provider fallback lose residency?", answer: "A semantically unsupported provider hook may use an owner-specific host fallback and restore the result. Malformed outputs and execution failures are errors, not fallback signals." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "mod", target: BuiltinDocumentationLinkTarget::Builtin("mod") },
    BuiltinDocumentationLink { label: "fix", target: BuiltinDocumentationLinkTarget::Builtin("fix") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/rounding/rem.rs") },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Remainder runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/rounding/rem.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Signs, zero divisors, classes, broadcasting, containers, and errors", location: "crates/runmat-runtime/src/builtins/math/rounding/rem.rs::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Owner-aware provider validation and fallback", location: "crates/runmat-runtime/src/builtins/math/rounding/rem.rs::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU floating and integer parity", location: "crates/runmat-runtime/src/builtins/math/rounding/rem.rs::tests" },
    ],
    notes: &[],
};

pub(super) const REM_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("rem"),
    slug: Some("rem"),
    summary: "Compute the truncation-based remainder of compatible real values.",
    description: "`rem` applies truncation-based remainder semantics with implicit expansion, exact fixed-width integer handling, supported containers, and validated provider execution.",
    keywords: &["rem", "remainder", "fix", "truncate", "integer", "table", "duration", "gpu"],
    related: &["mod", "fix", "floor", "gpuArray", "gather"],
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
