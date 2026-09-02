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
            "`mod(A, B)` computes `A - B .* floor(A ./ B)` element by element. The result is zero or has the divisor's sign. This differs from `rem` whenever a nonzero result combines operands with different signs.",
            "A zero divisor returns the corresponding dividend: `mod(A, 0)` is `A`. A finite dividend and infinite divisor follow the floor-based definition; `NaN` inputs and an infinite dividend with a finite divisor produce `NaN`.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Classes, shapes, and containers",
        paragraphs: &[
            "Real scalar, vector, matrix, and N-D operands use implicit expansion. Double and single operands retain their floating class, with single taking precedence when the floating inputs differ. Logical and character operands enter the double domain; characters contribute their Unicode code points. Complex and sparse inputs are not accepted.",
            "Matching fixed-width integer operands retain their class and use exact native storage. A documented scalar-double pairing is accepted where the integer class permits it. Wide integers are not converted through binary64.",
            "Tables and timetables apply `mod` to corresponding variables and retain their container metadata. A numeric scalar can be applied to every variable. Duration operands are measured in days for the calculation and return a duration with the source display format.",
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
        title: "Compute a positive modulus",
        program: "r = mod(17, 5)",
        display_output: Some("r = 2"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(r == 2);",
        },
    },
    BuiltinExample {
        id: "divisor-sign",
        title: "Follow the divisor's sign",
        program: "values = [-7 -3 4 9];\nr = mod(values, -4)",
        display_output: Some("r = [-3 -3 0 -3]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(r, [-3 -3 0 -3]));",
        },
    },
    BuiltinExample {
        id: "implicit-expansion",
        title: "Expand a column against a row",
        program: "A = [5; 8];\nB = [3 4 5];\nR = mod(A, B)",
        display_output: Some("R = [2 1 0; 2 0 3]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(R, [2 1 0; 2 0 3]));",
        },
    },
    BuiltinExample {
        id: "zero-divisor",
        title: "Return the dividend for a zero divisor",
        program: "A = [2 0 -2];\nR = mod(A, 0)",
        display_output: Some("R = [2 0 -2]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(R, A));",
        },
    },
    BuiltinExample {
        id: "integer-class",
        title: "Retain exact integer storage",
        program:
            "A = uint64([0x0020000000000001u64, 0xFFFFFFFFFFFFFFFFu64]);\nR = mod(A, uint64(2))",
        display_output: Some("R is uint64([1 1])"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(R, 'uint64')); assert(isequal(R, uint64([1 1])));",
        },
    },
    BuiltinExample {
        id: "single-class",
        title: "Retain single precision",
        program: "A = single([-3.5 4.5]);\nR = mod(A, 2)",
        display_output: Some("R is single([0.5 0.5])"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(R, 'single')); assert(isequal(R, single([0.5 0.5])));",
        },
    },
    BuiltinExample {
        id: "table",
        title: "Apply modulus to table variables",
        program: "T = table([5; 8], [7; 11], 'VariableNames', {'A', 'B'});\nR = mod(T, 3)",
        display_output: Some("R retains A and B with values [2;2] and [1;2]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source:
                "assert(istable(R)); assert(isequal(R.A, [2; 2])); assert(isequal(R.B, [1; 2]));",
        },
    },
    BuiltinExample {
        id: "duration",
        title: "Compute modulus of durations",
        program: "A = hours([25 50]);\nR = mod(A, hours(24))",
        display_output: Some("R = [01:00:00 02:00:00]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isduration(R)); assert(max(abs(hours(R) - [1 2])) < 1e-10);",
        },
    },
    BuiltinExample {
        id: "gpu-residency",
        title: "Compute on resident values",
        program: "G = gpuArray(-5:5);\nR = gather(mod(G, 4))",
        display_output: Some("R = [3 0 1 2 3 0 1 2 3 0 1]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(R, [3 0 1 2 3 0 1 2 3 0 1]));",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "How does mod differ from rem?", answer: "`mod` rounds the quotient toward negative infinity, so a nonzero result has the divisor's sign. `rem` rounds the quotient toward zero, so its nonzero result has the dividend's sign." },
    BuiltinDocumentationFaq { question: "What does a zero divisor return?", answer: "`mod(A, 0)` returns `A` element by element. This rule also applies to exact integer operands." },
    BuiltinDocumentationFaq { question: "Does mod accept complex values?", answer: "No. Both operands must be real." },
    BuiltinDocumentationFaq { question: "Are integer values converted to double?", answer: "No. Supported fixed-width integer forms retain their integer class and use exact native storage." },
    BuiltinDocumentationFaq { question: "How are tables handled?", answer: "The operation is applied variable by variable. Table and timetable identity, variable order, and timetable row-time metadata are retained." },
    BuiltinDocumentationFaq { question: "What unit is used for duration and numeric combinations?", answer: "Numeric values are interpreted as 24-hour days. The result is returned as a duration when either operand is a duration." },
    BuiltinDocumentationFaq { question: "Will provider fallback lose residency?", answer: "A semantically unsupported provider hook may use an owner-specific host fallback and restore the result. Malformed outputs and execution failures are errors, not fallback signals." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "rem", target: BuiltinDocumentationLinkTarget::Builtin("rem") },
    BuiltinDocumentationLink { label: "floor", target: BuiltinDocumentationLinkTarget::Builtin("floor") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/rounding/modulus.rs") },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Modulus runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/rounding/modulus.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Signs, zero divisors, classes, broadcasting, containers, and errors", location: "crates/runmat-runtime/src/builtins/math/rounding/modulus.rs::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Owner-aware provider validation and fallback", location: "crates/runmat-runtime/src/builtins/math/rounding/modulus.rs::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU floating and integer parity", location: "crates/runmat-runtime/src/builtins/math/rounding/modulus.rs::tests" },
    ],
    notes: &[],
};

pub(super) const MOD_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("mod"),
    slug: Some("mod"),
    summary: "Compute the floor-based modulus of compatible real values.",
    description: "`mod` applies floor-based remainder semantics with implicit expansion, exact fixed-width integer handling, supported containers, and validated provider execution.",
    keywords: &["mod", "modulus", "remainder", "floor", "integer", "table", "duration", "gpu"],
    related: &["rem", "floor", "fix", "gpuArray", "gather"],
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
