use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Inputs must behave like 2-D matrices (trailing singleton dimensions are accepted). `size(A, 1)` must match `size(B, 1)` after accounting for `opts.TRANSA`.",
            "When `opts.LT` or `opts.UT` are supplied, `linsolve` performs forward/back substitution instead of a full factorization. Singular pivots trigger the MATLAB error `\"linsolve: matrix is singular to working precision.\"`",
            "A logical `opts.TRANSA` is supported in compatibility mode (`false` means no transpose and `true` means conjugate transpose). Text values `'N'`, `'T'`, and `'C'` are a RunMat-only extension.",
            "`opts.POSDEF`, `opts.SYM`, and `opts.RECT` are parsed and forwarded to acceleration providers. On the host, nonsingular square systems use pivoted LU; singular and rectangular systems use the shared SVD minimum-norm solver. These hints do not select specialised structural host algorithms.",
            "For a nonsingular square system, the optional second output estimates reciprocal 1-norm condition using the same LU factors as the solve. For a rectangular system, the second output is the estimated rank.",
            "Logical and integer coefficient inputs are RunMat-only extensions and are promoted to double; integer values must be exactly representable in binary64. Complex floating inputs use complex arithmetic, while typed complex-integer inputs are rejected.",
            "Typed-integer values for `LT`, `UT`, `RECT`, `SYM`, and `POSDEF`, the `RCOND` option, and two-output calls with explicit `gpuArray` operands are RunMat-only extensions."
        ]
    },
    BuiltinDocumentationSection {
        heading: "Acceleration and residency",
        paragraphs: &[
            "Prefers the exact owner's `linsolve` hook. WGPU has selected real F32 triangular, transpose, positive-definite, square, and rectangular paths; unsupported cases gather to the shared host solver.",
            "For eligible real matrix operands, RunMat offers the solve to the exact owner's `linsolve` hook and validates that the returned solution is non-aliased, real, floating, correctly shaped, on the expected device, and owned by that provider. WGPU supports selected real F32 triangular, transpose, positive-definite, square, and rectangular routes. Provider-declined, complex, scalar, or otherwise unsupported cases gather to the shared host solver and return a host result. Explicit-GPU two-output calls are available only in RunMat extension mode.",
            "No additional residency management is required for supported real matrix operands. RunMat selects the exact provider that owns the first resident operand and requires every other resident operand to have that same owner. It validates provider results and preserves explicit or automatic provenance. If the provider declines the solve, or if the operands are complex or scalar, RunMat gathers through the normal host path; that fallback returns a host value rather than automatically re-uploading it."
        ]
    },
    BuiltinDocumentationSection {
        heading: "Implementation and validation",
        paragraphs: &[
            "Original GPU declarations: precisions f32, f64; elementwise false; reduction false; broadcasting none.",
            "Original fusion declarations: elementwise false; reduction false; max inputs 2; constants uniform."
        ]
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Solving a 2×2 linear system",
        program: "A = [4 -2; 1 3];\nb = [6; 7];\nx = linsolve(A, b)",
        display_output: Some("x = [16/7; 11/7]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(norm(A*x-b) < 1e-10);"
        }
    },
    BuiltinExample {
        id: "example-2",
        title: "Using a lower-triangular hint",
        program: "L = [3 0 0; -1 2 0; 4 1 5];\nb = [9; 1; 12];\nopts.LT = true;\nx = linsolve(L, b, opts)",
        display_output: Some("x = [3; 2; -2/5]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(norm(L*x-b) < 1e-10);"
        }
    },
    BuiltinExample {
        id: "example-3",
        title: "Solving the transposed system",
        program: "A = [2 1 0; 0 3 4; 0 0 5];\nb = [3; 11; 5];\nopts.UT = true;\nopts.TRANSA = 'T';\nx = linsolve(A, b, opts)",
        display_output: Some("x = [3/2; 19/6; -23/15]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(norm(A.'*x-b) < 1e-10);"
        }
    },
    BuiltinExample {
        id: "example-4",
        title: "Complex triangular solve",
        program: "U = [2+1i  -1i; 0  4-2i];\nb = [3+2i; 7];\nopts.UT = true;\nx = linsolve(U, b, opts)",
        display_output: Some("x = [1.6+0.9i; 1.4+0.7i]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(norm(U*x-b) < 1e-10);"
        }
    },
    BuiltinExample {
        id: "example-5",
        title: "Estimating the reciprocal condition number for a square system",
        program: "A = [1 1; 1 1+1e-12];\nb = [2; 2+1e-12];\n[x, rcond_est] = linsolve(A, b)",
        display_output: Some("x is approximately [1; 1]; rcond_est is small and positive"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(norm(A*x-b) < 1e-8); assert(norm(x-[1;1]) < 1e-6); assert(rcond_est > 0 && rcond_est < 1e-8);"
        }
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "What happens if I pass both `opts.LT` and `opts.UT`?",
        answer: "RunMat raises the MATLAB error `\"linsolve: LT and UT are mutually exclusive.\"`—a matrix cannot be simultaneously strictly lower- and upper-triangular."
    },
    BuiltinDocumentationFaq {
        question: "Does `opts.TRANSA` accept lowercase characters?",
        answer: "In RunMat extension mode, text values are case-insensitive and accept `'N'`, `'T'`, or `'C'`. In compatibility mode, use a logical scalar: `false` selects no transpose and `true` selects conjugate transpose."
    },
    BuiltinDocumentationFaq {
        question: "How is `opts.RCOND` used?",
        answer: "In RunMat extension mode, `opts.RCOND` provides a non-negative lower bound on the acceptable reciprocal condition estimate. If the estimate falls below it, the builtin raises `\"linsolve: matrix is singular to working precision.\"` Compatibility mode rejects this option."
    },
    BuiltinDocumentationFaq {
        question: "Do `opts.SYM` or `opts.POSDEF` change the algorithm today?",
        answer: "They are parsed and forwarded to providers. The host path uses pivoted LU for nonsingular square systems and SVD for singular or rectangular systems; these hints do not select specialised host algorithms. WGPU implements a selected real F32 positive-definite path; support for other cases remains provider-specific."
    },
    BuiltinDocumentationFaq {
        question: "Can I use higher-dimensional arrays?",
        answer: "Inputs must behave like matrices. Trailing singleton dimensions are permitted, but other higher-rank arrays should be reshaped before calling `linsolve`, just like in MATLAB."
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "mldivide",
        target: BuiltinDocumentationLinkTarget::Builtin("mldivide"),
    },
    BuiltinDocumentationLink {
        label: "mrdivide",
        target: BuiltinDocumentationLinkTarget::Builtin("mrdivide"),
    },
    BuiltinDocumentationLink {
        label: "lu",
        target: BuiltinDocumentationLinkTarget::Builtin("lu"),
    },
    BuiltinDocumentationLink {
        label: "chol",
        target: BuiltinDocumentationLinkTarget::Builtin("chol"),
    },
    BuiltinDocumentationLink {
        label: "gpuArray",
        target: BuiltinDocumentationLinkTarget::Builtin("gpuarray"),
    },
    BuiltinDocumentationLink {
        label: "gather",
        target: BuiltinDocumentationLinkTarget::Builtin("gather"),
    },
    BuiltinDocumentationLink {
        label: "cond",
        target: BuiltinDocumentationLinkTarget::Builtin("cond"),
    },
    BuiltinDocumentationLink {
        label: "det",
        target: BuiltinDocumentationLinkTarget::Builtin("det"),
    },
    BuiltinDocumentationLink {
        label: "inv",
        target: BuiltinDocumentationLinkTarget::Builtin("inv"),
    },
    BuiltinDocumentationLink {
        label: "norm",
        target: BuiltinDocumentationLinkTarget::Builtin("norm"),
    },
    BuiltinDocumentationLink {
        label: "pinv",
        target: BuiltinDocumentationLinkTarget::Builtin("pinv"),
    },
    BuiltinDocumentationLink {
        label: "rank",
        target: BuiltinDocumentationLinkTarget::Builtin("rank"),
    },
    BuiltinDocumentationLink {
        label: "rcond",
        target: BuiltinDocumentationLinkTarget::Builtin("rcond"),
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("linsolve"),
    slug: Some("linsolve"),
    summary: "Solve linear systems with structural hints in MATLAB and RunMat.",
    description: "`X = linsolve(A, B)` solves `A * X = B`. The implemented `opts` surface supports triangular, rectangular, symmetric, positive-definite, and transpose controls, with the extension boundaries described below.",
    keywords: &[
        "linsolve",
        "linear solve",
        "triangular system",
        "posdef",
        "gpu"
    ],
    related: &[
        "mldivide",
        "mrdivide",
        "lu",
        "chol",
        "gpuarray",
        "gather",
        "cond",
        "det",
        "inv",
        "norm",
        "pinv",
        "rank",
        "rcond"
    ],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "`crates/runmat-runtime/src/builtins/math/linalg/solve/linsolve.rs`",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/solve/linsolve.rs")
            }
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "unit",
                location: "builtins::math::linalg::solve::linsolve::tests"
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::ProviderTest,
                label: "gpu",
                location: "builtins::math::linalg::solve::linsolve::tests::gpu_round_trip_matches_cpu"
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::WgpuTest,
                label: "wgpu",
                location: "builtins::math::linalg::solve::linsolve::tests::wgpu_round_trip_matches_cpu"
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
