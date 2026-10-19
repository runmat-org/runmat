use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`cond(A)` and `cond(A, 2)` use the singular values of `A`, so rectangular matrices are supported.",
            "`cond(A, 1)`, `cond(A, Inf)`, and `cond(A, 'fro')` require a square, invertible matrix and use the definition `norm(A, p) * norm(inv(A), p)` with MATLAB's norm semantics.",
            "Scalars behave like 1x1 matrices. Non-zero scalars have condition number `1`, while `cond(0) = Inf`.",
            "Empty matrices return `0`, matching MATLAB's convention.",
            "Singular or rank-deficient matrices return `Inf`.",
            "Any NaN or infinite matrix element returns `NaN`, matching the current MATLAB nonfinite-input convention.",
            "Single-precision host input returns a native single scalar; double input returns double. Resident output retains the owning provider's selected precision, including integer gather-fallback results re-uploaded through that provider.",
            "Complex inputs are handled in full complex arithmetic.",
            "The documented matrix domain is single or double. Typed-integer and logical matrices are independent RunMat-only extensions; integer matrices must be exactly representable as binary64 before crossing the floating linear-algebra boundary.",
            "Typed-integer norm selectors are a separate RunMat-only structural extension. They are decoded exactly and accept only 1 or 2 without entering matrix arithmetic."
        ]
    },
    BuiltinDocumentationSection {
        heading: "Acceleration and residency",
        paragraphs: &[
            "RunMat uses the input handle's owning provider. When its cond hook is unavailable, RunMat gathers to host and re-uploads the scalar through that same provider; upload failures are reported rather than silently changing residency.",
            "When the input already lives on a GPU, RunMat first looks for an acceleration provider that exposes the custom `cond` hook registered below. Current providers gather the matrix to host memory, reuse the shared CPU implementation, and then re-upload the scalar so downstream GPU computations preserve residency. This mirrors MATLAB semantics while keeping the user-facing API uniform.",
            "```matlab:runnable\nG = gpuArray([]);      % Empty 0x0 matrix on the GPU\nk = cond(G);           % Returns 0 and keeps residency when possible\nresult = gather(k);\n```\nExpected output:\n```matlab\nresult = 0\n```"
        ]
    },
    BuiltinDocumentationSection {
        heading: "Implementation and validation",
        paragraphs: &[
            "Original GPU declarations: precisions f32, f64; elementwise false; reduction false; broadcasting none.",
            "Original fusion declarations: elementwise false; reduction false; max inputs 1; constants inline."
        ]
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Condition number of the identity matrix",
        program: "A = eye(3);\nk = cond(A)",
        display_output: Some("k = 1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(k == 1);"
        }
    },
    BuiltinExample {
        id: "example-2",
        title: "Diagnosing an ill-conditioned diagonal matrix",
        program: "D = diag([1, 1e-8]);\nk = cond(D)",
        display_output: Some("k = 1.0e+8"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(k - 1e8) / 1e8 < 1e-8);"
        }
    },
    BuiltinExample {
        id: "example-3",
        title: "Condition number of a rectangular matrix (2-norm)",
        program: "A = [1 0; 0 1; 1 1];\nk = cond(A, 2)",
        display_output: Some("k = 1.7321"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(k - sqrt(3)) < 1e-10);"
        }
    },
    BuiltinExample {
        id: "example-4",
        title: "Using a different norm specification",
        program: "A = [4 -1; 2 3];\nk1 = cond(A, 1)\nkInf = cond(A, Inf)",
        display_output: Some("k1   = 2.1429\nkInf = 2.1429"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(k1 - 15/7) < 1e-10); assert(abs(kInf - 15/7) < 1e-10);"
        }
    },
    BuiltinExample {
        id: "example-5",
        title: "Complex-valued matrices",
        program: "A = [1+2i 0; 3i 4-1i];\nk = cond(A)",
        display_output: Some("k = 3.0327"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(k - 3.0327) < 1e-3);"
        }
    },
    BuiltinExample {
        id: "example-6",
        title: "Empty inputs and GPU residency",
        program: "G = gpuArray([]);      % Empty 0x0 matrix on the GPU\nk = cond(G);           % Returns 0 and keeps residency when possible\nresult = gather(k)",
        display_output: Some("result = 0"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(result == 0);"
        }
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "What does a large condition number mean?",
        answer: "Large condition numbers (>> 1) indicate that small perturbations in the input can produce large changes in the solution of a linear system involving `A`. Values close to `1` indicate a well- conditioned matrix."
    },
    BuiltinDocumentationFaq {
        question: "Why does `cond` return `Inf` for singular matrices?",
        answer: "Singular matrices have at least one zero singular value (or an undefined inverse), so the condition number is mathematically infinite. RunMat mirrors MATLAB and returns `Inf` in these cases."
    },
    BuiltinDocumentationFaq {
        question: "Does `cond` support rectangular matrices?",
        answer: "Yes for the default 2-norm: `cond(A)` uses singular values and accepts any two-dimensional matrix. Norms `1`, `Inf`, and `'fro'` require a square, invertible matrix because they are defined using the matrix inverse."
    },
    BuiltinDocumentationFaq {
        question: "How does `cond` handle empty matrices?",
        answer: "All norm choices return `0` for empty matrices (`0x0`), matching MATLAB's behaviour."
    },
    BuiltinDocumentationFaq {
        question: "Will calling `cond` move my data off the GPU?",
        answer: "Only when the owning provider lacks a dedicated implementation. RunMat then gathers, computes on the host, and uploads the scalar through that same provider; an upload failure is reported rather than silently changing residency."
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "rcond",
        target: BuiltinDocumentationLinkTarget::Builtin("rcond"),
    },
    BuiltinDocumentationLink {
        label: "inv",
        target: BuiltinDocumentationLinkTarget::Builtin("inv"),
    },
    BuiltinDocumentationLink {
        label: "pinv",
        target: BuiltinDocumentationLinkTarget::Builtin("pinv"),
    },
    BuiltinDocumentationLink {
        label: "linsolve",
        target: BuiltinDocumentationLinkTarget::Builtin("linsolve"),
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
        label: "det",
        target: BuiltinDocumentationLinkTarget::Builtin("det"),
    },
    BuiltinDocumentationLink {
        label: "norm",
        target: BuiltinDocumentationLinkTarget::Builtin("norm"),
    },
    BuiltinDocumentationLink {
        label: "rank",
        target: BuiltinDocumentationLinkTarget::Builtin("rank"),
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("cond"),
    slug: Some("cond"),
    summary: "Compute matrix condition numbers in MATLAB and RunMat.",
    description: "`k = cond(A)` returns the condition number of matrix `A`, measuring sensitivity of linear-system solutions to perturbations. By default, it computes the 2-norm condition number from singular-value extrema, following MATLAB semantics.",
    keywords: &[
        "cond",
        "condition number",
        "norm",
        "linear algebra",
        "gpu"
    ],
    related: &[
        "rcond",
        "inv",
        "pinv",
        "linsolve",
        "gpuarray",
        "gather",
        "det",
        "norm",
        "rank"
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
                label: "crates/runmat-runtime/src/builtins/math/linalg/solve/cond.rs",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/solve/cond.rs")
            }
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "unit",
                location: "builtins::math::linalg::solve::cond::tests"
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::ProviderTest,
                label: "gpu",
                location: "builtins::math::linalg::solve::cond::tests::cond_gpu_round_trip_matches_cpu"
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::WgpuTest,
                label: "wgpu",
                location: "builtins::math::linalg::solve::cond::tests::cond_wgpu_matches_cpu"
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
