use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Inputs must behave like 2-D matrices. Trailing singleton dimensions are accepted; other higher ranks result in `\"rank: inputs must be 2-D matrices or vectors\"`.",
            "The default tolerance is `tol = max(size(A)) * eps(max(s))`, where `s` are the singular values from an SVD of `A`. You can override this by supplying a second argument: `rank(A, tol)`.",
            "When you provide an explicit tolerance it must be a finite, non-negative scalar. Non-scalars, `NaN`, `Inf`, or negative values raise MATLAB-compatible errors.",
            "Public MATLAB matrix classes are single and double. RunMat mode separately admits logical matrices and all eight real integer classes through named extensions; integer matrix values cross only after exact binary64 representability checks.",
            "`rank([])` returns `0`. Rank is always reported as a double scalar (e.g., `2.0`).",
            "Complex inputs use a complex SVD so that conjugate transposes and magnitudes follow MATLAB’s conventions.",
            "A typed-integer tolerance is a separate RunMat extension because the public scalar tolerance does not enumerate integer classes. It must be exactly representable as binary64 before threshold computation; logical tolerance is independently gated.",
            "Resident integer input bypasses floating provider rank kernels, gathers through its exact owner, and restores the double rank scalar to that owner when supported."
        ]
    },
    BuiltinDocumentationSection {
        heading: "Acceleration and residency",
        paragraphs: &[
            "Providers may implement a dedicated rank hook; current backends gather to the host and reuse the shared SVD path.",
            "When a GPU acceleration provider is active, RunMat first offers the computation through the reserved `rank` provider hook. Backends that implement it can stay fully on-device and return a `gpuTensor` scalar. Providers without that hook—including today’s WGPU backend—gather the matrix to host memory, reuse the shared SVD logic, and then re-upload the scalar rank so downstream kernels continue on the GPU without user intervention. Auto-offload treats the builtin as an eager sink, so any fused producers flush before `rank` executes and residency bookkeeping remains consistent.",
            "RunMat’s planner automatically keeps matrices on the GPU when a provider implements the `rank` hook. If the hook is missing, the builtin transparently gathers the matrix, computes the SVD on the CPU, and uploads the scalar result so later GPU work remains resident. You can still seed residency manually with `gpuArray` for MATLAB compatibility, but it is rarely required."
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
        title: "Determining the rank of a full matrix",
        program: "A = [1 2; 3 4];\nrk = rank(A)",
        display_output: Some("rk = 2"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(rk == 2);"
        }
    },
    BuiltinExample {
        id: "example-2",
        title: "Detecting rank deficiency in a singular matrix",
        program: "B = [1 2; 2 4];\nrk = rank(B)",
        display_output: Some("rk = 1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(rk == 1);"
        }
    },
    BuiltinExample {
        id: "example-3",
        title: "Applying a custom tolerance to suppress tiny singular values",
        program: "C = diag([1, 1e-12]);\nrk_default = rank(C);          % counts both singular values (rank 2)\nrk_custom  = rank(C, 1e-9);    % treats the small value as zero (rank 1)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(rk_default == 2 && rk_custom == 1);"
        }
    },
    BuiltinExample {
        id: "example-4",
        title: "Computing the rank of a tall matrix",
        program: "A = [1 0; 0 0; 0 1];\nrk = rank(A)",
        display_output: Some("rk = 2"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(rk == 2);"
        }
    },
    BuiltinExample {
        id: "example-5",
        title: "Evaluating the rank of a complex matrix",
        program: "Z = [1+1i 0; 0 2-3i];\nrk = rank(Z)",
        display_output: Some("rk = 2"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(rk == 2);"
        }
    },
    BuiltinExample {
        id: "example-6",
        title: "Checking the rank of an empty matrix",
        program: "E = [];\nrk = rank(E)",
        display_output: Some("rk = 0"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(rk == 0);"
        }
    },
    BuiltinExample {
        id: "example-7",
        title: "Using `rank` with `gpuArray` data",
        program: "G = gpuArray([1 2 3; 3 6 9; 0 1 0]);\nrk = rank(G);      % Computation stays on the GPU when the provider supports it\nrk_host = gather(rk)",
        display_output: Some("rk_host = 2"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(rk_host == 2);"
        }
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "How is the default tolerance chosen?",
        answer: "RunMat computes the default tolerance exactly as MATLAB: `max(size(A)) * eps(max(s))`, where `s` are the singular values of `A`. This scales the cutoff with matrix size and magnitude."
    },
    BuiltinDocumentationFaq {
        question: "What does `rank([])` return?",
        answer: "The rank of the empty matrix is `0`. This matches MATLAB’s convention that an empty product has neutral value."
    },
    BuiltinDocumentationFaq {
        question: "Does `rank` return an integer or a double?",
        answer: "`rank` returns a double-precision scalar, mirroring MATLAB’s numeric tower. The value is always an integer-valued double."
    },
    BuiltinDocumentationFaq {
        question: "How does `rank` behave for vectors or scalars?",
        answer: "Scalars are treated as `1×1` matrices. `rank([0])` returns `0`, while `rank([5])` returns `1`. Row or column vectors behave as matrices with one dimension equal to 1."
    },
    BuiltinDocumentationFaq {
        question: "Can `rank` detect symbolic rank or exact arithmetic?",
        answer: "No. Like MATLAB, RunMat’s `rank` relies on floating-point SVD and is subject to the chosen tolerance. For symbolic or exact arithmetic you would use a computer algebra system."
    },
    BuiltinDocumentationFaq {
        question: "Will `rank` participate in fusion or auto-offload?",
        answer: "No. `rank` is a residency sink that eagerly computes an SVD. Fusion groups terminate before the call, and the planner treats the builtin as a scalar reduction."
    },
    BuiltinDocumentationFaq {
        question: "Is the tolerance argument optional?",
        answer: "Yes. `rank(A)` uses the default tolerance and mirrors MATLAB. Supplying `rank(A, tol)` overrides the cutoff. Non-scalar or negative tolerances raise MATLAB-compatible errors."
    },
    BuiltinDocumentationFaq {
        question: "What happens if the matrix contains NaNs or Infs?",
        answer: "Singular values involving `NaN` propagate and typically produce a rank of `0`. Infinite values yield infinite singular values and therefore produce a rank equal to the number of infinite entries above tolerance, matching MATLAB’s behaviour."
    },
    BuiltinDocumentationFaq {
        question: "Does `rank` allocate large temporary buffers?",
        answer: "Only enough memory for the SVD factors. For host execution this is handled by `nalgebra` (and LAPACK when enabled). GPU providers are free to reuse buffers or stream the computation."
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "pinv",
        target: BuiltinDocumentationLinkTarget::Builtin("pinv"),
    },
    BuiltinDocumentationLink {
        label: "svd",
        target: BuiltinDocumentationLinkTarget::Builtin("svd"),
    },
    BuiltinDocumentationLink {
        label: "inv",
        target: BuiltinDocumentationLinkTarget::Builtin("inv"),
    },
    BuiltinDocumentationLink {
        label: "det",
        target: BuiltinDocumentationLinkTarget::Builtin("det"),
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
        label: "linsolve",
        target: BuiltinDocumentationLinkTarget::Builtin("linsolve"),
    },
    BuiltinDocumentationLink {
        label: "norm",
        target: BuiltinDocumentationLinkTarget::Builtin("norm"),
    },
    BuiltinDocumentationLink {
        label: "rcond",
        target: BuiltinDocumentationLinkTarget::Builtin("rcond"),
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("rank"),
    slug: Some("rank"),
    summary: "Compute the numerical rank of a matrix using SVD with MATLAB-compatible tolerance handling.",
    description: "`r = rank(A)` returns the numerical rank of a real or complex matrix `A`. The rank is the number of singular values above a tolerance derived from matrix size and scale, with MATLAB-compatible default and explicit-tolerance behavior.",
    keywords: &[
        "rank",
        "singular value decomposition",
        "tolerance",
        "matrix rank",
        "gpu"
    ],
    related: &[
        "pinv",
        "svd",
        "inv",
        "det",
        "gpuarray",
        "gather",
        "cond",
        "linsolve",
        "norm",
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
                label: "crates/runmat-runtime/src/builtins/math/linalg/solve/rank.rs",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/solve/rank.rs")
            }
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "unit",
                location: "builtins::math::linalg::solve::rank::tests"
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::ProviderTest,
                label: "gpu",
                location: "builtins::math::linalg::solve::rank::tests::rank_gpu_round_trip"
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::WgpuTest,
                label: "wgpu",
                location: "builtins::math::linalg::solve::rank::tests::rank_wgpu_matches_cpu"
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
