use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "The output has the same shape as `A` except the operated dimension has length 1.",
            "`p` may be any positive real scalar or `Inf`.",
            "The documented input array classes are single and double. Native integer `A` is a RunMat language extension that requires exact conversion to double and returns double output.",
            "Native integer storage for `p` is a separate RunMat language extension and must be exactly representable as double. Native integer `dim` is part of the compatibility surface and is decoded exactly as a structural dimension.",
            "Single and complex-single inputs return single output; double, complex-double, and native integer extension inputs return double output.",
            "Complex inputs are reduced using element magnitudes and return real non-negative values.",
            "NaN values propagate within the vector being reduced.",
            "When `dim` is greater than `ndims(A)`, or `size(A,dim)` is 1, the result is `abs(A)` with the original shape."
        ]
    },
    BuiltinDocumentationSection {
        heading: "Acceleration and residency",
        paragraphs: &[
            "When a provider does not supply a native kernel, RunMat gathers through the value's owning provider, computes the norm on the host, and restores the result to that provider when it can preserve the output's single or double precision.",
            "When no provider kernel is available, GPU inputs are gathered through their owning provider, vector-wise norms are computed with the same CPU logic as host arrays, and the result is restored to that provider when its physical precision matches the output class.",
            "The fallback includes NaNs and preserves compatibility-target shape semantics. If the provider cannot preserve the result class or restoration fails, RunMat returns the correctly typed host result rather than relabelling or changing the numeric answer.",
            "```matlab:runnable\nG = gpuArray([3 5; 4 12]);\nN = vecnorm(G)\n```\nExpected output:\n```matlab\nN = [5 13]\n```"
        ]
    },
    BuiltinDocumentationSection {
        heading: "Implementation and validation",
        paragraphs: &[
            "Original GPU declarations: precisions f32, f64; elementwise false; reduction true; broadcasting none.",
            "Original fusion declarations: elementwise false; reduction false; max inputs 1; constants inline."
        ]
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Computing column-wise Euclidean norms",
        program: "A = [2 0 1; -1 1 0; -3 3 0];\nN = vecnorm(A)",
        display_output: Some("N = [3.7417 3.1623 1]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(norm(N-[sqrt(14) sqrt(10) 1]) < 1e-4);",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Computing row-wise 1-norms",
        program: "A = [1 3; 2 4];\nN = vecnorm(A, 1, 2)",
        display_output: Some("N = [4; 6]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(norm(N-[4;6]) < 1e-10);",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Using a fractional positive norm order",
        program: "x = [1; 4; 9];\nN = vecnorm(x, 0.5)",
        display_output: Some("N = 36"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(N-36) < 1e-10);",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "Reducing complex values by magnitude",
        program: "z = [3+4i; 5+12i];\nN = vecnorm(z)",
        display_output: Some("N = 13.9284"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(N-sqrt(194)) < 1e-4);",
        },
    },
    BuiltinExample {
        id: "example-5",
        title: "Operating along a trailing dimension",
        program: "A = reshape(1:8, [2 2 2]);\nN = vecnorm(A, 2, 3)",
        display_output: Some("N is size 2x2x1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(size(N,1)==2 && size(N,2)==2 && size(N,3)==1);",
        },
    },
    BuiltinExample {
        id: "example-6",
        title: "Computing Vector Norms On GPU Data",
        program: "G = gpuArray([3 5; 4 12]);\nN = vecnorm(G)",
        display_output: Some("N = [5 13]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(norm(gather(N)-[5 13]) < 1e-10);",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "How is `vecnorm` different from `norm`?",
        answer: "`norm(A)` returns one scalar for a vector or matrix. `vecnorm(A)` reduces each vector along one dimension and returns an array of norms."
    },
    BuiltinDocumentationFaq {
        question: "What dimension does `vecnorm(A)` use by default?",
        answer: "It uses the first dimension whose size is not 1. For ordinary matrices this means column-wise norms."
    },
    BuiltinDocumentationFaq {
        question: "Does `vecnorm` support complex arrays?",
        answer: "Yes. Complex elements are converted to magnitudes before applying the selected vector norm."
    },
    BuiltinDocumentationFaq {
        question: "Can `p` be less than 1?",
        answer: "Yes. The compatibility target documents `p` as a positive scalar or `Inf`, and RunMat accepts positive fractional values."
    },
    BuiltinDocumentationFaq {
        question: "What happens for `dim` beyond the rank of the input?",
        answer: "RunMat follows the compatibility target by returning `abs(A)` with the original shape."
    },
    BuiltinDocumentationFaq {
        question: "Will GPU inputs stay on the GPU?",
        answer: "When a provider is active, RunMat gathers the input to compute the norm and uploads the result back to the provider. A future provider hook can replace this with a device-side kernel without changing user code."
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "norm",
        target: BuiltinDocumentationLinkTarget::Builtin("norm"),
    },
    BuiltinDocumentationLink {
        label: "sum",
        target: BuiltinDocumentationLinkTarget::Builtin("sum"),
    },
    BuiltinDocumentationLink {
        label: "hypot",
        target: BuiltinDocumentationLinkTarget::Builtin("hypot"),
    },
    BuiltinDocumentationLink {
        label: "pagenorm",
        target: BuiltinDocumentationLinkTarget::External(
            "https://www.mathworks.com/help/matlab/ref/pagenorm.html",
        ),
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
        label: "MATLAB Language Extensions",
        target: BuiltinDocumentationLinkTarget::External(
            "/docs/runtime/getting-started/language-extensions",
        ),
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("vecnorm"),
    slug: Some("vecnorm"),
    summary: "Compute vector-wise array norms.",
    description: "`vecnorm(A)` returns vector-wise 2-norms. Matrices reduce down columns by default, multidimensional arrays reduce along the first nonsingleton dimension, and `vecnorm(A,p,dim)` reduces along an explicit dimension.",
    keywords: &[
        "vecnorm",
        "vector norm",
        "array norm",
        "euclidean",
        "gpu"
    ],
    related: &[
        "norm",
        "sum",
        "hypot",
        "gpuarray",
        "gather"
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
                label: "`crates/runmat-runtime/src/builtins/math/linalg/solve/vecnorm.rs`",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/solve/vecnorm.rs")
            }
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "unit",
                location: "builtins::math::linalg::solve::vecnorm::tests"
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::ProviderTest,
                label: "gpu",
                location: "builtins::math::linalg::solve::vecnorm::tests::vecnorm_gpu_roundtrip_matches_cpu"
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::WgpuTest,
                label: "wgpu",
                location: "builtins::math::linalg::solve::vecnorm::tests::vecnorm_wgpu_matches_cpu"
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
