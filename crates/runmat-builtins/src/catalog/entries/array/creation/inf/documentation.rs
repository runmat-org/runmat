use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("inf"),
    slug: Some("inf"),
    summary: concat!(
        "Create arrays filled with IEEE positive ",
        "infinity values using MATLAB-compatible size, ",
        "class, and `like` forms.",
    ),
    description: concat!(
        "`inf(...)` creates positive-infinity-filled ",
        "arrays across scalar, vector, matrix, and N-D ",
        "size signatures, with MATLAB-compatible class ",
        "and `'like'` option behavior.",
    ),
    keywords: &["inf", "infinity", "array", "single", "gpu", "like"],
    related: &[],

    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "`inf()` returns the scalar `Inf`.",
                concat!(
                    "`inf(n)` returns an `n x n` double array ",
                    "filled with `Inf`.",
                ),
                concat!(
                    "`inf(m, n, ...)` returns a dense double array ",
                    "with the requested dimensions.",
                ),
                concat!(
                    "`inf(sz)` accepts a size vector and returns ",
                    "an array with `prod(sz)` Inf elements.",
                ),
                concat!(
                    "`inf(A)` interprets numeric scalar or ",
                    "size-vector values as dimensions; a general ",
                    "array is not an implicit shape prototype. Use ",
                    "`inf(size(A))` to copy its shape.",
                ),
                concat!(
                    "`inf(___, 'single')` returns a ",
                    "single-precision numeric array.",
                ),
                concat!(
                    "`inf(___, 'like', prototype)` matches numeric ",
                    "precision and device residency for supported ",
                    "prototype arrays.",
                ),
            ],
        },
        BuiltinDocumentationSection {
            heading: "GPU residency and provider behavior",
            paragraphs: &[
                concat!(
                    "Uses provider constant-fill hooks when ",
                    "available; otherwise falls back to uploading ",
                    "a host Inf tensor.",
                ),
                concat!(
                    "When a GPU prototype is supplied, RunMat ",
                    "preserves residency by allocating the output ",
                    "through the prototype handle's owning ",
                    "acceleration provider. If the provider cannot ",
                    "fill on device, RunMat falls back to a host ",
                    "Inf tensor and uploads it.",
                ),
                concat!(
                    "Provider-backed `inf` construction uses ",
                    "constant-fill hooks, so fully resident GPU ",
                    "pipelines can create Inf arrays without first ",
                    "materializing a host tensor when provider ",
                    "support is available.",
                ),
            ],
        },
    ],

    examples: &[
        BuiltinExample {
            id: "example-1",
            title: "Creating a 2x3 matrix of Inf values",
            program: "A = inf(2, 3)",
            display_output: Some("A = [Inf Inf Inf; Inf Inf Inf]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: concat!(
                    "assert(all(isinf(gather(A(:))))); ",
                    "assert(isequal(size(A), [2 3]));",
                ),
            },
        },
        BuiltinExample {
            id: "example-2",
            title: "Creating a single-precision Inf array",
            program: "A = inf(4, 1, 'single')",
            display_output: Some("A = [Inf; Inf; Inf; Inf]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: concat!(
                    "assert(all(isinf(gather(A(:))))); ",
                    "assert(isequal(size(A), [4 1])); ",
                    "assert(isa(A, 'single'));",
                ),
            },
        },
        BuiltinExample {
            id: "example-3",
            title: "Creating a GPU-resident Inf array like a prototype",
            program: "G = gpuArray(rand(2,2));\nA = inf(2, 2, 'like', G)",
            display_output: Some("A = [Inf Inf; Inf Inf]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Wgpu,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: concat!(
                    "host = gather(A); prototype = gather(G); ",
                    "assert(all(isinf(host(:)))); ",
                    "assert(strcmp(class(host), class(prototype))); ",
                    "assert(isequal(size(A), [2 2])); ",
                    "assert(isgpuarray(A));",
                ),
            },
        },
    ],
    example_exemption: None,

    faqs: &[
        BuiltinDocumentationFaq {
            question: "How is `inf` different from `Inf`?",
            answer: concat!(
                "Both bare identifiers resolve to the scalar ",
                "IEEE positive-infinity constant. The callable ",
                "form `inf(...)` constructs Inf-filled arrays.",
            ),
        },
        BuiltinDocumentationFaq {
            question: "Does `inf(n)` create a row vector?",
            answer: concat!(
                "No. MATLAB compatibility requires `inf(n)` to ",
                "create an `n x n` matrix. Use `inf(1, n)` or ",
                "`inf(n, 1)` for vectors.",
            ),
        },
        BuiltinDocumentationFaq {
            question: "Can `inf` create GPU arrays?",
            answer: concat!(
                "Yes. With a GPU prototype, `inf(..., 'like', ",
                "G)` asks the prototype handle's owning ",
                "acceleration provider to allocate an ",
                "Inf-filled GPU tensor.",
            ),
        },
    ],

    links: &[
        BuiltinDocumentationLink {
            label: "zeros",
            target: BuiltinDocumentationLinkTarget::Builtin("zeros"),
        },
        BuiltinDocumentationLink {
            label: "ones",
            target: BuiltinDocumentationLinkTarget::Builtin("ones"),
        },
        BuiltinDocumentationLink {
            label: "nan",
            target: BuiltinDocumentationLinkTarget::Builtin("nan"),
        },
        BuiltinDocumentationLink {
            label: "isinf",
            target: BuiltinDocumentationLinkTarget::Builtin("isinf"),
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
            label: concat!(
                "`crates/runmat-runtime/src/builtins/array/cre",
                "ation/inf.rs`",
            ),
            target: BuiltinDocumentationLinkTarget::Source(concat!(
                "https://github.com/runmat-org/runmat/blob/mai",
                "n/crates/runmat-runtime/src/builtins/array/cr",
                "eation/inf.rs",
            )),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Runtime",
            target: BuiltinDocumentationLinkTarget::Source(concat!(
                "https://github.com/runmat-org/runmat/blob/mai",
                "n/crates/runmat-runtime/src/builtins/array/cr",
                "eation/inf.rs",
            )),
        }],

        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Constructor semantic and failure tests",
            location: "builtins::array::creation::inf::tests",
        }],

        notes: &[concat!(
            "Legacy prose, behaviors, examples, FAQs, and ",
            "links are preserved here. Signatures, ",
            "compatibility, integer behavior, GPU policy ",
            "and fusion are authoritative typed ",
            "declarations; legacy requires_feature and ",
            "integration fields were null. Provider ",
            "hardware and native execution evidence are ",
            "separate integration gates.",
        )],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
