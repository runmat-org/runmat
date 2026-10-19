use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("nan"),
    slug: Some("nan"),
    summary: concat!(
        "Create arrays filled with IEEE NaN values ",
        "using MATLAB-compatible size, class, and ",
        "`like` forms.",
    ),
    description: concat!(
        "`nan(...)` creates NaN-filled arrays across ",
        "scalar, vector, matrix, and N-D size ",
        "signatures, with MATLAB-compatible class and ",
        "`'like'` option behavior.",
    ),
    keywords: &["nan", "array", "single", "gpu", "like"],
    related: &[],

    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "`nan()` returns the scalar `NaN`.",
                concat!(
                    "`nan(n)` returns an `n x n` double array ",
                    "filled with `NaN`.",
                ),
                concat!(
                    "`nan(m, n, ...)` returns a dense double array ",
                    "with the requested dimensions.",
                ),
                concat!(
                    "`nan(sz)` accepts a size vector and returns ",
                    "an array with `prod(sz)` NaN elements.",
                ),
                concat!(
                    "`nan(A)` interprets numeric scalar or ",
                    "size-vector values as dimensions; a general ",
                    "array is not an implicit shape prototype. Use ",
                    "`nan(size(A))` to copy its shape.",
                ),
                concat!(
                    "`nan(___, 'single')` returns a ",
                    "single-precision numeric array.",
                ),
                concat!(
                    "`nan(___, 'like', prototype)` matches numeric ",
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
                    "a host NaN tensor.",
                ),
                concat!(
                    "When a GPU prototype is supplied, RunMat ",
                    "preserves residency by allocating the output ",
                    "through the prototype handle's owning ",
                    "acceleration provider. If the provider cannot ",
                    "fill on device, RunMat falls back to a host ",
                    "NaN tensor and uploads it.",
                ),
                concat!(
                    "Provider-backed `nan` construction uses ",
                    "constant-fill hooks, so fully resident GPU ",
                    "pipelines can create NaN arrays without first ",
                    "materializing a host tensor when provider ",
                    "support is available.",
                ),
            ],
        },
    ],

    examples: &[
        BuiltinExample {
            id: "example-1",
            title: "Creating a 2x3 matrix of NaN values",
            program: "A = nan(2, 3)",
            display_output: Some("A = [NaN NaN NaN; NaN NaN NaN]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: concat!(
                    "assert(all(isnan(gather(A(:))))); ",
                    "assert(isequal(size(A), [2 3]));",
                ),
            },
        },
        BuiltinExample {
            id: "example-2",
            title: "Creating a single-precision NaN array",
            program: "A = nan(4, 1, 'single')",
            display_output: Some("A = [NaN; NaN; NaN; NaN]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: concat!(
                    "assert(all(isnan(gather(A(:))))); ",
                    "assert(isequal(size(A), [4 1])); ",
                    "assert(isa(A, 'single'));",
                ),
            },
        },
        BuiltinExample {
            id: "example-3",
            title: "Creating a GPU-resident NaN array like a prototype",
            program: "G = gpuArray(rand(2,2));\nA = nan(2, 2, 'like', G)",
            display_output: Some("A = [NaN NaN; NaN NaN]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Wgpu,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: concat!(
                    "host = gather(A); prototype = gather(G); ",
                    "assert(all(isnan(host(:)))); ",
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
            question: "How is `nan` different from `NaN`?",
            answer: concat!(
                "Both bare identifiers resolve to the scalar ",
                "IEEE NaN constant. The callable form ",
                "`nan(...)` constructs NaN-filled arrays.",
            ),
        },
        BuiltinDocumentationFaq {
            question: "Does `nan(n)` create a row vector?",
            answer: concat!(
                "No. MATLAB compatibility requires `nan(n)` to ",
                "create an `n x n` matrix. Use `nan(1, n)` or ",
                "`nan(n, 1)` for vectors.",
            ),
        },
        BuiltinDocumentationFaq {
            question: "Can `nan` create GPU arrays?",
            answer: concat!(
                "Yes. With a GPU prototype, `nan(..., 'like', ",
                "G)` asks the prototype handle's owning ",
                "acceleration provider to allocate a ",
                "NaN-filled GPU tensor.",
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
            label: "isnan",
            target: BuiltinDocumentationLinkTarget::Builtin("isnan"),
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
                "ation/nan.rs`",
            ),
            target: BuiltinDocumentationLinkTarget::Source(concat!(
                "https://github.com/runmat-org/runmat/blob/mai",
                "n/crates/runmat-runtime/src/builtins/array/cr",
                "eation/nan.rs",
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
                "eation/nan.rs",
            )),
        }],

        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Constructor semantic and failure tests",
            location: "builtins::array::creation::nan::tests",
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
