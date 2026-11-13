use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("false"),
    slug: Some("false"),
    summary: "Create logical arrays filled with false values.",
    description: concat!(
        "`false` creates logical arrays filled with ",
        "false values. MATLAB-compatible mode accepts ",
        "scalar dimensions, multiple scalar ",
        "dimensions, a row size vector, and `'like'`. ",
        "Integer dimensions are read exactly, negative ",
        "dimensions become zero, and trailing ",
        "singleton dimensions are normalized according ",
        "to MATLAB array-shape rules.",
    ),
    keywords: &["false", "logical", "array", "mask"],
    related: &[],

    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "`false()` returns the scalar logical `false`.",
                concat!(
                    "`false(n)` returns an n x n logical array of ",
                    "false values.",
                ),
                concat!(
                    "`false(m, n, ...)` returns a logical array ",
                    "with the requested dimensions.",
                ),
                concat!(
                    "`false(sz)` accepts one row size vector. A ",
                    "column vector or multiple vector size inputs ",
                    "are rejected.",
                ),
                concat!(
                    "All eight integer classes are exact size ",
                    "controls. Negative signed dimensions become ",
                    "zero; noninteger, nonfinite, or ",
                    "platform-unrepresentable dimensions reject ",
                    "before allocation.",
                ),
                concat!(
                    "`false(___, 'like', prototype)` preserves ",
                    "applicable sparsity or owning-provider ",
                    "residency while the result remains logical. ",
                    "The prototype never supplies output shape, so ",
                    "`false('like', prototype)` is 1 × 1.",
                ),
                concat!(
                    "`false(A)`, the `'logical'` class option, ",
                    "single-precision size controls, and resident ",
                    "size controls are independently gated RunMat ",
                    "extensions rather than MATLAB-compatible ",
                    "syntax. An admitted resident size control is ",
                    "gathered structurally and still produces host ",
                    "logical output unless residency is requested ",
                    "explicitly with `'like'`.",
                ),
            ],
        },
        BuiltinDocumentationSection {
            heading: "GPU residency and provider behavior",
            paragraphs: &[concat!(
                "A resident `'like'` prototype creates a ",
                "logical result on that handle's owning ",
                "provider. Numeric size arguments remain ",
                "structural controls and do not independently ",
                "request GPU output.",
            )],
        },
    ],

    examples: &[
        BuiltinExample {
            id: "example-1",
            title: "Creating a 2x3 logical array of false values",
            program: "mask = false(2, 3)",
            display_output: Some("mask = [0 0 0; 0 0 0]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: concat!(
                    "assert(islogical(mask)); ",
                    "assert(isequal(size(mask), [2 3])); ",
                    "assert(all(mask(:) == false));",
                ),
            },
        },
        BuiltinExample {
            id: "example-2",
            title: "Creating a logical array from a size vector",
            program: "dims = [1 4 2];\nmask = false(dims)",
            display_output: Some(concat!(
                "mask =\n  1x4x2 logical array\n     0     0     ",
                "0     0\n     0     0     0     0",
            )),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: concat!(
                    "assert(islogical(mask)); ",
                    "assert(isequal(size(mask), [1 4 2])); ",
                    "assert(all(mask(:) == false));",
                ),
            },
        },
        BuiltinExample {
            id: "example-3",
            title: concat!(
                "Creating a false mask with the same size and ",
                "residency as an existing array",
            ),
            program: "A = rand(2, 2);\nmask = false(size(A), 'like', A)",
            display_output: Some(concat!(
                "mask =\n  2x2 logical array\n     0     0\n     ",
                "0     0",
            )),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: concat!(
                    "assert(islogical(mask)); ",
                    "assert(isequal(size(mask), size(A))); ",
                    "assert(all(mask(:) == false));",
                ),
            },
        },
    ],
    example_exemption: None,

    faqs: &[
        BuiltinDocumentationFaq {
            question: "What does `false()` return?",
            answer: "It returns the logical scalar `false`.",
        },
        BuiltinDocumentationFaq {
            question: "Does `false(n)` create a square array?",
            answer: concat!(
                "Yes. `false(n)` returns an n x n logical ",
                "array of false values, matching MATLAB ",
                "behavior.",
            ),
        },
        BuiltinDocumentationFaq {
            question: "How does `false(sz)` interpret a size vector?",
            answer: concat!(
                "A row vector is treated as the full set of ",
                "dimensions, so `false([2 3 4])` yields a ",
                "2x3x4 logical array. Column vectors and ",
                "multiple vector inputs are rejected.",
            ),
        },
        BuiltinDocumentationFaq {
            question: "Does `false` keep results on the GPU?",
            answer: concat!(
                "Yes, when residency is requested explicitly ",
                "with `'like'`. RunMat constructs the logical ",
                "result through the prototype handle's owning ",
                "provider; a GPU value supplied as an ordinary ",
                "size control does not implicitly determine ",
                "shape and residency at once.",
            ),
        },
        BuiltinDocumentationFaq {
            question: "Is the `'logical'` option required?",
            answer: concat!(
                "No. `false` always returns logical values. ",
                "RunMat retains the redundant `'logical'` ",
                "spelling only in extension mode.",
            ),
        },
    ],

    links: &[
        BuiltinDocumentationLink {
            label: "true",
            target: BuiltinDocumentationLinkTarget::Builtin("true"),
        },
        BuiltinDocumentationLink {
            label: "logical",
            target: BuiltinDocumentationLinkTarget::Builtin("logical"),
        },
        BuiltinDocumentationLink {
            label: "islogical",
            target: BuiltinDocumentationLinkTarget::Builtin("islogical"),
        },
        BuiltinDocumentationLink {
            label: "zeros",
            target: BuiltinDocumentationLinkTarget::Builtin("zeros"),
        },
        BuiltinDocumentationLink {
            label: "colon",
            target: BuiltinDocumentationLinkTarget::Builtin("colon"),
        },
        BuiltinDocumentationLink {
            label: "eye",
            target: BuiltinDocumentationLinkTarget::Builtin("eye"),
        },
        BuiltinDocumentationLink {
            label: "fill",
            target: BuiltinDocumentationLinkTarget::Builtin("fill"),
        },
        BuiltinDocumentationLink {
            label: "linspace",
            target: BuiltinDocumentationLinkTarget::Builtin("linspace"),
        },
        BuiltinDocumentationLink {
            label: "logspace",
            target: BuiltinDocumentationLinkTarget::Builtin("logspace"),
        },
        BuiltinDocumentationLink {
            label: "magic",
            target: BuiltinDocumentationLinkTarget::Builtin("magic"),
        },
        BuiltinDocumentationLink {
            label: "meshgrid",
            target: BuiltinDocumentationLinkTarget::Builtin("meshgrid"),
        },
        BuiltinDocumentationLink {
            label: "ones",
            target: BuiltinDocumentationLinkTarget::Builtin("ones"),
        },
        BuiltinDocumentationLink {
            label: "rand",
            target: BuiltinDocumentationLinkTarget::Builtin("rand"),
        },
        BuiltinDocumentationLink {
            label: "randi",
            target: BuiltinDocumentationLinkTarget::Builtin("randi"),
        },
        BuiltinDocumentationLink {
            label: "randn",
            target: BuiltinDocumentationLinkTarget::Builtin("randn"),
        },
        BuiltinDocumentationLink {
            label: "randperm",
            target: BuiltinDocumentationLinkTarget::Builtin("randperm"),
        },
        BuiltinDocumentationLink {
            label: "range",
            target: BuiltinDocumentationLinkTarget::Builtin("range"),
        },
        BuiltinDocumentationLink {
            label: concat!(
                "`crates/runmat-runtime/src/builtins/array/cre",
                "ation/logical_constructor.rs`",
            ),
            target: BuiltinDocumentationLinkTarget::Source(concat!(
                "https://github.com/runmat-org/runmat/blob/mai",
                "n/crates/runmat-runtime/src/builtins/array/cr",
                "eation/logical_constructor.rs",
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
                "eation/logical_constructor.rs",
            )),
        }],

        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Constructor semantic and failure tests",
            location: "builtins::array::creation::logical_constructor::tests",
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
