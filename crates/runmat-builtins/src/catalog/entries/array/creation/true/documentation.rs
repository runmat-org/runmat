use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("true"),
    slug: Some("true"),
    summary: concat!(
        "Create logical arrays filled with `true` ",
        "values using MATLAB-compatible size forms.",
    ),
    description: concat!(
        "`true` creates logical arrays populated with ",
        "`true` values across scalar, vector, matrix, ",
        "and N-D size specifications, matching ",
        "MATLAB-compatible sizing behavior.",
    ),
    keywords: &["true", "logical", "array", "mask"],
    related: &[],

    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "`true()` returns the scalar logical `true`.",
                concat!(
                    "`true(n)` returns an n x n logical array of ",
                    "true values.",
                ),
                concat!(
                    "`true(m, n, ...)` returns a logical array ",
                    "with the requested dimensions.",
                ),
                concat!(
                    "`true(sz)` accepts a row size vector and ",
                    "returns an array with prod(sz) elements ",
                    "arranged using MATLAB column-major ordering.",
                ),
                concat!(
                    "`true(___, 'like', prototype)` uses the ",
                    "preceding size arguments and preserves ",
                    "supported sparse or gpuArray placement from ",
                    "the prototype; the output class remains ",
                    "logical.",
                ),
                concat!(
                    "All eight integer classes are accepted as ",
                    "exact size controls. Negative signed ",
                    "dimensions produce empty dimensions.",
                ),
                concat!(
                    "`true(A)` shape inference, single-precision ",
                    "size controls, resident size controls, and ",
                    "the trailing `'logical'` option are optional ",
                    "RunMat language extensions.",
                ),
            ],
        },
        BuiltinDocumentationSection {
            heading: "GPU residency and provider behavior",
            paragraphs: &[concat!(
                "A `like` prototype preserves supported sparse ",
                "or gpuArray placement while the output class ",
                "remains logical.",
            )],
        },
    ],

    examples: &[
        BuiltinExample {
            id: "example-1",
            title: "Creating a 2x3 logical array of true values",
            program: "mask = true(2, 3)",
            display_output: Some("mask = [1 1 1; 1 1 1]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: concat!(
                    "assert(islogical(mask)); ",
                    "assert(isequal(size(mask), [2 3])); ",
                    "assert(all(mask(:) == true));",
                ),
            },
        },
        BuiltinExample {
            id: "example-2",
            title: "Creating a logical array from a size vector",
            program: "dims = [2 1 3];\nmask = true(dims)",
            display_output: Some(concat!(
                "mask =\n  2x1x3 logical array\n     1\n     1\n   ",
                "  1\n     1\n     1\n     1",
            )),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: concat!(
                    "assert(islogical(mask)); ",
                    "assert(isequal(size(mask), [2 1 3])); ",
                    "assert(all(mask(:) == true));",
                ),
            },
        },
        BuiltinExample {
            id: "example-3",
            title: "Creating a true mask like an existing array",
            program: "A = rand(3, 2);\nmask = true(size(A), 'like', A)",
            display_output: Some(concat!(
                "mask =\n  3x2 logical array\n     1     1\n     ",
                "1     1\n     1     1",
            )),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: concat!(
                    "assert(islogical(mask)); ",
                    "assert(isequal(size(mask), size(A))); ",
                    "assert(all(mask(:) == true));",
                ),
            },
        },
    ],
    example_exemption: None,

    faqs: &[
        BuiltinDocumentationFaq {
            question: "What does `true()` return?",
            answer: "It returns the logical scalar `true`.",
        },
        BuiltinDocumentationFaq {
            question: "Does `true(n)` create a square array?",
            answer: concat!(
                "Yes. `true(n)` returns an n x n logical array ",
                "of true values, matching MATLAB behavior.",
            ),
        },
        BuiltinDocumentationFaq {
            question: "How does `true(sz)` interpret a size vector?",
            answer: concat!(
                "A row vector is treated as the full set of ",
                "dimensions, so `true([2 3 4])` yields a 2x3x4 ",
                "logical array.",
            ),
        },
        BuiltinDocumentationFaq {
            question: "Does `true` keep results on the GPU?",
            answer: concat!(
                "Yes. When a gpuArray is supplied after ",
                "`like`, the logical result is allocated ",
                "through that array's owning provider and ",
                "remains resident.",
            ),
        },
        BuiltinDocumentationFaq {
            question: "Is the `'logical'` option required?",
            answer: concat!(
                "No. `true` always returns logical values. ",
                "RunMat can accept the redundant option when ",
                "language extensions are enabled.",
            ),
        },
    ],

    links: &[
        BuiltinDocumentationLink {
            label: "false",
            target: BuiltinDocumentationLinkTarget::Builtin("false"),
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
