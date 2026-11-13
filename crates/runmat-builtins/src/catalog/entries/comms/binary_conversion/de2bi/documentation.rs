use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Runtime implementation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/comms/binary_conversion.rs",
    ),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("de2bi"),
    slug: Some("de2bi"),
    summary: "Convert decimal values to base-P digit rows.",
    description: "`de2bi(D)` expands each nonnegative integer value into a row of base-2 digits. Input matrices are flattened in column-major `D(:)` order. Optional width `N`, base `P`, and digit-order flag control the result.",
    keywords: &[
        "de2bi",
        "decimal",
        "binary",
        "base",
        "communications",
        "right-msb",
        "left-msb",
    ],
    related: &["bi2de"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Width, base, order, and class",
            paragraphs: &[
                "Output has one row per input element. If `N` is omitted or `[]`, width is the minimum needed for the largest value; zero needs one digit. Explicit `N` must be positive and wide enough for every value. `P` is at least two.",
                "`'right-msb'` puts the least significant digit in column one. `'left-msb'` puts the most significant digit there. The digit matrix retains the numeric class of `D` for double, single, and fixed-width integers.",
                "The documented input bound is 2^52. RunMat mode additionally expands larger exact native integers and flattens N-D numeric arrays in column-major order; those forms are rejected in MATLAB-compatible mode. A fixed-width integer width or base and logical input are also separately gated RunMat forms.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Execution",
            paragraphs: &[
                "Validation and digit expansion run on the host and form a fusion boundary. RunMat mode may gather a GPU-resident decimal array after compatibility admission; MATLAB-compatible mode rejects that form before provider access. A requested output exceeding the resource limit fails before allocation.",
                "This conversion has no callback, workspace, filesystem, network, randomness, or time effects.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "minimum-width",
            title: "Expand values to minimum-width binary rows",
            program: "b = de2bi([0;1;2;5])",
            display_output: Some("b = [0 0 0; 1 0 0; 0 1 0; 1 0 1]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(b,[0 0 0;1 0 0;0 1 0;1 0 1])); assert(isequal(size(b),[4 3]));",
            },
        },
        BuiltinExample {
            id: "fixed-width",
            title: "Use four left-MSB columns",
            program: "b = de2bi(5,4,'left-msb')",
            display_output: Some("b = [0 1 0 1]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(b,[0 1 0 1]));",
            },
        },
        BuiltinExample {
            id: "base-three",
            title: "Use an explicit base",
            program: "b = de2bi(21,4,3,'left-msb')",
            display_output: Some("b = [0 2 1 0]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(b,[0 2 1 0]));",
            },
        },
        BuiltinExample {
            id: "input-class",
            title: "Preserve the input class",
            program: "b = de2bi(uint8(5),4)",
            display_output: Some("b = uint8([1 0 1 0])"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(b,uint8([1 0 1 0]))); assert(strcmp(class(b),'uint8'));",
            },
        },
        BuiltinExample {
            id: "nd-extension",
            title: "Expand an N-D array in RunMat mode",
            program: "D = reshape([1 2],1,1,2); b = de2bi(D)",
            display_output: Some("b = [1 0; 0 1]"),
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(b),[2 2])); assert(isequal(b,[1 0;0 1]));",
            },
        },
        BuiltinExample {
            id: "invalid-width",
            title: "Reject an insufficient output width",
            program: "de2bi(8,3)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::ExpectedError {
                identifier: "MATLAB:de2bi:InvalidWidth",
            },
        },
        BuiltinExample {
            id: "native-aot-contract",
            title: "Compile digit expansion and error handling",
            program: "B = de2bi([0;1;2;5]); T = de2bi(21,4,3,'left-msb'); caught = false; try; bad = de2bi(8,3); catch failure; caught = strcmp(failure.identifier,'RunMat:de2bi:InvalidWidth'); end",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Native,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements {
                host: BuiltinExampleHostRequirement::NativeOnly,
                engine: BuiltinExampleEngine::Aot,
                ..BuiltinExampleRequirements::NONE
            },
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(B,[0 0 0;1 0 0;0 1 0;1 0 1]) && isequal(size(B),[4 3])); assert(isequal(T,[0 2 1 0]) && isa(T,'double') && caught);",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "How does de2bi choose the number of columns?",
            answer: "It uses the minimum number of digits needed for the largest input value unless `N` is supplied. Zero needs one digit.",
        },
        BuiltinDocumentationFaq {
            question: "Can I choose the base without specifying a width?",
            answer: "Yes. Use `[]` as the width placeholder, for example `de2bi(D,[],3)`.",
        },
        BuiltinDocumentationFaq {
            question: "Does de2bi change the input's numeric class?",
            answer: "No. Numeric output uses the same class as numeric input.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "bi2de",
            target: BuiltinDocumentationLinkTarget::Builtin("bi2de"),
        },
        BuiltinDocumentationLink {
            label: "MATLAB de2bi reference",
            target: BuiltinDocumentationLinkTarget::External(
                "https://www.mathworks.com/help/comm/ref/de2bi.html",
            ),
        },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Digit expansion, base, width, order, class, and invalid inputs",
                location: "crates/runmat-runtime/src/builtins/comms/binary_conversion.rs",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "VM conversion calls",
                location: "crates/runmat-vm/tests/comms.rs",
            },
        ],
        notes: &[
            "GPU, logical, N-D, integer controls, and exact wide inputs are separately classified RunMat forms.",
        ],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
