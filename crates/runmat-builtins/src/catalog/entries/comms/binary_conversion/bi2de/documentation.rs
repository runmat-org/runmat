use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Runtime implementation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/comms/binary_conversion.rs",
    ),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("bi2de"),
    slug: Some("bi2de"),
    summary: "Convert base-P digit rows to decimal values.",
    description: "`bi2de(B)` interprets each row of a real numeric or logical digit array as one base-2 number. The first column holds the least significant digit by default. Specify a base `P` and/or `'left-msb'` to change the interpretation.",
    keywords: &[
        "bi2de",
        "binary",
        "decimal",
        "base",
        "communications",
        "right-msb",
        "left-msb",
    ],
    related: &["de2bi"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Digits, shape, and class",
            paragraphs: &[
                "A row vector produces one scalar. Each row of a matrix produces one element of a column vector. The default base is two and the default order is `'right-msb'`; `'left-msb'` makes the first column most significant.",
                "Digits must be finite nonnegative integers less than `P`. Results above 2^52 reject rather than pass through an inexact decimal conversion. Double and logical inputs produce double values; single input produces single. Integer input retains its class if the result fits and otherwise widens within the same signedness.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Execution",
            paragraphs: &[
                "Digit validation and conversion run on the host; the operation is a fusion boundary and does not use a GPU kernel. RunMat mode can gather a GPU-resident digit matrix after admission. MATLAB-compatible mode rejects that form before provider access.",
                "A fixed-width integer base `P` is a RunMat extension. MATLAB-compatible mode accepts the documented single/double numeric base. Neither form reads the workspace, invokes callbacks, performs I/O, nor depends on time or randomness.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "right-msb",
            title: "Convert default right-MSB digits",
            program: "d = bi2de([1 0 1 0])",
            display_output: Some("d = 5"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(d,5));",
            },
        },
        BuiltinExample {
            id: "left-msb",
            title: "Read the first column as most significant",
            program: "d = bi2de([1 0 1 0], 'left-msb')",
            display_output: Some("d = 10"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(d,10));",
            },
        },
        BuiltinExample {
            id: "base-three-rows",
            title: "Convert two base-3 digit rows",
            program: "d = bi2de([2 1 0; 0 1 2], 3, 'left-msb')",
            display_output: Some("d = [21; 5]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(d,[21;5])); assert(isequal(size(d),[2 1]));",
            },
        },
        BuiltinExample {
            id: "integer-class",
            title: "Keep an integer result's class",
            program: "d = bi2de(uint8([1 0 1]))",
            display_output: Some("d = uint8(5)"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(d,uint8(5))); assert(strcmp(class(d),'uint8'));",
            },
        },
        BuiltinExample {
            id: "invalid-digit",
            title: "Reject a digit outside the base",
            program: "bi2de([1 2])",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::ExpectedError {
                identifier: "MATLAB:bi2de:InvalidInput",
            },
        },
        BuiltinExample {
            id: "native-aot-contract",
            title: "Compile digit conversion and error handling",
            program: "B = [1 0 1; 0 1 1]; D = bi2de(B); L = bi2de([1 0 1 0],'left-msb'); caught = false; try; bad = bi2de([1 2]); catch failure; caught = strcmp(failure.identifier,'RunMat:bi2de:InvalidInput'); end",
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
                source: "assert(isequal(D,[5;6]) && isequal(size(D),[2 1]) && isa(D,'double')); assert(L == 10 && caught);",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "What is the default digit ordering?",
            answer: "`'right-msb'`. The first column is least significant unless you pass `'left-msb'`.",
        },
        BuiltinDocumentationFaq {
            question: "Can bi2de convert bases other than two?",
            answer: "Yes. Pass an integer base `P` of at least two, such as `bi2de(B,3)`.",
        },
        BuiltinDocumentationFaq {
            question: "What happens if the decimal result is too large?",
            answer: "The conversion rejects results above 2^52 rather than silently losing exactness.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "de2bi",
            target: BuiltinDocumentationLinkTarget::Builtin("de2bi"),
        },
        BuiltinDocumentationLink {
            label: "MATLAB bi2de reference",
            target: BuiltinDocumentationLinkTarget::External(
                "https://www.mathworks.com/help/comm/ref/bi2de.html",
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
                label: "Digit conversion, bases, shapes, classes, and invalid inputs",
                location: "crates/runmat-runtime/src/builtins/comms/binary_conversion.rs",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "VM conversion calls",
                location: "crates/runmat-vm/tests/comms.rs",
            },
        ],
        notes: &[
            "The documented input/output class rules follow the public Communications Toolbox reference. GPU input and fixed-width integer P are explicitly RunMat-only forms.",
        ],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
