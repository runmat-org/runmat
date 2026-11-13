use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "String scalars and character row vectors are accepted as input text.",
        "The optional encoding argument can be a string scalar or character row vector.",
        "The default encoding is UTF-8.",
        "Encoding names are resolved through the runtime encoding library plus explicit Unicode encoders, including common aliases such as `UTF-8`, `latin1`, `US-ASCII`, `windows-1252`, `Shift_JIS`, `UTF-16`, and `UTF-32`.",
        "`UTF-16` emits a little-endian byte-order mark followed by UTF-16LE bytes, matching MATLAB's documented output form.",
        "`UTF-16LE` and `UTF-16BE` produce UTF-16 bytes without a byte-order mark.",
        "`UTF-32` emits a little-endian byte-order mark followed by UTF-32LE bytes. `UTF-32LE` and `UTF-32BE` produce UTF-32 bytes without a byte-order mark.",
        "Characters that cannot be represented in a legacy byte encoding are substituted with `?` (`63`) instead of HTML numeric character references.",
        "The result is always a host `uint8` tensor with shape `1xn`, including `1x0` for empty input.",
        "Non-scalar string arrays, multi-row character arrays, unsupported encodings, non-text inputs, and extra arguments raise descriptive errors.",
    ] },
    BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &[
        "Text encoding runs on the CPU. GPU-resident inputs are gathered before validation and encoding.",
    ] },
    BuiltinDocumentationSection { heading: "Syntax", paragraphs: &[
        "bytes = unicode2native(unicodestr)\nbytes = unicode2native(unicodestr, encoding)",
        "`unicodestr` must be a string scalar or character vector.",
        "`encoding` must be a string scalar or character vector containing an encoding name.",
        "Returns `bytes`, a `uint8` row vector.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Convert Text With The Default Encoding",
        program: "bytes = unicode2native(\"hello\")",
        display_output: Some("bytes =\n    1x5 uint8 row vector\n    104  101  108  108  111"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(bytes, uint8([104 101 108 108 111])));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Encode Text As UTF-8",
        program: "bytes = unicode2native(\"hé\", \"UTF-8\")",
        display_output: Some("bytes =\n    1x3 uint8 row vector\n    104  195  169"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(bytes, uint8([104 195 169])));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Encode Text As Latin-1",
        program: "bytes = unicode2native('café', 'latin1')",
        display_output: Some("bytes =\n    1x4 uint8 row vector\n    99  97  102  233"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(bytes, uint8([99 97 102 233])));",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "Encode Text As UTF-16",
        program: "bytes = unicode2native(\"hi\", \"UTF-16\")",
        display_output: Some("bytes =\n    1x6 uint8 row vector\n    255  254  104  0  105  0"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(bytes, uint8([255 254 104 0 105 0])));",
        },
    },
    BuiltinExample {
        id: "example-5",
        title: "Encode Text As UTF-32",
        program: "bytes = unicode2native(\"A\", \"UTF-32\")",
        display_output: Some("bytes =\n    1x8 uint8 row vector\n    255  254  0  0  65  0  0  0"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(bytes, uint8([255 254 0 0 65 0 0 0])));",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "What does `unicode2native` return?", answer: "It returns a `uint8` row vector containing the encoded byte values." },
    BuiltinDocumentationFaq { question: "What is the default encoding?", answer: "RunMat uses UTF-8 as the default native encoding." },
    BuiltinDocumentationFaq { question: "Can I pass a string array?", answer: "Only scalar string arrays are accepted. MATLAB documents `unicodestr` as a string scalar or character vector, so nonscalar string arrays raise an error." },
    BuiltinDocumentationFaq { question: "What happens when a character cannot be represented?", answer: "RunMat substitutes `?` (`63`) for characters that cannot be represented in byte-oriented legacy encodings. This avoids HTML numeric character-reference bytes and matches MATLAB-style byte substitution." },
    BuiltinDocumentationFaq { question: "Does this run on the GPU?", answer: "No. This is a text conversion sink: inputs are gathered to host memory and encoded on the CPU." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "native2unicode", target: BuiltinDocumentationLinkTarget::Builtin("native2unicode") },
    BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") },
    BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") },
    BuiltinDocumentationLink { label: "uint8", target: BuiltinDocumentationLinkTarget::Builtin("uint8") },
    BuiltinDocumentationLink { label: "fwrite", target: BuiltinDocumentationLinkTarget::Builtin("fwrite") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/unicode2native.rs") },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("unicode2native"), slug: Some("unicode2native"),
    summary: "Convert Unicode text to encoded uint8 bytes in MATLAB and RunMat.",
    description: "`unicode2native(unicodestr)` converts a string scalar or character vector into a `1xn uint8` row vector using UTF-8 by default. `unicode2native(unicodestr, encoding)` uses a named character encoding.",
    keywords: &["unicode2native", "encoding", "UTF-8", "latin1", "UTF-16", "UTF-32", "uint8", "text"],
    related: &["native2unicode", "string", "char", "uint8", "fwrite"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: FAQS, links: LINKS, media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/unicode2native.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Existing identity family tests", location: "builtins::strings::core::unicode2native::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Existing identity integration tests", location: "builtins::strings::core::unicode2native::tests::unicode2native_defaults_to_utf8_row_vector" },
        ],
        notes: &["Source-only canonical cutover; original JSON retained until compiled field and executable-example conservation."],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
