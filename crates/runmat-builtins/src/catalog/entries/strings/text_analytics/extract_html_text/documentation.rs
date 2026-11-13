use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("extractHTMLText"),
    slug: Some("extractHTMLText"),
    summary: "Extract visible text from HTML code or htmlTree objects.",
    description: "`extractHTMLText` extracts visible text from HTML code or from an `htmlTree` object. It supports the documented `ExtractionMethod` option values `\"tree\"`, `\"article\"`, and `\"all-text\"`.",
    keywords: &["extractHTMLText", "HTML", "text analytics", "parse", "htmlTree"],
    related: &["htmlTree", "findElement", "getAttribute", "erasePunctuation", "stopWords"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Supported forms",
            paragraphs: &["str = extractHTMLText(code)\nstr = extractHTMLText(tree)\nstr = extractHTMLText(___, 'ExtractionMethod', method)", "`str` is a string scalar or string array.", "`method` must be `\"tree\"`, `\"article\"`, or `\"all-text\"`.", "The option name is case-insensitive and must be `ExtractionMethod`."],
        },
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "Documented `extractHTMLText(code)` input is a string array, character vector, or cell array of character vectors. Row-wise character matrices and cells containing string scalars are RunMat extensions.",
                "`extractHTMLText(tree)` accepts RunMat `htmlTree` objects and homogeneous arrays of those objects.",
                "`extractHTMLText(___, 'ExtractionMethod', method)` accepts `tree`, `article`, or `all-text`.",
                "String arrays, cell arrays, htmlTree arrays, and extension-mode character matrices return string arrays with the input shape.",
                "Script, style, noscript, template, and head content is excluded from extracted text.",
                "Common HTML entities and numeric character references are decoded.",
                "`tree` and `article` currently use the same paragraph/block extraction heuristic. RunMat does not yet implement MathWorks' exact article scoring algorithm.",
                "`all-text` extracts normalized body text outside skipped elements. Whitespace is normalized for MATLAB-friendly string output.",
                "The builtin has no integer role: all eight integer classes, logical and complex values, and resident numeric handles reject without numeric-to-text conversion or provider access.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Execution and residency",
            paragraphs: &["`extractHTMLText` parses host text and htmlTree objects and does not use provider kernels. Numeric resident handles reject before provider access.", "`extractHTMLText` parses host-resident text and has no provider kernel."],
        },
        BuiltinDocumentationSection {
            heading: "Implementation boundary",
            paragraphs: &["The existing host HTML parser excludes nonvisible elements and decodes entities. Tree and article share a block heuristic, not the complete MATLAB article-scoring algorithm. Host parsing/scratch has no complete finite owner or provider text kernel."],
        },
        BuiltinDocumentationSection {
            heading: "Finite host allocation",
            paragraphs: &["The current implementation requires an Unbounded host allocation policy. Under Managed finite allocation, a shared family guard rejects before gather, parsing, model cloning, chart mutation or filesystem work because complete host scratch and metadata ownership is not yet implemented. Authority is checked again after suspension and before publication, and failures are terminal rather than permission to replay. This refusal is an implementation frontier, not completion of finite resource support."],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "retained-1",
            title: "Extract Text From HTML",
            program: "code = \"<html><body><h1>THE SONNETS</h1><p>by William Shakespeare</p></body></html>\";\nstr = extractHTMLText(code)",
            display_output: Some("str = \"THE SONNETS\\n\\nby William Shakespeare\""),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(strcmp(str, sprintf('THE SONNETS\\n\\nby William Shakespeare')));",
            },
        },
        BuiltinExample {
            id: "retained-2",
            title: "Use all-text Extraction",
            program: "code = \"<body><style>.x{}</style><p>visible</p><script>hidden()</script><span>tail</span></body>\";\nstr = extractHTMLText(code, \"ExtractionMethod\", \"all-text\")",
            display_output: Some("str = \"visible tail\""),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(strcmp(str, \"visible tail\"));",
            },
        },
        BuiltinExample {
            id: "retained-3",
            title: "Extract From htmlTree",
            program: "tree = htmlTree(\"<article><h2>Title</h2><p>Body &amp; tail</p></article>\");\nstr = extractHTMLText(tree)",
            display_output: Some("str = \"Title\\n\\nBody & tail\""),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(strcmp(str, sprintf('Title\\n\\nBody & tail')));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Are `tree` and `article` exact MATLAB article extraction modes?",
            answer: "No. Both modes currently use RunMat's paragraph/block extraction heuristic. Exact article scoring remains part of the broader Text Analytics compatibility work.",
        },
        BuiltinDocumentationFaq {
            question: "Does `extractHTMLText` support arrays?",
            answer: "Yes. Documented string arrays, cell arrays of character vectors, and htmlTree arrays produce string arrays with the same shape. Row-wise character matrices and broader or mixed cells require RunMat extension mode.",
        },
        BuiltinDocumentationFaq {
            question: "Does `extractHTMLText` execute on the GPU?",
            answer: "No. HTML extraction is host text processing and has no `runmat-accelerate` provider path.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "htmlTree",
            target: BuiltinDocumentationLinkTarget::Builtin("htmlTree"),
        },
        BuiltinDocumentationLink {
            label: "findElement",
            target: BuiltinDocumentationLinkTarget::Builtin("findElement"),
        },
        BuiltinDocumentationLink {
            label: "getAttribute",
            target: BuiltinDocumentationLinkTarget::Builtin("getAttribute"),
        },
        BuiltinDocumentationLink {
            label: "erasePunctuation",
            target: BuiltinDocumentationLinkTarget::Builtin("erasePunctuation"),
        },
        BuiltinDocumentationLink {
            label: "stopWords",
            target: BuiltinDocumentationLinkTarget::Builtin("stopWords"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/strings/text_analytics/html.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/text_analytics/html.rs"),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "unit",
            location: "builtins::strings::text_analytics::html::tests",
        }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
