use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("htmlTree"),
    slug: Some("htmlTree"),
    summary: "Parse HTML code into a lightweight htmlTree object.",
    description: "`htmlTree(code)` parses HTML text and returns an `htmlTree` object that can be passed to `extractHTMLText`. Scalar string and character-vector inputs produce a scalar object. String arrays, multi-row character arrays, and cell arrays of character vectors produce shaped cells containing `htmlTree` objects.",
    keywords: &[
        "htmlTree",
        "HTML",
        "DOM",
        "text analytics",
        "parse",
    ],
    related: &[
        "extractHTMLText",
        "findElement",
        "getAttribute",
        "stopWords",
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Supported form",
            paragraphs: &[
                "tree = htmlTree(code)",
                "`tree` is an `htmlTree` object or a shaped cell array of `htmlTree` objects.",
                "`code` must contain text HTML.",
                "Use `extractHTMLText(tree)` to extract visible text from a parsed tree.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "`code` can be a string scalar, character vector, string array, multi-row character array, or cell array of character vectors.",
                "The returned object stores `Name`, `Children`, `Parent`, `Attributes`, `RawHTML`, and `Text` properties.",
                "`Name` is the uppercase root element name when an element is present.",
                "`Children` is a column cell array of child `htmlTree` objects because RunMat does not yet have a first-class MATLAB object-array representation for this class.",
                "`Parent` is a typed missing string, displayed as `<missing>`, at the root. Child objects store the parent element name rather than a cyclic object reference.",
                "The parser accepts common well-formed and fragment HTML, skips comments and declarations, decodes common HTML entities, and preserves enough structure for text extraction, `findElement`, and `getAttribute`.",
                "`findElement` and `getAttribute` support common selector and attribute workflows over RunMat `htmlTree` objects, including CSS tag/id/class selectors, attribute operator selectors, descendant/child/sibling combinators, selector groups, `:empty`, `:first-child`, `:first-of-type`, and simple `:not(...)` pseudo-class forms.",
                "Full Text Analytics DOM parity remains incomplete: exact browser-grade malformed-HTML restructuring, cyclic parent object identity, MATLAB object-array identity semantics, and selector forms outside the documented RunMat subset remain tracked by the broader Text Analytics compatibility issue.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Execution and residency",
            paragraphs: &[
                "`htmlTree` parses host-resident text and does not use provider kernels.",
                "`htmlTree` parses host-resident text and has no provider kernel.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "retained-1",
            title: "Parse HTML",
            program: "tree = htmlTree(\"<html><body><h1>Title</h1><p>Body</p></body></html>\")",
            display_output: Some("`tree` is an `htmlTree` object with `Name` equal to `\"HTML\"`."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(strcmp(class(tree), \"htmlTree\")); assert(strcmp(tree.Name, \"HTML\"));",
            },
        },
        BuiltinExample {
            id: "retained-2",
            title: "Extract Text From Parsed HTML",
            program: "tree = htmlTree(\"<article><h2>Title</h2><p>Body &amp; tail</p></article>\");\nstr = extractHTMLText(tree)",
            display_output: Some("str = \"Title\\n\\nBody & tail\""),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(strcmp(str, sprintf(\"Title\\n\\nBody & tail\")));",
            },
        },
        BuiltinExample {
            id: "retained-3",
            title: "Find Elements",
            program: "tree = htmlTree(\"<div><a href='/docs'>Docs</a><p>Body</p></div>\");\nlinks = findElement(tree, \"a\");\nhref = getAttribute(links, \"href\")",
            display_output: Some("href = \"/docs\""),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(strcmp(href, \"/docs\"));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does RunMat implement the full MATLAB htmlTree DOM surface?",
            answer: "No. RunMat supports the core object representation, `extractHTMLText`, common `findElement` selectors, and `getAttribute`. Exact malformed-HTML restructuring, cyclic parent object identity, MATLAB object-array identity semantics, and selector forms outside the documented RunMat subset remain tracked in the Text Analytics umbrella issue.",
        },
        BuiltinDocumentationFaq {
            question: "Why is `Children` a cell array?",
            answer: "RunMat's current generic object value does not yet model MATLAB object arrays for `htmlTree`, so children are stored as a cell array of scalar `htmlTree` objects.",
        },
        BuiltinDocumentationFaq {
            question: "Does `htmlTree` execute on the GPU?",
            answer: "No. HTML parsing is host text processing and has no `runmat-accelerate` provider path.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "extractHTMLText",
            target: BuiltinDocumentationLinkTarget::Builtin("extractHTMLText")
        },
        BuiltinDocumentationLink {
            label: "findElement",
            target: BuiltinDocumentationLinkTarget::Builtin("findElement")
        },
        BuiltinDocumentationLink {
            label: "getAttribute",
            target: BuiltinDocumentationLinkTarget::Builtin("getAttribute")
        },
        BuiltinDocumentationLink {
            label: "stopWords",
            target: BuiltinDocumentationLinkTarget::Builtin("stopWords")
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "`crates/runmat-runtime/src/builtins/strings/text_analytics/html.rs`",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/text_analytics/html.rs")
            },
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "unit",
                location: "builtins::strings::text_analytics::html::tests",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
