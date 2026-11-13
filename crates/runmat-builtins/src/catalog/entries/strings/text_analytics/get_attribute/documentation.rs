use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("getAttribute"),
    slug: Some("getAttribute"),
    summary: "Read an HTML attribute from htmlTree objects.",
    description: "`getAttribute(tree, attr)` returns the value of an HTML attribute from a scalar RunMat `htmlTree` object or from each object in a cell array of `htmlTree` objects.",
    keywords: &["getAttribute", "htmlTree", "HTML", "DOM", "attribute", "text analytics"],
    related: &["htmlTree", "findElement", "extractHTMLText"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Supported form",
            paragraphs: &["str = getAttribute(tree, attr)", "`str` is a string scalar or string array.", "`tree` must be an `htmlTree` object or a cell array of `htmlTree` objects.", "`attr` must name one HTML attribute."],
        },
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &["`tree` must be a scalar RunMat `htmlTree` object or a cell array of `htmlTree` objects, such as the output of `findElement`.", "`attr` must be a nonempty string scalar or character vector.", "Attribute names are normalized to lowercase and matched against the parser's lowercase attribute map.", "A scalar `htmlTree` input returns a string scalar.", "A cell array of `htmlTree` objects returns a string array with the same shape unless there is only one cell element.", "If the requested attribute is absent, RunMat returns a typed missing string element, displayed as `<missing>`.", "Literal attribute text such as `<missing>` remains present text. Broader MATLAB object-array compatibility remains unqualified."],
        },
        BuiltinDocumentationSection {
            heading: "Execution and residency",
            paragraphs: &["`getAttribute` reads host-resident HTML metadata and does not use provider kernels.", "`getAttribute` reads host-resident HTML metadata and has no provider kernel."],
        },
        BuiltinDocumentationSection {
            heading: "Implementation boundary",
            paragraphs: &["Attribute lookup reparses the stored RawHTML through the existing HTML owner. Missing attributes and text nodes produce typed missing strings; present attribute values retain literal text. No provider kernel or complete finite parser/output owner is implemented."],
        },
        BuiltinDocumentationSection {
            heading: "Finite host allocation",
            paragraphs: &["The current implementation requires an Unbounded host allocation policy. Under Managed finite allocation, a shared family guard rejects before gather, parsing, model cloning, chart mutation or filesystem work because complete host scratch and metadata ownership is not yet implemented. Authority is checked again after suspension and before publication, and failures are terminal rather than permission to replay. This refusal is an implementation frontier, not completion of finite resource support."],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "retained-1",
            title: "Read Link Attributes",
            program: "tree = htmlTree(\"<a href='/home'>Home</a>\");\nhref = getAttribute(tree, \"href\")",
            display_output: Some("href = \"/home\""),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(strcmp(href, \"/home\"));",
            },
        },
        BuiltinExample {
            id: "retained-2",
            title: "Read Attributes From Matches",
            program: "tree = htmlTree(\"<div><a href='/home'>Home</a><a>Docs</a></div>\");\nlinks = findElement(tree, \"a\");\nhrefs = getAttribute(links, \"href\")",
            display_output: Some("hrefs = [\"/home\"; <missing>]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(hrefs), [2 1])); assert(isequal(hrefs(1), \"/home\")); assert(ismissing(hrefs(2)));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "What does RunMat return for a missing attribute?",
            answer: "RunMat returns a typed missing string for that element. Use `ismissing` to test absence; literal `<missing>` attribute text remains nonmissing. Broader MATLAB object compatibility remains unqualified.",
        },
        BuiltinDocumentationFaq {
            question: "Does `getAttribute` execute on the GPU?",
            answer: "No. Attribute lookup is host HTML metadata access and has no `runmat-accelerate` provider path.",
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
            label: "extractHTMLText",
            target: BuiltinDocumentationLinkTarget::Builtin("extractHTMLText"),
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
