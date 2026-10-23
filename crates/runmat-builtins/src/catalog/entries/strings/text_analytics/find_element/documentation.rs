use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("findElement"),
    slug: Some("findElement"),
    summary: "Find matching elements in an htmlTree object.",
    description: "`findElement(tree, selector)` searches a RunMat `htmlTree` object and returns matching subtrees as a column cell array of scalar `htmlTree` objects.",
    keywords: &["findElement", "htmlTree", "HTML", "DOM", "CSS selector", "text analytics"],
    related: &["htmlTree", "getAttribute", "extractHTMLText"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Supported form",
            paragraphs: &["subtrees = findElement(tree, selector)", "`subtrees` is a column cell array of scalar `htmlTree` objects.", "`tree` must be an `htmlTree` object or a cell array of `htmlTree` objects.", "`selector` must use the supported selector subset documented above."],
        },
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "`tree` must be a scalar RunMat `htmlTree` object or a cell array of `htmlTree` objects.",
                "`selector` must be a string scalar or character vector.",
                "Numeric, logical, complex, and provider-resident inputs are not htmlTree or selector values and reject before any provider access.",
                "Supported selectors include element tags, `*`, `.class`, `#id`, `[attr]`, `[attr=value]`, `[attr~=value]`, `[attr|=value]`, `[attr^=value]`, `[attr$=value]`, `[attr*=value]`, descendant combinators, child combinators (`>`), adjacent-sibling combinators (`+`), general-sibling combinators (`~`), comma-separated groups, `:empty`, `:first-child`, `:first-of-type`, and `:not(...)` for these supported pseudo-classes.",
                "Attribute names are matched case-insensitively through the parser's lowercase attribute map. Attribute values, ids, and classes use CSS-style exact, whitespace-list, dash-prefix, prefix, suffix, or substring string matching according to the selector operator.",
                "Sibling matching follows element siblings and ignores intervening whitespace text nodes.",
                "The result is a column cell array because RunMat does not yet have a first-class MATLAB object-array representation for `htmlTree`.",
                "Unsupported CSS pseudo-classes and selector forms raise an explicit error instead of silently returning partial results.",
                "Full CSS3 selector parity remains tracked by the broader Text Analytics compatibility issue.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Execution and residency",
            paragraphs: &["`findElement` traverses host-resident HTML metadata and does not use provider kernels.", "`findElement` traverses host-resident HTML metadata and has no provider kernel."],
        },
        BuiltinDocumentationSection {
            heading: "Implementation boundary",
            paragraphs: &["Selector traversal uses the existing host DOM and explicitly supported CSS grammar. Results remain a column cell array of scalar htmlTree objects, not a first-class MATLAB object array. Publication receives the actual RuntimeContext; DOM scratch and match metadata still lack complete finite ownership."],
        },
        BuiltinDocumentationSection {
            heading: "Finite host allocation",
            paragraphs: &["The current implementation requires an Unbounded host allocation policy. Under Managed finite allocation, a shared family guard rejects before gather, parsing, model cloning, chart mutation or filesystem work because complete host scratch and metadata ownership is not yet implemented. Authority is checked again after suspension and before publication, and failures are terminal rather than permission to replay. This refusal is an implementation frontier, not completion of finite resource support."],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "retained-1",
            title: "Find Links",
            program: "tree = htmlTree(\"<div><a class='nav' href='/home'>Home</a><a class='nav'>Docs</a></div>\");\nlinks = findElement(tree, \"a.nav\");\ntext = extractHTMLText(links)",
            display_output: Some("text = [\"Home\"; \"Docs\"]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(text, [\"Home\"; \"Docs\"])); assert(isequal(size(links), [2 1]));",
            },
        },
        BuiltinExample {
            id: "retained-2",
            title: "Use Combinators",
            program: "tree = htmlTree(\"<section><h1>Title</h1><p>Lead</p></section>\");\nlead = findElement(tree, \"h1 + p\")",
            display_output: Some("`lead` is a one-element cell array containing the matching `P` subtree."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(iscell(lead)); assert(isequal(size(lead), [1 1])); assert(strcmp(extractHTMLText(lead), \"Lead\"));",
            },
        },
        BuiltinExample {
            id: "retained-3",
            title: "Use CSS Attribute Operators",
            program: "tree = htmlTree(\"<div><a href='manual.pdf'>Manual</a><a href='guide.html'>Guide</a></div>\");\npdfs = findElement(tree, \"a[href$='.pdf']\")",
            display_output: Some("`pdfs` contains the link whose `href` ends with `.pdf`."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(numel(pdfs) == 1); assert(strcmp(getAttribute(pdfs, \"href\"), \"manual.pdf\"));",
            },
        },
        BuiltinExample {
            id: "retained-4",
            title: "Use Positional Pseudo-Classes",
            program: "tree = htmlTree(\"<section><p>Intro</p><a>First</a><a>Second</a></section>\");\nfirstLink = findElement(tree, \"a:first-of-type\")",
            display_output: Some("`firstLink` contains the first `A` element under the section."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(numel(firstLink) == 1); assert(strcmp(extractHTMLText(firstLink), \"First\"));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does `findElement` return a MATLAB object array?",
            answer: "No. RunMat currently returns a column cell array of scalar `htmlTree` objects because generic object-array identity semantics are still tracked separately.",
        },
        BuiltinDocumentationFaq {
            question: "Does RunMat support every CSS selector accepted by MATLAB?",
            answer: "No. Common tag, id, class, attribute operator, group, descendant, child, sibling, `:empty`, `:first-child`, `:first-of-type`, and simple `:not(...)` forms are supported. Full CSS3 parity remains tracked in the Text Analytics umbrella issue.",
        },
        BuiltinDocumentationFaq {
            question: "Does `findElement` execute on the GPU?",
            answer: "No. It traverses host HTML metadata and has no `runmat-accelerate` provider path.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "htmlTree",
            target: BuiltinDocumentationLinkTarget::Builtin("htmlTree"),
        },
        BuiltinDocumentationLink {
            label: "getAttribute",
            target: BuiltinDocumentationLinkTarget::Builtin("getAttribute"),
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
