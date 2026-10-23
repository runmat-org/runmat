use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,

    title: Some("tokenizedDocument"),

    slug: Some("tokenizedDocument"),

    summary: "Create tokenized document objects for Text Analytics workflows.",

    description: "`tokenizedDocument(str)` tokenizes English or German text into a `tokenizedDocument` object. It also accepts pre-tokenized words with `TokenizeMethod`, `\"none\"`, plus custom-token and regular-expression tokenization tables.",

    keywords: &["tokenizedDocument", "tokenize", "documents", "text analytics", "words"],
    related: &["bagOfWords", "removeShortWords", "normalizeWords", "tokenDetails"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Supported forms",
            paragraphs: &[
                "documents = tokenizedDocument\ndocuments = tokenizedDocument(str)\ndocuments = tokenizedDocument(str, 'TokenizeMethod', method)\ndocuments = tokenizedDocument(str, 'Language', language)\ndocuments = tokenizedDocument(str, 'DetectPatterns', patterns)\ndocuments = tokenizedDocument(str, 'TopLevelDomains', domains)\ndocuments = tokenizedDocument(str, 'CustomTokens', tokens)\ndocuments = tokenizedDocument(str, 'RegularExpressions', patterns)",
                "`documents` is a `tokenizedDocument` object.",
                "`method` must be `\"unicode\"` or `\"none\"`.",
                "`language` must currently be `\"en\"` or `\"de\"`.",
                "`patterns` may be `\"all\"`, `\"none\"`, or a text vector containing supported complex-token pattern names.",
                "`domains` may be a character vector, string array, or cell array of character vectors containing top-level-domain labels.",
                "`tokens` may be a text vector or table with `Token` and optional `Type` variables.",
                "`RegularExpressions` patterns may be a text vector or table with `Pattern` and optional `Type` variables.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "`tokenizedDocument` with no input creates a scalar object containing one empty document.",
                "`str` may be a string scalar, string array, character vector, multi-row character array, or cell array of scalar text.",
                "String arrays and cell arrays preserve their document-array shape in the object's `Shape` property. Character matrices are treated as one document per row.",
                "`tokenizedDocument(str, 'TokenizeMethod', 'unicode')` tokenizes text into words and simple punctuation tokens.",
                "`tokenizedDocument(words, 'TokenizeMethod', 'none')` accepts a string array, character vector, or cell form as pre-tokenized words. A cell array of string arrays represents multiple pre-tokenized documents.",
                "`Language` currently accepts `\"en\"` or `\"de\"`. Japanese, Korean, and MeCab tokenization are not implemented.",
                "`DetectPatterns` accepts `\"all\"`, `\"none\"`, or a string/cell vector containing one or more of `\"email-address\"`, `\"web-address\"`, `\"hashtag\"`, `\"at-mention\"`, and `\"emoticon\"`. RunMat only keeps enabled complex patterns as single tokens.",
                "`TopLevelDomains` accepts a character vector, string array, or cell array of character vectors and controls web-address recognition when `DetectPatterns` includes `\"web-address\"`. RunMat stores normalized TLD labels and uses custom lists for bare domains, `www.` addresses, and `http://`/`https://` addresses; with the default list, explicit `www.` and scheme URLs are accepted when the host has a valid domain shape.",
                "`CustomTokens` accepts a character vector, string scalar/vector, cell text vector, or table with a required `Token` variable and optional `Type` variable. Custom tokens are kept as single tokens, conflicting matches use the longest token, and missing or empty table `Type` values default to `\"custom\"`.",
                "`RegularExpressions` accepts a character vector, string scalar/vector, cell text vector, or table with a required `Pattern` variable and optional `Type` variable. Patterns are matched at the current tokenizer position, invalid or empty-matching patterns are rejected, conflicting regular-expression matches use the last matching pattern, and regular-expression tokens take precedence over custom tokens.",
                "The returned object stores `Vocabulary`, `Documents`, `NumDocuments`, `DocumentLengths`, `Shape`, `TokenizeMethod`, `Language`, `DetectPatterns`, and `TopLevelDomains`. Custom-token and regular-expression calls also store `TypeDetails` for `tokenDetails`.",
                "A document array is represented by one scalar object carrying document cells and `Shape` metadata. First-class object-array identity is not yet available for this class.",
                "Native numeric arrays are not document text or valid option values and are rejected before provider access. Document counts and lengths are stored as generated double metadata.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Execution and residency",
            paragraphs: &["`tokenizedDocument` tokenizes host-resident text and stores text metadata; it does not use provider kernels.", "`tokenizedDocument` tokenizes host-resident text and has no provider kernel."],
        },
        BuiltinDocumentationSection {
            heading: "Implementation boundary",
            paragraphs: &["RunMat uses its existing host tokenizer and compatibility-object representation, not the complete MATLAB language-analysis engine. Tokenization and custom/regular-expression rules are evaluated on host text; no numeric GPU kernel or fusion path is available."],
        },
        BuiltinDocumentationSection {
            heading: "Finite host allocation",
            paragraphs: &["The current implementation requires an Unbounded host allocation policy. Under Managed finite allocation, a shared family guard rejects before gather, model cloning, mutation or filesystem work because complete text-model scratch and metadata ownership is not yet implemented. Authority is checked again after suspension and before publication, and failures are terminal rather than permission to replay. This refusal is an implementation frontier, not completion of finite resource support."],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "retained-1",
            title: "Tokenize Text",
            program: "documents = tokenizedDocument([\"A short document\"; \"Another document\"])",
            display_output: Some("`documents.NumDocuments` is `2`, and `documents.Vocabulary` contains the unique tokens in first-seen order."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(documents.NumDocuments == 2); assert(isequal(documents.DocumentLengths, [3; 2])); assert(isequal(documents.Vocabulary, [\"A\" \"short\" \"document\" \"Another\"]));",
            },
        },
        BuiltinExample {
            id: "retained-2",
            title: "Use Pre-Tokenized Words",
            program: "documents = tokenizedDocument([\"alpha\" \"beta\" \"alpha\"], \"TokenizeMethod\", \"none\")",
            display_output: Some("`documents.DocumentLengths` is `3` for the single pre-tokenized document."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(documents.DocumentLengths == 3); assert(isequal(documents.Vocabulary, [\"alpha\" \"beta\"]));",
            },
        },
        BuiltinExample {
            id: "retained-3",
            title: "Choose Complex Token Patterns",
            program: "documents = tokenizedDocument(\"Mail a@example.com #MATLAB at https://example.com\", \"DetectPatterns\", [\"email-address\" \"hashtag\"])",
            display_output: Some("The email address and hashtag stay as single tokens; the web address is tokenized as ordinary text and punctuation."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(any(strcmp(documents.Vocabulary, \"a@example.com\"))); assert(any(strcmp(documents.Vocabulary, \"#MATLAB\"))); assert(~any(strcmp(documents.Vocabulary, \"https://example.com\")));",
            },
        },
        BuiltinExample {
            id: "retained-4",
            title: "Configure Web Address Domains",
            program: "documents = tokenizedDocument(\"See example.zz\", \"TopLevelDomains\", \"zz\")",
            display_output: Some("`example.zz` is recognized as a web-address token."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(any(strcmp(documents.Vocabulary, \"example.zz\")));",
            },
        },
        BuiltinExample {
            id: "retained-5",
            title: "Use Custom Token Types",
            program: "rules = table([\"Na+\"; \"H2O\"], [\"ion\"; \"formula\"], 'VariableNames', [\"Token\" \"Type\"])\ndocuments = tokenizedDocument(\"Na+ in H2O\", \"CustomTokens\", rules)",
            display_output: Some("`Na+` and `H2O` stay as single tokens, and `tokenDetails(documents)` reports `ion` and `formula` type labels for those tokens."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(documents.Vocabulary, [\"Na+\" \"in\" \"H2O\"])); details = tokenDetails(documents); assert(strcmp(details.Type(1), \"ion\")); assert(strcmp(details.Type(3), \"formula\"));",
            },
        },
        BuiltinExample {
            id: "retained-6",
            title: "Use Regular Expression Tokens",
            program: "patterns = table([\"[a-z]+\\\\d+\"; \"abc\\\\d+\"], [\"alnum\"; \"code\"], 'VariableNames', [\"Pattern\" \"Type\"])\ndocuments = tokenizedDocument(\"abc123 xyz9\", \"RegularExpressions\", patterns)",
            display_output: Some("`abc123` and `xyz9` stay as single tokens. The later matching `abc\\\\d+` pattern supplies the `code` type for `abc123`."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(documents.Vocabulary, [\"abc123\" \"xyz9\"])); details = tokenDetails(documents); assert(strcmp(details.Type(1), \"code\")); assert(strcmp(details.Type(2), \"alnum\"));",
            },
        },
    ],
    example_exemption: None,

    faqs: &[
        BuiltinDocumentationFaq {
            question: "Is RunMat's tokenizedDocument a full MATLAB object array?",
            answer: "No. It is one scalar object with document cells and a `Shape` property; first-class object-array identity is not yet available for this class.",
        },
        BuiltinDocumentationFaq {
            question: "Does `tokenizedDocument` support Japanese or Korean tokenization?",
            answer: "No. Japanese, Korean, and MeCab tokenization forms currently raise an unsupported-input error.",
        },
        BuiltinDocumentationFaq {
            question: "Does `tokenizedDocument` execute on the GPU?",
            answer: "No. Tokenization is host text processing and has no `runmat-accelerate` provider path.",
        },
        BuiltinDocumentationFaq {
            question: "Does RunMat ship the full MathWorks `topLevelDomains` list?",
            answer: "Not yet. RunMat supports user-specified `TopLevelDomains`, a compact built-in default list for common bare domains, and permissive default handling for explicit `www.`/scheme URLs; full default corpus parity remains tracked.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "bagOfWords",
            target: BuiltinDocumentationLinkTarget::Builtin("bagOfWords"),
        },
        BuiltinDocumentationLink {
            label: "removeShortWords",
            target: BuiltinDocumentationLinkTarget::Builtin("removeShortWords"),
        },
        BuiltinDocumentationLink {
            label: "normalizeWords",
            target: BuiltinDocumentationLinkTarget::Builtin("normalizeWords"),
        },
        BuiltinDocumentationLink {
            label: "tokenDetails",
            target: BuiltinDocumentationLinkTarget::Builtin("tokenDetails"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/strings/text_analytics/documents.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/text_analytics/documents.rs"),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "unit",
            location: "builtins::strings::text_analytics::documents::tests",
        }],
        notes: &[],
    },
    introduced: None,

    status: Some(BuiltinDocumentationStatus::Partial),
};
