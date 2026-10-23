use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("wordcloud"),
    slug: Some("wordcloud"),
    summary: "Create a word cloud chart from text, tokenized documents, bag models, categorical arrays, tables, or word-size data.",
    description: "`wordcloud` creates a `WordCloudChart` handle from raw text, tokenized documents, bag models, categorical arrays, explicit word sizes, or table variables.",
    keywords: &["wordcloud", "WordCloudChart", "text analytics", "tokenizedDocument", "bagOfWords", "bagOfNgrams", "categorical"],
    related: &["tokenizedDocument", "bagOfWords", "bagOfNgrams", "get", "set"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "`wordcloud(str)` tokenizes string, character, string-array, or cellstr input, removes common English stop words, counts words, and renders the most frequent words.",
                "`wordcloud(documents)` accepts `tokenizedDocument` objects and counts tokens from their `Documents` property.",
                "`wordcloud(bag)` accepts `bagOfWords` and `bagOfNgrams` objects and sums each column of `Counts`; n-gram labels are joined with spaces.",
                "`wordcloud(words,sizeData)` accepts a word vector and a numeric vector with the same element count. The compatibility target documents all eight native integer classes for `sizeData`; RunMat reads them from native storage before entering the chart's floating size domain. Duplicate words are aggregated and zero-sized entries are omitted.",
                "`wordcloud(tbl,wordVar,sizeVar)` accepts table and timetable objects where `wordVar` and `sizeVar` are variable names or scalar variable indices. Numeric selectors may use any native integer class and are decoded exactly as one-based structural indices.",
                "`wordcloud(C)` accepts categorical objects and uses category counts as word sizes.",
                "A leading encoded numeric figure or axes handle is a RunMat language extension available in `runmat` mode. The selected figure's active axes receives the chart.",
                "Supported name-value options are `MaxDisplayWords`, `Color`, `HighlightColor`, `Shape` (`oval` or `rectangle`), `LayoutNum`, `SizePower`, `Title`, `TitleFontName`, `Visible`, `FontName`, `Box`, `Units`, `Position`, `HandleVisibility`, `DisplayName`, and `Tag`.",
                "Native integer storage for numeric chart properties such as `MaxDisplayWords` and `LayoutNum` is a RunMat language extension available in `runmat` mode. Ordinary integer-valued doubles remain available in MATLAB-compatible modes.",
                "`MaxDisplayWords` is capped at 1000 rendered annotation slots to keep chart creation bounded.",
                "`get` supports `Type`, `Parent`, `Children`, `WordData`, `SizeData`, `WordVariable`, `SizeVariable`, `SourceTable`, `MaxDisplayWords`, `Color`, `HighlightColor`, `Shape`, `LayoutNum`, `SizePower`, `Title`, `TitleFontName`, `FontName`, `Visible`, `Box`, `Units`, `Position`, `HandleVisibility`, `DisplayName`, and `Tag`.",
                "`set` supports `WordData`, `SizeData`, `MaxDisplayWords`, `Color`, `HighlightColor`, `Shape`, `LayoutNum`, `SizePower`, `Title`, `TitleFontName`, `FontName`, `Visible`, `Box`, `Units`, `Position`, `HandleVisibility`, `DisplayName`, and `Tag`; `WordData` and `SizeData` assignments must preserve matching lengths.",
                "RunMat uses a deterministic annotation layout rather than attempting to reproduce another renderer's placement algorithm. `FontName`, `TitleFontName`, `Units`, `Position`, and `HandleVisibility` are stored as chart metadata; the current plotting renderer does not yet expose a separate chart-container layout or font-family path. LDA topic word clouds remain unsupported until `ldaModel` support exists.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Execution and residency",
            paragraphs: &["`wordcloud` is a chart/annotation rendering sink. Text, bag, table, and categorical inputs are materialized on the host to compute word labels and sizes, then rendered through the plotting annotation path."],
        },
        BuiltinDocumentationSection {
            heading: "Implementation boundary",
            paragraphs: &["The chart uses the existing deterministic host word-count/annotation path, not MATLAB renderer placement or LDA models. Rendering caps annotation slots, not all parser/model input allocation. Explicit gpuArray input remains unsupported. Complete finite chart scratch/state/render ownership and related property mutation remain separate obligations."],
        },
        BuiltinDocumentationSection {
            heading: "Finite host allocation",
            paragraphs: &["The current implementation requires an Unbounded host allocation policy. Under Managed finite allocation, a shared family guard rejects before gather, parsing, model cloning, chart mutation or filesystem work because complete host scratch and metadata ownership is not yet implemented. Authority is checked again after suspension and before publication, and failures are terminal rather than permission to replay. This refusal is an implementation frontier, not completion of finite resource support."],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "retained-1",
            title: "Create a word cloud from text",
            program: "txt = \"runmat computes fast math and runmat plots math\";\nwc = wordcloud(txt);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,
                assertions: "assert(strcmp(get(wc, \"Type\"), \"wordcloud\")); w = get(wc, \"WordData\"); s = get(wc, \"SizeData\"); assert(s(strcmp(w, \"runmat\")) == 2); assert(s(strcmp(w, \"math\")) == 2); assert(~any(strcmp(w, \"and\")));",
            },
        },
        BuiltinExample {
            id: "retained-2",
            title: "Create a word cloud from explicit word sizes",
            program: "words = [\"alpha\" \"beta\" \"gamma\"];\nsizes = [5 9 2];\nwc = wordcloud(words, sizes, 'MaxDisplayWords', 2, 'Shape', 'rectangle');\nget(wc, 'WordData')",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,
                assertions: "assert(isequal(get(wc, \"WordData\"), [\"alpha\"; \"beta\"; \"gamma\"])); assert(isequal(get(wc, \"SizeData\"), [5; 9; 2])); assert(get(wc, \"MaxDisplayWords\") == 2); assert(strcmp(get(wc, \"Shape\"), \"rectangle\"));",
            },
        },
        BuiltinExample {
            id: "retained-3",
            title: "Create a word cloud from a table",
            program: "tbl = table([\"alpha\";\"beta\"], [10;4], 'VariableNames', {'Word','Count'});\nwc = wordcloud(tbl, 'Word', 'Count');",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure {
                minimum_figures: 1,
                assertions: "assert(isequal(get(wc, \"WordData\"), [\"alpha\"; \"beta\"])); assert(isequal(get(wc, \"SizeData\"), [10; 4])); assert(strcmp(get(wc, \"WordVariable\"), \"Word\")); assert(strcmp(get(wc, \"SizeVariable\"), \"Count\"));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does wordcloud return a graphics object?",
            answer: "Yes. RunMat returns a first-class `wordcloud` chart handle with readable and mutable chart properties through `get` and `set`.",
        },
        BuiltinDocumentationFaq {
            question: "Does RunMat match MATLAB's exact word placement?",
            answer: "No. RunMat computes compatible word labels, sizes, and chart properties, then uses its own deterministic annotation layout.",
        },
        BuiltinDocumentationFaq {
            question: "Can wordcloud preserve gpuArray residency?",
            answer: "No. Word clouds are host chart-rendering sinks over text, table, categorical, and bag metadata. Explicit `gpuArray` input is not supported.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "MATLAB language extensions",
            target: BuiltinDocumentationLinkTarget::Documentation("/docs/runtime/getting-started/language-extensions"),
        },
        BuiltinDocumentationLink {
            label: "tokenizedDocument",
            target: BuiltinDocumentationLinkTarget::Builtin("tokenizedDocument"),
        },
        BuiltinDocumentationLink {
            label: "bagOfWords",
            target: BuiltinDocumentationLinkTarget::Builtin("bagOfWords"),
        },
        BuiltinDocumentationLink {
            label: "bagOfNgrams",
            target: BuiltinDocumentationLinkTarget::Builtin("bagOfNgrams"),
        },
        BuiltinDocumentationLink {
            label: "get",
            target: BuiltinDocumentationLinkTarget::Builtin("get"),
        },
        BuiltinDocumentationLink {
            label: "set",
            target: BuiltinDocumentationLinkTarget::Builtin("set"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/plotting/ops/wordcloud.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/wordcloud.rs"),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "unit",
            location: "builtins::plotting::wordcloud::tests",
        }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
