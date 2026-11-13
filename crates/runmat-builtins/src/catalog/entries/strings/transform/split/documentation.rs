use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "The default delimiter is whitespace (`isspace`), and consecutive whitespace is treated as a single separator (equivalent to `'CollapseDelimiters', true`).",
            "When you supply explicit delimiters, they can be a string scalar, string array, character array (rows), or a cell array of character vectors. Delimiters are matched left to right and the longest delimiter wins when several candidates match at the same position.",
            "RunMat mode retains the advanced `'CollapseDelimiters'` and `'IncludeDelimiters'` options behind one compatibility gate; they are not presented as MATLAB-compatible `split` options.",
            "A documented positive double `dim` controls output orientation. Typed integer dimensions are available only in RunMat mode because the compatibility target does not enumerate their storage classes. RunMat rejects dimensions above its supported rank ceiling of 1024 before allocating shape metadata.",
            "Without an explicit dimension, scalar input produces an N×1 string column, a column vector produces M×N, and a multidimensional input appends the token dimension. An explicit dimension inserts or replaces that dimension using column-major ordering.",
            "When requested, the second output contains the matched delimiters with the same orientation rules. Padding slots contain typed missing strings, displayed as `<missing>`.",
            "Missing string scalars propagate unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Placement and effects",
        paragraphs: &[
            "Text is host-only. Automatically placed double dimension controls gather transparently; explicit resident dimensions and typed integer dimensions are separately gated RunMat extensions.",
            "This pure host operation may throw and may suspend at admitted gathers. It performs no callbacks, workspace/environment mutation, filesystem operations, randomness, or Clock reads. There is no GPU kernel, fusion route, or distributed support. VM and native calls share the canonical portable Runtime binding; these declarations are not execution evidence.",
            "Text and delimiters are host-only. Automatically placed double dimensions gather transparently; explicit resident dimensions are compatibility-gated before gather.",
            "`split` executes on the CPU. Numeric resident values in text or delimiter roles reject without provider access; only an admitted dimension control may gather.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Current implementation frontier",
        paragraphs: &[
            "Character arrays with non-singleton dimensions beyond two, including those inside cells and delimiter controls, reject before row extraction. String and cell array geometry follows the per-entry behavior described here.",
            "Current input text geometry is scalar or two-dimensional; non-singleton dimensions beyond two reject before flattening. Output orientation may add a token axis, and two output prefixes are supported. General N-D input support and managed text allocation remain incomplete.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Split A String On Whitespace",
        program: "txt = \"RunMat Accelerate Planner\";\npieces = split(txt)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(pieces,[\"RunMat\";\"Accelerate\";\"Planner\"]));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Split A String Using A Custom Delimiter",
        program: "csv = \"alpha,beta,gamma\";\ntokens = split(csv, \",\")",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tokens,[\"alpha\";\"beta\";\"gamma\"]));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Include Delimiters In The Output",
        program: "expr = \"A+B-C\";\nsegments = split(expr, [\"+\", \"-\"], \"IncludeDelimiters\", true)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(segments,[\"A\";\"+\";\"B\";\"-\";\"C\"]));",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "Preserve Empty Segments When CollapseDelimiters Is False",
        program: "values = \"one,,three,\";\nparts = split(values, \",\", \"CollapseDelimiters\", false)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(parts,[\"one\";\"\";\"three\";\"\"]));",
        },
    },
    BuiltinExample {
        id: "example-5",
        title: "Split Each Row Of A Character Array",
        program: "rows = char(\"GPU Accelerate\", \"VM Interpreter\");\nresult = split(rows)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(result,[\"GPU\",\"Accelerate\";\"VM\",\"Interpreter\"]));",
        },
    },
    BuiltinExample {
        id: "example-6",
        title: "Split Elements Of A Cell Array",
        program: "C = {'RunMat Snapshot'; \"Fusion Planner\"};\nout = split(C, \" \")",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(out,[\"RunMat\",\"Snapshot\";\"Fusion\",\"Planner\"]));",
        },
    },
    BuiltinExample {
        id: "example-7",
        title: "Handle Missing String Inputs",
        program: "names = [\"RunMat\"; string(missing); \"Accelerate Engine\"];\nsplit_names = split(names)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(split_names),[3 2])); assert(isequal(ismissing(split_names),logical([0 1;1 1;0 0]))); assert(isequal(split_names(1,1),\"RunMat\")); assert(isequal(split_names(3,:),[\"Accelerate\",\"Engine\"]));",
        },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("split"),
    slug: Some("split"),
    summary: "Split text values into substrings using delimiter rules with MATLAB-compatible container behavior.",
    description: "`split(text)` breaks text into substrings separated by delimiters across string, char, and cell-text inputs. It follows MATLAB-compatible default-delimiter and output-container behavior, including whitespace splitting when no delimiter is provided.",
    keywords: &["split", "string split", "text split", "delimiters", "collapse delimiters", "include delimiters"],
    related: &["strsplit", "replace", "lower", "upper", "strip", "erase", "eraseBetween", "extractBetween", "join", "pad", "strcat", "strrep", "strtrim"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "What delimiters does `split` use by default?",
            answer: "When you omit the second argument, `split` treats any Unicode whitespace as a delimiter and collapses consecutive whitespace runs so they produce a single split point.",
        },
        BuiltinDocumentationFaq {
            question: "How do explicit delimiters change the defaults?",
            answer: "Providing explicit delimiters switches the internal collapse default to false. RunMat mode can override it with the compatibility-gated advanced option.",
        },
        BuiltinDocumentationFaq {
            question: "What happens when `'IncludeDelimiters'` is `true`?",
            answer: "This is a RunMat-mode extension. Matched delimiters are inserted between tokens in original order, with missing values used for padding.",
        },
        BuiltinDocumentationFaq {
            question: "How is the output sized for string arrays?",
            answer: "Default orientation follows the documented input-shape rules: scalar input becomes N×1, a column vector becomes M×N, and multidimensional input gains a trailing token dimension. An explicit positive dimension controls where that token axis is placed.",
        },
        BuiltinDocumentationFaq {
            question: "How does `split` handle missing strings?",
            answer: "Typed missing string scalars propagate unchanged. Padding slots also contain typed missing strings; literal `<missing>` text is split as ordinary text.",
        },
        BuiltinDocumentationFaq {
            question: "Can I provide empty delimiters?",
            answer: "No. Empty delimiters are disallowed, matching MATLAB's input validation. Specify at least one character per delimiter.",
        },
        BuiltinDocumentationFaq {
            question: "Which argument types are accepted as delimiters?",
            answer: "You may pass string scalars, string arrays, character arrays (each row is a delimiter), or cell arrays containing string scalars or character vectors.",
        },
        BuiltinDocumentationFaq {
            question: "How is `split` different from `strsplit`?",
            answer: "`split` is the vectorized string-focused builtin in RunMat. Use `strsplit` when you need the MATLAB-style scalar-text API with optional `matches` output and `DelimiterType` handling.",
        },
    ],
    links: &[
        BuiltinDocumentationLink { label: "strsplit", target: BuiltinDocumentationLinkTarget::Builtin("strsplit") },
        BuiltinDocumentationLink { label: "replace", target: BuiltinDocumentationLinkTarget::Builtin("replace") },
        BuiltinDocumentationLink { label: "lower", target: BuiltinDocumentationLinkTarget::Builtin("lower") },
        BuiltinDocumentationLink { label: "upper", target: BuiltinDocumentationLinkTarget::Builtin("upper") },
        BuiltinDocumentationLink { label: "strip", target: BuiltinDocumentationLinkTarget::Builtin("strip") },
        BuiltinDocumentationLink { label: "erase", target: BuiltinDocumentationLinkTarget::Builtin("erase") },
        BuiltinDocumentationLink { label: "eraseBetween", target: BuiltinDocumentationLinkTarget::Builtin("eraseBetween") },
        BuiltinDocumentationLink { label: "extractBetween", target: BuiltinDocumentationLinkTarget::Builtin("extractBetween") },
        BuiltinDocumentationLink { label: "join", target: BuiltinDocumentationLinkTarget::Builtin("join") },
        BuiltinDocumentationLink { label: "pad", target: BuiltinDocumentationLinkTarget::Builtin("pad") },
        BuiltinDocumentationLink { label: "strcat", target: BuiltinDocumentationLinkTarget::Builtin("strcat") },
        BuiltinDocumentationLink { label: "strrep", target: BuiltinDocumentationLinkTarget::Builtin("strrep") },
        BuiltinDocumentationLink { label: "strtrim", target: BuiltinDocumentationLinkTarget::Builtin("strtrim") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/strings/transform/split.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/transform/split.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Runtime text semantics", location: "crates/runmat-runtime/src/builtins/strings/transform/split.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed identity facts", location: "crates/runmat-builtins/src/catalog/entries/strings/transform/split/inference.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Original unit source reference (not execution evidence)", location: "builtins::strings::transform::split::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Original integration source reference (not execution evidence)", location: "crates/runmat-vm/tests/integer_pattern_and_string_transform_semantics.rs" },
        ],
        notes: &["Original examples are preserved as standalone asserted programs except the impossible GPU text example, whose host equivalent is explicit. Historical display outputs were unverified presentation claims and are retired rather than invented as transcripts. Source-only migration requires centralized compiler, catalog, documentation, and backend qualification."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
