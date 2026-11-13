use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "When you omit the dimension, `join` operates along the **last dimension whose size is not 1**. If all dimensions are singleton, it uses dimension 2.",
            "The default delimiter is a single space character. You can pass a scalar delimiter (string or character vector) or supply a string/cell array whose shape matches the input, with the join dimension reduced by one, to customise delimiters for each gap.",
            "Inputs may be string scalars, string arrays (including N-D), character arrays, or cell arrays of character vectors. Cell inputs return cell arrays; all other inputs return string scalars or string arrays.",
            "If any element participating in a join is a typed missing string, the result for that slice is also missing. Literal `<missing>` text joins as ordinary text.",
            "Joining along a dimension greater than `ndims(str)` leaves the input unchanged.",
            "The documented dimension is a positive integer-valued double scalar. Native typed-integer dimensions and the reversed `join(str,dim,delimiter)` argument order are accepted only in RunMat compatibility mode.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Placement and effects",
        paragraphs: &[
            "Executes on the CPU. Automatic internal residency gathers transparently; typed-integer dimensions and reversed `join(str,dim,delimiter)` syntax are mode-gated RunMat extensions.",
            "This pure host operation may throw and may suspend at admitted gathers. It performs no callbacks, workspace/environment mutation, filesystem operations, randomness, or Clock reads. There is no GPU kernel, fusion route, or distributed support. VM and native calls share the canonical portable Runtime binding; these declarations are not execution evidence.",
            "Text manipulation runs on the CPU. Automatic internal residency gathers transparently. Explicit resident typed-integer dimension controls are subject to the same RunMat-only compatibility gate as host typed dimensions.",
            "`join` executes on the CPU. When text or delimiters reside on the GPU, RunMat gathers them to host memory before performing the concatenation, ensuring identical results to MATLAB. Providers do not need to implement custom kernels for this builtin today.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Current implementation frontier",
        paragraphs: &[
            "Character arrays with non-singleton dimensions beyond two, including those inside cells and delimiter controls, reject before row extraction. String and cell array geometry follows the per-entry behavior described here.",
            "Host text joining supports N-D string and cell geometry. Typed integer dimensions, reversed dimension/delimiter order, and explicit resident controls are the declared RunMat extensions. General managed output allocation and independent MATLAB/browser/native qualification remain open.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Combine Strings In Each Row Of A Matrix",
        program: "names = [\"Carlos\" \"Sada\"; \"Ella\" \"Olsen\"; \"Diana\" \"Lee\"];\nfullNames = join(names)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(fullNames,[\"Carlos Sada\";\"Ella Olsen\";\"Diana Lee\"]));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Insert A Custom Delimiter Between Elements",
        program: "labels = [\"x\" \"y\" \"z\"; \"a\" \"b\" \"c\"];\njoined = join(labels, \"-\")",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(joined,[\"x-y-z\";\"a-b-c\"]));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Provide A Delimiter Array That Varies Per Row",
        program: "str = [\"x\" \"y\" \"z\"; \"a\" \"b\" \"c\"];\ndelims = [\" + \" \" = \"; \" - \" \" = \"];\nequations = join(str, delims)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(equations,[\"x + y = z\";\"a - b = c\"]));",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "Join Along A Specific Dimension",
        program: "scores = [\"Alice\" \"Bob\"; \"92\" \"88\"; \"85\" \"90\"];\nbyColumn = join(scores, 1)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(byColumn,[\"Alice 92 85\",\"Bob 88 90\"]));",
        },
    },
    BuiltinExample {
        id: "example-5",
        title: "Join A Cell Array Of Character Vectors",
        program: "C = {'GPU', 'Accelerate'; 'VM', 'Interpreter'};\nresult = join(C, \", \")",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(result,{'GPU, Accelerate';'VM, Interpreter'}));",
        },
    },
    BuiltinExample {
        id: "example-6",
        title: "Join Using A Dimension Argument As The Second Input",
        program: "words = [\"RunMat\"; \"Accelerate\"; \"Planner\"];\nsentence = join(words, 1)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(sentence,\"RunMat Accelerate Planner\"));",
        },
    },
    BuiltinExample {
        id: "example-7",
        title: "Join Rows Of An Empty String Array",
        program: "emptyRows = strings(2, 0);\nout = join(emptyRows)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(out),[2 1])); assert(isequal(out,[\"\";\"\"]));",
        },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("join"),
    slug: Some("join"),
    summary: "Join text elements with delimiters in MATLAB and RunMat.",
    description: "`join(text, delimiter, dim)` concatenates text along a chosen dimension and inserts delimiters between neighboring elements. Default dimension, delimiter broadcasting, and missing-string behavior follow MATLAB semantics.",
    keywords: &["join", "string join", "concatenate strings", "delimiters", "cell array join", "dimension join"],
    related: &["split", "compose", "string", "erase", "eraseBetween", "extractBetween", "lower", "pad", "replace", "strcat", "strip", "strrep", "strtrim", "upper"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "How does `join` choose the dimension when I do not specify one?",
            answer: "It looks for the last dimension whose size is not 1 and joins along that axis. If every dimension has size 1, it uses dimension 2.",
        },
        BuiltinDocumentationFaq {
            question: "Can I use different delimiters between separate pairs of strings?",
            answer: "Yes. Supply a string array or a cell array of character vectors with the same size as `str`, except that the join dimension must be one element shorter. Values of size 1 in other dimensions broadcast.",
        },
        BuiltinDocumentationFaq {
            question: "What happens when `str` contains a missing string?",
            answer: "The result for that slice is a typed missing string, displayed as `<missing>`. Literal text with that spelling remains nonmissing.",
        },
        BuiltinDocumentationFaq {
            question: "Can I pass GPU-resident text or delimiters?",
            answer: "Values placed on the device by RunMat's internal planner gather transparently. Explicit resident numeric handles are not text; typed resident dimension controls are accepted only through the corresponding RunMat extension.",
        },
        BuiltinDocumentationFaq {
            question: "What if I request a dimension larger than `ndims(str)`?",
            answer: "`join` returns the original text unchanged, matching MATLAB semantics.",
        },
        BuiltinDocumentationFaq {
            question: "Does `join` support numeric or logical inputs?",
            answer: "No. Convert them to strings first (e.g., with `string` or `compose`), then call `join`.",
        },
        BuiltinDocumentationFaq {
            question: "How do I join every element into a single string?",
            answer: "Specify the dimension explicitly. For column vectors, use `join(str, 1)`; for higher dimensional arrays, choose the axis that spans the elements you want to combine.",
        },
        BuiltinDocumentationFaq {
            question: "Are cell array outputs returned as strings?",
            answer: "No. When the input is a cell array, the output is a cell array of character vectors, keeping parity with MATLAB.",
        },
    ],
    links: &[
        BuiltinDocumentationLink { label: "split", target: BuiltinDocumentationLinkTarget::Builtin("split") },
        BuiltinDocumentationLink { label: "compose", target: BuiltinDocumentationLinkTarget::Builtin("compose") },
        BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") },
        BuiltinDocumentationLink { label: "erase", target: BuiltinDocumentationLinkTarget::Builtin("erase") },
        BuiltinDocumentationLink { label: "eraseBetween", target: BuiltinDocumentationLinkTarget::Builtin("eraseBetween") },
        BuiltinDocumentationLink { label: "extractBetween", target: BuiltinDocumentationLinkTarget::Builtin("extractBetween") },
        BuiltinDocumentationLink { label: "lower", target: BuiltinDocumentationLinkTarget::Builtin("lower") },
        BuiltinDocumentationLink { label: "pad", target: BuiltinDocumentationLinkTarget::Builtin("pad") },
        BuiltinDocumentationLink { label: "replace", target: BuiltinDocumentationLinkTarget::Builtin("replace") },
        BuiltinDocumentationLink { label: "strcat", target: BuiltinDocumentationLinkTarget::Builtin("strcat") },
        BuiltinDocumentationLink { label: "strip", target: BuiltinDocumentationLinkTarget::Builtin("strip") },
        BuiltinDocumentationLink { label: "strrep", target: BuiltinDocumentationLinkTarget::Builtin("strrep") },
        BuiltinDocumentationLink { label: "strtrim", target: BuiltinDocumentationLinkTarget::Builtin("strtrim") },
        BuiltinDocumentationLink { label: "upper", target: BuiltinDocumentationLinkTarget::Builtin("upper") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/strings/transform/join.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/transform/join.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Runtime text semantics", location: "crates/runmat-runtime/src/builtins/strings/transform/join.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed identity facts", location: "crates/runmat-builtins/src/catalog/entries/strings/transform/join/inference.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Original unit source reference (not execution evidence)", location: "builtins::strings::transform::join::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Original integration source reference (not execution evidence)", location: "builtins::strings::transform::join::tests::join_cell_array_of_char_vectors" },
        ],
        notes: &["Original examples are preserved as standalone asserted programs except the impossible GPU text example, whose host equivalent is explicit. Historical display outputs were unverified presentation claims and are retired rather than invented as transcripts. Source-only migration requires centralized compiler, catalog, documentation, and backend qualification."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
