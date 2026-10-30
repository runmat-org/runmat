use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Inputs are concatenated element-wise. Scalars expand across arrays of matching dimensions using MATLAB's implicit expansion rules.",
            "When at least one input is a string array (or string scalar), the result is a string array. Typed missing string values propagate, so any missing operand yields a missing result for that element. Literal `<missing>` text concatenates as ordinary text.",
            "When no string arrays are present but any input is a cell array of character vectors, the result is a cell array whose elements are character vectors.",
            "Otherwise, the result is a character array. Character and character-cell operands have trailing ASCII whitespace removed before concatenation; leading and internal spaces remain intact.",
            "Cell array elements must be character vectors (or string scalars). Mixing cell arrays with unsupported content raises a MATLAB-compatible error.",
            "Empty inputs broadcast naturally: an operand with a zero-length dimension yields an empty output after broadcasting.",
            "Integer and other numeric arrays are not implicitly formatted as text; use `string` first when numeric-to-text conversion is intentional.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Placement and effects",
        paragraphs: &[
            "Text concatenation executes on the CPU. Numeric and provider-resident values are unsupported text inputs and reject before provider access.",
            "This pure host operation may throw and may suspend at admitted gathers. It performs no callbacks, workspace/environment mutation, filesystem operations, randomness, or Clock reads. There is no GPU kernel, fusion route, or distributed support. VM and native calls share the canonical portable Runtime binding; these declarations are not execution evidence.",
            "No. `strcat` accepts host text and returns host text. Provider-resident numeric values are unsupported text operands and reject before access.",
            "RunMat performs text concatenation on the CPU. Unsupported resident numeric operands reject before provider lookup rather than being gathered and implicitly formatted.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Current implementation frontier",
        paragraphs: &[
            "Character arrays with non-singleton dimensions beyond two, including those inside cells and delimiter controls, reject before row extraction. String and cell array geometry follows the per-entry behavior described here.",
            "Character trimming currently removes trailing ASCII whitespace, not only spaces. Any string operand selects a string result, including when mixed with character arrays; old display-only character transcripts were incorrect. General managed text allocation remains incomplete.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Concatenate string scalars element-wise",
        program: "greeting = strcat(\"Run\", \"Mat\")",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(greeting,\"RunMat\"));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Concatenate a string scalar with a string array",
        program: "names = [\"VM\", \"JIT\", \"Accelerate\"];\ntagged = strcat(\"runmat-\", names)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tagged,[\"runmat-VM\",\"runmat-JIT\",\"runmat-Accelerate\"]));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Concatenate character arrays while trimming trailing spaces",
        program: "A = char(\"GPU \", \"Planner\");\nB = char(\"Accel\", \" Stage \");\nresult = strcat(A, B)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(ischar(result)); assert(isequal(size(result),[2 13])); assert(strcmp(result(1,:),'GPUAccel     ')); assert(strcmp(result(2,:),'Planner Stage'));",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "Concatenate cell arrays of character vectors",
        program: "C = {'Run ', 'Plan '; 'Fuse ', 'Cache '};\nsuffix = {'Mat', 'Ops'; 'Kernels', 'Stats'};\ncombined = strcat(C, suffix)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(combined,{'RunMat','PlanOps';'FuseKernels','CacheStats'}));",
        },
    },
    BuiltinExample {
        id: "example-5",
        title: "Propagate missing strings during concatenation",
        program: "values = [string(missing) \"ready\"];\nout = strcat(\"job-\", values)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(ismissing(out(1))); assert(isequal(out(2),\"job-ready\"));",
        },
    },
    BuiltinExample {
        id: "example-6",
        title: "Broadcast a scalar character vector across a character array",
        program: "labels = char(\"core\", \"runtime\", \"planner\");\nprefixed = strcat(\"runmat-\", labels)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isstring(prefixed)); assert(isequal(prefixed,[\"runmat-core\";\"runmat-runtime\";\"runmat-planner\"]));",
        },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("strcat"),
    slug: Some("strcat"),
    summary: "Concatenate text inputs element-wise with MATLAB-compatible implicit expansion and trailing-space handling.",
    description: "`strcat` concatenates text inputs element-wise across compatible array sizes. It supports MATLAB-compatible string, char, and cellstr container forms with implicit expansion semantics.",
    keywords: &["strcat", "string concatenation", "character arrays", "cell arrays", "trailing spaces"],
    related: &["string", "plus", "join", "cellstr", "erase", "eraseBetween", "extractBetween", "lower", "pad", "replace", "split", "strip", "strrep", "strtrim", "upper"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does `strcat` remove spaces between words?",
            answer: "No. `strcat` only strips trailing **space characters** from character inputs before concatenating. Spaces in the middle of a string remain untouched. To insert separators explicitly, concatenate the desired delimiter or use `join`.",
        },
        BuiltinDocumentationFaq {
            question: "How are missing strings handled?",
            answer: "Missing string scalars (`string(missing)`) propagate. If any operand is missing for a specific element, the resulting element is `<missing>`.",
        },
        BuiltinDocumentationFaq {
            question: "What happens when I mix strings and character arrays?",
            answer: "The output is a string array. Character inputs are converted to strings (after trimming trailing spaces) and combined element-wise with the string operands.",
        },
        BuiltinDocumentationFaq {
            question: "Can I concatenate cell arrays with string arrays?",
            answer: "Yes. Inputs are implicitly converted to strings when any operand is a string array, so the result is a string array. Cell array elements must still contain character vectors (or scalar strings).",
        },
        BuiltinDocumentationFaq {
            question: "What if I pass numeric or logical inputs?",
            answer: "`strcat` only accepts strings, character arrays, character vectors, or cell arrays of character vectors. Passing unsupported types raises a MATLAB-compatible error.",
        },
        BuiltinDocumentationFaq {
            question: "How are empty inputs treated?",
            answer: "Dimensions with length zero propagate through implicit expansion. For example, concatenating with an empty string array returns an empty array with the broadcasted shape.",
        },
    ],
    links: &[
        BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") },
        BuiltinDocumentationLink { label: "plus", target: BuiltinDocumentationLinkTarget::Builtin("plus") },
        BuiltinDocumentationLink { label: "join", target: BuiltinDocumentationLinkTarget::Builtin("join") },
        BuiltinDocumentationLink { label: "cellstr", target: BuiltinDocumentationLinkTarget::Builtin("cellstr") },
        BuiltinDocumentationLink { label: "erase", target: BuiltinDocumentationLinkTarget::Builtin("erase") },
        BuiltinDocumentationLink { label: "eraseBetween", target: BuiltinDocumentationLinkTarget::Builtin("eraseBetween") },
        BuiltinDocumentationLink { label: "extractBetween", target: BuiltinDocumentationLinkTarget::Builtin("extractBetween") },
        BuiltinDocumentationLink { label: "lower", target: BuiltinDocumentationLinkTarget::Builtin("lower") },
        BuiltinDocumentationLink { label: "pad", target: BuiltinDocumentationLinkTarget::Builtin("pad") },
        BuiltinDocumentationLink { label: "replace", target: BuiltinDocumentationLinkTarget::Builtin("replace") },
        BuiltinDocumentationLink { label: "split", target: BuiltinDocumentationLinkTarget::Builtin("split") },
        BuiltinDocumentationLink { label: "strip", target: BuiltinDocumentationLinkTarget::Builtin("strip") },
        BuiltinDocumentationLink { label: "strrep", target: BuiltinDocumentationLinkTarget::Builtin("strrep") },
        BuiltinDocumentationLink { label: "strtrim", target: BuiltinDocumentationLinkTarget::Builtin("strtrim") },
        BuiltinDocumentationLink { label: "upper", target: BuiltinDocumentationLinkTarget::Builtin("upper") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/strings/transform/strcat.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/transform/strcat.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Runtime text semantics", location: "crates/runmat-runtime/src/builtins/strings/transform/strcat.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed identity facts", location: "crates/runmat-builtins/src/catalog/entries/strings/transform/strcat/inference.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Original unit source reference (not execution evidence)", location: "builtins::strings::transform::strcat::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Original integration source reference (not execution evidence)", location: "builtins::strings::transform::strcat::tests::strcat_cell_array_trims_trailing_spaces" },
        ],
        notes: &["Original examples are preserved as standalone asserted programs except the impossible GPU text example, whose host equivalent is explicit. Historical display outputs were unverified presentation claims and are retired rather than invented as transcripts. Source-only migration requires centralized compiler, catalog, documentation, and backend qualification."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
