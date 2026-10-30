use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "String scalars remain strings. Typed missing string values (displayed as `<missing>`) propagate unchanged; literal `<missing>` text remains ordinary text.",
            "String arrays are processed element-wise while preserving their full shape and orientation.",
            "Character arrays are handled row by row. Rows expand or shrink as needed and are padded with spaces so the result stays a rectangular char array, just like MATLAB.",
            "Cell arrays must contain character vectors or string scalars. The result is a cell array of identical size where each element has had the replacement applied.",
            "The `old` and `new` arguments must be string scalars or character vectors of the same data type.",
            "`old` can be empty. In that case, `strrep` inserts `new` before the first character, between existing characters, and after the final character.",
            "Integer values in any of the three text roles are rejected rather than treated as character codes or formatted numbers.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Placement and effects",
        paragraphs: &[
            "Replacement runs on host text. Numeric and provider-resident inputs are unsupported and reject before provider access.",
            "This pure host operation may throw and may suspend at admitted gathers. It performs no callbacks, workspace/environment mutation, filesystem operations, randomness, or Clock reads. There is no GPU kernel, fusion route, or distributed support. VM and native calls share the canonical portable Runtime binding; these declarations are not execution evidence.",
            "No. `strrep` accepts host text and returns host text. Provider-resident numeric values are unsupported and reject before access.",
            "`strrep` computes replacements on host text and returns a host text container of the corresponding kind. Numeric resident values reject before provider access rather than being gathered and implicitly converted to text.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Current implementation frontier",
        paragraphs: &[
            "Character arrays with non-singleton dimensions beyond two, including those inside cells and delimiter controls, reject before row extraction. String and cell array geometry follows the per-entry behavior described here.",
            "The host implementation also processes string arrays and character matrices inside cells, broader than scalar cellstr prose. Typed missing strings remain distinct from literal text. The retired GPU example was not executable: gpuArray has no text representation. Its host equivalent is asserted below. General managed text allocation remains incomplete.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Replacing a word inside a string scalar",
        program: "txt = \"RunMat turbo mode\";\nresult = strrep(txt, \"turbo\", \"accelerate\")",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(result,\"RunMat accelerate mode\"));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Updating every element of a string array",
        program: "labels = [\"GPU planner\", \"CPU planner\"];\nupdated = strrep(labels, \"planner\", \"pipeline\")",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(updated,[\"GPU pipeline\",\"CPU pipeline\"]));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Preserving rectangular shape in character arrays",
        program: "chars = char(\"alpha\", \"beta \");\nout = strrep(chars, \"a\", \"A\")",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(ischar(out)); assert(isequal(size(out),[2 5])); assert(strcmp(out(1,:),'AlphA')); assert(strcmp(out(2,:),'betA '));",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "Applying replacements inside a cell array of character vectors",
        program: "C = {'Kernel Fusion', 'GPU Planner'};\nrenamed = strrep(C, ' ', '_')",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(renamed,{'Kernel_Fusion','GPU_Planner'}));",
        },
    },
    BuiltinExample {
        id: "example-5",
        title: "Inserting text with an empty search pattern",
        program: "stub = \"abc\";\nexpanded = strrep(stub, \"\", \"-\")",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(expanded,\"-a-b-c-\"));",
        },
    },
    BuiltinExample {
        id: "example-6",
        title: "Leaving missing string values untouched",
        program: "vals = [\"RunMat\", string(missing), \"Accelerate\"];\nout = strrep(vals, \"RunMat\", \"RUNMAT\")",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(out(1),\"RUNMAT\")); assert(ismissing(out(2))); assert(isequal(out(3),\"Accelerate\"));",
        },
    },
    BuiltinExample {
        id: "example-7",
        title: "Replace host text (corrected GPU text example)",
        program: "g = \"native-code\";\nhost = strrep(g, \"code\", \"path\")",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(host,\"native-path\"));",
        },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("strrep"),
    slug: Some("strrep"),
    summary: "Replace non-overlapping substring occurrences in text inputs with MATLAB-compatible container behavior.",
    description: "`strrep(str, old, new)` replaces every non-overlapping occurrence of `old` with `new`. It supports MATLAB-compatible string, char, and cellstr-style inputs with element-wise replacement semantics.",
    keywords: &["strrep", "string replace", "character array replace", "text replacement", "substring replace"],
    related: &["replace", "regexprep", "string", "char", "join", "erase", "eraseBetween", "extractBetween", "lower", "pad", "split", "strcat", "strip", "strtrim", "upper"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Which input types does `strrep` accept?",
            answer: "String scalars, string arrays, character vectors, character arrays, and cell arrays of character vectors. The `old` and `new` arguments must be string scalars or character vectors of the same data type.",
        },
        BuiltinDocumentationFaq {
            question: "Does `strrep` support multiple search terms at once?",
            answer: "No. Use the newer `replace` builtin if you need to substitute several search terms in a single call.",
        },
        BuiltinDocumentationFaq {
            question: "How does `strrep` handle missing strings?",
            answer: "Missing string scalars remain `<missing>` and are returned unchanged, even when the search pattern matches ordinary text.",
        },
        BuiltinDocumentationFaq {
            question: "Will rows of a character array stay aligned?",
            answer: "Yes. Each row is replaced individually, then padded with spaces so that the overall array stays rectangular, matching MATLAB exactly.",
        },
        BuiltinDocumentationFaq {
            question: "What happens when `old` is empty?",
            answer: "RunMat mirrors MATLAB: `new` is inserted before the first character, between every existing character, and after the last character.",
        },
        BuiltinDocumentationFaq {
            question: "Does `strrep` run on the GPU?",
            answer: "No GPU text representation is supported. Numeric GPU values are not text and reject before provider access; convert supported host data to strings explicitly before calling `strrep`.",
        },
        BuiltinDocumentationFaq {
            question: "Can I mix strings and character vectors for `old` and `new`?",
            answer: "No. MATLAB requires `old` and `new` to share the same data type. RunMat enforces the same rule and raises a descriptive error when they differ.",
        },
        BuiltinDocumentationFaq {
            question: "How do I replace text stored inside cell arrays?",
            answer: "`strrep` traverses the cell array, applying the replacement to each character vector or string scalar element and returning a cell array of the same shape.",
        },
    ],
    links: &[
        BuiltinDocumentationLink { label: "replace", target: BuiltinDocumentationLinkTarget::Builtin("replace") },
        BuiltinDocumentationLink { label: "regexprep", target: BuiltinDocumentationLinkTarget::Builtin("regexprep") },
        BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") },
        BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") },
        BuiltinDocumentationLink { label: "join", target: BuiltinDocumentationLinkTarget::Builtin("join") },
        BuiltinDocumentationLink { label: "erase", target: BuiltinDocumentationLinkTarget::Builtin("erase") },
        BuiltinDocumentationLink { label: "eraseBetween", target: BuiltinDocumentationLinkTarget::Builtin("eraseBetween") },
        BuiltinDocumentationLink { label: "extractBetween", target: BuiltinDocumentationLinkTarget::Builtin("extractBetween") },
        BuiltinDocumentationLink { label: "lower", target: BuiltinDocumentationLinkTarget::Builtin("lower") },
        BuiltinDocumentationLink { label: "pad", target: BuiltinDocumentationLinkTarget::Builtin("pad") },
        BuiltinDocumentationLink { label: "split", target: BuiltinDocumentationLinkTarget::Builtin("split") },
        BuiltinDocumentationLink { label: "strcat", target: BuiltinDocumentationLinkTarget::Builtin("strcat") },
        BuiltinDocumentationLink { label: "strip", target: BuiltinDocumentationLinkTarget::Builtin("strip") },
        BuiltinDocumentationLink { label: "strtrim", target: BuiltinDocumentationLinkTarget::Builtin("strtrim") },
        BuiltinDocumentationLink { label: "upper", target: BuiltinDocumentationLinkTarget::Builtin("upper") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/strings/transform/strrep.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/transform/strrep.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Runtime text semantics", location: "crates/runmat-runtime/src/builtins/strings/transform/strrep.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed identity facts", location: "crates/runmat-builtins/src/catalog/entries/strings/transform/strrep/inference.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Original unit source reference (not execution evidence)", location: "builtins::strings::transform::strrep::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Original integration source reference (not execution evidence)", location: "builtins::strings::transform::strrep::tests::strrep_cell_array_char_vectors, builtins::strings::transform::strrep::tests::strrep_wgpu_provider_fallback" },
        ],
        notes: &["Original examples are preserved as standalone asserted programs except the impossible GPU text example, whose host equivalent is explicit. Historical display outputs were unverified presentation claims and are retired rather than invented as transcripts. Source-only migration requires centralized compiler, catalog, documentation, and backend qualification."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
