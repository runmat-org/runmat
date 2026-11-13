use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "String scalars remain strings. Typed missing string scalars (displayed as `<missing>`) propagate unchanged; literal `<missing>` text remains ordinary text.",
            "String arrays are processed element wise while preserving shape.",
            "Character arrays are handled row by row. Rows expand or shrink as needed and are padded with spaces so the result remains a rectangular char array, just like MATLAB.",
            "Cell arrays must contain string scalars or character vectors. The result is a cell array with the same size and element types mirrored after replacement.",
            "The `old` and `new` inputs can be string scalars, string arrays, character arrays, or cell arrays of character vectors / strings. `new` must be a scalar or match the number of search terms in `old`.",
            "Non-text inputs (numeric, logical, structs, GPU tensors, etc.) produce MATLAB-compatible errors.",
            "Integer semantics: `replace` has no numeric input role. Numeric, logical, symbolic, and provider-resident values are rejected before gather or provider access.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Placement and effects",
        paragraphs: &[
            "Replacement executes on host text. Numeric resident values are not text and reject before provider access.",
            "This pure host operation may throw and may suspend at admitted gathers. It performs no callbacks, workspace/environment mutation, filesystem operations, randomness, or Clock reads. There is no GPU kernel, fusion route, or distributed support. VM and native calls share the canonical portable Runtime binding; these declarations are not execution evidence.",
            "Host text remains on the CPU; numeric resident values are not converted to text or gathered as textual operands.",
            "Replacement is a host-only fusion boundary. Unsupported resident numeric values reject before provider access; there is no device text kernel.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Current implementation frontier",
        paragraphs: &[
            "Character arrays with non-singleton dimensions beyond two, including those inside cells and delimiter controls, reject before row extraction. String and cell array geometry follows the per-entry behavior described here.",
            "Pattern objects are accepted in the search role. Replacements preserve string and cell shapes, including N-D cells; character rows may change width. Earlier GPU-gather prose was incorrect. General managed text allocation and independent MATLAB parity remain incomplete.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Replace all instances of a word in a string",
        program: "txt = \"RunMat accelerates MATLAB code\";\nresult = replace(txt, \"RunMat\", \"RunMat Accelerate\")",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(result,\"RunMat Accelerate accelerates MATLAB code\"));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Replace multiple terms in a string array",
        program: "labels = [\"GPU pipeline\"; \"CPU pipeline\"];\nresult = replace(labels, [\"GPU\", \"CPU\"], [\"Device\", \"Host\"])",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(result,[\"Device pipeline\";\"Host pipeline\"]));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Replace substrings in a character array while preserving padding",
        program: "chars = char(\"alpha\", \"beta \");\nout = replace(chars, \"a\", \"A\")",
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
        title: "Replace text within a cell array of character vectors",
        program: "C = {'Kernel Fusion', 'GPU Planner'};\nupdated = replace(C, {'Kernel', 'GPU'}, {'Shader', 'Device'})",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(updated,{'Shader Fusion','Device Planner'}));",
        },
    },
    BuiltinExample {
        id: "example-5",
        title: "Remove substrings by replacing with empty text",
        program: "paths = [\"runmat/bin\", \"runmat/lib\"];\nclean = replace(paths, \"runmat/\", \"\")",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(clean,[\"bin\",\"lib\"]));",
        },
    },
    BuiltinExample {
        id: "example-6",
        title: "Replace using scalar replacement for multiple search terms",
        program: "message = \"OpenCL or CUDA or Vulkan\";\nunified = replace(message, [\"OpenCL\", \"CUDA\", \"Vulkan\"], \"GPU backend\")",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(unified,\"GPU backend or GPU backend or GPU backend\"));",
        },
    },
    BuiltinExample {
        id: "example-7",
        title: "Replace text stored inside a cell array of strings",
        program: "cells = { \"Snapshot\", \"VM Interpreter\" };\nrenamed = replace(cells, \" \", \"_\")",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(renamed,{\"Snapshot\",\"VM_Interpreter\"}));",
        },
    },
    BuiltinExample {
        id: "example-8",
        title: "Preserve missing strings during replacement",
        program: "vals = [\"runmat\", string(missing), \"accelerate\"];\nout = replace(vals, \"runmat\", \"RunMat\")",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(out(1),\"RunMat\")); assert(ismissing(out(2))); assert(isequal(out(3),\"accelerate\"));",
        },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("replace"),
    slug: Some("replace"),
    summary: "Replace substring occurrences in string, char, and cell-text data with MATLAB-compatible mapping behavior.",
    description: "`replace(str, old, new)` substitutes every occurrence of `old` in `str` with `new`. It supports string scalars/arrays, character arrays, and cell arrays of text, including MATLAB-compatible multi-pattern replacement mapping.",
    keywords: &["replace", "substring replace", "string replace", "strrep", "text replace", "character array replace"],
    related: &["regexprep", "string", "char", "strtrim", "strip", "erase", "eraseBetween", "extractBetween", "join", "lower", "pad", "split", "strcat", "strrep", "upper"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "What sizes are allowed for `old` and `new` inputs?",
            answer: "`old` must contain at least one search term. `new` may be a scalar or contain the same number of elements as `old`. Otherwise, `replace` raises a size-mismatch error matching MATLAB behaviour.",
        },
        BuiltinDocumentationFaq {
            question: "Does `replace` modify the original input?",
            answer: "No. The builtin returns a new value with substitutions applied. The original inputs are left untouched.",
        },
        BuiltinDocumentationFaq {
            question: "How are character arrays padded after replacement?",
            answer: "Each row is expanded or truncated according to the longest resulting row. Shorter rows are padded with space characters so the output remains a proper char matrix.",
        },
        BuiltinDocumentationFaq {
            question: "How are missing strings handled?",
            answer: "Typed missing string scalars propagate unchanged. Replacements never convert a missing value into a non-missing string; literal `<missing>` text remains ordinary text.",
        },
        BuiltinDocumentationFaq {
            question: "Can I replace with an empty string?",
            answer: "Yes. Provide `\"\"` (empty string) or `''` as the replacement to remove matched substrings entirely.",
        },
        BuiltinDocumentationFaq {
            question: "Does `replace` support overlapping matches?",
            answer: "Replacements are non-overlapping and proceed from left to right, matching MATLAB’s behaviour for `replace`.",
        },
        BuiltinDocumentationFaq {
            question: "How does `replace` behave with GPU data?",
            answer: "There is no GPU text representation or kernel. Numeric resident operands reject before provider access. The result is host text.",
        },
    ],
    links: &[
        BuiltinDocumentationLink { label: "regexprep", target: BuiltinDocumentationLinkTarget::Builtin("regexprep") },
        BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") },
        BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") },
        BuiltinDocumentationLink { label: "strtrim", target: BuiltinDocumentationLinkTarget::Builtin("strtrim") },
        BuiltinDocumentationLink { label: "strip", target: BuiltinDocumentationLinkTarget::Builtin("strip") },
        BuiltinDocumentationLink { label: "erase", target: BuiltinDocumentationLinkTarget::Builtin("erase") },
        BuiltinDocumentationLink { label: "eraseBetween", target: BuiltinDocumentationLinkTarget::Builtin("eraseBetween") },
        BuiltinDocumentationLink { label: "extractBetween", target: BuiltinDocumentationLinkTarget::Builtin("extractBetween") },
        BuiltinDocumentationLink { label: "join", target: BuiltinDocumentationLinkTarget::Builtin("join") },
        BuiltinDocumentationLink { label: "lower", target: BuiltinDocumentationLinkTarget::Builtin("lower") },
        BuiltinDocumentationLink { label: "pad", target: BuiltinDocumentationLinkTarget::Builtin("pad") },
        BuiltinDocumentationLink { label: "split", target: BuiltinDocumentationLinkTarget::Builtin("split") },
        BuiltinDocumentationLink { label: "strcat", target: BuiltinDocumentationLinkTarget::Builtin("strcat") },
        BuiltinDocumentationLink { label: "strrep", target: BuiltinDocumentationLinkTarget::Builtin("strrep") },
        BuiltinDocumentationLink { label: "upper", target: BuiltinDocumentationLinkTarget::Builtin("upper") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/strings/transform/replace.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/transform/replace.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Runtime text semantics", location: "crates/runmat-runtime/src/builtins/strings/transform/replace.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed identity facts", location: "crates/runmat-builtins/src/catalog/entries/strings/transform/replace/inference.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Original unit source reference (not execution evidence)", location: "builtins::strings::transform::replace::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Original integration source reference (not execution evidence)", location: "builtins::strings::transform::replace::tests::replace_cell_array_mixed_content" },
        ],
        notes: &["Original examples are preserved as standalone asserted programs except the impossible GPU text example, whose host equivalent is explicit. Historical display outputs were unverified presentation claims and are retired rather than invented as transcripts. Source-only migration requires centralized compiler, catalog, documentation, and backend qualification."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
