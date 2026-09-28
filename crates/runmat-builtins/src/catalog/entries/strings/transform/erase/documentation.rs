use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "String inputs stay strings and missing string scalars propagate unchanged. String arrays preserve size and orientation; each element has every supplied pattern removed.",
        "In RunMat mode, multirow character arrays are processed row by row. Rows can shrink, then receive trailing spaces so the result is rectangular. Cell arrays retain their shape; cells containing string scalars are also a RunMat extension. MATLAB compatibility mode accepts character vectors and cell arrays of character vectors, but rejects these extended forms.",
        "The pattern can be a string scalar or array, character array, cell array of text scalars, or a pattern object. Pattern lists are applied in order; an empty list leaves the input unchanged, and empty literal patterns have no effect. Text patterns are literal, while pattern objects use their compiled expression. The current pattern engine does not support every MATLAB expression construct, including lookaround and backreferences.",
        "Numeric and logical values, including all typed-integer classes and nested resident numeric values, reject before provider access. The input variables are not modified.",
    ] },
    BuiltinDocumentationSection { heading: "Placement and effects", paragraphs: &[
        "erase is a host text operation with no resident text representation, provider kernel or fusion route. GPU-resident numeric arguments are invalid text and reject before gathering.",
        "The operation is pure apart from validation or pattern errors. It has no callback, workspace, filesystem, random, time or distributed execution effect.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "legacy-word", title: "Remove one word from a string", program: "txt = \"RunMat accelerates MATLAB code\";\nclean = erase(txt, \"accelerates \")", display_output: Some("clean = \"RunMat MATLAB code\""), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(clean, \"RunMat MATLAB code\"));" } },
    BuiltinExample { id: "legacy-array", title: "Remove several substrings from string arrays", program: "labels = [\"GPU pipeline\"; \"CPU pipeline\"];\nresult = erase(labels, [\"GPU \", \"CPU \"])", display_output: Some("result = [\"pipeline\"; \"pipeline\"]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(result), [2 1])); assert(isequal(result, [\"pipeline\"; \"pipeline\"]));" } },
    BuiltinExample { id: "legacy-char", title: "Erase characters while retaining a rectangular char array", program: "chars = char(\"workspace\", \"snapshots\");\ntrimmed = erase(chars, \"s\")", display_output: Some("trimmed = ['workpace'; 'naphot  ']"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(ischar(trimmed)); assert(isequal(size(trimmed), [2 8])); assert(strcmp(trimmed(1,:), 'workpace')); assert(strcmp(trimmed(2,:), 'naphot  '));" } },
    BuiltinExample { id: "legacy-cell", title: "Remove substrings from a mixed text cell", program: "C = {'Kernel Planner', \"GPU Fusion\"};\nout = erase(C, [\"Kernel \", \"GPU \"])", display_output: Some("out = {'Planner', \"Fusion\"}"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(out), [1 2])); assert(ischar(out{1})); assert(strcmp(out{1}, 'Planner')); assert(isequal(out{2}, \"Fusion\"));" } },
    BuiltinExample { id: "legacy-empty", title: "Keep text unchanged with an empty pattern list", program: "data = [\"alpha\", \"beta\"];\nunchanged = erase(data, string.empty)", display_output: Some("unchanged = [\"alpha\", \"beta\"]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(unchanged, data));" } },
    BuiltinExample { id: "legacy-split", title: "Remove delimiters before splitting text", program: "path = \"runmat/bin:runmat/lib\";\nclean = erase(path, \":\")\nparts = split(clean, \"runmat/\")", display_output: Some("clean = \"runmat/binrunmat/lib\"; parts = [\"\"; \"bin\"; \"lib\"]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(clean, \"runmat/binrunmat/lib\")); assert(isequal(parts, [\"\"; \"bin\"; \"lib\"]));" } },
];

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/transform/erase.rs") };
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("erase"), slug: Some("erase"),
    summary: "Remove substring occurrences from text inputs in MATLAB and RunMat.",
    description: "erase(text, pattern) removes every occurrence of pattern from text. Supported text and pattern forms include string, character, cell text and pattern objects.",
    keywords: &["erase", "remove substring", "delete text", "string manipulation", "strings", "character array", "text"],
    related: &["replace", "strrep", "split", "regexprep", "string", "eraseBetween", "extractBetween", "join", "lower", "pad", "strcat", "strip", "strtrim", "upper"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Can I remove multiple patterns at once?", answer: "Yes. Supply a string or cell array of patterns. Each is removed in order from every input element." },
        BuiltinDocumentationFaq { question: "What happens if pattern is empty?", answer: "An empty pattern list leaves input unchanged. Empty literal patterns are ignored." },
        BuiltinDocumentationFaq { question: "Does erase modify the original data?", answer: "No. It returns a new value; the inputs remain unchanged." },
        BuiltinDocumentationFaq { question: "How are missing strings handled?", answer: "Missing string scalars propagate unchanged." },
        BuiltinDocumentationFaq { question: "Can erase operate on GPU-resident data?", answer: "No resident text representation or GPU kernel is available. GPU-resident numeric values are not text and reject before provider access; there is no implicit gather." },
        BuiltinDocumentationFaq { question: "How do I remove text using regular expressions?", answer: "Use regexprep for regular-expression replacement; erase accepts literal text patterns or RunMat pattern objects." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "replace", target: BuiltinDocumentationLinkTarget::Builtin("replace") },
        BuiltinDocumentationLink { label: "strrep", target: BuiltinDocumentationLinkTarget::Builtin("strrep") },
        BuiltinDocumentationLink { label: "split", target: BuiltinDocumentationLinkTarget::Builtin("split") },
        BuiltinDocumentationLink { label: "regexprep", target: BuiltinDocumentationLinkTarget::Builtin("regexprep") },
        BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") },
        BuiltinDocumentationLink { label: "eraseBetween", target: BuiltinDocumentationLinkTarget::Builtin("eraseBetween") },
        BuiltinDocumentationLink { label: "extractBetween", target: BuiltinDocumentationLinkTarget::Builtin("extractBetween") },
        BuiltinDocumentationLink { label: "join", target: BuiltinDocumentationLinkTarget::Builtin("join") },
        BuiltinDocumentationLink { label: "lower", target: BuiltinDocumentationLinkTarget::Builtin("lower") },
        BuiltinDocumentationLink { label: "pad", target: BuiltinDocumentationLinkTarget::Builtin("pad") },
        BuiltinDocumentationLink { label: "strcat", target: BuiltinDocumentationLinkTarget::Builtin("strcat") },
        BuiltinDocumentationLink { label: "strip", target: BuiltinDocumentationLinkTarget::Builtin("strip") },
        BuiltinDocumentationLink { label: "strtrim", target: BuiltinDocumentationLinkTarget::Builtin("strtrim") },
        BuiltinDocumentationLink { label: "upper", target: BuiltinDocumentationLinkTarget::Builtin("upper") },
        SOURCE,
    ], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "text, arrays, errors and integer admission", location: "builtins::strings::transform::erase::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "identity inference", location: "crates/runmat-builtins/src/catalog/entries/strings/transform/erase/inference.rs" }], notes: &["The character-array example checks rectangular padding and the split example checks column orientation. Resident numeric arguments reject before gathering.", "Pattern objects use Rust-regex-backed expressions; unsupported lookaround and backreference constructs remain a parity limitation, so complete MATLAB pattern-language coverage is not claimed."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
