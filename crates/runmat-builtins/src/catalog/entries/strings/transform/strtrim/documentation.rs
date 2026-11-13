use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Text containers", paragraphs: &[
        "strtrim(text) removes leading and trailing whitespace without changing internal spacing. MATLAB-compatible mode retains significant whitespace U+0085, U+00A0, U+2007 and U+202F at either end. RunMat mode retains its broader Unicode-whitespace trimming behavior.",
        "String scalars and arrays preserve type, dimensions and missing values. Character arrays are trimmed row by row: the row count stays fixed, and the width becomes that of the longest result, with shorter rows space-padded.",
        "Cell arrays retain their layout. Character-vector cells are supported in both modes; cells containing string scalars are a RunMat-only extension. Numeric, logical, struct and integer data are not interpreted as character codes.",
    ] },
    BuiltinDocumentationSection { heading: "Placement and effects", paragraphs: &[
        "strtrim runs on host text with no provider kernel or fusion route. Numeric or resident values reject before provider access rather than being implicitly converted or gathered.",
        "The builtin uses no callbacks or external resources. Malformed input and output construction can throw.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "legacy-scalar", title: "Trim a string scalar", program: "name = \"   RunMat   \";\nclean = strtrim(name)", display_output: Some("clean = \"RunMat\""), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(clean, \"RunMat\"));" } },
    BuiltinExample { id: "legacy-array", title: "Trim each element of a string array", program: "labels = [\"  Alpha  \"; \"Beta   \"; \"   Gamma\"];\ntrimmed = strtrim(labels)", display_output: Some("trimmed = [\"Alpha\"; \"Beta\"; \"Gamma\"]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(trimmed), [3 1])); assert(isequal(trimmed, [\"Alpha\"; \"Beta\"; \"Gamma\"]));" } },
    BuiltinExample { id: "legacy-char", title: "Trim character-array rows", program: "animals = char('  cat   ', 'dog', ' cow ');\nresult = strtrim(animals)", display_output: Some("result = ['cat'; 'dog'; 'cow']"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(ischar(result)); assert(isequal(size(result), [3 3])); assert(strcmp(result(1,:), 'cat')); assert(strcmp(result(2,:), 'dog')); assert(strcmp(result(3,:), 'cow'));" } },
    BuiltinExample { id: "legacy-tabs", title: "Trim tabs and newlines", program: "text = string([char(9) 'Metrics ' char(10)]);\nclean = strtrim(text)", display_output: Some("clean = \"Metrics\""), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(clean, \"Metrics\"));" } },
    BuiltinExample { id: "legacy-cell", title: "Trim a cell array of text", program: "pieces = {'  GPU  ', \" Accelerate\", 'RunMat   '};\nout = strtrim(pieces)", display_output: Some("out = {'GPU', \"Accelerate\", 'RunMat'}"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(out), [1 3])); assert(strcmp(out{1}, 'GPU')); assert(isequal(out{2}, \"Accelerate\")); assert(strcmp(out{3}, 'RunMat'));" } },
    BuiltinExample { id: "legacy-missing", title: "Preserve missing strings", program: "vals = [\" ok \"; string(missing); \" trimmed \" ];\ntrimmed = strtrim(vals)", display_output: Some("trimmed = [\"ok\"; <missing>; \"trimmed\"]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(trimmed), [3 1])); assert(isequal(trimmed(1), \"ok\")); assert(ismissing(trimmed(2))); assert(isequal(trimmed(3), \"trimmed\"));" } },
    BuiltinExample { id: "significant-whitespace", title: "Retain a nonbreaking space in MATLAB-compatible mode", program: "text = ['  Keep' char(160) '  '];\ntrimmed = strtrim(text)", display_output: Some("trimmed = ['Keep' char(160)]"), compatibility: BuiltinExampleCompatibility::Strict, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(trimmed, ['Keep' char(160)]));" } },
];
const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/transform/strtrim.rs") };
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("strtrim"), slug: Some("strtrim"),
    summary: "Remove leading and trailing whitespace from text inputs with MATLAB-compatible container semantics.",
    description: "strtrim(text) removes leading and trailing whitespace while preserving internal spacing. It supports string, character and cellstr-style input containers.",
    keywords: &["strtrim", "trim whitespace", "leading spaces", "trailing spaces", "character arrays", "string arrays"],
    related: &["strip", "upper", "lower", "erase", "eraseBetween", "extractBetween", "join", "pad", "replace", "split", "strcat", "strrep"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does strtrim modify internal whitespace?", answer: "No. Only leading and trailing whitespace is removed; interior spacing remains intact." },
        BuiltinDocumentationFaq { question: "Which characters count as whitespace?", answer: "Ordinary spaces, tabs and line breaks are trimmed. MATLAB-compatible mode preserves significant whitespace U+0085, U+00A0, U+2007 and U+202F; RunMat mode trims those too under its broader Unicode rule." },
        BuiltinDocumentationFaq { question: "How are character arrays resized?", answer: "Each row is trimmed independently. The output keeps its row count and uses the width of the longest trimmed row, padding shorter rows." },
        BuiltinDocumentationFaq { question: "What happens to missing strings?", answer: "Typed missing string elements remain missing. Literal text such as `<missing>` is ordinary text and is trimmed without changing its presence." },
        BuiltinDocumentationFaq { question: "Can numeric or logical arrays be passed to strtrim?", answer: "No. Non-text inputs raise a text-input error; integer arrays are not interpreted as character codes." },
        BuiltinDocumentationFaq { question: "How does strtrim differ from strip?", answer: "strtrim always removes whitespace at both ends. strip also supports custom characters and directional trimming." },
        BuiltinDocumentationFaq { question: "What does strtrim do in MATLAB?", answer: "It removes leading and trailing whitespace from a string or character array, preserving interior whitespace." },
        BuiltinDocumentationFaq { question: "How is strtrim different from strip in MATLAB?", answer: "strip also lets you choose left, right or both ends and supply the characters to remove." },
        BuiltinDocumentationFaq { question: "Does strtrim work with string arrays?", answer: "Yes. It trims each element and preserves the array shape and missing strings." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "strip", target: BuiltinDocumentationLinkTarget::Builtin("strip") },
        BuiltinDocumentationLink { label: "upper", target: BuiltinDocumentationLinkTarget::Builtin("upper") },
        BuiltinDocumentationLink { label: "lower", target: BuiltinDocumentationLinkTarget::Builtin("lower") },
        BuiltinDocumentationLink { label: "erase", target: BuiltinDocumentationLinkTarget::Builtin("erase") },
        BuiltinDocumentationLink { label: "eraseBetween", target: BuiltinDocumentationLinkTarget::Builtin("eraseBetween") },
        BuiltinDocumentationLink { label: "extractBetween", target: BuiltinDocumentationLinkTarget::Builtin("extractBetween") },
        BuiltinDocumentationLink { label: "join", target: BuiltinDocumentationLinkTarget::Builtin("join") },
        BuiltinDocumentationLink { label: "pad", target: BuiltinDocumentationLinkTarget::Builtin("pad") },
        BuiltinDocumentationLink { label: "replace", target: BuiltinDocumentationLinkTarget::Builtin("replace") },
        BuiltinDocumentationLink { label: "split", target: BuiltinDocumentationLinkTarget::Builtin("split") },
        BuiltinDocumentationLink { label: "strcat", target: BuiltinDocumentationLinkTarget::Builtin("strcat") },
        BuiltinDocumentationLink { label: "strrep", target: BuiltinDocumentationLinkTarget::Builtin("strrep") },
        BuiltinDocumentationLink { label: "MATLAB strtrim significant-whitespace rules", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/strtrim.html") },
        SOURCE,
    ], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "text, Unicode and shape semantics", location: "builtins::strings::transform::strtrim::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "static text facts", location: "crates/runmat-builtins/src/catalog/entries/strings/transform/strtrim/inference.rs" },
    ], notes: &["MATLAB-compatible mode preserves the four documented significant-whitespace code points while RunMat mode retains broader Unicode trimming. Typed missing values are distinct from every literal spelling. Cells containing string scalars require RunMat compatibility mode."], },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
