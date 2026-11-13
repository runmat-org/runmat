use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Text containers", paragraphs: &[
        "upper(text) converts alphabetic code points to uppercase. Non-alphabetic characters pass through unchanged; string arrays preserve shape, orientation and missing values.",
        "Character arrays are processed row by row. MATLAB documents Latin-1 character-array case conversion and Unicode string-array conversion. RunMat currently applies Unicode uppercase mappings to character arrays as well; mappings beyond the documented MATLAB character set are not claimed as verified MATLAB parity.",
        "Cell arrays retain their dimensions. Character-vector cells are supported in both compatibility modes; cells containing string scalars are a RunMat-only extension. Numeric, logical, struct and resident numeric inputs are not text.",
    ] },
    BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
        "upper runs on host text with no provider kernel or fusion path. It does not implicitly convert or gather a numeric GPU handle; convert code points to host text explicitly first.",
        "It reads no workspace or external resources and invokes no callbacks. Malformed input and output construction may throw.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "legacy-scalar", title: "Convert a string scalar", program: "txt = \"RunMat\";\nresult = upper(txt)", display_output: Some("result = \"RUNMAT\""), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(result, \"RUNMAT\"));" } },
    BuiltinExample { id: "legacy-array", title: "Convert a string array", program: "labels = [\"north\" \"South\"; \"East\" \"west\"];\nuppered = upper(labels)", display_output: Some("uppered = [\"NORTH\" \"SOUTH\"; \"EAST\" \"WEST\"]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(uppered), [2 2])); assert(isequal(uppered, [\"NORTH\" \"SOUTH\"; \"EAST\" \"WEST\"]));" } },
    BuiltinExample { id: "legacy-char", title: "Convert character-array rows", program: "animals = char(\"cat\", \"doge\");\nresult = upper(animals)", display_output: Some("result = ['CAT '; 'DOGE']"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(ischar(result)); assert(isequal(size(result), [2 4])); assert(strcmp(result(1,:), 'CAT ')); assert(strcmp(result(2,:), 'DOGE'));" } },
    BuiltinExample { id: "legacy-cell", title: "Convert cell text", program: "C = {'hello', 'World'};\nout = upper(C)", display_output: Some("out = {'HELLO', 'WORLD'}"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(out), [1 2])); assert(strcmp(out{1}, 'HELLO')); assert(strcmp(out{2}, 'WORLD'));" } },
    BuiltinExample { id: "legacy-missing", title: "Preserve a missing string", program: "vals = [\"data\" string(missing) \"gpu\"];\nconverted = upper(vals)", display_output: Some("converted = [\"DATA\" <missing> \"GPU\"]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(converted), [1 3])); assert(isequal(converted(1), \"DATA\")); assert(ismissing(converted(2))); assert(isequal(converted(3), \"GPU\"));" } },
    BuiltinExample { id: "legacy-gpu-codes", title: "Gather numeric code points before conversion", program: "codes = gpuArray(uint16('runmat'));\ntxt = char(gather(codes));\nresult = upper(txt)", display_output: Some("result = 'RUNMAT'"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(result, 'RUNMAT'));" } },
];
const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/transform/upper.rs") };
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("upper"), slug: Some("upper"),
    summary: "Convert text to uppercase in MATLAB and RunMat.",
    description: "upper(text) converts alphabetic characters to uppercase while retaining the input text container and non-alphabetic content.",
    keywords: &["upper", "uppercase", "convert to uppercase", "string case", "character arrays"],
    related: &["lower", "string", "char", "regexprep", "strcmpi", "erase", "eraseBetween", "extractBetween", "join", "pad", "replace", "split", "strcat", "strip", "strrep", "strtrim"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does upper change non-alphabetic characters?", answer: "No. Digits, punctuation, whitespace and symbols remain untouched; alphabetic code points with uppercase mappings change." },
        BuiltinDocumentationFaq { question: "What happens to character-array dimensions?", answer: "Rows stay rectangular. An uppercase mapping may expand a character, in which case the output gains columns and shorter rows are padded with spaces." },
        BuiltinDocumentationFaq { question: "Can numeric arrays be passed to upper?", answer: "No. Numeric, logical and struct inputs raise a type error. Convert them to strings or character arrays explicitly first." },
        BuiltinDocumentationFaq { question: "How are missing strings handled?", answer: "Typed missing string elements remain missing. Literal `<missing>` text is ordinary text and follows the same case mapping as other text." },
        BuiltinDocumentationFaq { question: "Does upper execute on the GPU?", answer: "No. There is no resident-text representation or provider string kernel. Numeric GPU handles reject without implicit gathering; gather code points and convert them to text explicitly first." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "lower", target: BuiltinDocumentationLinkTarget::Builtin("lower") },
        BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") },
        BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") },
        BuiltinDocumentationLink { label: "regexprep", target: BuiltinDocumentationLinkTarget::Builtin("regexprep") },
        BuiltinDocumentationLink { label: "strcmpi", target: BuiltinDocumentationLinkTarget::Builtin("strcmpi") },
        BuiltinDocumentationLink { label: "erase", target: BuiltinDocumentationLinkTarget::Builtin("erase") },
        BuiltinDocumentationLink { label: "eraseBetween", target: BuiltinDocumentationLinkTarget::Builtin("eraseBetween") },
        BuiltinDocumentationLink { label: "extractBetween", target: BuiltinDocumentationLinkTarget::Builtin("extractBetween") },
        BuiltinDocumentationLink { label: "join", target: BuiltinDocumentationLinkTarget::Builtin("join") },
        BuiltinDocumentationLink { label: "pad", target: BuiltinDocumentationLinkTarget::Builtin("pad") },
        BuiltinDocumentationLink { label: "replace", target: BuiltinDocumentationLinkTarget::Builtin("replace") },
        BuiltinDocumentationLink { label: "split", target: BuiltinDocumentationLinkTarget::Builtin("split") },
        BuiltinDocumentationLink { label: "strcat", target: BuiltinDocumentationLinkTarget::Builtin("strcat") },
        BuiltinDocumentationLink { label: "strip", target: BuiltinDocumentationLinkTarget::Builtin("strip") },
        BuiltinDocumentationLink { label: "strrep", target: BuiltinDocumentationLinkTarget::Builtin("strrep") },
        BuiltinDocumentationLink { label: "strtrim", target: BuiltinDocumentationLinkTarget::Builtin("strtrim") },
        BuiltinDocumentationLink { label: "MATLAB upper character-set support", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/string.upper.html") },
        SOURCE,
    ], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "text containers and errors", location: "builtins::strings::transform::upper::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "static facts", location: "crates/runmat-builtins/src/catalog/entries/strings/transform/upper/inference.rs" },
    ], notes: &["Character-array mappings outside MATLAB's documented Latin-1 support remain unverified; RunMat currently uses Unicode case mapping. Typed missing strings are distinct from literal text. Cells containing string scalars require RunMat compatibility mode."], },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
