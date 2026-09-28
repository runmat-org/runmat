use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Text containers", paragraphs: &[
        "lower(text) converts alphabetic code points to lowercase. Non-alphabetic characters pass through unchanged; string arrays preserve shape, orientation and missing values.",
        "Character arrays are processed row by row. A lowercase mapping can expand a character (for example İ); output width can grow and shorter rows are padded to keep the array rectangular.",
        "Cell arrays retain their dimensions and must contain string scalars or character row vectors. Numeric, logical, struct and resident numeric inputs are not text and raise an input error.",
    ] },
    BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
        "lower runs on host text with no provider kernel or fusion path. It does not implicitly convert or gather a numeric GPU handle; convert code points to host text explicitly first.",
        "It reads no workspace or external resources and invokes no callbacks. Malformed input and output construction may throw.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "legacy-scalar", title: "Convert a string scalar", program: "txt = \"RunMat\";\nresult = lower(txt)", display_output: Some("result = \"runmat\""), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(result, \"runmat\"));" } },
    BuiltinExample { id: "legacy-array", title: "Convert a string array", program: "labels = [\"NORTH\" \"South\"; \"EaSt\" \"WEST\"];\nlowered = lower(labels)", display_output: Some("lowered = [\"north\" \"south\"; \"east\" \"west\"]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(lowered), [2 2])); assert(isequal(lowered, [\"north\" \"south\"; \"east\" \"west\"]));" } },
    BuiltinExample { id: "legacy-char", title: "Convert character-array rows", program: "animals = char(\"CAT\", \"DOGE\");\nresult = lower(animals)", display_output: Some("result = ['cat '; 'doge']"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(ischar(result)); assert(isequal(size(result), [2 4])); assert(strcmp(result(1,:), 'cat ')); assert(strcmp(result(2,:), 'doge'));" } },
    BuiltinExample { id: "legacy-cell", title: "Convert cell text", program: "C = {'HELLO', 'World'};\nout = lower(C)", display_output: Some("out = {'hello', 'world'}"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(out), [1 2])); assert(strcmp(out{1}, 'hello')); assert(strcmp(out{2}, 'world'));" } },
    BuiltinExample { id: "legacy-missing", title: "Preserve a missing string", program: "vals = [\"DATA\" string(missing) \"GPU\"];\nconverted = lower(vals)", display_output: Some("converted = [\"data\" <missing> \"gpu\"]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(converted), [1 3])); assert(isequal(converted(1), \"data\")); assert(ismissing(converted(2))); assert(isequal(converted(3), \"gpu\"));" } },
    BuiltinExample { id: "legacy-gpu-codes", title: "Gather numeric code points before conversion", program: "codes = gpuArray(uint16('RUNMAT'));\ntxt = char(gather(codes));\nresult = lower(txt)", display_output: Some("result = 'runmat'"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(result, 'runmat'));" } },
];
const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/transform/lower.rs") };
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("lower"), slug: Some("lower"),
    summary: "Convert text to lowercase in MATLAB and RunMat.",
    description: "lower(text) converts alphabetic characters to lowercase while retaining the input text container and non-alphabetic content.",
    keywords: &["lower", "lowercase", "convert to lowercase", "string case", "character arrays"],
    related: &["upper", "string", "char", "regexprep", "strcmpi", "erase", "eraseBetween", "extractBetween", "join", "pad", "replace", "split", "strcat", "strip", "strrep", "strtrim"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does lower change non-alphabetic characters?", answer: "No. Digits, punctuation, whitespace and symbols remain untouched; alphabetic code points with lowercase mappings change." },
        BuiltinDocumentationFaq { question: "What happens to character-array dimensions?", answer: "Rows stay rectangular. A lowercase mapping may expand a character, in which case the output gains columns and shorter rows are padded with spaces." },
        BuiltinDocumentationFaq { question: "Can numeric arrays be passed to lower?", answer: "No. Numeric, logical and struct inputs raise a type error. Convert them to strings or character arrays explicitly first." },
        BuiltinDocumentationFaq { question: "How are missing strings handled?", answer: "A missing string remains missing; it is not the literal text <missing>." },
        BuiltinDocumentationFaq { question: "Does lower execute on the GPU?", answer: "No. There is no resident-text representation or provider string kernel. Numeric GPU handles reject without implicit gathering; gather code points and convert them to text explicitly first." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "upper", target: BuiltinDocumentationLinkTarget::Builtin("upper") },
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
        SOURCE,
    ], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "text containers and errors", location: "builtins::strings::transform::lower::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "static facts", location: "crates/runmat-builtins/src/catalog/entries/strings/transform/lower/inference.rs" },
    ], notes: &["All six legacy examples, five FAQs, 16 related links and text/GPU behavior fields are represented. The old missing example used a literal <missing> string; this version constructs a missing value. The old GPU FAQ incorrectly claimed implicit gathering, contradicting runtime admission and the sidecar residency section."], },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
