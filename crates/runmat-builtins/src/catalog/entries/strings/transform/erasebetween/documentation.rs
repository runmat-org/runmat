use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Text and boundaries", paragraphs: &[
        "eraseBetween accepts string scalars, string arrays, character arrays processed row by row, and cell arrays containing string scalars or character vectors. Output keeps the text container kind; string output can expand with boundary arrays, while character and cell outputs cannot expand beyond their text-element shape.",
        "Both boundaries in one call must be text or pattern markers, or both must be numeric positions. Scalar pattern objects are supported; broader pattern-array containers depend on shared pattern-array representation and are not claimed here.",
        "Text boundaries are exclusive by default: the markers remain while the enclosed text is removed. Numeric positions are inclusive by default: the characters at both positions and between them are removed. Boundaries can be changed to inclusive or exclusive with a case-insensitive name-value option.",
        "Missing string text or markers propagate a missing result. If a boundary is not found, the original element is returned unchanged. Numeric positions use exact positive-integer parsing and MATLAB-style one-based indexing; every typed-integer class is accepted as a structural control.",
        "MATLAB-compatible boundary arrays must be scalar or the same size as text. RunMat mode also allows broadcast-compatible boundaries. Character matrices, cells containing string scalars, and resident numeric position arrays are separately gated RunMat extensions.",
    ] },
    BuiltinDocumentationSection { heading: "Placement and effects", paragraphs: &[
        "The text input and output reside on the host. A resident numeric position control can gather through its owning provider only after RunMat extension admission; strict compatibility rejects it before provider access. Numeric or resident text is invalid and rejects before gathering.",
        "This operation does not run a GPU text kernel or join a fusion plan. It may suspend when gathering an admitted resident control, and it can throw for invalid type, shape, position, or option values. No callback, filesystem, workspace, randomness or time service is involved.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "legacy-substrings", title: "Remove text between substring markers", program: "txt = \"The quick brown fox\";\nresult = eraseBetween(txt, \"quick\", \" fox\")", display_output: Some("result = \"The quick fox\""), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(result, \"The quick fox\"));" } },
    BuiltinExample { id: "legacy-array", title: "Delete enclosed text across a string array", program: "str = [\"The quick brown fox jumps\"; \"over the lazy dog\"];\nstarts = [\"quick\"; \"the\"];\nends = [\" fox\"; \" dog\"];\ntrimmed = eraseBetween(str, starts, ends)", display_output: Some("trimmed = [\"The quick fox jumps\"; \"over the dog\"]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(trimmed), [2 1])); assert(isequal(trimmed, [\"The quick fox jumps\"; \"over the dog\"]));" } },
    BuiltinExample { id: "legacy-positions", title: "Erase inclusive numeric positions", program: "name = \"Edgar Allen Poe\";\nshort = eraseBetween(name, 6, 11)", display_output: Some("short = \"Edgar Poe\""), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(short, \"Edgar Poe\"));" } },
    BuiltinExample { id: "legacy-inclusive", title: "Include text boundary markers in deletion", program: "sentence = \"The quick brown fox jumps over the lazy dog\";\ncollapsed = eraseBetween(sentence, \" brown\", \"lazy\", \"Boundaries\", \"inclusive\")", display_output: Some("collapsed = \"The quick dog\""), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(collapsed, \"The quick dog\"));" } },
    BuiltinExample { id: "legacy-char", title: "Process padded character rows", program: "chars = char(\"Server<GPU>\", \"Engine<CPU>\");\ntrimmed = eraseBetween(chars, \"<\", \">\", \"Boundaries\", \"inclusive\")", display_output: Some("trimmed = ['Server'; 'Engine']"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(ischar(trimmed)); assert(isequal(size(trimmed), [2 6])); assert(strcmp(trimmed(1,:), 'Server')); assert(strcmp(trimmed(2,:), 'Engine'));" } },
    BuiltinExample { id: "legacy-cell", title: "Preserve element types in text cells", program: "C = {'alpha<1>', \"beta<2>\";\n     'gamma<3>', \"delta<4>\"};\nclean = eraseBetween(C, \"<\", \">\", \"Boundaries\", \"inclusive\")", display_output: Some("clean = {'alpha', \"beta\"; 'gamma', \"delta\"}"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(clean), [2 2])); assert(ischar(clean{1,1})); assert(strcmp(clean{1,1}, 'alpha')); assert(isequal(clean{1,2}, \"beta\")); assert(strcmp(clean{2,1}, 'gamma')); assert(isequal(clean{2,2}, \"delta\"));" } },
    BuiltinExample { id: "legacy-missing", title: "Propagate missing strings", program: "texts = [missing, \"Planner<GPU>\"];\nresult = eraseBetween(texts, \"<\", \">\")", display_output: Some("result = [<missing>, \"Planner<>\"]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(ismissing(result(1))); assert(isequal(result(2), \"Planner<>\"));" } },
];
const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/transform/erasebetween.rs") };
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("eraseBetween"), slug: Some("erasebetween"),
    summary: "Delete text between boundary markers in MATLAB and RunMat.",
    description: "eraseBetween(text, start, stop) removes text between two boundary markers. Text or numeric positions, expansion rules, and the Boundaries option determine the result.",
    keywords: &["eraseBetween", "delete", "delete substring", "boundaries", "inclusive", "exclusive", "string array", "strings"],
    related: &["extractBetween", "erase", "replace", "split", "join", "lower", "pad", "strcat", "strip", "strrep", "strtrim", "upper"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Which argument types does eraseBetween accept?", answer: "Text can be a string scalar or array, character array, or cell of character vectors or string scalars. Both boundaries must be text or pattern markers, or both numeric positions." },
        BuiltinDocumentationFaq { question: "What if a boundary is not found?", answer: "The original text element is returned unchanged. Missing string values also propagate." },
        BuiltinDocumentationFaq { question: "How does Boundaries inclusive work with text markers?", answer: "Inclusive mode deletes both matched markers and the enclosed text; exclusive mode keeps the markers." },
        BuiltinDocumentationFaq { question: "Can scalar boundaries apply to an array?", answer: "Yes. MATLAB-compatible mode also accepts arrays matching text size. General broadcasting is a separately gated RunMat extension." },
        BuiltinDocumentationFaq { question: "Are GPU inputs supported?", answer: "Text is host-only. In RunMat mode, admitted GPU-resident numeric position controls may gather to host; strict mode rejects that extension before provider access. Numeric or resident text is invalid." },
        BuiltinDocumentationFaq { question: "Are numeric positions validated?", answer: "Yes. Positions are parsed as exact positive integers with one-based indexing. Every out-of-range detail is not claimed as an independently verified MATLAB guarantee." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "extractBetween", target: BuiltinDocumentationLinkTarget::Builtin("extractBetween") },
        BuiltinDocumentationLink { label: "erase", target: BuiltinDocumentationLinkTarget::Builtin("erase") },
        BuiltinDocumentationLink { label: "replace", target: BuiltinDocumentationLinkTarget::Builtin("replace") },
        BuiltinDocumentationLink { label: "split", target: BuiltinDocumentationLinkTarget::Builtin("split") },
        BuiltinDocumentationLink { label: "join", target: BuiltinDocumentationLinkTarget::Builtin("join") },
        BuiltinDocumentationLink { label: "lower", target: BuiltinDocumentationLinkTarget::Builtin("lower") },
        BuiltinDocumentationLink { label: "pad", target: BuiltinDocumentationLinkTarget::Builtin("pad") },
        BuiltinDocumentationLink { label: "strcat", target: BuiltinDocumentationLinkTarget::Builtin("strcat") },
        BuiltinDocumentationLink { label: "strip", target: BuiltinDocumentationLinkTarget::Builtin("strip") },
        BuiltinDocumentationLink { label: "strrep", target: BuiltinDocumentationLinkTarget::Builtin("strrep") },
        BuiltinDocumentationLink { label: "strtrim", target: BuiltinDocumentationLinkTarget::Builtin("strtrim") },
        BuiltinDocumentationLink { label: "upper", target: BuiltinDocumentationLinkTarget::Builtin("upper") },
        SOURCE,
    ], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "text, pattern, exact integer and compatibility behavior", location: "builtins::strings::transform::erasebetween::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "identity inference", location: "crates/runmat-builtins/src/catalog/entries/strings/transform/erasebetween/inference.rs" }], notes: &["Exclusive text boundaries retain their markers, as checked by the missing-string example. Only admitted numeric position controls may gather; invalid numeric or resident text does not. Broader pattern-array representation remains an explicit partial gap."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
