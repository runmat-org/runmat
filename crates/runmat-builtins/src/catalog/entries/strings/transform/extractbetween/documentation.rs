use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Text and boundaries", paragraphs: &[
        "extractBetween(str,start,stop) accepts two text/pattern markers or two numeric positions. A string input produces strings; a character-vector input produces a cell containing a character vector; a cell input preserves its cell layout and element text kinds. Row-wise character matrices are a RunMat extension.",
        "Text markers are excluded by default. Numeric endpoint positions are included by default. The Boundaries option selects inclusive or exclusive handling explicitly. The first start marker and the first following stop marker determine each result; an unmatched marker produces empty text in the current implementation.",
        "Documented boundary arrays are scalar or the same size as the text input. RunMat mode additionally permits compatible broadcasting for string output. Positions use exact one-based integer parsing. Mixed text and numeric boundary domains are rejected.",
        "Typed missing subject or marker elements propagate a missing string result. Every literal spelling, including `<missing>`, is ordinary text. Independent MATLAB parity for missing-marker overloads remains unqualified.",
    ] },
    BuiltinDocumentationSection { heading: "Placement and effects", paragraphs: &[
        "Extraction executes on host text without a GPU kernel or fusion route. Numeric or resident text input is rejected before provider access. RunMat mode can admit a resident numeric position control and gather it before host extraction; MATLAB-compatible mode rejects that extension before gathering.",
        "The operation is pure, may throw on invalid inputs, options or output construction, and may suspend while gathering an admitted resident position. It does not access the workspace, filesystem, clock, randomness, or callbacks.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "legacy-substrings", title: "Extract between text markers", program: "txt = \"RunMat accelerates MATLAB workloads\";\nsegment = extractBetween(txt, \"RunMat \", \" workloads\")", display_output: Some("segment = \"accelerates MATLAB\""), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(segment, \"accelerates MATLAB\"));" } },
    BuiltinExample { id: "legacy-inclusive", title: "Include text markers", program: "path = \"snapshots/run/fusion.mat\";\nwithMarkers = extractBetween(path, \"snapshots/\", \".mat\", \"Boundaries\", \"inclusive\")", display_output: Some("withMarkers = \"snapshots/run/fusion.mat\""), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(withMarkers, path));" } },
    BuiltinExample { id: "legacy-positions", title: "Use one-based numeric positions", program: "name = \"Accelerator\";\nmiddle = extractBetween(name, 3, 7)", display_output: Some("middle = \"celer\""), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(middle, \"celer\"));" } },
    BuiltinExample { id: "legacy-array", title: "Apply scalar markers to an array", program: "files = [\"runmat_accel.rs\", \"runmat_gc.rs\"; \"runmat_plot.rs\", \"runmat_cli.rs\"];\nstems = extractBetween(files, \"runmat_\", \".rs\")", display_output: Some("stems = [\"accel\", \"gc\"; \"plot\", \"cli\"]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(stems), [2 2])); assert(isequal(stems, [\"accel\", \"gc\"; \"plot\", \"cli\"]));" } },
    BuiltinExample { id: "legacy-char", title: "Extract row by row from a character matrix", program: "chars = char(\"Device<GPU>\", \"Planner<Fusion>\");\ntokens = extractBetween(chars, \"<\", \">\")", display_output: Some("tokens = ['GPU   '; 'Fusion']"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(ischar(tokens)); assert(isequal(size(tokens), [2 6])); assert(strcmp(tokens(1,:), 'GPU   ')); assert(strcmp(tokens(2,:), 'Fusion'));" } },
    BuiltinExample { id: "legacy-cell", title: "Preserve cell element text kinds", program: "C = {'<missing>', 'A<B>C'; \"Planner <Fusion>\", \"Device<GPU>\"};\nout = extractBetween(C, \"<\", \">\")", display_output: Some("out = {'missing', 'B'; \"Fusion\", \"GPU\"}"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(out), [2 2])); assert(ischar(out{1,1})); assert(strcmp(out{1,1}, 'missing')); assert(ischar(out{1,2})); assert(strcmp(out{1,2}, 'B')); assert(isequal(out{2,1}, \"Fusion\")); assert(isequal(out{2,2}, \"GPU\"));" } },
    BuiltinExample { id: "legacy-missing", title: "Propagate missing string values", program: "txt = [missing, \"Planner<GPU>\"];\ntokens = extractBetween(txt, \"<\", \">\")", display_output: Some("tokens = [<missing>, \"GPU\"]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(ismissing(tokens(1))); assert(isequal(tokens(2), \"GPU\"));" } },
    BuiltinExample { id: "missing-marker", title: "Observe an unmatched marker", program: "value = extractBetween(\"abc\", \"x\", \"c\")", display_output: Some("value = \"\""), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(value, \"\"));" } },
];

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/transform/extractbetween.rs") };

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("extractBetween"), slug: Some("extractbetween"),
    summary: "Extract substrings between boundary markers in MATLAB and RunMat.",
    description: "extractBetween(text, start, stop) returns text between two markers. Numeric positions accept single, double, and all eight integer classes as one-based structural controls.",
    keywords: &["extractBetween", "substring", "boundaries", "inclusive", "exclusive", "string array", "character vector"],
    related: &["replace", "split", "join", "contains", "strfind", "erase", "eraseBetween", "lower", "pad", "strcat", "strip", "strrep", "strtrim", "upper"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Which argument types does extractBetween accept?", answer: "The documented text input is a string array, character vector, or cell array of character vectors. Both boundaries are text/pattern markers or both are numeric positions; all eight integer classes are accepted as exact position controls. RunMat mode also admits row-wise character matrices and broader text cells." },
        BuiltinDocumentationFaq { question: "Can the start and end arguments mix text and numeric positions?", answer: "No. Both boundaries must use the same domain." },
        BuiltinDocumentationFaq { question: "What happens when a boundary is not found?", answer: "The current implementation returns empty extracted text. This behavior is recorded as RunMat behavior rather than independently verified MATLAB parity." },
        BuiltinDocumentationFaq { question: "How does Boundaries inclusive work with numeric positions?", answer: "Inclusive mode includes both indexed characters. Exclusive mode returns only the characters strictly between them." },
        BuiltinDocumentationFaq { question: "Does extractBetween support implicit expansion?", answer: "MATLAB-compatible mode permits scalar or same-size boundaries. General compatible broadcasting is a gated RunMat extension and cannot expand character or cell text output." },
        BuiltinDocumentationFaq { question: "Are GPU inputs supported?", answer: "There is no GPU text kernel. RunMat mode may gather a resident numeric position control; resident numeric text is rejected without provider access." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "replace", target: BuiltinDocumentationLinkTarget::Builtin("replace") },
        BuiltinDocumentationLink { label: "split", target: BuiltinDocumentationLinkTarget::Builtin("split") },
        BuiltinDocumentationLink { label: "join", target: BuiltinDocumentationLinkTarget::Builtin("join") },
        BuiltinDocumentationLink { label: "contains", target: BuiltinDocumentationLinkTarget::Builtin("contains") },
        BuiltinDocumentationLink { label: "strfind", target: BuiltinDocumentationLinkTarget::Builtin("strfind") },
        BuiltinDocumentationLink { label: "erase", target: BuiltinDocumentationLinkTarget::Builtin("erase") },
        BuiltinDocumentationLink { label: "eraseBetween", target: BuiltinDocumentationLinkTarget::Builtin("eraseBetween") },
        BuiltinDocumentationLink { label: "lower", target: BuiltinDocumentationLinkTarget::Builtin("lower") },
        BuiltinDocumentationLink { label: "pad", target: BuiltinDocumentationLinkTarget::Builtin("pad") },
        BuiltinDocumentationLink { label: "strcat", target: BuiltinDocumentationLinkTarget::Builtin("strcat") },
        BuiltinDocumentationLink { label: "strip", target: BuiltinDocumentationLinkTarget::Builtin("strip") },
        BuiltinDocumentationLink { label: "strrep", target: BuiltinDocumentationLinkTarget::Builtin("strrep") },
        BuiltinDocumentationLink { label: "strtrim", target: BuiltinDocumentationLinkTarget::Builtin("strtrim") },
        BuiltinDocumentationLink { label: "upper", target: BuiltinDocumentationLinkTarget::Builtin("upper") },
        BuiltinDocumentationLink { label: "MATLAB extractBetween", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/string.extractbetween.html") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "text extraction, shapes, and compatibility", location: "builtins::strings::transform::extractbetween::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "typed static facts", location: "crates/runmat-builtins/src/catalog/entries/strings/transform/extractbetween/inference.rs" },
    ], notes: &["Typed missing strings retain their presence through extraction. Broader pattern-array representation and full missing-marker parity remain incomplete."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
