use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Text and boundary semantics", paragraphs: &[
        "insertBefore(str,patOrPos,newText) inserts newText before every matching text or pattern marker, or before a one-based numeric position. A scalar marker/replacement applies to each element; nonscalar controls must match the text-element shape.",
        "The current implementation supports scalar or same-size boundary/replacement arrays, first-class pattern objects as scalar markers, and host text. An absent marker or out-of-range numeric position raises a boundary error. Typed missing subject elements propagate unchanged; missing boundary or replacement controls are rejected. Literal `<missing>` text remains ordinary text.",
        "MATLAB-compatible text inputs are string arrays, character vectors, and cell arrays of character vectors. RunMat mode additionally admits row-wise character matrices and cells containing string scalars or nested cells, subject to each identity's extension gate.",
        "Text and pattern markers and real numeric positions are distinct boundary forms. All eight fixed-width integer classes are accepted as exact structural position controls; nonintegral, zero, and unsupported controls raise errors.",
    ] },
    BuiltinDocumentationSection { heading: "Placement and effects", paragraphs: &[
        "Source and insertion values are host text. A GPU-resident position control can gather only after its RunMat extension is admitted; numeric or resident source/replacement text rejects before provider access.",
        "This host text operation does not fuse or run a provider kernel. It may suspend when gathering an admitted position, can throw on malformed inputs or boundary errors, and does not use callbacks, workspace state, files, time, or randomness.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "legacy-position", title: "Insert at the beginning", program: "value = insertBefore(\"run\", 1, \"pre\")", display_output: Some("value = \"prerun\""), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(value, \"prerun\"));" } },
    BuiltinExample { id: "legacy-scalar", title: "Insert before every text match", program: "part = insertBefore(\"bread cheese wine\", \" \", \",\")", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(part, \"bread, cheese, wine\"));" } },
    BuiltinExample { id: "numeric-position", title: "Insert before a one-based position", program: "part = insertBefore(\"James Maxwell\", 7, \"Clerk \")", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(part, \"James Clerk Maxwell\"));" } },
    BuiltinExample { id: "array", title: "Insert elementwise in a string array", program: "parts = insertBefore([\"a.txt\"; \"b.csv\"], \".\", [\"-x\"; \"-y\"])", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(parts), [2 1])); assert(isequal(parts, [\"a-x.txt\"; \"b-y.csv\"]));" } },
    BuiltinExample { id: "character", title: "Preserve a character vector", program: "part = insertBefore('cat dog', ' dog', ',')", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(ischar(part)); assert(strcmp(part, 'cat, dog'));" } },
];
const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/transform/insertbefore.rs") };

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("insertBefore"), slug: Some("insertbefore"),
    summary: "Insert text before each matching marker or a one-based position.",
    description: "insertBefore(str,patOrPos,newText) inserts newText before every matching text or pattern marker, or before a one-based numeric position. A scalar marker/replacement applies to each element; nonscalar controls must match the text-element shape.",
    keywords: &["insertBefore", "text", "string array", "character vector", "boundary", "position", "insert"],
    related: &["string", "char", "extractBefore", "extractAfter", "extractBetween", "insertBefore", "insertAfter", "replaceBetween", "eraseBetween"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[
    BuiltinDocumentationFaq { question: "Does insertion affect every marker match?", answer: "Yes. A text or pattern marker inserts newText at every match in each text element. A numeric position inserts once." },
    BuiltinDocumentationFaq { question: "What if the marker is absent?", answer: "The current RunMat implementation raises a boundary error." },
    BuiltinDocumentationFaq { question: "Can replacement text vary by element?", answer: "Yes. newText may be scalar or have the same shape as the source text elements." },
    BuiltinDocumentationFaq { question: "Does insertion preserve input type?", answer: "Yes for supported host string, character, and cell text containers." },
    BuiltinDocumentationFaq { question: "Can numeric arrays be source or replacement text?", answer: "No. Integer arrays are valid only as exact one-based position controls." },
    BuiltinDocumentationFaq { question: "Does insertion run on GPU?", answer: "No GPU text kernel exists. RunMat mode may gather an admitted resident numeric position; text remains on host." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") },
        BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") },
        BuiltinDocumentationLink { label: "MATLAB insertBefore", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/string.insertbefore.html") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "boundary text and position semantics", location: "builtins::strings::transform::compat::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "typed output and diagnostics", location: "crates/runmat-builtins/src/catalog/entries/strings/transform/insertbefore/inference.rs" },
    ], notes: &["Pattern arrays and full missing-value parity remain unverified. Compatibility extensions are declared individually and require RunMat mode."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
