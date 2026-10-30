use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Text and boundary semantics", paragraphs: &[
        "extractAfter(str,patOrPos) returns text after the first matching text or pattern marker, or after a one-based numeric position. The matched marker or indexed character is excluded. Nonscalar boundary controls must match the text-element shape.",
        "The current implementation handles the first marker match and preserves the text container kind. An unmatched text or pattern marker raises a boundary error. Typed missing subject elements propagate unchanged; missing boundary controls are rejected. Literal `<missing>` text remains ordinary text.",
        "MATLAB-compatible text inputs are string arrays, character vectors, and cell arrays of character vectors. RunMat mode additionally admits row-wise character matrices and cells containing string scalars or nested cells, subject to each identity's extension gate.",
        "Text and pattern markers and real numeric positions are distinct boundary forms. All eight fixed-width integer classes are accepted as exact structural position controls; nonintegral, zero, and unsupported controls raise errors.",
    ] },
    BuiltinDocumentationSection { heading: "Placement and effects", paragraphs: &[
        "Source text stays on the host. A GPU-resident numeric position control can gather only after its RunMat extension is admitted; numeric or resident source text rejects before provider access.",
        "This host text operation does not fuse or run a provider kernel. It may suspend when gathering an admitted position, can throw on malformed inputs or boundary errors, and does not use callbacks, workspace state, files, time, or randomness.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "legacy-scalar",
        title: "Extract after a marker",
        program: "part = extractAfter(\"alpha.beta\", \".\")",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(part, \"beta\"));",
        },
    },
    BuiltinExample {
        id: "numeric-position",
        title: "Extract after a one-based position",
        program: "part = extractAfter(\"RunMat\", 3)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(part, \"Mat\"));",
        },
    },
    BuiltinExample {
        id: "first-marker",
        title: "Use the first matching marker",
        program: "part = extractAfter(\"a:b:c\", \":\")",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(part, \"b:c\"));",
        },
    },
    BuiltinExample {
        id: "array",
        title: "Preserve a string array shape",
        program: "parts = extractAfter([\"a.txt\"; \"b.csv\"], \".\")",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source:
                "assert(isequal(size(parts), [2 1])); assert(isequal(parts, [\"txt\"; \"csv\"]));",
        },
    },
];
const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/transform/extractafter.rs") };

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("extractAfter"), slug: Some("extractafter"),
    summary: "Extract text after the first matching marker or a one-based position.",
    description: "extractAfter(str,patOrPos) returns text after the first matching text or pattern marker, or after a one-based numeric position. The matched marker or indexed character is excluded. Nonscalar boundary controls must match the text-element shape.",
    keywords: &["extractAfter", "text", "string array", "character vector", "boundary", "position", "extract"],
    related: &["string", "char", "extractBefore", "extractAfter", "extractBetween", "insertBefore", "insertAfter", "replaceBetween", "eraseBetween"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[
    BuiltinDocumentationFaq { question: "Which marker occurrence is used?", answer: "The first matching text or pattern marker in each text element." },
    BuiltinDocumentationFaq { question: "What if the marker is absent?", answer: "The current RunMat implementation raises a boundary error." },
    BuiltinDocumentationFaq { question: "Is the marker included in the output?", answer: "No. The marker or indexed character is excluded from the extracted text." },
    BuiltinDocumentationFaq { question: "Can positions vary across a string array?", answer: "Yes. Use a scalar position or a numeric array the same size as the text array." },
    BuiltinDocumentationFaq { question: "Does extraction preserve input type?", answer: "Yes for supported string, character, and cell text containers." },
    BuiltinDocumentationFaq { question: "Does extraction run on GPU?", answer: "No GPU text kernel exists. RunMat mode may gather an admitted resident numeric position; text remains on host." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") },
        BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") },
        BuiltinDocumentationLink { label: "MATLAB extractAfter", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/string.extractafter.html") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "boundary text and position semantics", location: "builtins::strings::transform::compat::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "typed output and diagnostics", location: "crates/runmat-builtins/src/catalog/entries/strings/transform/extractafter/inference.rs" },
    ], notes: &["Pattern arrays and full missing-value parity remain unverified. Compatibility extensions are declared individually and require RunMat mode."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
