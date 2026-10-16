use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Parse formatted values from text.",
        "Supports repeated literal/whitespace-aware scan formats for %f/%e/%g, %d/%i/%u/%o/%x, %s, %c, field widths, assignment suppression, optional size truncation, and requested outputs for count, errmsg, and nextindex.",
        "A format containing only %ld or %li conversions returns native int64 storage, and a format containing only %lu, %lo, or %lx conversions returns native uint64 storage. Mixed numeric formats return double.",
        "The optional size argument accepts single, double, and all eight integer classes as a scalar or two-element vector and affects output shape without converting typed dimensions through double.",
        "String and character scan specifiers return character code values in the numeric output array.",
    ] },
    BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &[
        "No GPU kernel is registered. Text and format must be host text; numeric/resident values in those roles reject before transfer. A resident numeric size control gathers through its owning provider before scanning.",
        "Not eligible for fusion; this builtin materializes host numeric results and metadata outputs.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "example-1", title: "sscanf parses two values and reports the first unconsumed character",
        program: "[A,count,errmsg,nextindex] = sscanf(\"1 2 x\", \"%f\")", display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(A, [1;2])); assert(count == 2); assert(strcmp(errmsg, '')); assert(nextindex == 5);" },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") },
    BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/compat.rs") },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("sscanf"), slug: Some("sscanf"),
    summary: "Parse formatted values from text.",
    description: "[A, count, errmsg, nextindex] = sscanf(text, format, size) - Parse formatted values from text.",
    keywords: &["sscanf", "strings", "core", "matlab", "compatibility"],
    related: &["string", "char"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: FAQS, links: LINKS, media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/compat.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Existing identity family tests", location: "builtins::strings::core::compat::tests" },
        ],
        notes: &["Source-only canonical cutover; original JSON retained until compiled field and executable-example conservation."],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
