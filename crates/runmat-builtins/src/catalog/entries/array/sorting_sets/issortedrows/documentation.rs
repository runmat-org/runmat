use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &["Supports the same row ordering arguments as `sortrows`, including column and direction arguments for supported inputs."],
    },
    BuiltinDocumentationSection {
        heading: "Types, placement and limitations",
        paragraphs: &["The result is a host logical scalar. Matrix and character rows use direct lexicographic checks; table rows use the table sort permutation. String arrays are not supported. GPU input is a declared RunMat extension and is gathered before checking."],
    },
    BuiltinDocumentationSection {
        heading: "Resident execution",
        paragraphs: &["These operations are eager fusion boundaries, not elementwise kernels or reductions, and do not broadcast. Provider availability does not imply a device-native sorting or set kernel: current providers may stage host work. An unsupported input can use exact host fallback before provider work starts. Errors after an attempt, including cancellation, are returned without repeating the operation on the host. Supported resident outputs return to their owning provider; predicates and all-indices cells have the host placement described above. Distributed execution is not supported."],
    },
];
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "example-1",

    title: "Check sorted rows",

    program: "tf = issortedrows([1 2; 2 3])",

    display_output: Some("tf = true"),

    compatibility: BuiltinExampleCompatibility::Matlab,

    harness: BuiltinExampleHarness::Portable,

    fixture: BuiltinExampleFixture::None,

    requirements: BuiltinExampleRequirements::NONE,

    verification: BuiltinExampleVerification::Assertions {
        source: "assert(tf && islogical(tf));",
    },
}];
const FAQS: &[BuiltinDocumentationFaq] = &[];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "sortrows",

        target: BuiltinDocumentationLinkTarget::Builtin("sortrows"),
    },
    BuiltinDocumentationLink {
        label: "MATLAB reference",

        target: BuiltinDocumentationLinkTarget::External(
            "https://www.mathworks.com/help/matlab/ref/issortedrows.html",
        ),
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("issortedrows"),
    slug: Some("issortedrows"),
    summary: "Determine whether matrix or table rows are sorted.",
    description: "`issortedrows(A)` returns true when `A` is already in the row order produced by `sortrows(A)`.",
    keywords: &["issortedrows", "sortrows", "rows", "table"],
    related: &["sortrows"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/array/sorting_sets/issortedrows.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/sorting_sets/issortedrows.rs"),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Runtime value, option, shape and error tests",
            location: "builtins::array::sorting_sets::issortedrows::tests",
        }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
