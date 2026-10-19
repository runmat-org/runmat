use crate::*;

const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "selected-write-through",
    title: "Write one exact mapped element without replacing other bytes",
    program: "m = memmapfile('samples.bin', 'Format', {'uint16', [2 1], 'samples'}, 'Writable', true);\nm.Data.samples(2) = uint16(7);\nn = memmapfile('samples.bin', 'Format', {'uint16', [2 1], 'samples'});\nassert(isequal(n.Data.samples, uint16([1; 7])));",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
        id: BuiltinExampleFixtureId { local_name: "mapped-method-write" },
        root: BuiltinFilesystemRoot::IsolatedWorkspace,
        entries: super::super::documentation::SAMPLE,
    }),
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isequal(n.Data.samples, uint16([1; 7])));",
    },
}];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("memmapfile.subsasgn"),
    slug: Some("memmapfile-subsasgn"),
    summary: "Update controls or selected bytes through a sealed writable memmapfile receiver.",
    description: "Ordinary assignment to m.Writable, m.Offset, m.Format, m.Repeat, or m.Data invokes memmapfile.subsasgn. Selected Data indexing uses the canonical fixed-shape index plan and writes only affected opened byte ranges. A mapped write may partly commit before a later error; it is not a transaction.",
    keywords: &["memmapfile", "subsasgn", "mapped write", "indexed assignment"],
    related: &["memmapfile", "memmapfile.subsref", "fwrite"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Assignment forms",
            paragraphs: &[
                "The runtime calls m = memmapfile.subsasgn(m, S, rhs) for property and indexed Data assignment. S is a standard object subscript structure. A writable receiver requires an opened backing from the selected filesystem provider; a detached snapshot is read-only.",
                "Writable, Offset, Repeat, and Format are receiver controls copied independently when the object is cloned. The opened backing remains shared. Repeat accepts a nonnegative count or numeric Inf; Format determines the exact type and record layout.",
                "Data assignment requires the declared numeric or character class and the selected shape. A fixed mapped field or record cannot grow into an adjacent byte range. Selected updates leave other fields and records untouched.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Effects and limits",
            paragraphs: &[
                "The receiver's selected provider, opened file identity, and session authority remain captured. A pending resident rhs or index may materialize before file effects; unsupported or invalid geometry rejects before writes.",
                "A multi-chunk update is not atomic. The source of a mapped write error reports whether the current chunk committed bytes or has indeterminate effects, while earlier successful chunks remain committed. Synchronous codec and provider phases may delay cancellation, and large assignments may use substantial host memory.",
                "A malformed standard S reports InvalidObjectSubscriptPath; a well-formed S naming an unsupported mapped property reports the memmapfile format diagnostic.",
            ],
        },
    ],
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink {
        label: "memmapfile",
        target: BuiltinDocumentationLinkTarget::Builtin("memmapfile"),
    }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Sealed assignment method", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/compat/memmapfile/live.rs") },
            BuiltinDocumentationLink { label: "Fixed-shape mapped range plan", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/compat/memmapfile/live/ranges/plan.rs") },
            BuiltinDocumentationLink { label: "Selected write effects", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/compat/memmapfile/live/ranges/effects.rs") },
        ],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Indexed byte/field preservation, fixed-shape rejection and provider ownership source fixtures",
            location: "builtins::io::repl_fs::compat::memmapfile::live::tests",
        }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
