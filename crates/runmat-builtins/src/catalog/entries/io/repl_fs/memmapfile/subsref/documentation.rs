use crate::*;

const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "selected-exact-integer",
    title: "Read one exact mapped UInt16 element",
    program: "m = memmapfile('samples.bin', 'Format', {'uint16', [2 1], 'samples'});\nx = m.Data.samples(2);\nassert(isequal(x, uint16(2)));\nassert(strcmp(class(x), 'uint16'));",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
        id: BuiltinExampleFixtureId { local_name: "mapped-method-read" },
        root: BuiltinFilesystemRoot::IsolatedWorkspace,
        entries: super::super::documentation::SAMPLE,
    }),
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isequal(x, uint16(2))); assert(strcmp(class(x), 'uint16'));",
    },
}];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("memmapfile.subsref"),
    slug: Some("memmapfile-subsref"),
    summary: "Read a property or selected bytes from a sealed memmapfile receiver.",
    description: "Object property and indexed Data access use the memmapfile class's subsref method. Opened-range providers read only selected byte ranges for indexed Data; whole Data returns a decoded host snapshot. A detached read-only receiver indexes its explicit host snapshot.",
    keywords: &["memmapfile", "subsref", "mapped data", "indexed read"],
    related: &["memmapfile", "memmapfile.subsasgn", "fread"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Forms and authority",
            paragraphs: &[
                "The runtime calls value = memmapfile.subsref(m, S) for ordinary m.Property and m.Data(indices) syntax. S is a standard object subscript structure; a class name alone does not authorize access without the receiver's sealed state.",
                "Filename, Writable, Offset, Format, and Repeat expose receiver controls. Data is decoded from the opened backing on access, preserving the declared numeric class and exact integer bits. Indexing a field or record uses fixed-size, column-major selection and does not read unrelated mapped bytes.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Execution and limits",
            paragraphs: &[
                "The selected filesystem provider and session owner remain captured across suspension. Provider-resident indices may materialize before a mapped read. The result is a host value; no GPU kernel or distributed transfer of the opened file handle is implied.",
                "Whole Data and codec decoding can allocate substantial host memory, and synchronous provider or codec work may delay cancellation. A file extent limits offsets, not host memory use.",
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
            BuiltinDocumentationLink { label: "Sealed method dispatch", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/compat/memmapfile/live.rs") },
            BuiltinDocumentationLink { label: "Indexed range planning and reads", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/compat/memmapfile/live/ranges/plan.rs") },
        ],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Live selected byte, field, owner and forged-class witnesses",
            location: "builtins::io::repl_fs::compat::memmapfile::live::tests",
        }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
