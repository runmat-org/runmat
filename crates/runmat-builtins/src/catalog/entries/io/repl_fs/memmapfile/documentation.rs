use crate::*;

const SAMPLE: &[BuiltinFilesystemEntry] = &[BuiltinFilesystemEntry::File {
    relative_path: "samples.bin",
    content: BuiltinFixtureContent::Bytes(&[1, 0, 2, 0]),
}];
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "typed-record",
    title: "Read exact UInt16 samples from a format record",
    program: "m = memmapfile('samples.bin', 'Format', {'uint16', [2 1], 'samples'});\nassert(isequal(m.Data.samples, uint16([1; 2])));\nassert(strcmp(class(m.Data.samples), 'uint16'));",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
        id: BuiltinExampleFixtureId {
            local_name: "typed-record",
        },
        root: BuiltinFilesystemRoot::IsolatedWorkspace,
        entries: SAMPLE,
    }),
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isequal(m.Data.samples, uint16([1; 2]))); assert(strcmp(class(m.Data.samples), 'uint16'));",
    },
}];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("memmapfile"),
    slug: Some("memmapfile"),
    summary: "Read binary records using a declared native numeric format.",
    description: "`memmapfile` reads file bytes and returns an object with Filename, Writable, Offset, Format, Repeat, and Data properties. Numeric payloads retain their declared class, including exact 64-bit integers. The current Data value is materialized in memory; Writable does not yet provide write-through mapping.",
    keywords: &["memmapfile", "binary", "format", "mapping"],
    related: &["fread", "fwrite", "fileattrib"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Forms and record geometry",
            paragraphs: &[
                "`m = memmapfile(filename)` reads UInt8 data. Name/Value pairs configure Writable, Offset, Format, and Repeat; incomplete or unknown pairs are errors.",
                "Format accepts a native type name or an n-by-3 cell table. Each table row gives a type, positive dimension vector, and unique field name. File records contain those fields in row order; Data is a structure array when more than one complete record is read.",
                "Supported numeric names are int8/int16/int32/int64, uint8/uint16/uint32/uint64, single, double, and char. Native integer storage and Float32 bit patterns are read without a Double conversion.",
                "Offset selects the starting byte. Repeat limits complete records. Checked record and Repeat geometry rejects overflow before reading the file.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Current limitations",
            paragraphs: &[
                "The current implementation reads the selected file into host memory instead of retaining a live memory-mapped backing. Writable is reported as a property but changes to Data are not written through to the file.",
                "Automatic resident structural controls may be downloaded through their provider. Explicit gpuArray structural controls require the RunMat extension; memmapfile does not run a GPU kernel.",
                "Large files and records may require substantial host memory. Resource-bounded admission and cancellation throughout decoding are not yet available.",
            ],
        },
    ],
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does setting Writable update the file when Data changes?",
            answer: "Not yet. Writable is retained as metadata, while Data is a host-memory copy.",
        },
        BuiltinDocumentationFaq {
            question: "Can Format describe records with different field types?",
            answer: "Yes. Use an n-by-3 cell table with one type, dimension vector, and unique field name per row.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "MATLAB memmapfile reference",
            target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/memmapfile.html"),
        },
        BuiltinDocumentationLink {
            label: "fread",
            target: BuiltinDocumentationLinkTarget::Builtin("fread"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "Runtime binding and filesystem capture",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/compat.rs"),
            },
            BuiltinDocumentationLink {
                label: "Format admission",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/compat/memmapfile.rs"),
            },
            BuiltinDocumentationLink {
                label: "Typed record decoding",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/compat/memmapfile/decode.rs"),
            },
        ],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Integer payload, format table, overflow and provider ownership source fixtures",
            location: "builtins::io::repl_fs::compat::tests",
        }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
