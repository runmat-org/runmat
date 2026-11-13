use crate::*;

pub(super) const SAMPLE: &[BuiltinFilesystemEntry] = &[BuiltinFilesystemEntry::File {
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
}, BuiltinExample {
    id: "indexed-write-through",
    title: "Update one mapped record and observe the opened file",
    program: "m = memmapfile('samples.bin', 'Format', {'uint16', [2 1], 'samples'}, 'Writable', true);\nm.Data.samples(2) = uint16(7);\nn = memmapfile('samples.bin', 'Format', {'uint16', [2 1], 'samples'});\nassert(isequal(n.Data.samples, uint16([1; 7])));",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
        id: BuiltinExampleFixtureId { local_name: "indexed-write-through" },
        root: BuiltinFilesystemRoot::IsolatedWorkspace,
        entries: SAMPLE,
    }),
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isequal(n.Data.samples, uint16([1; 7])));",
    },
}];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("memmapfile"),
    slug: Some("memmapfile"),
    summary: "Read binary records using a declared native numeric format.",
    description: "`memmapfile` returns an object with Filename, Writable, Offset, Format, Repeat, and Data properties. Providers with opened-range support retain one captured file backing: reading Data decodes its current bytes, and Writable permits checked write-through assignment. Numeric payloads retain their declared class, including exact 64-bit integers.",
    keywords: &["memmapfile", "binary", "format", "mapping"],
    related: &["fread", "fwrite", "fileattrib"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Forms and record geometry",
            paragraphs: &[
                "`m = memmapfile(filename)` exposes UInt8 data. Name/Value pairs configure Writable, Offset, Format, and Repeat; incomplete or unknown pairs are errors.",
                "Format accepts a native type name or an n-by-3 cell table. Each table row gives a type, positive dimension vector, and unique field name. File records contain those fields in row order; Data is a structure array when more than one complete record is read.",
                "Supported numeric names are int8/int16/int32/int64, uint8/uint16/uint32/uint64, single, double, and char. Native integer storage and Float32 bit patterns are read without a Double conversion.",
                "The char format returns character Data. An unnamed cell Format retains its declared dimension vector in the returned Format property.",
                "Offset selects the starting byte. Repeat limits complete records; numeric Inf, the default, selects complete records through the end of the file. Repeat is queried and assigned as a numeric control, not the text 'Inf'. RunMat also retains its zero-repeat empty-map form. Checked record and Repeat geometry rejects overflow before reading the file. Counts above 2^53 are presented as exact UInt64 properties rather than rounded doubles.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Current limitations",
            paragraphs: &[
                "The native and in-memory filesystem providers support opened-range reads and writes. A selected provider without that capability can still return a detached read-only Data snapshot; a writable mapping request is rejected. A mapped Data access materializes a host value, and assigning a changed value writes its encoded bytes to the opened backing. A multi-chunk write is not an atomic file transaction.",
                "Automatic resident structural controls may be downloaded through their provider. Explicit gpuArray structural controls require the RunMat extension; memmapfile does not run a GPU kernel.",
                "Large files and records may require substantial host memory. The opened extent limits file offsets, not process memory use; synchronous provider and value-codec phases may still delay cancellation.",
            ],
        },
    ],
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does setting Writable update the file when Data changes?",
            answer: "Yes, when the selected provider supplies an opened writable backing. Assignment through Data encodes the value and writes to that backing. If the provider cannot supply opened ranges, writable construction fails rather than reporting a false mapping.",
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
            BuiltinDocumentationLink {
                label: "Sealed live receiver and class methods",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/compat/memmapfile/live.rs"),
            },
            BuiltinDocumentationLink {
                label: "Provider-owned opened-range interface",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-filesystem/src/mapped_backing.rs"),
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
