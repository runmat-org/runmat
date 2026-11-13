use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("h5write"),
    slug: Some("h5write"),
    summary: "Write dense numeric, logical, string, or char data to an HDF5 dataset.",
    description: "`h5write(filename, dataset, data)` creates a dense dataset (and intermediate groups) or overwrites an existing dataset of matching shape. Start/count and optional stride select a hyperslab in an existing dataset.",
    keywords: &["h5write", "hdf5", "dataset", "file", "io"],
    related: &["h5read", "h5writeatt"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["Double, single, exact signed/unsigned integer, logical, string, and char values are admitted; native numeric storage is preserved. Numeric, logical, and string array dimensions are preserved in HDF5 row-major storage. String values and character rows are written as UTF-8 variable-length strings. Partial writes require data shape to match `count`.", "Sparse, complex, cell, struct, object, and handle values report `RunMat:hdf5:UnsupportedType`. Known-invalid data and selection controls reject before native file access. Admitted GPU-resident numeric data is materialized to host; native HDF5 I/O has no GPU kernel or fusion route. Browser execution rejects with `RunMat:hdf5:UnsupportedTarget`."] },
        BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &["Native HDF5 file access is synchronous host work. It terminates fusion and does not launch a GPU kernel. The active native filesystem path authority and HDF5 library must be available; a missing native route reports a typed error rather than falling back to an unrelated ambient path."] },
    ],
    examples: &[BuiltinExample {
        id: "native-h5write",
        title: "h5write with a deterministic native HDF5 file",
        program: "h5write('out.h5', '/A', magic(3));\nA = h5read('out.h5', '/A');\nA",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(A, magic(3)));" },
    }],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Can this run in a browser?", answer: "The current HDF5 backend requires a native HDF5 library; browser calls return RunMat:hdf5:UnsupportedTarget." },
    ],
    links: &[BuiltinDocumentationLink { label: "h5read", target: BuiltinDocumentationLinkTarget::Builtin("h5read") },
        BuiltinDocumentationLink { label: "h5writeatt", target: BuiltinDocumentationLinkTarget::Builtin("h5writeatt") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "mod.rs", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/hdf5/mod.rs") },
            BuiltinDocumentationLink { label: "geometry.rs", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/hdf5/geometry.rs") },
            BuiltinDocumentationLink { label: "path.rs", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/hdf5/path.rs") },
            BuiltinDocumentationLink { label: "write.rs", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/hdf5/native/write.rs") },
        ],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Native HDF5 behavior", location: "builtins::io::hdf5::tests::h5write_and_h5read_round_trip_column_major_numeric" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
