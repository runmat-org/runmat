use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("hdf5read"),
    slug: Some("hdf5read"),
    summary: "Legacy alias for reading dense HDF5 datasets.",
    description: "`hdf5read(filename, dataset)` is the legacy spelling of `h5read` and supports the same dense types and optional start/count/stride selections.",
    keywords: &["hdf5read", "h5read", "hdf5", "dataset", "file", "io", "legacy"],
    related: &["h5read"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["It preserves exact integer classes, logical/string values, nonsingleton array dimensions, and HDF5-to-RunMat layout conversion; singleton results use scalar forms. Unsupported HDF5 types report `RunMat:hdf5:UnsupportedType`.", "The alias uses the same host materialization and native HDF5 route as `h5read`; browser execution reports `RunMat:hdf5:UnsupportedTarget`."] },
        BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &["Native HDF5 file access is synchronous host work. It terminates fusion and does not launch a GPU kernel. The active native filesystem path authority and HDF5 library must be available; a missing native route reports a typed error rather than falling back to an unrelated ambient path."] },
    ],
    examples: &[BuiltinExample {
        id: "native-hdf5read",
        title: "hdf5read with a deterministic native HDF5 file",
        program: "h5write('measurements.h5', '/A', [1 2; 3 4]);\nA = hdf5read('measurements.h5', '/A');\nA",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(A, [1 2; 3 4]));" },
    }],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Can this run in a browser?", answer: "The current HDF5 backend requires a native HDF5 library; browser calls return RunMat:hdf5:UnsupportedTarget." },
    ],
    links: &[BuiltinDocumentationLink { label: "h5read", target: BuiltinDocumentationLinkTarget::Builtin("h5read") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "mod.rs", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/hdf5/mod.rs") },
            BuiltinDocumentationLink { label: "geometry.rs", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/hdf5/geometry.rs") },
            BuiltinDocumentationLink { label: "path.rs", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/hdf5/path.rs") },
            BuiltinDocumentationLink { label: "read.rs", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/hdf5/native/read.rs") },
        ],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Native HDF5 behavior", location: "builtins::io::hdf5::alias_tests::legacy_read_write_and_info_share_exact_native_dataset_semantics" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
