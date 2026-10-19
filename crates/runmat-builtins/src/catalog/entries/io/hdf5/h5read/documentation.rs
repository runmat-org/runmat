use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("h5read"),
    slug: Some("h5read"),
    summary: "Read dense numeric, logical, or string datasets from an HDF5 file.",
    description: "`h5read(filename, dataset)` reads a native HDF5 dataset. `h5read(filename, dataset, start, count)` and the five-input stride form read one-based hyperslabs.",
    keywords: &["h5read", "hdf5", "dataset", "file", "io"],
    related: &["h5write", "h5info"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["Dense floating-point, exact signed and unsigned integer, logical, and variable-length string datasets map to RunMat values. Nonsingleton array dimensions are preserved while HDF5 row-major buffers become RunMat column-major values; singleton results use scalar value forms. The result class and shape depend on external dataset metadata.", "Compound, reference, enum, fixed-array, and nested variable-length HDF5 types report `RunMat:hdf5:UnsupportedType`. Selection entries must be positive finite integers, with bounds checked against the opened dataset. GPU-resident numeric selection controls are gathered after host-known admission; output is host-resident. Browser execution rejects with `RunMat:hdf5:UnsupportedTarget`."] },
        BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &["Native HDF5 file access is synchronous host work. It terminates fusion and does not launch a GPU kernel. The active native filesystem path authority and HDF5 library must be available; a missing native route reports a typed error rather than falling back to an unrelated ambient path."] },
    ],
    examples: &[BuiltinExample {
        id: "native-h5read",
        title: "h5read with a deterministic native HDF5 file",
        program: "h5write('measurements.h5', '/group/A', [1 2; 3 4]);\nA = h5read('measurements.h5', '/group/A');\nA",
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
    links: &[BuiltinDocumentationLink { label: "h5write", target: BuiltinDocumentationLinkTarget::Builtin("h5write") },
        BuiltinDocumentationLink { label: "h5info", target: BuiltinDocumentationLinkTarget::Builtin("h5info") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "mod.rs", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/hdf5/mod.rs") },
            BuiltinDocumentationLink { label: "geometry.rs", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/hdf5/geometry.rs") },
            BuiltinDocumentationLink { label: "path.rs", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/hdf5/path.rs") },
            BuiltinDocumentationLink { label: "read.rs", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/hdf5/native/read.rs") },
        ],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Native HDF5 behavior", location: "builtins::io::hdf5::tests::h5write_and_h5read_round_trip_column_major_numeric" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
