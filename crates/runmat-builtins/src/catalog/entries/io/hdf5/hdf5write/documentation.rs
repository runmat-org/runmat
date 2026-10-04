use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("hdf5write"),
    slug: Some("hdf5write"),
    summary: "Legacy alias for writing dense HDF5 datasets.",
    description: "`hdf5write(filename, dataset, data)` is the legacy spelling of `h5write`; it supports the same whole-dataset and start/count/stride hyperslab forms.",
    keywords: &["hdf5write", "h5write", "hdf5", "dataset", "file", "io", "legacy"],
    related: &["h5write"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["Partial writes use native HDF5 hyperslabs. Dense real numeric (including exact integers), logical, string, and char values are admitted; sparse, complex, cell, struct, object, and handle values report `RunMat:hdf5:UnsupportedType`.", "The alias uses the same host materialization and native HDF5 route as `h5write`; browser execution reports `RunMat:hdf5:UnsupportedTarget`."] },
        BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &["Native HDF5 file access is synchronous host work. It terminates fusion and does not launch a GPU kernel. The active native filesystem path authority and HDF5 library must be available; a missing native route reports a typed error rather than falling back to an unrelated ambient path."] },
    ],
    examples: &[BuiltinExample {
        id: "native-hdf5write",
        title: "hdf5write with a deterministic native HDF5 file",
        program: "A = [1 2; 3 4];\nhdf5write('out.h5', '/A', A);\nB = h5read('out.h5', '/A');\nB",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(B, A));" },
    }],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Can this run in a browser?", answer: "The current HDF5 backend requires a native HDF5 library; browser calls return RunMat:hdf5:UnsupportedTarget." },
    ],
    links: &[BuiltinDocumentationLink { label: "h5write", target: BuiltinDocumentationLinkTarget::Builtin("h5write") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "mod.rs", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/hdf5/mod.rs") },
            BuiltinDocumentationLink { label: "geometry.rs", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/hdf5/geometry.rs") },
            BuiltinDocumentationLink { label: "path.rs", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/hdf5/path.rs") },
            BuiltinDocumentationLink { label: "write.rs", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/hdf5/native/write.rs") },
        ],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Native HDF5 behavior", location: "builtins::io::hdf5::alias_tests::legacy_read_write_and_info_share_exact_native_dataset_semantics" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
